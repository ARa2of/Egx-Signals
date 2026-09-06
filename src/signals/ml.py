"""
ML Price Prediction for EGX Signals — V2
Ensemble (XGBoost + LightGBM) quantile regression + direction classification
Richer features, rolling walk-forward, optional Optuna tuning
"""
import warnings
import logging
import random
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, accuracy_score
from typing import Dict, Optional, List
from src.config import load_params

warnings.filterwarnings("ignore")

# ── Reproducibility: seed all RNGs ──
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

log = logging.getLogger(__name__)

params = load_params()
ml_cfg = params["ml"]
HORIZON = ml_cfg["horizon_days"]
TRAIN_RATIO = ml_cfg["train_ratio"]
MIN_ROWS = ml_cfg["min_rows"]
QUANTILES = ml_cfg["quantiles"]
XGB_PARAMS = ml_cfg["xgb_params"]
LGBM_PARAMS = ml_cfg["lgbm_params"]
ENSEMBLE_W = ml_cfg.get("ensemble_weights", {"xgb": 0.5, "lgbm": 0.5})
WF_CFG = ml_cfg.get("walk_forward", {"n_splits": 5, "test_ratio": 0.20})
OPTUNA_CFG = ml_cfg.get("optuna", {"enabled": False, "n_trials": 30, "timeout": 120})
MAX_PEERS = ml_cfg.get("max_peers", 4)
SECTOR_GROUPS_FILE = ml_cfg.get("sector_groups_file", "")

# ─────────────────────────────────────────────────────────────
# Sector-Based Peer Selection
# ─────────────────────────────────────────────────────────────
_sector_cache = None

def _load_sector_groups() -> Dict[str, str]:
    global _sector_cache
    if _sector_cache is not None:
        return _sector_cache

    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    file_path = project_root / SECTOR_GROUPS_FILE

    if not file_path.exists():
        log.warning("Sector groups file not found: %s", file_path)
        _sector_cache = {}
        return _sector_cache

    try:
        df = pd.read_excel(file_path)
        _sector_cache = dict(zip(df["Ticker"], df["Sector"]))
        log.info("Loaded sector groups: %d tickers across %d sectors", len(_sector_cache), df["Sector"].nunique())
    except Exception as e:
        log.warning("Failed to load sector groups: %s", e)
        _sector_cache = {}

    return _sector_cache

def get_sector_peers(target: str, all_available: List[str], max_peers: int = None) -> List[str]:
    if max_peers is None:
        max_peers = MAX_PEERS

    sector_map = _load_sector_groups()
    if not sector_map:
        return all_available[:max_peers]

    target_sector = sector_map.get(target)
    if not target_sector:
        log.info("%s: No sector found, using first %d available tickers", target, max_peers)
        return all_available[:max_peers]

    peers = [t for t in all_available
             if t != target and sector_map.get(t) == target_sector and t in all_available]

    log.info("%s: Sector=%s, found %d peers, using %d", target, target_sector, len(peers), min(len(peers), max_peers))
    return peers[:max_peers]

# ─────────────────────────────────────────────────────────────
# Feature Engineering — V2 (richer)
# ─────────────────────────────────────────────────────────────
def _safe_return(series: pd.Series, periods: int) -> pd.Series:
    r = np.log(series / series.shift(periods))
    return r.clip(-0.5, 0.5)

def _normalize_series(series: pd.Series, window: int = 252) -> pd.Series:
    roll_min = series.rolling(window, min_periods=20).min()
    roll_max = series.rolling(window, min_periods=20).max()
    denom = (roll_max - roll_min).replace(0, np.nan)
    normed = (series - roll_min) / denom
    return normed.fillna(0.5)

def _garman_klass_vol(high: pd.Series, low: pd.Series, open_: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / open_) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return gk.rolling(window, min_periods=5).mean().apply(lambda x: np.sqrt(max(x, 0)))

def build_features(target_df: pd.DataFrame,
                   peer_close: pd.DataFrame,
                   target: str,
                   is_commodity_map: Dict = None) -> pd.DataFrame:
    feat = pd.DataFrame(index=target_df.index)
    close = target_df["Close"]
    is_commodity_map = is_commodity_map or {}

    # ── Lag returns ──
    for lag in [1, 2, 3, 5, 10]:
        feat[f"ret_{lag}d"] = _safe_return(close, lag)

    # ── SMA distance ──
    for ma in ["SMA_20", "SMA_50", "SMA_200"]:
        if ma in target_df.columns:
            feat[f"dist_{ma}"] = (close - target_df[ma]) / target_df[ma]

    # ── RSI ──
    if "RSI" in target_df.columns:
        feat["rsi"] = target_df["RSI"] / 100.0
        feat["rsi_ob"] = (target_df["RSI"] > 70).astype(float)
        feat["rsi_os"] = (target_df["RSI"] < 30).astype(float)

    # ── Bollinger Band % ──
    if "BB_pct" in target_df.columns:
        feat["bb_pct"] = target_df["BB_pct"].clip(-1, 2)

    # ── ATR ──
    if "ATR" in target_df.columns:
        feat["atr_pct"] = target_df["ATR"] / close

    # ── MACD ──
    if "MACD" in target_df.columns and "MACD_sig" in target_df.columns:
        feat["macd_cross"] = (target_df["MACD"] - target_df["MACD_sig"]) / close
        feat["macd_hist"] = target_df.get("MACD_hist", pd.Series(0, index=close.index)) / close

    # ── ADX ──
    if "ADX" in target_df.columns:
        feat["adx"] = target_df["ADX"] / 100.0
        feat["adx_strong"] = (target_df["ADX"] > 25).astype(float)

    # ── HL range ──
    if "High" in target_df.columns and "Low" in target_df.columns:
        feat["hl_range"] = (target_df["High"] - target_df["Low"]) / close

    # ── Momentum: ROC ──
    for period in [5, 10, 20]:
        feat[f"roc_{period}"] = (close / close.shift(period) - 1).clip(-0.5, 0.5)

    # ── Momentum: Stochastic %K ──
    if "High" in target_df.columns and "Low" in target_df.columns:
        for period in [14]:
            low_min = target_df["Low"].rolling(period).min()
            high_max = target_df["High"].rolling(period).max()
            feat[f"stoch_k_{period}"] = ((close - low_min) / (high_max - low_min + 1e-10)).fillna(0.5)

    # ── Volatility: Historical vol ──
    ret_1d = _safe_return(close, 1)
    for window in [10, 20]:
        feat[f"hvol_{window}"] = ret_1d.rolling(window, min_periods=5).std()

    # ── Volatility: Garman-Klass ──
    if all(c in target_df.columns for c in ["High", "Low", "Open", "Close"]):
        feat["gk_vol"] = _garman_klass_vol(target_df["High"], target_df["Low"], target_df["Open"], close)

    # ── Volume: OBV trend ──
    if "Volume" in target_df.columns:
        vol = target_df["Volume"].replace(0, np.nan)
        direction = np.sign(close.diff()).fillna(0)
        obv = (direction * vol).cumsum()
        feat["obv_slope_10"] = obv.rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0, raw=True
        )
        feat["obv_slope_20"] = obv.rolling(20).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0, raw=True
        )
        vol_ma20 = vol.rolling(20).mean()
        feat["vol_ratio"] = (vol / vol_ma20).clip(0, 5)

    # ── Distance to VWAP ──
    if "VWAP" in target_df.columns:
        feat["dist_vwap"] = (close - target_df["VWAP"]) / target_df["VWAP"]

    # ── Peer returns ──
    peer_tickers = [c for c in peer_close.columns if c != target]
    for peer in peer_tickers:
        s = peer_close[peer]
        if is_commodity_map.get(peer, False):
            s_norm = _normalize_series(s)
            feat[f"{peer}_ret1d"] = _safe_return(s_norm, 1)
            feat[f"{peer}_ret5d"] = _safe_return(s_norm, 5)
            feat[f"{peer}_ret10d"] = _safe_return(s_norm, 10)
            feat[f"{peer}_ret1d_raw"] = _safe_return(s, 1)
        else:
            feat[f"{peer}_ret1d"] = _safe_return(s, 1)
            feat[f"{peer}_ret5d"] = _safe_return(s, 5)
            feat[f"{peer}_ret10d"] = _safe_return(s, 10)

    # ── Sector-level features ──
    if peer_tickers:
        peer_df = peer_close[peer_tickers]
        peer_ret5 = pd.DataFrame(index=peer_df.index)
        for peer in peer_tickers:
            s = peer_df[peer]
            if is_commodity_map.get(peer, False):
                s_norm = _normalize_series(s)
                peer_ret5[peer] = _safe_return(s_norm, 5)
            else:
                peer_ret5[peer] = _safe_return(s, 5)

        feat["sector_ret5d"] = peer_ret5.mean(axis=1)

        def above_sma20(s):
            sma = s.rolling(20).mean()
            return (s > sma).astype(float)
        feat["sector_breadth"] = peer_df.apply(above_sma20).mean(axis=1)

        target_ret5 = _safe_return(close, 5)
        feat["rel_strength"] = target_ret5 - feat["sector_ret5d"]
        feat["sector_dispersion"] = peer_ret5.std(axis=1)

    # ── Calendar features ──
    idx = feat.index
    feat["day_of_week"] = idx.dayofweek / 4.0
    feat["month"] = idx.month / 12.0
    feat["days_to_month_end"] = (
        idx.to_period("M").to_timestamp("M") - idx
    ).days / 31.0

    return feat

def build_target(close: pd.Series, horizon: int = HORIZON) -> pd.Series:
    return np.log(close.shift(-horizon) / close).clip(-0.5, 0.5)

def build_direction(close: pd.Series, horizon: int = HORIZON) -> pd.Series:
    return (close.shift(-horizon) > close).astype(int)

# ─────────────────────────────────────────────────────────────
# Rolling Walk-Forward Validation
# ─────────────────────────────────────────────────────────────
def _train_quantile_models(X_tr, y_tr, X_te, y_te, model_type="xgb", params=None):
    results = {}
    preds = {}
    models = {}
    for name, q in QUANTILES.items():
        if model_type == "lgbm":
            model = LGBMRegressor(objective="quantile", alpha=q, **params)
        else:
            model = XGBRegressor(objective="reg:quantileerror", quantile_alpha=q, **params)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        results[name] = {"mae": float(mean_absolute_error(y_te, pred)), "model": model}
        preds[name] = pd.Series(pred, index=X_te.index)
        models[name] = model
    return results, preds, models

def walk_forward_eval(X: pd.DataFrame, y: pd.Series, train_ratio: float = TRAIN_RATIO) -> Dict:
    n_splits = WF_CFG.get("n_splits", 5)
    test_ratio = WF_CFG.get("test_ratio", 0.20)

    all_mae = {"xgb": {q: [] for q in QUANTILES}, "lgbm": {q: [] for q in QUANTILES}}

    total_test = int(len(X) * test_ratio)
    step = max(total_test // n_splits, 1)

    for i in range(n_splits):
        test_end = len(X) - i * step
        test_start = test_end - step
        if test_start <= 0:
            break
        train_end = test_start
        if train_end < MIN_ROWS:
            break

        X_tr, X_te = X.iloc[:train_end], X.iloc[test_start:test_end]
        y_tr, y_te = y.iloc[:train_end], y.iloc[test_start:test_end]

        for model_type, model_params in [("xgb", XGB_PARAMS), ("lgbm", LGBM_PARAMS)]:
            # Inject random_state into params to guarantee reproducibility
            mp = {**model_params, "random_state": RANDOM_SEED}
            if model_type == "lgbm":
                mp["verbosity"] = -1
            else:
                mp["tree_method"] = "hist"
                mp["verbosity"] = 0
            results, preds, _ = _train_quantile_models(X_tr, y_tr, X_te, y_te, model_type, mp)
            for q_name in QUANTILES:
                all_mae[model_type][q_name].append(results[q_name]["mae"])

    avg_mae = {}
    for model_type in ["xgb", "lgbm"]:
        avg_mae[model_type] = {}
        for q_name in QUANTILES:
            vals = all_mae[model_type][q_name]
            avg_mae[model_type][q_name] = float(np.mean(vals)) if vals else 0

    split = int(len(X) * train_ratio)
    X_tr_final, X_te_final = X.iloc[:split], X.iloc[split:]
    y_tr_final, y_te_final = y.iloc[:split], y.iloc[split:]

    final_results = {}
    final_preds = {}
    for model_type, model_params in [("xgb", XGB_PARAMS), ("lgbm", LGBM_PARAMS)]:
        mp = {**model_params, "random_state": RANDOM_SEED}
        if model_type == "lgbm":
            mp["verbosity"] = -1
        else:
            mp["tree_method"] = "hist"
            mp["verbosity"] = 0
        results, preds, _ = _train_quantile_models(X_tr_final, y_tr_final, X_te_final, y_te_final, model_type, mp)
        final_results[model_type] = results
        final_preds[model_type] = preds

    return {
        "avg_mae": avg_mae,
        "results": final_results,
        "y_test": y_te_final,
        "predictions": final_preds,
        "X_train": X_tr_final,
        "y_train": y_tr_final,
        "X_test": X_te_final,
        "split_date": X_te_final.index[0],
        "n_splits_actual": n_splits,
    }

# ─────────────────────────────────────────────────────────────
# Optuna Hyperparameter Tuning
# ─────────────────────────────────────────────────────────────
def _optuna_tune(X, y, model_type="xgb"):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    split = int(len(X) * 0.8)
    X_tr, X_val = X.iloc[:split], X.iloc[split:]
    y_tr, y_val = y.iloc[:split], y.iloc[split:]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "random_state": 42,
        }
        if model_type == "lgbm":
            params["verbosity"] = -1
            model = LGBMRegressor(objective="quantile", alpha=0.5, **params)
        else:
            params["tree_method"] = "hist"
            params["verbosity"] = 0
            model = XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **params)

        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        return mean_absolute_error(y_val, pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=OPTUNA_CFG.get("n_trials", 30),
                   timeout=OPTUNA_CFG.get("timeout", 120))

    best = study.best_params
    if model_type == "lgbm":
        best["random_state"] = 42
        best["verbosity"] = -1
    else:
        best["random_state"] = 42
        best["tree_method"] = "hist"
        best["verbosity"] = 0
    return best

# ─────────────────────────────────────────────────────────────
# Ensemble Training & Forecast
# ─────────────────────────────────────────────────────────────
def _train_ensemble(X, y, xgb_params=None, lgbm_params=None):
    xgb_params = xgb_params or XGB_PARAMS
    lgbm_params = lgbm_params or LGBM_PARAMS

    # Inject random_state for reproducibility
    xgb_params = {**xgb_params, "random_state": RANDOM_SEED}
    lgbm_params = {**lgbm_params, "random_state": RANDOM_SEED}

    models = {"xgb": {}, "lgbm": {}}
    for name, q in QUANTILES.items():
        xgb_m = XGBRegressor(objective="reg:quantileerror", quantile_alpha=q, **xgb_params)
        xgb_m.fit(X, y)
        models["xgb"][name] = xgb_m

        lgbm_m = LGBMRegressor(objective="quantile", alpha=q, **lgbm_params)
        lgbm_m.fit(X, y)
        models["lgbm"][name] = lgbm_m

    return models

def _ensemble_predict(models, last_row):
    pred_returns = {}
    w_xgb = ENSEMBLE_W.get("xgb", 0.5)
    w_lgbm = ENSEMBLE_W.get("lgbm", 0.5)

    for name in QUANTILES:
        xgb_pred = float(models["xgb"][name].predict(last_row)[0])
        lgbm_pred = float(models["lgbm"][name].predict(last_row)[0])
        ensemble_pred = w_xgb * xgb_pred + w_lgbm * lgbm_pred
        pred_returns[name] = {
            "xgb": xgb_pred,
            "lgbm": lgbm_pred,
            "ensemble": ensemble_pred,
        }

    return pred_returns

def _feature_importance(models, feature_names):
    xgb_imp = pd.Series(models["xgb"]["Medium"].feature_importances_, index=feature_names)
    lgbm_imp = pd.Series(models["lgbm"]["Medium"].feature_importances_, index=feature_names)
    combined = (xgb_imp + lgbm_imp) / 2
    return combined.sort_values(ascending=False).reset_index().rename(columns={"index": "Feature", 0: "Importance"})

def train_and_forecast(X: pd.DataFrame, y: pd.Series, last_row: pd.DataFrame,
                       last_price: float, xgb_params=None, lgbm_params=None) -> Dict:
    models = _train_ensemble(X, y, xgb_params, lgbm_params)
    pred_data = _ensemble_predict(models, last_row)

    forecast = {}
    for name in QUANTILES:
        ret = pred_data[name]["ensemble"]
        price = last_price * np.exp(ret)
        forecast[name] = {
            "return_pct": ret * 100,
            "price": round(price, 3),
            "xgb_return": pred_data[name]["xgb"] * 100,
            "lgbm_return": pred_data[name]["lgbm"] * 100,
        }

    imp_df = _feature_importance(models, X.columns)

    return {
        "forecast": forecast,
        "models": models,
        "importances": imp_df,
        "pred_returns": pred_data,
    }

# ─────────────────────────────────────────────────────────────
# Master Pipeline
# ─────────────────────────────────────────────────────────────
def run_ml_pipeline(target_df_enriched: pd.DataFrame,
                    peer_close: pd.DataFrame,
                    target: str,
                    is_commodity_map: Dict = None,
                    tv_close: float = None) -> Dict:
    try:
        features = build_features(target_df_enriched, peer_close, target, is_commodity_map)
        target_y = build_target(target_df_enriched["Close"])
        target_dir = build_direction(target_df_enriched["Close"])

        common = features.index.intersection(target_y.dropna().index).intersection(target_dir.dropna().index)
        X = features.loc[common].copy()
        y = target_y.loc[common].copy()
        y_dir = target_dir.loc[common].copy()

        mask = X.notna().all(axis=1)
        X, y, y_dir = X[mask], y[mask], y_dir[mask]

        if len(X) < MIN_ROWS:
            return {"error": f"Not enough data to train ({len(X)} rows, need {MIN_ROWS})."}

        scaler = RobustScaler()
        scaler.fit(X)
        X_scaled = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)

        # Optional Optuna tuning
        xgb_params_tuned = None
        lgbm_params_tuned = None
        if OPTUNA_CFG.get("enabled", False):
            log.info("%s: Running Optuna tuning...", target)
            try:
                xgb_params_tuned = _optuna_tune(X_scaled, y, "xgb")
                lgbm_params_tuned = _optuna_tune(X_scaled, y, "lgbm")
                log.info("%s: Optuna tuning complete", target)
            except Exception as e:
                log.warning("%s: Optuna failed, using defaults: %s", target, e)

        # Walk-forward evaluation
        eval_result = walk_forward_eval(X_scaled, y)

        # Direction classification accuracy
        split = int(len(X_scaled) * TRAIN_RATIO)
        X_tr_dir, X_te_dir = X_scaled.iloc[:split], X_scaled.iloc[split:]
        y_tr_dir, y_te_dir = y_dir.iloc[:split], y_dir.iloc[split:]

        dir_acc = {}
        for model_type, model_params in [("xgb", xgb_params_tuned or XGB_PARAMS),
                                          ("lgbm", lgbm_params_tuned or LGBM_PARAMS)]:
            mp = {**model_params, "random_state": RANDOM_SEED}
            if model_type == "lgbm":
                mp["verbosity"] = -1
                dir_model = LGBMRegressor(objective="binary", **mp)
            else:
                mp["tree_method"] = "hist"
                mp["verbosity"] = 0
                dir_model = XGBRegressor(objective="binary:logistic", **mp)
            dir_model.fit(X_tr_dir, y_tr_dir)
            dir_pred = (dir_model.predict(X_te_dir) > 0.5).astype(int)
            dir_acc[model_type] = float(accuracy_score(y_te_dir, dir_pred))

        # Forecast
        raw_close = target_df_enriched["Close"].dropna()
        last_price = float(raw_close.iloc[-1])
        last_date = raw_close.index[-1]
        last_price_source = "yfinance"

        # Override with TradingView real-time close if available
        if tv_close is not None and tv_close > 0:
            last_price = float(tv_close)
            last_price_source = "TradingView"

        features_full = build_features(target_df_enriched, peer_close, target, is_commodity_map)
        feat_mask = features_full.notna().all(axis=1)
        features_full_clean = features_full[feat_mask]

        if features_full_clean.empty:
            return {"error": "Could not build a complete feature row for forecasting."}

        last_feat_row = pd.DataFrame(
            scaler.transform(features_full_clean.iloc[[-1]]),
            columns=features_full_clean.columns,
        )

        forecast_result = train_and_forecast(
            X_scaled, y, last_feat_row, last_price,
            xgb_params_tuned, lgbm_params_tuned
        )

        next_bdays = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=HORIZON)
        forecast_start = next_bdays[0]
        forecast_end = next_bdays[-1]

        return {
            "features": X,
            "eval": eval_result,
            "forecast": forecast_result["forecast"],
            "importances": forecast_result["importances"],
            "pred_returns": forecast_result["pred_returns"],
            "last_price": last_price,
            "last_price_source": last_price_source,
            "last_date": last_date,
            "forecast_start": forecast_start,
            "forecast_end": forecast_end,
            "n_features": len(X.columns),
            "n_train_rows": int(len(X) * TRAIN_RATIO),
            "n_total_rows": len(X),
            "scaler": scaler,
            "direction_accuracy": dir_acc,
            "ensemble_weights": ENSEMBLE_W,
            "tuned": xgb_params_tuned is not None,
        }

    except Exception as e:
        return {"error": str(e)}

# ─────────────────────────────────────────────────────────────
# ML Conviction Scoring — V2 (ensemble-aware)
# ─────────────────────────────────────────────────────────────
def compute_ml_conviction(forecast: Dict, last_price: float,
                          mae_pct: float = None, direction_accuracy: Dict = None,
                          ensemble_weights: Dict = None) -> Dict:
    med_price = forecast.get("Medium", {}).get("price", last_price)
    high_price = forecast.get("High", {}).get("price", last_price)
    low_price = forecast.get("Low", {}).get("price", last_price)

    upside_pct = (med_price - last_price) / last_price * 100
    cone_width = high_price - low_price
    cone_pct = cone_width / last_price * 100
    up_from_now = high_price - last_price
    down_from_now = last_price - low_price
    asymmetry = up_from_now / max(down_from_now, 0.0001)

    score = 0

    # Upside magnitude (35 pts)
    if upside_pct >= 5.0: score += 35
    elif upside_pct >= 3.0: score += 28
    elif upside_pct >= 1.5: score += 20
    elif upside_pct >= 0.5: score += 10
    elif upside_pct > 0: score += 3

    # Cone tightness (25 pts)
    if cone_pct <= 3: score += 25
    elif cone_pct <= 6: score += 18
    elif cone_pct <= 10: score += 10
    elif cone_pct <= 15: score += 5

    # Asymmetry (20 pts)
    if asymmetry >= 3.0: score += 20
    elif asymmetry >= 2.0: score += 14
    elif asymmetry >= 1.5: score += 8
    elif asymmetry >= 1.0: score += 3

    # MAE penalty (up to -20 pts)
    if mae_pct is not None and abs(upside_pct) > 0:
        mae_ratio = mae_pct / max(abs(upside_pct), 0.01)
        if mae_ratio > 2.0: score -= 20
        elif mae_ratio > 1.5: score -= 12
        elif mae_ratio > 1.0: score -= 6

    # Direction accuracy bonus (up to +20 pts)
    if direction_accuracy:
        avg_dir_acc = np.mean(list(direction_accuracy.values()))
        if avg_dir_acc >= 0.60: score += 20
        elif avg_dir_acc >= 0.55: score += 12
        elif avg_dir_acc >= 0.52: score += 6

    # Ensemble agreement bonus (up to +10 pts)
    if forecast.get("Medium"):
        xgb_ret = forecast["Medium"].get("xgb_return", 0)
        lgbm_ret = forecast["Medium"].get("lgbm_return", 0)
        if xgb_ret > 0 and lgbm_ret > 0:
            score += 10
        elif (xgb_ret > 0) != (lgbm_ret > 0):
            score -= 5

    score = min(100, max(0, score))

    if score >= 70:
        label = "High Conviction [GREEN]"
        rec = "Ensemble strongly agrees - use full position size."
    elif score >= 45:
        label = "Moderate Conviction [YELLOW]"
        rec = "Ensemble mildly bullish - consider 50-75% of normal position size."
    elif score >= 25:
        label = "Low Conviction [RED]"
        rec = "Ensemble uncertain - reduce position size to 25-50% or wait for a clearer signal."
    else:
        label = "No Conviction [BLOCKED]"
        rec = "Ensemble sees little/no upside - avoid new longs or stay out entirely."

    return {
        "conviction_score": score,
        "conviction_label": label,
        "upside_pct": round(upside_pct, 2),
        "cone_pct": round(cone_pct, 2),
        "asymmetry": round(asymmetry, 2),
        "recommendation": rec,
        "direction_accuracy": direction_accuracy,
    }
