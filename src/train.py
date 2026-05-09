import pandas as pd
import numpy as np
import joblib
import optuna
import warnings
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss, f1_score
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROC = Path("data/processed")
MODELS = Path("data/models")

FEATURE_COLS = [
    "SRS_DIFF", "ORTG_DIFF", "DRTG_DIFF",
    "NET_RTG_DIFF", "TS_DIFF", "TOV_DIFF", "OREB_DIFF",
    "HOME_SRS", "HOME_ORTG", "HOME_DRTG", "HOME_TS", "HOME_TOV", "HOME_OREB",
    "AWAY_SRS", "AWAY_ORTG", "AWAY_DRTG", "AWAY_TS", "AWAY_TOV", "AWAY_OREB",
]


def load_data():
    df = pd.read_parquet(PROC / "features.parquet")
    df.sort_values("GAME_DATE", inplace=True)
    X = df[FEATURE_COLS].fillna(0).values
    y = df["HOME_WIN"].values
    return X, y, df


def tune_xgb(X, y, n_trials=40):
    tscv = TimeSeriesSplit(n_splits=5)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("lr", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X, y, cv=tscv, scoring="roc_auc")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print(f"Best XGB AUC: {study.best_value:.4f}")
    return study.best_params


def tune_lgbm(X, y, n_trials=40):
    tscv = TimeSeriesSplit(n_splits=5)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("lr", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model = LGBMClassifier(**params)
        scores = cross_val_score(model, X, y, cv=tscv, scoring="roc_auc")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print(f"Best LGBM AUC: {study.best_value:.4f}")
    return study.best_params


def build_stack(xgb_params, lgbm_params):
    from sklearn.model_selection import KFold
    xgb = XGBClassifier(
        **xgb_params,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    lgbm = LGBMClassifier(
        **lgbm_params,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    stack = StackingClassifier(
        estimators=[("xgb", xgb), ("lgbm", lgbm)],
        final_estimator=meta,
        cv=KFold(n_splits=5, shuffle=False),
        passthrough=False,
        n_jobs=1,
    )
    return stack

def evaluate(model, X, y):
    tscv = TimeSeriesSplit(n_splits=5)
    aucs, lls, f1s = [], [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        model.fit(X[train_idx], y[train_idx])
        probs = model.predict_proba(X[val_idx])[:, 1]
        preds = (probs >= 0.5).astype(int)
        aucs.append(roc_auc_score(y[val_idx], probs))
        lls.append(log_loss(y[val_idx], probs))
        f1s.append(f1_score(y[val_idx], preds))
        print(f"  Fold {fold}: AUC={aucs[-1]:.4f}  LogLoss={lls[-1]:.4f}  F1={f1s[-1]:.4f}")

    print(f"\nMean ROC-AUC : {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    print(f"Mean Log-loss: {np.mean(lls):.4f}")
    print(f"Mean F1      : {np.mean(f1s):.4f}")
    return np.mean(aucs)


if __name__ == "__main__":
    print("Loading data...")
    X, y, df = load_data()
    print(f"Dataset: {X.shape[0]} games, {X.shape[1]} features")
    print(f"Home win rate: {y.mean():.2%}\n")

    print("Tuning XGBoost...")
    xgb_params = tune_xgb(X, y, n_trials=40)

    print("\nTuning LightGBM...")
    lgbm_params = tune_lgbm(X, y, n_trials=40)

    print("\nBuilding stacking ensemble...")
    stack = build_stack(xgb_params, lgbm_params)

    print("Evaluating...")
    evaluate(stack, X, y)

    print("\nFitting final model on all data...")
    stack.fit(X, y)
    
    print("Saving model...")
    joblib.dump(stack, MODELS / "model.pkl")
    joblib.dump(xgb_params, MODELS / "xgb_params.pkl")
    joblib.dump(lgbm_params, MODELS / "lgbm_params.pkl")
    print("\nSaved data/models/model.pkl — training complete.")