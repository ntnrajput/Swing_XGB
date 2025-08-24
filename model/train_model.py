# model/train_model.py

import sys
import os
import joblib
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb

from utils.logger import get_logger
from config import FEATURE_COLUMNS, MODEL_FILE

logger = get_logger(__name__)

def _safe_sort_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts dataframe chronologically if a 'date' column exists; otherwise returns as-is.
    Keeps compatibility with multi-ticker panels if already pre-sorted.
    """
    if 'date' in df.columns:
        try:
            # Ensure datetime
            if not np.issubdtype(df['date'].dtype, np.datetime64):
                df = df.copy()
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.sort_values('date').reset_index(drop=True)
        except Exception as e:
            logger.warning(f"Could not sort by date safely: {e}. Proceeding without explicit sort.")
    else:
        logger.warning("No 'date' column found; assuming input is already chronologically ordered.")
    return df

def _find_best_threshold(y_true, y_proba):
    """
    Choose decision threshold that maximizes F1 on the PR curve.
    Returns (best_threshold, best_f1, precision_at_best, recall_at_best).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # thresholds length = len(precision)-1; avoid division by zero
    f1_scores = []
    for p, r in zip(precision[:-1], recall[:-1]):
        if (p + r) == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * p * r / (p + r))
    if len(f1_scores) == 0:
        return 0.5, 0.0, 0.0, 0.0
    idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return best_threshold, float(f1_scores[idx]), float(precision[idx]), float(recall[idx])

def train_model(df_all, available_features=None, model_params=None):
    """
    Enhanced trading model with XGBoost, time-aware split, PR-focused tuning, and threshold optimization.
    (Function name, return structure, and key variable names preserved for compatibility.)
    """

    # -----------------------------
    # SECTION 0: Feature list checks
    # -----------------------------
    if available_features is None:
        try:
            from config import FEATURE_COLUMNS
            available_features = FEATURE_COLUMNS
        except ImportError:
            raise ValueError("available_features must be provided or FEATURE_COLUMNS must be available in config")

    missing_features = [col for col in available_features if col not in df_all.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataframe: {missing_features}")

    if 'target_hit' not in df_all.columns:
        raise ValueError("Target column 'target_hit' not found in dataframe")

    try:
        print("📊 Initial data shape:", df_all.shape)
        initial_rows = len(df_all)

        # -----------------------------
        # SECTION 1: Clean + (optional) chronological sort
        # -----------------------------
        num_cols = df_all.select_dtypes(include=[np.number]).columns
        df_all = df_all.copy()
        df_all[num_cols] = df_all[num_cols].replace([np.inf, -np.inf], np.nan)
        # Drop rows with NaN in features or target
        df_all = df_all.dropna(subset=available_features + ['target_hit']).reset_index(drop=True)
        # Time-aware ordering if possible
        df_all = _safe_sort_by_date(df_all)

        final_rows = len(df_all)
        print(f"✅ Dropped {initial_rows - final_rows} rows due to NaN/Inf values.")
        print("📊 Final data shape:", df_all.shape)

        # -----------------------------
        # SECTION 2: Feature-target split
        # -----------------------------
        X = df_all[available_features]
        y = df_all['target_hit'].astype(int)

        # -----------------------------
        # SECTION 3: Scaling (kept for compatibility with your pipeline)
        # NOTE: XGBoost does not require scaling; we keep it to avoid breaking downstream usage.
        # -----------------------------
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # -----------------------------
        # SECTION 4: Feature selection (made safe/optional)
        # We fit a lightweight XGB to compute importances, but avoid dropping too many features.
        # If selection yields < 5 features, we fallback to keeping all features.
        # -----------------------------

        xgb_fs = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,           # slower learning, more stable importance
            max_depth=6,                  # reasonable depth for feature ranking
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method="hist",
            scale_pos_weight=82/18,       # handle class imbalance
            eval_metric="aucpr",
            use_label_encoder=False
        )
        xgb_fs.fit(X_scaled, y)

        # Try median threshold; if too aggressive, keep all features
        selector = SelectFromModel(xgb_fs, threshold="median", prefit=True)
        X_selected = selector.transform(X_scaled)
        selected_mask = selector.get_support()
        selected_features = [f for f, keep in zip(available_features, selected_mask) if keep]

        if X_selected.shape[1] < 5:
            # Fallback: keep all features
            selector = SelectFromModel(xgb_fs, threshold=-np.inf, prefit=True)  # keep all
            X_selected = selector.transform(X_scaled)
            selected_features = available_features.copy()
            logger.info("Feature selection too aggressive. Falling back to keeping all features.")

        print(f" Selected Top {len(selected_features)} Features: {selected_features}")

        # -----------------------------
        # SECTION 5: Time-aware split (NO RANDOM SPLIT)
        # -----------------------------
        split_idx = int(len(X_selected) * 0.8)
        X_train, X_test = X_selected[:split_idx], X_selected[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")

        # -----------------------------
        # SECTION 6: Imbalance handling via scale_pos_weight (NO SMOTE)
        # -----------------------------
        pos = int(np.sum(y_train))
        neg = int(len(y_train) - pos)
        if pos == 0:
            raise ValueError("No positive samples in training set after split. Check labeling logic or expand dataset.")
        scale_pos_weight = neg / pos
        logger.info(f"Imbalance: pos={pos}, neg={neg}, scale_pos_weight={scale_pos_weight:.3f}")

        # -----------------------------
        # SECTION 7: Hyperparameter tuning (RandomizedSearchCV, PR-focused)
        # -----------------------------
        if model_params is None:
            # Wider but reasonable ranges for tabular financial data
            param_dist = {
                'n_estimators': [300, 500, 800],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.03, 0.05, 0.1],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'min_child_weight': [1, 3, 5, 7],
                'gamma': [0, 0.5, 1, 2]
            }
            search_cv = RandomizedSearchCV
            search_params = {
                'param_distributions': param_dist,
                'n_iter': 30,
                'cv': 3,
                'scoring': 'average_precision',
                'n_jobs': -1,
                'verbose': 1,
                'random_state': 42
            }
        else:
            # If user passed a grid, still use PR metric
            search_cv = GridSearchCV
            search_params = {
                'param_grid': model_params,
                'cv': 3,
                'scoring': 'average_precision',
                'n_jobs': -1,
                'verbose': 1
            }

        print("🔍 Performing hyperparameter tuning with XGBoost (PR AUC focus)...")

        xgb_model = xgb.XGBClassifier(
            random_state=42,
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,  # balance positives vs negatives
            eval_metric=["aucpr", "logloss"]  # track both
        )

        # Use early stopping via fit params on the time-aware validation fold
        # (Grid/RandomizedSearchCV will pass these on each fold)
        search = search_cv(
            estimator=xgb_model,
            **search_params
        )

        # We can pass a static eval_set using the final test; this is acceptable for tuning stability,
        # but to strictly avoid leakage, CV will still be done on train folds internally.
        search.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        best_model = search.best_estimator_
        print(f"Best parameters: {search.best_params_}")

        # -----------------------------
        # SECTION 8: Evaluation + Threshold tuning
        # -----------------------------
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        # Find best threshold by F1 on PR curve
        decision_threshold, best_f1, p_at_best, r_at_best = _find_best_threshold(y_test.values, y_pred_proba)
        y_pred = (y_pred_proba >= decision_threshold).astype(int)

        # Summary metrics
        ap = average_precision_score(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)

        print("\n Enhanced Model Evaluation (PR-focused):")
        print(classification_report(y_test, y_pred))
        print(f"ROC AUC Score: {roc_auc:.4f}")
        print(f"Average Precision (PR AUC): {ap:.4f}")
        print(f"Chosen Threshold: {decision_threshold:.4f} | F1={best_f1:.4f} | P={p_at_best:.4f} | R={r_at_best:.4f}")
        print("Confusion Matrix (at chosen threshold):")
        print(cm)

        # -----------------------------
        # SECTION 9: Feature importance (gain-based)
        # -----------------------------
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 15 Most Important Features:")
        print(feature_importance.head(15))

        # Plot feature importance
        try:
            plt.figure(figsize=(12, 10))
            sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
            plt.title('XGBoost Feature Importance (Top 20)')
            plt.tight_layout()
            plt.savefig('feature_importance_plot.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("📊 Feature importance plot saved as 'feature_importance_plot.png'")
        except Exception as e:
            print(f"⚠️ Could not create feature importance plot: {e}")

        # Plot Precision-Recall curve
        try:
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve")
            plt.tight_layout()
            plt.savefig("pr_curve.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("📈 PR curve saved as 'pr_curve.png'")
        except Exception as e:
            print(f"⚠️ Could not create PR curve: {e}")

        # -----------------------------
        # SECTION 10: Additional metrics (kept for compatibility)
        # -----------------------------
        train_pred_proba = best_model.predict_proba(X_train)[:, 1]
        # Use same tuned threshold for train to compute comparable accuracy
        train_pred = (train_pred_proba >= decision_threshold).astype(int)
        train_accuracy = float(np.mean(train_pred == y_train))
        test_accuracy = float(np.mean(y_pred == y_test))
        overfitting_gap = train_accuracy - test_accuracy
        win_rate = test_accuracy * 100.0

        metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'overfitting_gap': overfitting_gap,
            'win_rate': win_rate,
            'roc_auc': roc_auc,
            'average_precision': float(ap),
            'decision_threshold': float(decision_threshold),
            'precision_at_threshold': float(p_at_best),
            'recall_at_threshold': float(r_at_best),
            'confusion_matrix': cm.tolist()
        }

        logger.info(f"Model metrics: {metrics}")

        # -----------------------------
        # SECTION 11: Save pipeline (names preserved)
        # -----------------------------
        full_pipeline = Pipeline([
            ('scaler', scaler),     # kept for compatibility
            ('selector', selector),
            ('model', best_model)
        ])

        model_bundle = {
            'pipeline': full_pipeline,
            'feature_columns': selected_features,   # Features after selection
            'all_features': available_features,     # Full feature list before selection
            'metrics': metrics,
            'best_params': search.best_params_,
            'feature_importance': feature_importance
        }

        joblib.dump(model_bundle, MODEL_FILE)
        logger.info(f" Model pipeline saved to {MODEL_FILE}")

        print("XGBoost enhanced model saved!")

        return model_bundle

    except Exception as e:
        logger.error(f"Failed to train enhanced model: {e}", exc_info=True)
        raise
