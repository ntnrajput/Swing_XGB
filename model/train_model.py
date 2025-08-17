# model/train_model.py

import sys
import os
import joblib
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
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

def train_model(df_all, available_features=None, model_params=None):
    """
    Enhanced trading model with XGBoost, hyperparameter tuning, and volume analysis
    """

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

        num_cols = df_all.select_dtypes(include=[np.number]).columns
        df_all[num_cols] = df_all[num_cols].replace([np.inf, -np.inf], np.nan)
        df_all = df_all.dropna(subset=available_features + ['target_hit']).reset_index(drop=True)

        final_rows = len(df_all)
        print(f"✅ Dropped {initial_rows - final_rows} rows due to NaN/Inf values.")
        print("📊 Final data shape:", df_all.shape)

        X = df_all[available_features]
        y = df_all['target_hit']

        # Balance classes
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
        print("After balancing classes:", X.shape, "Class distribution:", y.value_counts())

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Feature selection with XGBoost importance
        xgb_fs = xgb.XGBClassifier(
            n_estimators=200,
            random_state=42,
            tree_method="hist"  # change to "gpu_hist" if GPU build available
        )
        xgb_fs.fit(X_scaled, y)
        selector = SelectFromModel(xgb_fs, threshold="median", prefit=True)
        X_selected = selector.transform(X_scaled)
        selected_mask = selector.get_support()
        selected_features = [f for f, keep in zip(available_features, selected_mask) if keep]
        print(f" Selected Top {len(selected_features)} Features: {selected_features}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, stratify=y, test_size=0.2, random_state=42
        )
        logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")

        # Hyperparameter tuning
        if model_params is None:
            model_params = {
                'n_estimators': [200],
                'max_depth': [6, 10],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }

        print("🔍 Performing hyperparameter tuning with XGBoost...")
        xgb_model = xgb.XGBClassifier(
            random_state=42,
            tree_method="hist",  # change to "gpu_hist" if GPU build available
            eval_metric="logloss",
            use_label_encoder=False
        )

        grid_search = GridSearchCV(xgb_model, model_params, cv=3, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")

        # Evaluate
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]

        print("\n Enhanced Model Evaluation:")
        print(classification_report(y_test, y_pred))
        print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 15 Most Important Features:")
        print(feature_importance.head(15))

        try:
            plt.figure(figsize=(12, 10))
            sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
            plt.title('XGBoost Feature Importance with Volume Analysis')
            plt.tight_layout()
            plt.savefig('feature_importance_plot.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("📊 Feature importance plot saved as 'feature_importance_plot.png'")
        except Exception as e:
            print(f"⚠️ Could not create feature importance plot: {e}")

        # Metrics
        train_pred = best_model.predict(X_train)
        train_accuracy = np.mean(train_pred == y_train)
        test_accuracy = np.mean(y_pred == y_test)
        overfitting_gap = train_accuracy - test_accuracy
        win_rate = test_accuracy * 100

        metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'overfitting_gap': overfitting_gap,
            'win_rate': win_rate,
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }

        logger.info(f"Model metrics: {metrics}")

        # Save pipeline
        full_pipeline = Pipeline([
            ('scaler', scaler),
            ('selector', selector),
            ('model', best_model)
        ])

        model_bundle = {
            'pipeline': full_pipeline,
            'feature_columns': selected_features,
            'all_features': available_features,
            'metrics': metrics,
            'best_params': grid_search.best_params_,
            'feature_importance': feature_importance
        }

        joblib.dump(model_bundle, MODEL_FILE)
        logger.info(f" Model pipeline saved to {MODEL_FILE}")

        print("XGBoost enhanced model saved!")

        return model_bundle

    except Exception as e:
        logger.error(f"Failed to train enhanced model: {e}", exc_info=True)
        raise
