"""
pipeline.py
===========
End-to-End Scikit-Learn Pipeline dengan MLflow Experiment Tracking.
Dataset: Student Placement (features + target terpisah)

Jalankan:
    pip install mlflow scikit-learn pandas hyperopt
    python pipeline.py

MLflow UI:
    mlflow ui          # buka http://localhost:5000
"""

import os
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (RobustScaler, LabelEncoder,
                                   OneHotEncoder, OrdinalEncoder)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score,
                             mean_absolute_error, mean_squared_error, r2_score)

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

mlflow.set_tracking_uri("file:./mlruns")

SEED = 123
np.random.seed(SEED)

# 1. DATA INGESTION

def load_data(features_path: str, target_path: str) -> pd.DataFrame:
    """Muat dan merge dataset fitur + target berdasarkan Student_ID."""
    df_feat   = pd.read_csv(features_path)
    df_target = pd.read_csv(target_path)
    df = pd.merge(df_feat, df_target, on='Student_ID')
    df = df.drop(columns=['Student_ID'])
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[load_data] Shape: {df.shape}")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Imputasi kolom kategorik dengan modus."""
    mode_val = df['extracurricular_involvement'].mode()[0]
    df['extracurricular_involvement'] = (
        df['extracurricular_involvement'].fillna(mode_val)
    )
    print(f"[handle_missing] Missing total: {df.isnull().sum().sum()}")
    return df


# 2. PREPROCESSING CONFIG

NUM_COLS = [
    'cgpa', 'tenth_percentage', 'twelfth_percentage',
    'study_hours_per_day', 'attendance_percentage',
    'projects_completed', 'internships_completed',
    'coding_skill_rating', 'communication_skill_rating',
    'aptitude_skill_rating', 'hackathons_participated',
    'certifications_count', 'sleep_hours', 'backlogs', 'stress_level'
]

# Binary cols — di-encode manual sebelum pipeline
BINARY_COLS  = ['gender', 'part_time_job', 'internet_access']
ORDINAL_COLS = ['family_income_level', 'city_tier', 'extracurricular_involvement']
NOMINAL_COLS = ['branch']

ORDINAL_CATEGORIES = [
    ['Low', 'Medium', 'High'],
    ['Tier 3', 'Tier 2', 'Tier 1'],
    ['None', 'Low', 'Medium', 'High'],
]


def encode_binary(df: pd.DataFrame):
    """LabelEncode kolom binary, kembalikan df + dict encoder."""
    bin_enc_dict = {}
    for col in BINARY_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        bin_enc_dict[col] = le
    return df, bin_enc_dict


def build_preprocessor():
    """Bangun ColumnTransformer untuk dipakai di dalam Pipeline."""
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  RobustScaler()),
    ])
    ord_transformer = OrdinalEncoder(
        categories=ORDINAL_CATEGORIES,
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )
    ohe_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer,  NUM_COLS),
        ('ord', ord_transformer,  ORDINAL_COLS),
        ('ohe', ohe_transformer,  NOMINAL_COLS),
    ], remainder='passthrough')

    return preprocessor


# 3. EVALUATION

def eval_classification(y_true, y_pred, y_proba=None) -> dict:
    metrics = {
        'accuracy' : accuracy_score(y_true, y_pred),
        'recall'   : recall_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'f1_score' : f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
    return metrics


def eval_regression(y_true, y_pred) -> dict:
    return {
        'mae' : mean_absolute_error(y_true, y_pred),
        'mse' : mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2'  : r2_score(y_true, y_pred),
    }


# 4. CLASSIFICATION PIPELINE

def run_classification(df: pd.DataFrame, exp_path: str):
    """Latih 3+ model klasifikasi, log ke MLflow, simpan model terbaik."""

    df = df.copy()
    df['placement_enc'] = (df['placement_status'] == 'Placed').astype(int)

    df, bin_enc_dict = encode_binary(df)

    feature_cols = [c for c in df.columns
                    if c not in ['placement_status', 'salary_lpa', 'placement_enc']]

    X = df[feature_cols]
    y = df['placement_enc']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    cw_arr  = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw_dict = dict(enumerate(cw_arr))

    preprocessor = build_preprocessor()

    models = {
        'LogisticRegression': LogisticRegression(
            class_weight=cw_dict, max_iter=1000, random_state=SEED),
        'DecisionTree': DecisionTreeClassifier(
            criterion='gini', max_depth=7,
            class_weight=cw_dict, random_state=SEED),
        'RandomForest': RandomForestClassifier(
            n_estimators=100, max_depth=10,
            class_weight=cw_dict, random_state=SEED, n_jobs=-1),
    }

    best_f1    = -1
    best_model = None
    best_name  = ''

    mlflow.set_experiment('placement_classification')

    for name, clf in models.items():
        pipe = Pipeline([('prep', preprocessor), ('clf', clf)])

        with mlflow.start_run(run_name=name):
            pipe.fit(X_train, y_train)
            y_pred  = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1]
            metrics = eval_classification(y_test, y_pred, y_proba)

            mlflow.log_params({'model': name, 'test_size': 0.2, 'seed': SEED})
            mlflow.log_metrics(metrics)

            sig = infer_signature(X_train, y_pred)
            mlflow.sklearn.log_model(pipe, artifact_path='model', signature=sig)

            print(f"[CLF] {name:25s} | F1={metrics['f1_score']:.4f} | AUC={metrics['auc_roc']:.4f}")

            if metrics['f1_score'] > best_f1:
                best_f1    = metrics['f1_score']
                best_model = pipe
                best_name  = name

    # ── HyperOpt tuning untuk RandomForest ──
    print("\n[CLF] HyperOpt tuning RandomForest ...")

    def objective_clf(params):
        clf = RandomForestClassifier(
            n_estimators     = int(params['n_estimators']),
            max_depth        = int(params['max_depth']),
            min_samples_leaf = int(params['min_samples_leaf']),
            criterion        = params['criterion'],
            class_weight     = cw_dict,
            random_state     = SEED, n_jobs=-1
        )
        pipe = Pipeline([('prep', preprocessor), ('clf', clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        return {'loss': 1 - recall_score(y_test, y_pred, zero_division=0),
                'status': STATUS_OK}

    space_clf = {
        'n_estimators'    : hp.choice('n_estimators',    [50, 100, 200]),
        'max_depth'       : hp.choice('max_depth',        [5, 10, 15, 20]),
        'min_samples_leaf': hp.choice('min_samples_leaf', [1, 3, 5]),
        'criterion'       : hp.choice('criterion',        ['gini', 'entropy']),
    }
    trials = Trials()
    best_idx = fmin(objective_clf, space=space_clf, algo=tpe.suggest,
                    max_evals=20, trials=trials,
                    rstate=np.random.default_rng(SEED))

    n_est_opts = [50, 100, 200]
    depth_opts = [5, 10, 15, 20]
    leaf_opts  = [1, 3, 5]
    crit_opts  = ['gini', 'entropy']

    tuned_clf = RandomForestClassifier(
        n_estimators     = n_est_opts[best_idx['n_estimators']],
        max_depth        = depth_opts[best_idx['max_depth']],
        min_samples_leaf = leaf_opts[best_idx['min_samples_leaf']],
        criterion        = crit_opts[best_idx['criterion']],
        class_weight     = cw_dict,
        random_state     = SEED, n_jobs=-1
    )
    tuned_pipe = Pipeline([('prep', preprocessor), ('clf', tuned_clf)])

    with mlflow.start_run(run_name='RF_HyperOpt_Tuned'):
        tuned_pipe.fit(X_train, y_train)
        y_pred  = tuned_pipe.predict(X_test)
        y_proba = tuned_pipe.predict_proba(X_test)[:, 1]
        metrics = eval_classification(y_test, y_pred, y_proba)

        mlflow.log_params({
            'model'           : 'RF_HyperOpt',
            'n_estimators'    : n_est_opts[best_idx['n_estimators']],
            'max_depth'       : depth_opts[best_idx['max_depth']],
            'min_samples_leaf': leaf_opts[best_idx['min_samples_leaf']],
            'criterion'       : crit_opts[best_idx['criterion']],
        })
        mlflow.log_metrics(metrics)
        sig = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(tuned_pipe, artifact_path='model', signature=sig)

        print(f"[CLF] RF_HyperOpt_Tuned        | F1={metrics['f1_score']:.4f} | AUC={metrics['auc_roc']:.4f}")

        if metrics['f1_score'] > best_f1:
            best_f1    = metrics['f1_score']
            best_model = tuned_pipe
            best_name  = 'RF_HyperOpt_Tuned'

    # Simpan model terbaik sebagai pickle
    os.makedirs(exp_path, exist_ok=True)
    clf_path = os.path.join(exp_path, 'best_clf_pipeline.pkl')
    with open(clf_path, 'wb') as f:
        pickle.dump(best_model, f)

    with open(os.path.join(exp_path, 'bin_enc_dict.pkl'), 'wb') as f:
        pickle.dump(bin_enc_dict, f)

    with open(os.path.join(exp_path, 'clf_feature_cols.pkl'), 'wb') as f:
        pickle.dump(feature_cols, f)

    print(f"\n[CLF] Best model: {best_name} (F1={best_f1:.4f})")
    print(f"[CLF] Saved to  : {clf_path}")
    return best_model, feature_cols, bin_enc_dict


# 5. REGRESSION PIPELINE

def run_regression(df: pd.DataFrame, exp_path: str):
    """Latih 3+ model regresi, log ke MLflow, simpan model terbaik."""

    df = df[df['placement_status'] == 'Placed'].copy().reset_index(drop=True)
    print(f"\n[REG] Data Placed: {len(df)} baris")

    # Encode binary
    df, bin_enc_dict_reg = encode_binary(df)

    feature_cols = [c for c in df.columns
                    if c not in ['placement_status', 'salary_lpa', 'placement_enc']]
    if 'placement_enc' in df.columns:
        feature_cols = [c for c in feature_cols if c != 'placement_enc']

    X = df[feature_cols]
    y = df['salary_lpa']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    preprocessor = build_preprocessor()

    models = {
        'LinearRegression'     : LinearRegression(),
        'RidgeRegression'      : Ridge(alpha=1.0),
        'DecisionTreeRegressor': DecisionTreeRegressor(max_depth=7, random_state=SEED),
        'RandomForestRegressor': RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1),
    }

    best_r2    = -np.inf
    best_model = None
    best_name  = ''

    mlflow.set_experiment('placement_regression')

    for name, reg in models.items():
        pipe = Pipeline([('prep', preprocessor), ('reg', reg)])

        with mlflow.start_run(run_name=name):
            pipe.fit(X_train, y_train)
            y_pred  = pipe.predict(X_test)
            metrics = eval_regression(y_test, y_pred)

            mlflow.log_params({'model': name, 'test_size': 0.2, 'seed': SEED})
            mlflow.log_metrics(metrics)
            sig = infer_signature(X_train, y_pred)
            mlflow.sklearn.log_model(pipe, artifact_path='model', signature=sig)

            print(f"[REG] {name:28s} | R2={metrics['r2']:.4f} | RMSE={metrics['rmse']:.4f}")

            if metrics['r2'] > best_r2:
                best_r2    = metrics['r2']
                best_model = pipe
                best_name  = name

    # ── HyperOpt tuning RF Regressor ──
    print("\n[REG] HyperOpt tuning RF Regressor ...")

    def objective_reg(params):
        reg = RandomForestRegressor(
            n_estimators     = int(params['n_estimators']),
            max_depth        = params['max_depth'],
            min_samples_leaf = int(params['min_samples_leaf']),
            random_state     = SEED, n_jobs=-1
        )
        pipe = Pipeline([('prep', preprocessor), ('reg', reg)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        return {'loss': mean_squared_error(y_test, y_pred), 'status': STATUS_OK}

    space_reg = {
        'n_estimators'    : hp.choice('n_estimators',    [50, 100, 200]),
        'max_depth'       : hp.choice('max_depth',        [5, 10, 15, None]),
        'min_samples_leaf': hp.choice('min_samples_leaf', [1, 3, 5]),
    }
    trials_r = Trials()
    best_ridx = fmin(objective_reg, space=space_reg, algo=tpe.suggest,
                     max_evals=20, trials=trials_r,
                     rstate=np.random.default_rng(SEED))

    n_est_opts = [50, 100, 200]
    depth_opts = [5, 10, 15, None]
    leaf_opts  = [1, 3, 5]

    tuned_reg = RandomForestRegressor(
        n_estimators     = n_est_opts[best_ridx['n_estimators']],
        max_depth        = depth_opts[best_ridx['max_depth']],
        min_samples_leaf = leaf_opts[best_ridx['min_samples_leaf']],
        random_state     = SEED, n_jobs=-1
    )
    tuned_pipe_r = Pipeline([('prep', preprocessor), ('reg', tuned_reg)])

    with mlflow.start_run(run_name='RF_HyperOpt_Tuned'):
        tuned_pipe_r.fit(X_train, y_train)
        y_pred  = tuned_pipe_r.predict(X_test)
        metrics = eval_regression(y_test, y_pred)

        mlflow.log_params({
            'model'           : 'RF_HyperOpt',
            'n_estimators'    : n_est_opts[best_ridx['n_estimators']],
            'max_depth'       : str(depth_opts[best_ridx['max_depth']]),
            'min_samples_leaf': leaf_opts[best_ridx['min_samples_leaf']],
        })
        mlflow.log_metrics(metrics)
        sig = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(tuned_pipe_r, artifact_path='model', signature=sig)

        print(f"[REG] RF_HyperOpt_Tuned            | R2={metrics['r2']:.4f} | RMSE={metrics['rmse']:.4f}")

        if metrics['r2'] > best_r2:
            best_r2    = metrics['r2']
            best_model = tuned_pipe_r
            best_name  = 'RF_HyperOpt_Tuned'

    # Simpan model terbaik
    reg_path = os.path.join(exp_path, 'best_reg_pipeline.pkl')
    with open(reg_path, 'wb') as f:
        pickle.dump(best_model, f)

    with open(os.path.join(exp_path, 'bin_enc_dict_reg.pkl'), 'wb') as f:
        pickle.dump(bin_enc_dict_reg, f)

    with open(os.path.join(exp_path, 'reg_feature_cols.pkl'), 'wb') as f:
        pickle.dump(feature_cols, f)

    print(f"\n[REG] Best model: {best_name} (R2={best_r2:.4f})")
    print(f"[REG] Saved to  : {reg_path}")
    return best_model, feature_cols, bin_enc_dict_reg


# 6. MAIN

if __name__ == '__main__':
    FEATURES_PATH = './Dataset/A.csv'
    TARGET_PATH   = './Dataset/A_targets.csv'
    EXP_PATH      = './exp/placement/'

    # Load & preprocess
    df = load_data(FEATURES_PATH, TARGET_PATH)
    df = handle_missing(df)

    print("\n" + "="*55)
    print("  KLASIFIKASI — Prediksi Placement Status")
    print("="*55)
    clf_model, clf_feat, bin_enc_clf = run_classification(df, EXP_PATH)

    print("\n" + "="*55)
    print("  REGRESI — Prediksi Salary LPA")
    print("="*55)
    reg_model, reg_feat, bin_enc_reg = run_regression(df, EXP_PATH)

    print("\n" + "="*55)
    print("  SELESAI")
    print(f"  Artifacts : {EXP_PATH}")
    print("  MLflow UI : mlflow ui  ->  http://localhost:5000")
    print("="*55)
