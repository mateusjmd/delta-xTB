# -*- coding: latin-1 -*-

# Importações
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_selection import VarianceThreshold, SequentialFeatureSelector
from optuna import create_study
from optuna.pruners import MedianPruner
from optuna.exceptions import TrialPruned


# Definições Globais
RANDOM_SEED = 88
PATH = Path('../../dataset_processing/xtb_dataset.csv')
TRAIN_SIZE = 0.8
STUDY_NAME = 'xgboost'

# ### Leitura dos Dados
df = pd.read_csv(PATH)

X = df.drop(columns=['Delta'])
y = df['Delta']

# Pré-processamento dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE, random_state=RANDOM_SEED)

# Função para sugerir os valores dos hiperparâmetros
def suggest_xgb_params(trial):
    params = {}

    params['booster'] = trial.suggest_categorical('booster', ['gbtree', 'dart'])
    params['objective'] = 'reg:squarederror'
    params['verbosity'] = 0
    params['random_state'] = trial.number
    params['n_jobs'] = 1

    params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)

    params['n_estimators'] = trial.suggest_int('n_estimators', 100, 3000)

    params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
    params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.4, 1.0)
    params['colsample_bylevel'] = trial.suggest_float('colsample_bylevel', 0.4, 1.0)
    params['colsample_bynode'] = trial.suggest_float('colsample_bynode', 0.4, 1.0)

    params['min_child_weight'] = trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True)
    params['gamma'] = trial.suggest_float('gamma', 1e-8, 10.0, log=True)

    params['reg_alpha'] = trial.suggest_float('reg_alpha', 1e-8, 100.0, log=True)
    params['reg_lambda'] = trial.suggest_float('reg_lambda', 1e-8, 100.0, log=True)

    params['max_delta_step'] = trial.suggest_int('max_delta_step', 0, 10)

    params['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
    if params['grow_policy'] == 'depthwise':
        params['max_depth'] = trial.suggest_int('max_depth', 3, 12)
        params['max_leaves'] = 0
    else:
        params['max_depth'] = 0
        params['max_leaves'] = trial.suggest_int('max_leaves', 8, 1024)

    if params['booster'] == 'dart':
        params['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
        params['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
        params['rate_drop'] = trial.suggest_float('rate_drop', 0.0, 0.3)
        params['skip_drop'] = trial.suggest_float('skip_drop', 0.0, 0.3)
    else:
        params['sample_type'] = 'uniform'
        params['normalize_type'] = 'tree'
        params['rate_drop'] = 0.0
        params['skip_drop'] = 0.0

    return params


# Função para instanciar o modelo dentro do Optuna
def inst_xgboost(trial):
    params = suggest_xgb_params(trial)

    # Métodos que fazem sentido para XGBoost
    pre_processing = trial.suggest_categorical(
        'pre_processing',
        ['none', 'vt', 'sfs']
    )

    # Sem pré-processamento explícito
    if pre_processing == 'none':
        model = XGBRegressor(**params)

    # Variance Threshold
    elif pre_processing == 'vt':
        threshold = trial.suggest_float(
            'variance_threshold',
            0.0,
            0.1
        )
        model = make_pipeline(
            VarianceThreshold(threshold=threshold),
            XGBRegressor(**params)
        )

    # Sequential Feature Selection
    elif pre_processing == 'sfs':
        n_features_to_select = trial.suggest_int(
            'sfs_features',
            2,
            9
        )

        sfs = SequentialFeatureSelector(
            estimator=XGBRegressor(**params),
            n_features_to_select=n_features_to_select,
            direction='forward',
            scoring='neg_mean_squared_error',
            cv=3,
            n_jobs=-1
        )

        model = make_pipeline(
            sfs,
            XGBRegressor(**params)
        )

    return model


# Função objetivo para o Optuna
def objective_function(trial, X, y, num_folds, instantiator, random_state=RANDOM_SEED):
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    model = instantiator(trial)

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_arr)):
        X_train, X_val = X_arr[train_idx], X_arr[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        rmse = root_mean_squared_error(y_val, preds)
        fold_scores.append(rmse)

        trial.report(rmse, step=fold_idx)
        if trial.should_prune():
            raise TrialPruned()

    return float(np.mean(fold_scores))


# Função de otimização para o Optuna
def run_optuna(study_name, X, y, instanciador, num_folds=5, n_trials=500):
    study = create_study(
        direction='minimize',
        study_name=study_name,
        storage=f'sqlite:///{study_name}.db',
        load_if_exists=True,
        pruner=MedianPruner()
    )

    def objetivo_parcial(trial):
        return objective_function(trial, X, y, num_folds, instanciador)

    study.optimize(objetivo_parcial, n_trials=n_trials, show_progress_bar=True, n_jobs=3)
    return study

# Execução dos estudos de otimização de hiperparâmetros
print(f'##### ESTUDOS INICIADOS #####')
study = run_optuna(f'{STUDY_NAME}', X_train, y_train, inst_xgboost)
print(f'##### ESTUDOS FINALIZADOS #####')