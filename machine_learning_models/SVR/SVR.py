# -*- coding: latin-1 -*-

import pandas as pd
import numpy as np
import os
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR, LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.exceptions import ConvergenceWarning

from optuna import create_study
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from optuna.trial import FixedTrial
from optuna.exceptions import TrialPruned

import warnings


# Definições Globais
RANDOM_SEED = 88
PATH = Path('../dataset_processing/xtb_dataset.csv')
TRAIN_SIZE = 0.8
STUDY_NAME = 'svr'

# ### Leitura dos Dados
df = pd.read_csv(PATH)

X = df.drop(columns=['Delta'])
y = df['Delta']

# Pré-processamento dos Dados
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE, random_state=RANDOM_SEED)

# ### Treino & Otimização de Hiperparâmetros


# Definição de classe para aplicação do método de redução de dimensionalidade VIF (*Variance Inflation Factor*):
class VIFSelector(BaseEstimator, TransformerMixin):
    def __init__(self, thresh=10.0):
        self.thresh = thresh

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        self.n_features_in_ = X_df.shape[1]
        self.features_ = list(X_df.columns)

        while True:
            vif = pd.Series(
                [
                    variance_inflation_factor(X_df.values, i)
                    for i in range(X_df.shape[1])
                ],
                index=X_df.columns
            )

            max_vif = vif.replace([np.inf, -np.inf], np.nan).max()

            if max_vif is not None and max_vif > self.thresh:
                drop_col = vif.idxmax()
                X_df = X_df.drop(columns=drop_col)
                self.features_.remove(drop_col)
            else:
                break

        self.features_ = np.array(self.features_, dtype=int)
        return self

    def transform(self, X):
        X = np.asarray(X)
        return X[:, self.features_]

# Instância do modelo com todos os hiperparâmetros e tratamentos a serem usados pelo `optuna`.
def inst_svr(trial, n_features):

    # -------------------------------
    # Kernel
    # -------------------------------
    kernel = trial.suggest_categorical(
        'kernel', ['linear', 'rbf', 'poly']
    )

    # -------------------------------
    # Parâmetros comuns
    # -------------------------------
    params = {
        'C': trial.suggest_float('C', 1e-1, 1e2, log=True),
        'epsilon': trial.suggest_float('epsilon', 1e-2, 1.0, log=True),
        'kernel': kernel,
        'max_iter': 500_000
    }

    # -------------------------------
    # Parâmetros específicos por kernel
    # -------------------------------
    if kernel in ['rbf', 'poly']:
        params['gamma'] = trial.suggest_float(
            'gamma', 1e-3, 10.0, log=True
        )
    else:
        params['gamma'] = 'scale'

    if kernel == 'poly':
        params.update({
            'degree': trial.suggest_int('degree', 2, 4),
            'coef0': trial.suggest_float('coef0', 0.0, 1.0)
        })

    # -------------------------------
    # Espaço de pré-processamento condicionado ao kernel
    # -------------------------------
    pre_processing = trial.suggest_categorical('pre_processing', [None, 'std', 'VT', 'PCA', 'RFE', 'VIF'])

    # -------------------------------
    # Restrições de compatibilidade
    # -------------------------------
    if kernel in ["rbf", "poly"] and pre_processing in ["RFE", "VIF"]:
        raise TrialPruned()

    if kernel == "linear" and pre_processing == "PCA":
        raise TrialPruned()

    steps = []

    # -------------------------------
    # Escalonamento (quase sempre necessário)
    # -------------------------------
    if pre_processing is not None or kernel in ['rbf', 'poly']:
        steps.append(('scale', StandardScaler()))

    # -------------------------------
    # Variance Threshold
    # -------------------------------
    if pre_processing == 'VT':
        threshold = trial.suggest_float(
            'variance_threshold', 0.0, 0.1
        )
        steps.append((
            'vt', VarianceThreshold(threshold)
        ))

    # -------------------------------
    # PCA
    # -------------------------------
    elif pre_processing == 'PCA':
        max_comp = min(n_features, 50)
        n_comp = trial.suggest_int(
            'pca_components', 2, max_comp
        )
        steps.append((
            'pca', PCA(n_components=n_comp)
        ))

    # -------------------------------
    # RFE (apenas kernel linear)
    # -------------------------------
    elif pre_processing == 'RFE':
        n_sel = trial.suggest_int(
            'rfe_features', 2, min(50, n_features)
        )
        estimator = LinearSVR(
            dual='auto',
            max_iter=10_000
        )
        steps.append((
            'rfe', RFE(
                estimator=estimator,
                n_features_to_select=n_sel
            )
        ))

    # -------------------------------
    # VIF (apenas sentido em espaço linear)
    # -------------------------------
    elif pre_processing == 'VIF':
        vif_thresh = trial.suggest_float(
            'vif_threshold', 5.0, 20.0
        )
        steps.append((
            'vif', VIFSelector(thresh=vif_thresh)
        ))

    # -------------------------------
    # Modelo final
    # -------------------------------
    steps.append(('svr', SVR(**params)))

    model = make_pipeline(*[s[1] for s in steps])

    return model

# Função objetivo para validação cruzada:
def objective_function(trial, X, y, NUM_FOLDS=5):
    n_features = X.shape[1]
    cv = KFold(
        n_splits=NUM_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    rmse_folds = []

    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        model = inst_svr(trial, n_features=n_features)

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = root_mean_squared_error(y_test, y_pred)

        rmse_folds.append(rmse)

        trial.report(np.mean(rmse_folds), step=i + 1)
        if trial.should_prune():
            raise TrialPruned()

    return float(np.mean(rmse_folds))

def nested_cv_fold(fold_idx, X, y, outer_splits, inner_splits, n_trials, studies_folder):
    outer_cv = KFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    for fold, (idx_train, idx_test) in enumerate(outer_cv.split(X, y)):
        if fold != fold_idx:
            continue

        print(f'\nFold externo {fold + 1}/{outer_splits}')

        X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
        y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]

        db_path = os.path.join(
            studies_folder,
            f'{STUDY_NAME}_fold_{fold + 1}.db'
        )

        def inner_objective(trial):
            return objective_function(
                trial,
                X_train,
                y_train,
                NUM_FOLDS=inner_splits
            )

        study = create_study(
            study_name=f'{STUDY_NAME}_fold_{fold + 1}',
            direction='minimize',
            sampler=TPESampler(seed=RANDOM_SEED),
            pruner=HyperbandPruner(
                min_resource=1,
                max_resource=inner_splits,
                reduction_factor=2
            ),
            storage=f'sqlite:///{db_path}',
            load_if_exists=True
        )

        study.optimize(
            inner_objective,
            n_trials=n_trials,
            n_jobs=1,
            show_progress_bar=False
        )

        print('Melhores hiperparâmetros encontrados:')
        print(study.best_params)

        best_model = inst_svr(
            FixedTrial(study.best_params),
            n_features=X_train.shape[1]
        )

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            rmse_test = root_mean_squared_error(y_test, y_pred)

        print(f'RMSE de teste externo (fold {fold + 1}): {rmse_test:.4f}')
        return rmse_test

def nested_cv(X, y, outer_splits=5, inner_splits=3, n_trials=200, studies_folder='estudos_optuna_svr_final'):
    os.makedirs(studies_folder, exist_ok=True)
    outer_scores = []

    for fold in range(outer_splits):
        rmse_fold = nested_cv_fold(
            fold_idx=fold,
            X=X,
            y=y,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            n_trials=n_trials,
            studies_folder=studies_folder
        )
        outer_scores.append(rmse_fold)

    outer_scores = np.array(outer_scores)

    print('\n========================')
    print('Resultados do Nested CV:')
    print(f'RMSE médio: {outer_scores.mean():.4f}')
    print(f'Desvio padrão: {outer_scores.std():.4f}')
    print('========================')

    return outer_scores

print('##### ESTUDOS INICIADOS #####')
results = nested_cv(
    X_train, y_train,
    outer_splits=5,
    inner_splits=3,
    n_trials=200
)
print('##### ESTUDOS FINALIZADOS #####')
