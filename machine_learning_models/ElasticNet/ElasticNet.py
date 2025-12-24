# -*- coding: latin-1 -*-

# Importações
import pandas as pd
from pathlib import Path

from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.stats.outliers_influence import variance_inflation_factor

from optuna import create_study
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna import TrialPruned


# Definições Globais
RANDOM_SEED = 88
PATH = Path('../dataset_processing/xtb_dataset.csv')
TRAIN_SIZE = 0.8

# Leitura dos dados
df = pd.read_csv(PATH)

X = df.drop(columns=['Delta'])
y = df['Delta']

# Pré-processamento dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE, random_state=RANDOM_SEED)


# Classe para o método de seleção de features VIF
class VIFSelector(BaseEstimator, TransformerMixin):
    def __init__(self, thresh=10.0):
        self.thresh = thresh

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.features_ = list(range(X_df.shape[1]))

        while True:
            vif = pd.Series(
                [variance_inflation_factor(X_df.values, i)
                 for i in range(X_df.shape[1])],
                index=X_df.columns
            )

            max_vif = vif.max()
            if max_vif > self.thresh:
                drop_col = vif.idxmax()
                X_df = X_df.drop(columns=[drop_col])
                self.features_.remove(drop_col)
            else:
                break

        return self

    def transform(self, X):
        return X[:, self.features_]


# Função para instanciar o modelo dentro do Optuna
def inst_elasticnet(trial):
    params = {
        'alpha': trial.suggest_float('alpha', 1e-5, 1e3, log=True),
        'l1_ratio': trial.suggest_float('l1_ratio', 0, 1),
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        'positive': trial.suggest_categorical('positive', [False, True]),
        'max_iter': trial.suggest_int('max_iter', 1_000, 50_000),
        'tol': trial.suggest_float('tol', 1e-8, 1e-3, log=True),
        'selection': trial.suggest_categorical('selection', ['cyclic', 'random'])
    }

    pre_processing = trial.suggest_categorical('pre_processing', [None, 'stds', 'PCA', 'VT', 'RFE', 'VIF'])

    if pre_processing == 'stds':
        modelo = make_pipeline(
            StandardScaler(),
            ElasticNet(**params)
        )

    elif pre_processing == 'PCA':
        max_components = min(X.shape[0] - 1, X.shape[1])

        if max_components < 2:
            raise TrialPruned()

        components = trial.suggest_int(
            'pca_components',
            2,
            max_components
        )

        modelo = make_pipeline(
            StandardScaler(),
            PCA(n_components=components),
            ElasticNet(**params)
        )


    elif pre_processing == 'VT':
        threshold = trial.suggest_float('variance_threshold', 0, 0.1)
        modelo = make_pipeline(
            StandardScaler(),
            VarianceThreshold(threshold),
            ElasticNet(**params)
        )

    elif pre_processing == 'RFE':
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        n_features = trial.suggest_int('rfe_features', 2, 50)
        modelo = make_pipeline(
            StandardScaler(),
            RFE(estimator=estimator, n_features_to_select=n_features),
            ElasticNet(**params)
        )

    elif pre_processing == 'VIF':
        vif_thresh = trial.suggest_float('vif_threshold', 5.0, 20.0)
        modelo = make_pipeline(
            StandardScaler(),
            VIFSelector(thresh=vif_thresh),
            ElasticNet(**params)
        )

    else:
        modelo = ElasticNet(**params)

    return modelo


# Função objetivo para o Optuna
def objective_function(trial, X, y, num_folds, instantiator):
    model = instantiator(trial)

    try:
        scores = -cross_val_score(
            model,
            X,
            y,
            cv=num_folds,
            n_jobs=-1,
            scoring='neg_root_mean_squared_error',
            error_score='raise'
        )
        return scores.mean()

    except ValueError:
        raise TrialPruned()


# Função de otimização para o Optuna
def run_optuna(study_name, X, y, instantiator, num_folds=5, n_trials=1_000):

    sampler = TPESampler(
        n_startup_trials=50,
        multivariate=True,
        seed=RANDOM_SEED
    )

    pruner = MedianPruner(
        n_startup_trials=50
    )

    study = create_study(
        direction='minimize',
        study_name=study_name,
        storage=f'sqlite:///{study_name}.db',
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner
    )

    def objective(trial):
        return objective_function(trial, X, y, num_folds, instantiator)

    study.optimize(objective, n_trials=n_trials)
    return study

# Execução dos estudos de otimização de hiperparâmetros
print("##### ESTUDOS INICIADOS #####")
study_en = run_optuna('elasticnet', X_train, y_train, inst_elasticnet)
print("##### ESTUDOS FINALIZADOS #####")
