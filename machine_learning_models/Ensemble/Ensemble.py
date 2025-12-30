
# -*- coding: latin-1 -*-

# Importações
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge, ElasticNet, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor, StackingRegressor

import optuna
from optuna import load_study

import joblib


#  Definições Globais
RANDOM_SEED = 88
PATH = '../../dataset_processing/xtb_dataset.csv'
TRAIN_SIZE = 0.8
STUDY_NAME = 'ensemble'


#  Leitura dos Dados
df = pd.read_csv(PATH)

X = df.drop(columns=['Delta'])
y = df['Delta']


#  Pré-processamento dos Dados
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE, random_state=RANDOM_SEED)


#  Treino & Otimização de Hiperparâmetros
# Definição dos modelos base
# ElasticNet
best_model_en = make_pipeline(
    StandardScaler(),
    VarianceThreshold(0.04137779740016238),
    ElasticNet(alpha=4.742486275093751e-05,
               l1_ratio=0.018589191548930475,
               fit_intercept=True,
               positive=False,
               max_iter=41963,
               tol=6.982053770448695e-07,
               selection='random'))

# kNN
best_model_knn = make_pipeline(
    StandardScaler(), 
    KNeighborsRegressor(n_neighbors=4,
                        weights='distance',
                        metric='manhattan'))

# SGD
best_model_sgd = make_pipeline(
    StandardScaler(),
    VarianceThreshold(0.0003095313058332159),
    SGDRegressor(loss='squared_error',
                 penalty='l2',
                 alpha=6.2269177801281935e-06,
                 learning_rate='adaptive',
                 eta0=0.0644834328258094))

# SVR
best_model_svr = make_pipeline(
    StandardScaler(),
    SVR(kernel='linear',
        C=13.978181360804484,
        epsilon=0.010033199456911295))

best_model_xgb = XGBRegressor(booster='gbtree',
                              learning_rate=0.021284051324216856,
                              n_estimators=1930,
                              subsample=0.7577937243056768,
                              colsample_bytree=0.8379115844846744,
                              colsample_bylevel=0.636877116450914,
                              colsample_bynode=0.8794677006047796,
                              min_child_weight=0.002797687917122989,
                              gamma=1.701085854648532e-07,
                              reg_alpha=4.3309175266571056e-08,
                              reg_lambda=0.0006700533858310995,
                              max_delta_step=8,
                              grow_policy='depthwise',
                              max_depth=7,
)

base_models = {
    "elasticnet": best_model_en,
    "knn": best_model_knn,
    "sgd": best_model_sgd,
    "svr": best_model_svr,
    "xgb": best_model_xgb
}


# Função objetivo do `optuna`
def objective(trial, X, y):

    ensemble_type = trial.suggest_categorical(
        "ensemble_type", ["voting", "stacking"]
    )


    # VOTING REGRESSOR
    if ensemble_type == "voting":

        voting_strategy = trial.suggest_categorical(
            "voting_strategy", ["hard", "soft"]
        )

        selected_models = []
        weights = []

        for name, model in base_models.items():
            use_model = trial.suggest_categorical(f"use_{name}", [0, 1])
            if use_model:
                selected_models.append((name, model))
                if voting_strategy == "soft":
                    weights.append(
                        trial.suggest_float(f"w_{name}", 0.1, 5.0)
                    )

        if len(selected_models) < 2:
            raise optuna.exceptions.TrialPruned()

        if voting_strategy == "hard":
            ensemble = VotingRegressor(
                estimators=selected_models
            )
        else:
            ensemble = VotingRegressor(
                estimators=selected_models,
                weights=weights
            )


    # STACKING REGRESSOR
    else:
        selected_models = []

        for name, model in base_models.items():
            use_model = trial.suggest_categorical(f"use_{name}", [0, 1])
            if use_model:
                selected_models.append((name, model))

        if len(selected_models) < 2:
            raise optuna.exceptions.TrialPruned()

        meta_model_name = trial.suggest_categorical(
            "meta_model", ["ridge", "elasticnet"]
        )

        if meta_model_name == "ridge":
            meta_model = Ridge(
                alpha=trial.suggest_float("meta_alpha", 1e-3, 10, log=True)
            )
        else:
            meta_model = ElasticNet(
                alpha=trial.suggest_float("meta_alpha", 1e-4, 1e-1, log=True),
                l1_ratio=trial.suggest_float("meta_l1_ratio", 0.0, 1.0)
            )

        ensemble = StackingRegressor(
            estimators=selected_models,
            final_estimator=meta_model,
            passthrough=False,
            n_jobs=-1
        )


    # AVALIAÇÃO
    score = cross_val_score(
        ensemble,
        X,
        y,
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1
    ).mean()

    return -score

# Execução dos estudos de otimização do optuna
print('===== ESTUDOS INICIADOS =====')
study = optuna.create_study(
     direction="minimize",
     study_name=STUDY_NAME,
     storage=f'sqlite:///{STUDY_NAME}.db',
     load_if_exists=True
 )
 
study.optimize(
     lambda trial: objective(trial, X, y),
     n_trials=300,
     n_jobs=1
 )
print('===== ESTUDOS FINALIZADOS =====')
