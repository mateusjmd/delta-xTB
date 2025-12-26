# -*- coding: latin-1 -*-

# ImportaÃ§Ãµes
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_selection import VarianceThreshold
from sklearn.exceptions import ConvergenceWarning

import warnings

from optuna import create_study
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from optuna.trial import FixedTrial
from optuna.exceptions import TrialPruned

import os


# DefiniÃ§Ãµes Globais
RANDOM_SEED = 88
PATH = '../../dataset_processing/xtb_dataset.csv'
TRAIN_SIZE = 0.8
STUDY_NAME = 'knn'


# Leitura dos Dados
df = pd.read_csv(PATH)

X = df.drop(columns=['Delta'])
y = df['Delta']

# PrÃ©-processamento dos Dados
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE, random_state=RANDOM_SEED)

# Treino & OtimizaÃ§Ã£o de HiperparÃ¢metros
# InstÃ¢ncia do modelo com todos os hiperparÃ¢metros e tratamentos a serem usados pelo `optuna`.
def inst_knn(trial, n_features):


    # PrÃ©-processamento disponÃ­vel
    pre_processing = trial.suggest_categorical(
        "pre_processing", [None, "VT", "PCA"]
    )

    steps = []


    # NormalizaÃ§Ã£o
    steps.append(("scale", StandardScaler()))


    # Variance Threshold
    if pre_processing == "VT":
        threshold = trial.suggest_float(
            "variance_threshold", 0.0, 1e-3
        )
        steps.append((
            "vt", VarianceThreshold(threshold=threshold)
        ))


    # PCA
    elif pre_processing == "PCA":
        max_comp = min(n_features, 8)
        n_comp = trial.suggest_int(
            "pca_components", 2, max_comp
        )
        steps.append((
            "pca", PCA(n_components=n_comp)
        ))


    # HiperparÃ¢metros do kNN
    n_neighbors = trial.suggest_int(
        "n_neighbors", 3, 50
    )

    weights = trial.suggest_categorical(
        "weights", ["uniform", "distance"]
    )

    metric = trial.suggest_categorical(
        "metric", ["euclidean", "manhattan", "minkowski"]
    )

    p = 2
    if metric == "minkowski":
        p = trial.suggest_int("p", 1, 2)

    # Modelo kNN
    steps.append((
        "knn", KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            p=p,
            n_jobs=1
        )
    ))

    model = make_pipeline(*[s[1] for s in steps])

    return model

# FunÃ§Ã£o objetivo para validaÃ§Ã£o cruzada
def objective_function(trial, X, y, NUM_FOLDS=5):
    n_features = X.shape[1]
    cv = KFold(
        n_splits=NUM_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    rmse_folds = []

    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        model = inst_knn(trial, n_features=n_features)

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = root_mean_squared_error(y_test, y_pred)

        rmse_folds.append(rmse)

        trial.report(np.mean(rmse_folds), step=i + 1)
        if trial.should_prune():
            raise TrialPruned()

    return float(np.mean(rmse_folds))


# FunÃ§Ãµes para validaÃ§Ã£o cruzada aninhada
def nested_cv_fold(fold_idx, X, y, outer_splits, inner_splits, n_trials, studies_folder):
    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=RANDOM_SEED)

    for fold, (idx_train, idx_test) in enumerate(outer_cv.split(X, y)):
        if fold != fold_idx:
            continue

        print(f"\nFold externo {fold + 1}/{outer_splits}")

        X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
        y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]

        db_path = os.path.join(
            studies_folder,
            f"{STUDY_NAME}_fold_{fold + 1}.db"
        )

        def inner_objective(trial):
            return objective_function(
                trial,
                X_train,
                y_train,
                NUM_FOLDS=inner_splits
            )

        study = create_study(
            study_name=f"{STUDY_NAME}_fold_{fold + 1}",
            direction="minimize",
            sampler=TPESampler(seed=RANDOM_SEED),
            pruner=HyperbandPruner(
                min_resource=1,
                max_resource=inner_splits,
                reduction_factor=2
            ),
            storage=f"sqlite:///{db_path}",
            load_if_exists=True
        )

        study.optimize(
            inner_objective,
            n_trials=n_trials,
            n_jobs=1,
            show_progress_bar=False
        )

        print("Melhores hiperparÃ¢metros encontrados:")
        print(study.best_params)

        best_model = inst_knn(
            FixedTrial(study.best_params),
            n_features=X_train.shape[1]
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            rmse_test = root_mean_squared_error(y_test, y_pred)

        print(f"RMSE de teste externo (fold {fold + 1}): {rmse_test:.4f}")
        return rmse_test


def nested_cv(X, y, outer_splits=5, inner_splits=3, n_trials=200, studies_folder=f"optuna_{STUDY_NAME}_studies"):
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

    print("\n========================")
    print("Resultados do Nested CV:")
    print(f"RMSE mÃ©dio: {outer_scores.mean():.4f}")
    print(f"Desvio padrÃ£o: {outer_scores.std():.4f}")
    print("========================")

    return outer_scores


print(f'##### ESTUDOS INICIADOS ######')
study = nested_cv(X_train, y_train, outer_splits=5, inner_splits=3, n_trials=100)
print(f'##### ESTUDOS FINALIZADOS #####')
