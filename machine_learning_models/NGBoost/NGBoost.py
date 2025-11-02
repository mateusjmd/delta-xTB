# -*- coding: latin-1 -*-
# Leitura dos dados
import pandas as pd
import numpy as np

# Aprendizado de máquina
from ngboost import NGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error

# Otimização de hiperparâmetros
from optuna import create_study, TrialPruned
from optuna.pruners import MedianPruner

# Visualização dos resultados
from tabulate import tabulate

# Exportação do modelo
import pickle


SEED = 88 # Semente aleatória

# =================
# LEITURA DOS DADOS
# =================
df = pd.read_csv('/home/mateus25032/work/Projeto_Final_ML/Data Processing/xtb_dataset.csv')

PERCENTAGE = 1.0
subset_df = df.sample(frac=PERCENTAGE, random_state=SEED)
X = subset_df.iloc[:, :-1]  # Features
y = subset_df.iloc[:, -1]   # Target

# Divisão em treino e teste
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=SEED)



# ===========
# SELEÇÃO VIF
# ===========
class VIFSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=10.0):
        self.threshold = threshold
        self.features_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.features_ = X.columns.tolist()
        dropped = True

        while dropped:
            dropped = False
            vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            max_vif = max(vif)
            if max_vif > self.threshold:
                maxloc = vif.index(max_vif)
                X = X.drop(X.columns[maxloc], axis=1)
                dropped = True

        self.features_ = X.columns.tolist()
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X[self.features_].values


# ======================
# FUNÇÃO PARA O PIPELINE
# ======================
def make_pipeline_ngboost(use_pca, n_components, n_features, base_learner, ngb_params):
    steps = [
        ('var', VarianceThreshold(threshold=0.0)),
        ('vif', VIFSelector(threshold=10.0)),
        ('scale', StandardScaler())
    ]

    if use_pca:
        steps.append(('reduce', PCA(n_components=n_components)))
    else:
        steps.append(('reduce', RFE(LinearRegression(), n_features_to_select=n_features, step=1)))

    steps.append(('ngb', NGBRegressor(Base=base_learner, **ngb_params)))
    return Pipeline(steps)


# Validação externa
K_FOLDS_OUTER = 10
K_FOLDS_INNER = 5

outer_cv = KFold(n_splits=K_FOLDS_OUTER, shuffle=True, random_state=SEED)

# Armazena os resultados das métricas de desempenho
mae_scores, rmse_scores, mape_scores = [], [], []

# Armazena os melhores resultados de hiperparâmetros obtidos
best_results = []

# ==============
# LOOP PRINCIPAL
# ==============
# ===== OUTER CV =====
for fold, (train_idx, valid_idx) in enumerate(outer_cv.split(X_trainval)):
    print(f"\n[Fold {fold+1}/{K_FOLDS_OUTER}]")

    X_train, X_valid = X_trainval.iloc[train_idx], X_trainval.iloc[valid_idx]
    y_train, y_valid = y_trainval.iloc[train_idx], y_trainval.iloc[valid_idx]

    def funcao_objetivo(trial, X, y, NUM_FOLDS=5):
        scores = [] # Armazena os resultados obtidos da métrica usada para o INNER CV
        # Escolha da redução de dimensionalidade
        use_pca = trial.suggest_categorical("use_pca", [True, False])
        if use_pca:
            n_components = trial.suggest_float("n_components", 0.8, 0.99)
            n_features = None
        else:
            n_components = None
            n_features = trial.suggest_int("n_features", 5, min(50, X.shape[1]))

        # Hiperparâmetros do NGBoost
        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
        minibatch_frac = trial.suggest_float("minibatch_frac", 0.5, 1.0)
        col_sample = trial.suggest_float("col_sample", 0.5, 1.0)

        # Hiperparâmetros da árvore base
        max_depth = trial.suggest_int("max_depth", 2, 8)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

        base_learner = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=SEED
        )

        ngb_params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            minibatch_frac=minibatch_frac,
            col_sample=col_sample,
            natural_gradient=True,
            random_state=SEED,
            verbose=False
        )

        pipeline = make_pipeline_ngboost(use_pca, n_components, n_features, base_learner, ngb_params)
        inner_cv = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

        # ===== CV INTERNO =====
        for i, (train_idx, valid_idx) in enumerate(inner_cv.split(X, y)):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_valid)
            fold_score = root_mean_squared_error(y_valid, y_pred)
            scores.append(fold_score)

            trial.report(np.mean(scores), step=i)
            if trial.should_prune():
                raise TrialPruned()

        return np.mean(scores)
    

    study = create_study(
                        direction="minimize",
                        study_name=f'NGBoost_testing{fold + 1}',
                        storage=f'sqlite:///study_fold_{fold + 1}.db',
                        load_if_exists=True,
                        pruner=MedianPruner(n_startup_trials=20, n_warmup_steps=1)
                        )
    study.optimize(lambda trial: funcao_objetivo(trial, X_train, y_train), n_trials=30, show_progress_bar=True)

    # ===== OPTUNA =====
    # Treina o melhor modelo com os hiperparâmetros ótimos
    best_params = study.best_params

    best_results.append({
    "fold": fold + 1,
    "params": study.best_params,
    "score": study.best_value
    })


    best_pipeline = make_pipeline_ngboost(
    use_pca=best_params["use_pca"],
    n_components=best_params.get("n_components"),
    n_features=best_params.get("n_features"),
    base_learner=DecisionTreeRegressor(
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=SEED
    ),
    ngb_params=dict(
        n_estimators=best_params["n_estimators"],
        learning_rate=best_params["learning_rate"],
        minibatch_frac=best_params["minibatch_frac"],
        col_sample=best_params["col_sample"],
        natural_gradient=True,
        random_state=SEED,
        verbose=False
    ))

    # Avalia o conjunto de validação externo
    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_valid)

    # Métricas de desempenho
    mae = mean_absolute_error(y_valid, y_pred)
    rmse = root_mean_squared_error(y_valid, y_pred)
    mape = mean_absolute_percentage_error(y_valid, y_pred) * 100

    # Armazena os resultados das métricas para cada fold externo
    mae_scores.append(mae)
    rmse_scores.append(rmse)
    mape_scores.append(mape)

    print(study.best_params)

    if study.best_params.get("use_pca"):
        print(f"? Método: PCA ({study.best_params['n_components']:.2f} var. explicada)")
    else:
        print(f"? Método: RFE ({study.best_params['n_features']} features)")

    print(f"? RMSE médio (inner CV): {study.best_value:.4f}")
    print(f"? Melhor conjunto de hiperparâmetros no Fold {fold+1}:")
    for k, v in study.best_params.items():
        print(f"   {k}: {v}")



# ===========================
# VISUALIZAÇÃO DOS RESULTADOS
# ===========================
headers = ["Métrica", "Média", "Desvio Padrão"]

# Extrai os valores finais de cada lista
results_table = [
    ["MAE", np.mean(mae_scores), np.std(mae_scores)],
    ["RMSE", np.mean(rmse_scores), np.std(rmse_scores)],
    ["MAPE (%)", np.mean(mape_scores), np.mean(mape_scores)],
]


# Exibe a tabela formatada
print("\n===== Resultados Finais =====")
print(tabulate(results_table, headers=headers, floatfmt=".4f", tablefmt="github"))

# Identifica o dicionário com melhor desempenho
best_fold = min(best_results, key=lambda x: x["score"])
best_params = best_fold["params"]

print("\n===== Melhor conjunto global de hiperparâmetros =====")
for k, v in best_params.items():
    print(f"{k}: {v}")



# ==========
# EXPORTAÇÃO
# ==========
# Treino final para exportação
final_pipeline = make_pipeline_ngboost(
    use_pca=best_params["use_pca"],
    n_components=best_params.get("n_components"),
    n_features=best_params.get("n_features"),
    base_learner=DecisionTreeRegressor(
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=SEED
    ),
    ngb_params=dict(
        n_estimators=best_params["n_estimators"],
        learning_rate=best_params["learning_rate"],
        minibatch_frac=best_params["minibatch_frac"],
        col_sample=best_params["col_sample"],
        natural_gradient=True,
        random_state=SEED,
        verbose=False
    )
)

print("\nTreinando modelo final com todos os dados disponíveis...")
final_pipeline.fit(X_trainval, y_trainval)

# Exporta o modelo pré-treinado
with open("best_ngboost_model.pkl", "wb") as f:
    pickle.dump(final_pipeline, f)

print("\n? Modelo final salvo como 'best_ngboost_model.pkl'")
