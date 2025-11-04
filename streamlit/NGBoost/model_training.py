# model_training.py
import joblib
import pandas as pd
from ngboost import NGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

def train_ngboost(dataset_path: str = r"C:\Users\mateus25032\STREAMLIT_TEST\xtb_data\xtb_dataset.csv"):
    """Treina o modelo NGBoost com os dados de propriedades do xTB."""
    df = pd.read_csv(dataset_path)
    X = df.drop(columns=["U0", "SMILES"], errors="ignore")
    y = df["U0"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = NGBRegressor(n_estimators=500, learning_rate=0.01, verbose=True)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"RMSE no conjunto de teste: {rmse:.6f} Eh")

    joblib.dump(model, "ngboost_model.pkl")
    joblib.dump(list(X.columns), "columns_ref.pkl")
    print("Modelo e colunas salvos com sucesso.")

    return model
