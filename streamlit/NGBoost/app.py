# app.py
import streamlit as st
import joblib
import pandas as pd
import tempfile
from pathlib import Path
from xtb_interface import smiles_to_xyz, run_xtb, extract_xtb_features


# ===========
# SELEÇÃO VIF
# ===========
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.stats.outliers_influence import variance_inflation_factor
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






st.set_page_config(page_title="Total Energy Predictor", page_icon="⚛️")

st.title("⚛️ Predição de Energia Total via xTB + NGBoost")
st.markdown("Forneça um SMILES e obtenha a energia total corrigida via Δ-learning.")

smiles = st.text_input("Digite o SMILES da molécula:")

if smiles:
    with st.spinner("Gerando geometria e executando xTB..."):
        tmpdir = Path(tempfile.mkdtemp())
        xyz_path = smiles_to_xyz(smiles, tmpdir)
        out_path = run_xtb(xyz_path)
        features = extract_xtb_features(out_path)

    if not features:
        st.error("Falha ao extrair propriedades do xTB.")
    else:
        st.success("Propriedades extraídas com sucesso!")
        st.write(features)

        model = joblib.load("ngboost_model.pkl")
        columns_ref = joblib.load("columns_ref.pkl")

        X_new = pd.DataFrame([features])
        for col in columns_ref:
            if col not in X_new.columns:
                X_new[col] = 0.0
        X_new = X_new[columns_ref]

        pred_energy = model.predict(X_new)[0]
        st.info(f"**Energia Total (xTB + Δ-learning):** {pred_energy:.6f} Eh")
