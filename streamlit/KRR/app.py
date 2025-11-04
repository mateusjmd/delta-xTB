import streamlit as st
import pandas as pd
import joblib
import tempfile
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from PIL import Image

# Importa funções do módulo local
from xtb_interface import smiles_to_xyz, run_xtb, extract_xtb_features

# =========== 
# CLASSE VIF 
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


# ===========
# CONFIGURAÇÃO VISUAL
# ===========
st.set_page_config(page_title="Total Energy Predictor", page_icon="⚛️")

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# Estado inicial da página
if "pagina" not in st.session_state:
    st.session_state.pagina = "Início"

# ===========
# SIDEBAR
# ===========
st.sidebar.title("Navegação")
st.sidebar.button("Início", on_click=lambda: st.session_state.update({"pagina": "Início"}))
st.sidebar.button("Como usar", on_click=lambda: st.session_state.update({"pagina": "Como usar"}))
st.sidebar.button("Total Energy", on_click=lambda: st.session_state.update({"pagina": "Total Energy"}))
st.sidebar.button("Mais informações", on_click=lambda: st.session_state.update({"pagina": "MI"}))

# ===========
# CONTEÚDO DAS PÁGINAS
# ===========
if st.session_state.pagina == "Início":
    st.markdown("<h1 style='text-align: center;'>⚛️ Total Energy Predictor</h1>", unsafe_allow_html=True)
    st.subheader("Predição precisa da energia total molecular a partir de SMILES.")
    st.markdown("---")

    st.header("Equipe")
    equipe = [
        {
            "nome": "Edélio G. M. de Jesus",
            "resumo": "Cursando bacharelado em Ciência e Tecnologia na Ilum - Escola de Ciência.",
            "imagem": "https://github.com/Velky2/R2D2/blob/main/images/edelio.jpeg?raw=true",
            "link": "https://github.com/EdelioGabriel"
        },
        {
            "nome": "Mateus de J. Mendes",
            "resumo": "Cursando bacharelado em Ciência e Tecnologia na Ilum - Escola de Ciência.",
            "imagem": "https://github.com/Velky2/R2D2/blob/main/images/mateus_mendes.jpg?raw=true",
            "link": "https://github.com/mateusjmd"
        },
        {
            "nome": "Matheus P. V. da Silveira",
            "resumo": "Cursando bacharelado em Ciência e Tecnologia na Ilum - Escola de Ciência.",
            "imagem": "https://github.com/Velky2/R2D2/blob/main/images/matheus_velloso.jpg?raw=true",
            "link": "https://github.com/Velky2"
        }
    ]
    cols2 = st.columns(3)
    for col, pessoa in zip(cols2, equipe):
        with col:
            st.markdown(
    f'<img src="{pessoa["imagem"]}" width="150">',
    unsafe_allow_html=True
)
            st.markdown(f"### {pessoa['nome']}")
            st.markdown(f"<p style='text-align: justify;'>{pessoa['resumo']}</p>", unsafe_allow_html=True)
            st.markdown(
                f'<a href="{pessoa["link"]}" target="_blank">'
                '<img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="50">'
                '</a>',
                unsafe_allow_html=True
            )

elif st.session_state.pagina == "Como usar":
    st.markdown("<h1 style='text-align: center;'>Como usar</h1>", unsafe_allow_html=True)
    st.subheader("Aprenda a utilizar nossa ferramenta de forma simples.")
    st.markdown("---")
    st.markdown("""
    1. Digite o **SMILES** da molécula que deseja estudar.  
    2. O aplicativo irá gerar a geometria 3D, otimizar via xTB e extrair as propriedades.  
    3. Um modelo de **aprendizado de máquina (KRR)** fará a correção Δ-learning, exibindo a energia total final.
    """)

elif st.session_state.pagina == "Total Energy":
    st.markdown("<h1 style='text-align: center;'>Calcular Total Energy</h1>", unsafe_allow_html=True)
    st.markdown("---")

    smiles = st.text_input("Digite o SMILES da molécula:")
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error("SMILES inválido. Verifique a entrada.")
        else:
            mol_h = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
            AllChem.UFFOptimizeMolecule(mol_h)
            img = Draw.MolToImage(mol_h, size=(400, 300))
            st.image(img, caption="Visualização 3D aproximada (estrutura otimizada)")

            with st.spinner("Executando xTB e extraindo propriedades..."):
                tmpdir = Path(tempfile.mkdtemp())
                xyz_path = smiles_to_xyz(smiles, tmpdir)
                out_path = run_xtb(xyz_path)
                features = extract_xtb_features(out_path)

            if not features:
                st.error("Falha ao extrair propriedades do xTB.")
            else:
                st.success("Propriedades extraídas com sucesso!")
                st.write(features)

                try:
                    model = joblib.load("best_krr_model.pkl")
                    columns_ref = joblib.load("columns_ref.pkl")

                    X_new = pd.DataFrame([features])
                    for col in columns_ref:
                        if col not in X_new.columns:
                            X_new[col] = 0.0
                    X_new = X_new[columns_ref]

                    pred_energy = model.predict(X_new)[0]
                    st.info(f"**Energia Total (xTB + Δ-learning):** {pred_energy:.6f} Eh")
                except Exception as e:
                    st.error(f"Erro ao carregar o modelo: {e}")

elif st.session_state.pagina == "MI":
    st.markdown("<h1 style='text-align: center;'>Mais informações</h1>", unsafe_allow_html=True)
    st.subheader("Veja o projeto completo no GitHub:")
    st.markdown("---")
    st.markdown(
        f'<a href="https://github.com/Velky2/R2D2" target="_blank">'
        '<img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="150">'
        '</a>',
        unsafe_allow_html=True
    )
