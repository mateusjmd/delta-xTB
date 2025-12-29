import streamlit as st
import pandas as pd
import joblib
import tempfile
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import io

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
BASE_DIR = Path(__file__).resolve().parent
MEDIA_DIR = BASE_DIR / "media"

page_bg_style = """
<style>
/* Header transparente */
[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);  /* mantém compatível com dark */
}

/* Caixa principal com padding e background com gradiente suave */
[data-testid="stAppViewContainer"] {
    background-image: linear-gradient(to right bottom, #000000, #000000, #000000, #000000, #000000, #120408, #1c0711, #240b18, #370d29, #480e3e, #551157, #5e1a75);
}

/* Conteúdo principal com padding */
[data-testid="stAppViewContainer"] > .main {
    padding: 2rem 4rem;
}
</style>
"""

st.markdown(page_bg_style, unsafe_allow_html=True)
st.set_page_config(page_title="Δ-xTB", page_icon=MEDIA_DIR / "icon.png")
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)


tab1, tab2, tab3, tab4 = st.tabs(['Home', 'Tutorial', 'Δ-xTB', 'Info'])

# ===========
# CONTEÚDO DAS ABAS
# ===========
with tab1:
    st.image(MEDIA_DIR / 'banner.png')
    st.markdown("<h3 style='text-align: center;'><i>Machine Learning for Cheminformatics<i></h3>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown(
        """
        <style>
            .justified-text {
                text-align: justify;
                text-justify: inter-word;
                hyphens: auto;
                max-width: 900px;
            }
        </style>
        """,
        unsafe_allow_html=True)
    st.markdown(
    """
    <style>
        a {
            color: #C0392B;          /* cor padrão do link */
            text-decoration: none;  /* remove sublinhado (opcional) */
        }

        a:hover {
            color: #fa8f02;         /* cor ao passar o mouse */
            text-decoration: underline;
        }
    </style>
    """,
    unsafe_allow_html=True)


    st.markdown(
    """
    <div class="justified-text">

    ### O Projeto $\Delta$-xTB

    O $\Delta$-xTB consiste em uma aplicação desenvolvida como projeto final da disciplina de *Machine Learning* do curso de Bacharelado
    em Ciência e Tecnologia da Ilum — Escola de Ciência. Seu objetivo central é empregar métodos supervisionados de *Machine Learning*
    para a predição de propriedades termodinâmicas e eletrônicas de espécies químicas a partir de sua representação estrutural em
    **[SMILES](https://pubs.acs.org/doi/10.1021/ci00057a005)**.

    A proposta insere-se no contexto contemporâneo da química computacional, em que a crescente complexidade dos sistemas de interesse
    demanda abordagens capazes de conciliar rigor físico-químico, eficiência numérica e viabilidade computacional.

    ### Aspectos Metodológicos

    A motivação fundamental que orienta a concepção e o desenvolvimento do $\Delta$-xTB reside no elevado custo computacional associado
    a técnicas de simulação molecular de alta fidelidade, notadamente a **[Teoria do Funcional da Densidade](
    https://pubs.acs.org/doi/10.1021/jp960669l)** (DFT — *Density Functional Theory*). Embora tais métodos assegurem elevada 
    confiabilidade físico-química, sua aplicação sistemática em grandes espaços químicos torna-se frequentemente proibitiva.

    Nesse sentido, o propósito do $\Delta$-xTB é oferecer uma alternativa de menor custo computacional, mantendo uma adequação
    satisfatória aos princípios físico-químicos e às exigências numéricas subjacentes. Para tanto, o projeto fundamenta-se na indução
    de modelos supervisionados clássicos de *Machine Learning*, capazes de aprender relações não triviais entre descritores moleculares
    e propriedades de interesse a partir de dados previamente calculados.

    Foram explorados modelos baseados nos algoritmos ElasticNet, $k$-NN ($k$ *Nearest Neighbors*), SGD (*Stochastic Gradient Descent*),
    SVR (*Support Vector Regression*) e XGBoost (*Extreme Gradient Boosting*), todos treinados a partir do *dataset* 
    **[QM9](https://www.nature.com/articles/sdata201422)**. Esse conjunto de dados é composto por geometrias moleculares de pequenas 
    moléculas orgânicas, contendo até nove átomos pesados de `C`, `H`, `O`, `N` e `F`, cujas propriedades foram originalmente obtidas 
    por meio de cálculos em nível de DFT.

    O *dataset* foi reconstituído com o auxílio dos módulos `rdkit` e `xTB`, a partir dos quais foram extraídas as seguintes propriedades
    termodinâmicas e eletrônicas:

    - Momento de dipolo  
    - Energia do HOMO ($E_{\\text{HOMO}}$)  
    - Energia do LUMO ($E_{\\text{LUMO}}$)  
    - *Gap* HOMO–LUMO  
    - Energia de ponto zero (ZPE)  
    - Entalpia ($H$)  
    - Energia interna ($U$)  
    - Energia interna corrigida ($U_0$)  
    - Energia livre de Gibbs ($G$)

    Por fim, a diferença entre a energia interna total calculada via métodos semiempíricos (`xTB`) e o valor de referência obtido por
    DFT (conforme disponibilizado no QM9) foi definida como *target* do problema, caracterizando uma tarefa de
    $\Delta$-*learning*. Dessa forma, ao estimar essa correção energética por meio de *Machine Learning*, torna-se possível combinar a
    eficiência computacional dos métodos aproximados com a precisão associada aos cálculos de referência, resultando em predições
    robustas e computacionalmente acessíveis para diferentes espécies químicas.

    ### Resultados

    </div>
        """,
        unsafe_allow_html=True)


    st.markdown("---")
    st.image(MEDIA_DIR / 'footer.webp')


with tab2:
    st.markdown("<h1 style='text-align: center;'>Tutorial</h1>", unsafe_allow_html=True)
    st.markdown(
    """
    Utilizar o $\Delta$-xTB é simples e rápido! Siga as instruções e desfrute da ferramenta:

    1. Acesse a aba **$\Delta$-xTB**.
    2. Informe a notação **[SMILES](https://en.wikipedia.org/wiki/Simplified_Molecular_Input_Line_Entry_System#Examples)** da espécie química desejada.  
    3. A geometria otimizada é calculada e exibida, extraindo-se as propriedades: `HOMO`, `LUMO`, `U0`, `gap_HOMO_LUMO`.  
    4. Por fim, um modelo de **_Machine Learning_** fará a correção **$\Delta$-_learning_**, exibindo a energia total final calculada 
        para a espécie química.
    """)

    st.markdown('Vídeo demonstrativo:')
    st.video(MEDIA_DIR / 'tutorial.mp4')

    st.markdown("---")
    st.image(MEDIA_DIR / 'footer.webp')



with tab3:
    st.markdown("<h1 style='text-align: center;'>Calcular Total Energy</h1>", unsafe_allow_html=True)

    smiles = st.text_input("Digite o SMILES da molécula:")

    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error("SMILES inválido. Verifique a entrada.")
        else:
            # Contagens estruturais
            n_heavy = mol.GetNumHeavyAtoms()
            n_h = sum(atom.GetTotalNumHs() for atom in mol.GetAtoms())
            n_atoms = n_heavy + n_h

            # Heurística de decisão
            show_h = not (
                n_h > 20 or
                n_atoms > 35 or
                (n_h / n_heavy) > 2.0
            )

            if show_h:
                mol_vis = Chem.AddHs(mol)
            else:
                mol_vis = mol

            AllChem.EmbedMolecule(mol_vis, AllChem.ETKDGv3())
            AllChem.UFFOptimizeMolecule(mol_vis)

            img = Draw.MolToImage(mol_vis, size=(1200, 800))
            st.image(
                img,
                caption=f"Visualização 3D aproximada ({'com' if show_h else 'sem'} hidrogênios)",
                use_container_width=True
            )
            

            st.markdown(
                """
                <style>
                div[data-testid="stDownloadButton"] {
                    display: flex;
                    justify-content: center;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            st.download_button(
                    label="Download",
                    data=buf,
                    file_name="molecule.png",
                    mime="image/png"
                )


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
                    model = joblib.load("model.pkl")
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

    st.markdown("---")
    st.image(MEDIA_DIR / 'footer.webp')


with tab4:
    st.markdown("<h1 style='text-align: center;'>Informações</h1>", unsafe_allow_html=True)
    st.markdown("Verifique o projeto completo no GitHub: **[$\Delta$-xTB](https://github.com/mateusjmd/delta-xTB)**")

    st.subheader("Desenvolvedores")
    st.markdown(
    """
    - Edélio Gabriel Magalhães de Jesus:
        - Email: **[edelio25024@ilum.cnpem.br](mailto:edelio25024@ilum.cnpem.br)**
        - GitHub: **[EdelioGabriel](https://github.com/EdelioGabriel)**
    - Mateus de Jesus Mendes:
        - Email: **[mateus25032@ilum.cnpem.br](mailto:mateus25032@ilum.cnpem.br)**
        - GitHub: **[mateusjmd](https://github.com/mateusjmd)**
    - Matheus Pereira Velloso da Silveira
        - Email: **[matheus25022@ilum.cnpem.br](mailto:matheus25022@ilum.cnpem.br)**
        - GitHub: **[Velky2](https://github.com/Velky2)**
        """)
    st.subheader('Orientador')
    st.markdown(
    """
    - Daniel Roberto Cassar:
        - Email: **[daniel.cassar@ilum.cnpem.br](mailto:daniel.cassar@ilum.cnpem.br)**
    """)

    st.subheader('Agradecimentos')
    st.markdown(
    """
    Os desenvolvedores do presente projeto agredecem ao suporte técnico-acadêmico ofertado pelo Prof. Dr. Daniel Roberto Cassar ao
    longo do semestre em que se deu o desenvolvimento desse trabalho. Ademais, os auxílios infraestruturais fornecidos pelo **[Centro
    Nacional de Pesquisa em Energia e Materiais (CNPEM)](https://cnpem.br)** a partir da 
    **[Ilum - Escola de Ciência](https://ilum.cnpem.br)** foram cruciais para a própria concepção e execução de tal projeto.    
    """)

    st.markdown("---")
    st.image(MEDIA_DIR / 'footer.webp')
