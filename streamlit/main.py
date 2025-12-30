import streamlit as st
import pandas as pd
import joblib
import tempfile
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import io
import base64

# Importa funções do módulo local
from xtb_interface import smiles_to_xyz, run_xtb, extract_xtb_features


# ==================
# CONFIGURAÇÃO VISUAL
# ===================
BASE_DIR = Path(__file__).resolve().parent
MEDIA_DIR = BASE_DIR / 'media'

# Configurações visuais gerais em CSS
page_bg_style = """
<style>
/* Header transparente */
[data-testid='stHeader'] {
    background-color: rgba(0,0,0,0);
}

/* Background com gradiente suave */
[data-testid='stAppViewContainer'] {
    background-image: linear-gradient(to right bottom, 
                                      #000000, 
                                      #000000, 
                                      #000000, 
                                      #000000, 
                                      #000000, 
                                      #120408, 
                                      #1c0711, 
                                      #240b18, 
                                      #370d29,
                                      #480e3e, 
                                      #551157, 
                                      #5e1a75);
}
</style>
"""
st.markdown(page_bg_style, unsafe_allow_html=True)
st.set_page_config(page_title='Δ-xTB', page_icon=MEDIA_DIR / 'icon.png')

# Oculta o menu de configurações
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# 
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

# Customiza cor do hover para as abas e links
st.markdown(
    """
    <style>
        a {
            color: #C0392B;          /* cor padrão do link */
        }

        a:hover {
            color: #fa8f02;         /* cor ao passar o mouse */
            text-decoration: underline;
        }
    </style>
    """,
    unsafe_allow_html=True)


# Função para renderizar imagens sem interação com zoom
def render_image_no_zoom(path, width='100%'):
    with open(path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <img src='data:image/webp;base64,{encoded}'
             style='
                 width: {width};
                 pointer-events: none;
                 user-select: none;
             '>
        """,
        unsafe_allow_html=True
    )



# =================
# CONTEÚDO DAS ABAS
# =================
# Abas de navegação
tab1, tab2, tab3, tab4 = st.tabs(['Home', 'Tutorial', 'Δ-xTB', 'Info'])


# Home
with tab1:
    render_image_no_zoom(MEDIA_DIR / 'banner.png')
    st.markdown("<h3 style='text-align: center;'><i>Machine Learning for Cheminformatics<i></h3>", unsafe_allow_html=True)
    st.markdown('---')

    st.markdown(
    """
    <div class='justified-text'>

    ### O Projeto $\Delta$-xTB

    O **$\Delta$-xTB** é uma aplicação desenvolvida como projeto final da disciplina de *Machine Learning* do curso de Bacharelado 
    em Ciência e Tecnologia da **Ilum — Escola de Ciência**. Seu objetivo central é empregar métodos supervisionados de *Machine 
    Learning* para a predição de propriedades termodinâmicas e eletrônicas de espécies químicas a partir de sua representação estrutural
    em **[SMILES](https://pubs.acs.org/doi/10.1021/ci00057a005)**.

    A proposta insere-se no cenário contemporâneo da química computacional, no qual a ampliação do espaço químico explorável — 
    tanto em diversidade estrutural quanto em complexidade eletrônica — impõe desafios significativos em termos de custo computacional, 
    escalabilidade e tempo de resposta. Nesse contexto, torna-se essencial o desenvolvimento de abordagens que conciliem rigor 
    físico-químico, eficiência numérica e viabilidade computacional, sem comprometer a confiabilidade das predições.

    ---

    ### Aspectos Metodológicos

    A motivação fundamental que orienta a concepção e o desenvolvimento do $\Delta$-xTB decorre do elevado custo computacional associado a métodos de simulação molecular de alta fidelidade, em especial a **[Teoria do Funcional da Densidade](https://pubs.acs.org/doi/10.1021/jp960669l)** (DFT — *Density Functional Theory*). Embora tais abordagens forneçam descrições precisas das propriedades eletrônicas e energéticas dos sistemas químicos, sua aplicação sistemática em grandes conjuntos moleculares torna-se, na prática, computacionalmente proibitiva.

    Diante disso, o $\Delta$-xTB propõe uma estratégia alternativa baseada em *Machine Learning*, cujo objetivo é reduzir significativamente o custo computacional mantendo uma aderência satisfatória aos princípios físico-químicos subjacentes. Para tanto, o projeto fundamenta-se na indução de modelos supervisionados clássicos, capazes de aprender relações não lineares e de alta complexidade entre descritores moleculares e propriedades de interesse, a partir de dados previamente calculados em nível de referência.

    Foram explorados modelos baseados nos algoritmos ElasticNet, $k$-NN (*$k$ Nearest Neighbors*), SGD (*Stochastic Gradient Descent*), SVR (*Support Vector Regression*) e XGBoost (*Extreme Gradient Boosting*), todos treinados a partir do *dataset* **[QM9](https://www.nature.com/articles/sdata201422)**. Esse conjunto de dados é composto por geometrias moleculares de pequenas moléculas orgânicas — contendo até nove átomos pesados de `C`, `H`, `O`, `N` e `F` — cujas propriedades termodinâmicas e eletrônicas foram originalmente obtidas por meio de cálculos de DFT.

    O *dataset* foi reconstituído com o auxílio dos módulos `rdkit` e `xTB`, permitindo a extração sistemática das seguintes propriedades:

    * Momento de dipolo
    * Energia do HOMO ($E_{\text{HOMO}}$)
    * Energia do LUMO ($E_{\text{LUMO}}$)
    * *Gap* HOMO–LUMO
    * Energia de ponto zero (ZPE)
    * Entalpia ($H$)
    * Energia interna ($U$)
    * Energia interna corrigida ($U_0$)
    * Energia livre de Gibbs ($G$)

    A diferença entre a energia interna total calculada via métodos semiempíricos (`xTB`) e o valor de referência obtido por DFT (conforme disponibilizado no QM9) foi então definida como *target* do problema, caracterizando formalmente uma tarefa de **$\Delta$-*learning***. Nessa abordagem, o modelo não aprende diretamente a propriedade absoluta, mas sim a correção necessária para alinhar o resultado aproximado ao nível de referência. Dessa forma, é possível combinar a eficiência computacional dos métodos semiempíricos com a precisão associada aos cálculos de alta fidelidade, resultando em predições robustas, escaláveis e computacionalmente acessíveis para diferentes espécies químicas.

    ---

    ### Resultados

    O *ensemble* do tipo *stacking*, obtido após a otimização de hiperparâmetros via `optuna`, apresentou desempenho consistente e estatisticamente estável. Observou-se um erro quadrático médio (RMSE) da ordem de $10^{-4}$ durante o processo de indução (treino e validação) e entre $10^{-4}$ e $10^{-3}$ no conjunto de teste, indicando um aumento moderado — e esperado — do erro absoluto ao se avaliar dados não vistos.

    Esse comportamento é desejável do ponto de vista estatístico e evidencia:

    * Ausência de *overfitting* severo;
    * Estabilidade do modelo frente à variabilidade amostral;
    * Boa capacidade de generalização, com preservação da ordem de grandeza do erro.

    Em particular, a diferença entre $\mathrm{RMSE} \\approx 0{,}0001$ no treinamento/validação e $\mathrm{RMSE} \\approx 0{,}0002$ no teste sugere que o *ensemble* foi capaz de capturar estruturas reais do problema físico-químico subjacente, em vez de apenas ajustar regularidades espúrias específicas do conjunto de treinamento. Esse resultado reforça a adequação da abordagem $\Delta$-*learning* como um compromisso eficiente entre custo computacional e precisão preditiva no contexto da química computacional orientada por dados.
    </div>
    """,
        unsafe_allow_html=True)


    st.markdown('---')
    st.image(MEDIA_DIR / 'footer.webp')


# Tutorial
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

    st.markdown('---')
    render_image_no_zoom(MEDIA_DIR / "footer.webp")


# Δ-xTB
with tab3:
    st.markdown("<h1 style='text-align: center;'>Previsão Energética</h1>", unsafe_allow_html=True)

    smiles = st.text_input('Digite o SMILES da molécula:')

    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error('SMILES inválido. Verifique a entrada.')
        else:
            # Contagens estruturais
            n_heavy = mol.GetNumHeavyAtoms()
            n_h = sum(atom.GetTotalNumHs() for atom in mol.GetAtoms())
            n_atoms = n_heavy + n_h

            # Heurística de decisão para visualização de hidrogênios da estrutura molecular
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
                caption=f'Visualização 3D aproximada ({'com' if show_h else 'sem'} hidrogênios) - Sujeita a distorções estruturais',
                use_container_width=True
            )
            
            # Botão para download da visualização estrutural
            st.markdown(
                """
                <style>
                div[data-testid='stDownloadButton'] {
                    display: flex;
                    justify-content: center;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            buf.seek(0)
            st.download_button(
                    label='Download',
                    data=buf,
                    file_name='molecule.png',
                    mime='image/png',
                )

            # Execução do xTB
            with st.spinner('Executando xTB e extraindo propriedades...'):
                tmpdir = Path(tempfile.mkdtemp())
                xyz_path = smiles_to_xyz(smiles, tmpdir)
                out_path = run_xtb(xyz_path)
                features = extract_xtb_features(out_path)

            if not features:
                st.error('Falha ao extrair propriedades do xTB.')
            else:
                st.success('Propriedades extraídas com sucesso!')
                st.write(features)

                try:
                    model = joblib.load(BASE_DIR / 'model.pkl')
                    columns_ref = joblib.load(BASE_DIR / 'columns_ref.pkl')

                    X_new = pd.DataFrame([features])
                    for col in columns_ref:
                        if col not in X_new.columns:
                            X_new[col] = 0.0
                    X_new = X_new[columns_ref]

                    pred_energy = model.predict(X_new)[0]
                    st.info(f'**Energia Total (xTB + Δ-learning): {pred_energy:.6f} Eh**')
                except Exception as e:
                    st.error(f'Erro ao carregar o modelo: {e}')

    st.markdown('---')
    render_image_no_zoom(MEDIA_DIR / "footer.webp")


# Info
with tab4:
    st.markdown("<h1 style='text-align: center;'>Informações</h1>", unsafe_allow_html=True)
    st.markdown('Verifique o projeto completo no GitHub: **[$\Delta$-xTB](https://github.com/mateusjmd/delta-xTB)**')

    st.subheader('Desenvolvedores')
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

    st.markdown('---')
    render_image_no_zoom(MEDIA_DIR / "footer.webp")
