import streamlit as st
import pandas as pd
import traceback
import io
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import AllChem
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import os
import shlex

def smiles_to_xyz_single(smiles: str, tmpdir: Path) -> Path | None:
    mol = Chem.MolFromSmiles(smiles)
    tmpdir_path = Path(tmpdir)
    
    # Criar o diretório se não existir
    tmpdir_path.mkdir(parents=True, exist_ok=True)
    
    if mol is None:
        return None
    
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    res = AllChem.EmbedMolecule(mol, params)
    
    if res != 0:
        res = AllChem.EmbedMolecule(mol, useRandomCoords=True)
        if res != 0:
            return None
    
    try:
        AllChem.UFFOptimizeMolecule(mol)
    except Exception:
        return None
    
    xyz_block = Chem.MolToXYZBlock(mol)
    tmp_xyz = tmpdir_path / "molecule.xyz"
    tmp_xyz.write_text(xyz_block)
    return tmp_xyz






#Deixar barra de opções invisível
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# Inicializa a sessão (se não existir)
if "pagina" not in st.session_state:
    st.session_state.pagina = "Início"

# Sidebar com botões
st.sidebar.button("Início", on_click=lambda: st.session_state.update({"pagina": "Início"}))
st.sidebar.button("Como usar", on_click=lambda: st.session_state.update({"pagina": "Como usar"}))
st.sidebar.button("Total Energy", on_click=lambda: st.session_state.update({"pagina": "Total Energy"}))
st.sidebar.button("Mais informações", on_click=lambda: st.session_state.update({"pagina": "MI"}))


# Mostra conteúdo com base no estado
if st.session_state.pagina == "Início":
    #Cabeçalho
    st.markdown("<h1 style='text-align: center;'>Total Energy</h1>", unsafe_allow_html=True)
    st.subheader('Você poderá calcular rapidamente e com alta precisão a energia total de uma molécula a partir de seu SMILES')
    st.markdown("---", unsafe_allow_html=True)
            
elif st.session_state.pagina == "Como usar":
    st.markdown("<h1 style='text-align: center;'>Como usar</h1>", unsafe_allow_html=True)
    st.subheader("Aprenda a utilizar nossa ferramenta de forma simples")
    st.markdown("---", unsafe_allow_html=True)

elif st.session_state.pagina == "Total Energy":
    st.markdown("<h1 style='text-align: center;'>Calcular Total Energy</h1>", unsafe_allow_html=True)
    st.markdown("---", unsafe_allow_html=True)
    smiles = st.text_input("Digite seu smiles")
    if (smiles):
        smiles_to_xyz_single(smiles=smiles, tmpdir=Path('./mols'))
        st.text(f"Seu smiles é {smiles}")

elif st.session_state.pagina == "MI":
    st.markdown("<h1 style='text-align: center;'>Mais informações</h1>", unsafe_allow_html=True)
    st.subheader("Veja todo o processo utilizado pelo site para obter a energia total a partir do SMILES")
    st.markdown("---", unsafe_allow_html=True)