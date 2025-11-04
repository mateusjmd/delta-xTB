# data_pipeline.py
import pandas as pd
from pathlib import Path
from xtb_interface import smiles_to_xyz, run_xtb, extract_xtb_features

def build_xtb_dataset(smiles_list: list[str], output_dir: Path = Path(r"C:\Users\mateus25032\STREAMLIT_TEST\xtb_data")) -> pd.DataFrame:
    """Gera dataset de propriedades xTB a partir de uma lista de SMILES."""
    output_dir.mkdir(exist_ok=True)
    data = []

    for i, smiles in enumerate(smiles_list, 1):
        mol_dir = output_dir / f"mol_{i}"
        mol_dir.mkdir(exist_ok=True)

        xyz_path = smiles_to_xyz(smiles, mol_dir)
        if xyz_path is None:
            print(f"[ERRO] Falha na geração da geometria: {smiles}")
            continue

        out_path = run_xtb(xyz_path)
        if out_path is None:
            print(f"[ERRO] xTB falhou: {smiles}")
            continue

        features = extract_xtb_features(out_path)
        if features:
            features["SMILES"] = smiles
            data.append(features)

    df = pd.DataFrame(data)
    df.to_csv(output_dir / "xtb_dataset.csv", index=False)
    return df
