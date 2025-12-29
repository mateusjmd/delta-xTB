# xtb_interface.py
import subprocess
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
import subprocess
from pathlib import Path

RANDOM_SEED = 88

# ============================================================
# 1. GERAÇÃO DA GEOMETRIA 3D (.xyz) A PARTIR DO SMILES
# ============================================================
def smiles_to_xyz(smiles: str, output_dir: Path) -> Path | None:
    """
    Converte um SMILES para um arquivo .xyz com geometria otimizada.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = RANDOM_SEED
    res = AllChem.EmbedMolecule(mol, params)
    if res != 0:
        res = AllChem.EmbedMolecule(mol, useRandomCoords=True)
        if res != 0:
            return None
    AllChem.UFFOptimizeMolecule(mol)

    xyz_block = Chem.MolToXYZBlock(mol)
    xyz_path = output_dir / "molecule.xyz"
    xyz_path.write_text(xyz_block)
    return xyz_path

# ============================================================
# 2. EXECUÇÃO DO xTB
# ============================================================
import subprocess
import platform
import os
from pathlib import Path


def run_xtb(xyz_path: Path, gfn: int = 2) -> Path | None:
    """
    Executa o xTB de forma multiplataforma (Windows / Linux)
    e retorna o caminho do arquivo .out.
    """

    xyz_path = Path(xyz_path).resolve()
    workdir = xyz_path.parent
    out_path = workdir / "xtb.out"

    # Diretório base do projeto
    BASE_DIR = Path(__file__).resolve().parents[1]

    # Identificação do sistema operacional
    system = platform.system()

    if system == "Windows":
        xtb_bin = BASE_DIR / "xtb-windows" / "bin" / "xtb.exe"
    elif system == "Linux":
        xtb_bin = BASE_DIR / "xtb-linux" / "bin" / "xtb"
    else:
        raise RuntimeError(f"Sistema operacional não suportado: {system}")

    # Validações explícitas
    if not xtb_bin.exists():
        raise FileNotFoundError(f"xTB não encontrado em: {xtb_bin}")

    if not workdir.exists():
        raise FileNotFoundError(f"Diretório inexistente: {workdir}")

    if system == "Linux" and not os.access(xtb_bin, os.X_OK):
        raise PermissionError(
            f"Binário do xTB existe, mas não possui permissão de execução: {xtb_bin}"
        )

    # Montagem do comando (forma segura para ambos os sistemas)
    cmd = [
        str(xtb_bin),
        str(xyz_path),
        "--opt",
        "--gfn",
        str(gfn),
    ]

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=str(workdir),
                check=True,
                timeout=600,
            )

        return out_path if out_path.exists() else None

    except subprocess.TimeoutExpired:
        print("[ERRO] Execução do xTB excedeu o tempo limite.")
        return None

    except subprocess.CalledProcessError as e:
        print(f"[ERRO] Falha na execução do xTB: {e}")
        return None



# ============================================================
# 3. EXTRAÇÃO DE PROPRIEDADES DO ARQUIVO .OUT
# ============================================================
def parse_xtb_property(prop: str, out_path: Path) -> float | None:
    """
    Extrai uma propriedade específica do arquivo .out do xTB.
    """

    try:
        with open(out_path, "r", encoding="latin-1") as f:
            for line in f:
                match prop:
                    case "dipole":
                        if "molecular dipole" in line and "Debye" in line:
                            return float(line.split()[-2])
                    case "HOMO":
                        if "(HOMO)" in line:
                            return float(line.split()[-2])
                    case "LUMO":
                        if "(LUMO)" in line:
                            return float(line.split()[-2])
                    case "ZPE":
                        if "zero point energy" in line:
                            return float(line.split()[-3])
                    case "H":
                        if "TOTAL ENTHALPY" in line:
                            return float(line.split()[-3])
                    case "U0":
                        if "TOTAL ENERGY" in line:
                            return float(line.split()[3])
                    case "G":
                        if "TOTAL FREE ENERGY" in line:
                            return float(line.split()[-3])
        return None
    except Exception:
        return None


def extract_xtb_features(out_path: Path) -> dict[str, float]:
    """
    Extrai todas as features relevantes do .out e retorna um dicionário.
    """

    features = {}
    for prop in ["dipole", "HOMO", "LUMO", "ZPE", "H", "U0", "G"]:
        val = parse_xtb_property(prop, out_path)
        if val is not None:
            features[prop] = val

    # Calcula propriedades derivadas
    if "HOMO" in features and "LUMO" in features:
        features["gap_HOMO-LUMO"] = features["HOMO"] - features["LUMO"]
    return features
