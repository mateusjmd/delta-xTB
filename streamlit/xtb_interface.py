# xtb_interface.py
import subprocess
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
import subprocess
from pathlib import Path
import platform
import tarfile
import urllib.request
import os

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
def get_xtb_binary() -> Path:
    """
    Obtém o binário multiplataforma (Windows/Linux) do xTB 
    """
    system = platform.system()

    # Windows: usa o binário versionado
    if system == "Windows":
        base = Path(__file__).resolve().parents[1]
        xtb = base / "xtb-windows" / "bin" / "xtb.exe"

        if not xtb.exists():
            raise FileNotFoundError("xTB.exe não encontrado no projeto.")

        return xtb

    # Linux: baixa dinamicamente
    elif system == "Linux":
        tmp_dir = Path("/tmp/xtb")
        bin_path = tmp_dir / "bin" / "xtb"

        if bin_path.exists():
            return bin_path

        tmp_dir.mkdir(parents=True, exist_ok=True)

        url = (
            "https://github.com/grimme-lab/xtb/releases/download/v6.7.1/xtb-6.7.1-linux-x86_64.tar.xz"
        )

        tar_path = tmp_dir / "xtb.tar.xz"

        # Download
        urllib.request.urlretrieve(url, tar_path)

        # Extração
        with tarfile.open(tar_path) as tar:
            tar.extractall(tmp_dir)

        # O tar cria algo como xtb-6.7.1/bin/xtb
        extracted = next(tmp_dir.glob("xtb-dist/bin/xtb"))

        os.chmod(extracted, 0o755)

        return extracted

    else:
        raise RuntimeError(f"Sistema operacional não suportado: {system}")


def run_xtb(xyz_path: Path, gfn: int = 2) -> Path | None:
    """
    Executa o xTB de forma multiplataforma (Windows / Linux)
    e retorna o caminho do arquivo .out.
    """

    xyz_path = Path(xyz_path).resolve()
    workdir = xyz_path.parent
    out_path = workdir / "xtb.out"

    xtb_bin = get_xtb_binary()

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

    except Exception as e:
        print(f"[ERRO] Execução do xTB falhou: {e}")
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
