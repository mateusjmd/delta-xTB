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
def _get_xtb_windows() -> Path:
    base = Path(__file__).resolve().parents[1]
    xtb = base / "xtb-windows" / "bin" / "xtb.exe"

    if not xtb.exists():
        raise FileNotFoundError(
            "xTB.exe não encontrado. Verifique a pasta xtb-windows."
        )

    return xtb


def _get_xtb_linux_local() -> Path:
    base = Path(__file__).resolve().parents[1]
    xtb = base / "xtb-linux" / "bin" / "xtb"

    if not xtb.exists():
        raise FileNotFoundError("xTB Linux local não encontrado.")

    if not os.access(xtb, os.X_OK):
        raise PermissionError(
            "xTB Linux encontrado, mas não é executável (chmod +x necessário)."
        )

    return xtb


def _get_xtb_linux_dynamic() -> Path:
    tmp_dir = Path("/tmp/xtb")
    bin_path = tmp_dir / "xtb-dist" / "bin" / "xtb"

    if bin_path.exists():
        return bin_path

    tmp_dir.mkdir(parents=True, exist_ok=True)

    url = (
        "https://github.com/grimme-lab/xtb/releases/download/v6.7.1/xtb-6.7.1-linux-x86_64.tar.xz"
    )

    tar_path = tmp_dir / "xtb.tar.xz"

    urllib.request.urlretrieve(url, tar_path)

    with tarfile.open(tar_path) as tar:
        tar.extractall(tmp_dir)

    if not bin_path.exists():
        raise RuntimeError("Falha ao localizar o binário do xTB após extração.")

    os.chmod(bin_path, 0o755)

    return bin_path


def get_xtb_binary() -> Path:
    """
    Resolve o binário do xTB de forma multiplataforma.

    Estratégia:
    - Windows: usa binário versionado no repositório
    - Linux:
        1) tenta usar binário Linux local (execução local)
        2) fallback: baixa binário em diretório temporário (cloud/servidor)
    """
    system = platform.system()

    if system == "Windows":
        return _get_xtb_windows()

    elif system == "Linux":
        try:
            return _get_xtb_linux_local()
        except (FileNotFoundError, PermissionError):
            return _get_xtb_linux_dynamic()

    else:
        raise RuntimeError(f"Sistema operacional não suportado: {system}")


def run_xtb(xyz_path: Path, gfn: int = 2) -> Path | None:
    """
    Executa o xTB de forma multiplataforma e retorna o caminho do arquivo .out.
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
