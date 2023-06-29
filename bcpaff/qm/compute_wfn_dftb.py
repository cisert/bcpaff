"""
© 2023, ETH Zurich
"""

import argparse
import json
import os
import subprocess
from typing import List, Optional

from openbabel.pybel import readfile
from rdkit import Chem

from bcpaff.utils import DFTBPLUS_DATA_PATH, SEED

periodic_table = Chem.GetPeriodicTable()
hubbard_derivs_dict = {
    1: "H = -0.1857",
    8: "O = -0.1575",
    35: "Br = -0.0573",
    6: "C = -0.1492",
    17: "Cl = -0.0697",
    9: "F = -0.1623",
    53: "I = -0.0433",
    7: "N = -0.1535",
    15: "P = -0.14",
    16: "S = -0.11",
    20: "Ca	= -0.0340",
    19: "K= -0.0339",
    12: "Mg	= -0.02",
    11: "Na	= -0.0454",
    30: "Zn = -0.03",
}  # https://dftb.org/parameters/download/3ob/3ob-3-1-cc (accessed 06.02.23)


def get_max_angular_momentum(atomic_num: int) -> str:
    """Figure out maximum angular momentum we should take into account

    Parameters
    ----------
    atomic_num : int
        atomic number

    Returns
    -------
    str
        descriptions of maximum angular momentum for given atom type
    """
    element_symbol = periodic_table.GetElementSymbol(atomic_num)
    if atomic_num <= 2:  # first period
        max_angular_momentum = "s"
    elif atomic_num <= 10:  # second period
        max_angular_momentum = "p"
    elif atomic_num <= 18:  # third period
        max_angular_momentum = "d"
    elif atomic_num > 18:
        max_angular_momentum = "f"
    return f"{element_symbol} = {max_angular_momentum}"


def get_hubbard_derivs(atomicnums: List[int]) -> str:
    """Get Hubbard values for set of atomic numbers"""
    hubbard_str = "    "
    for a in atomicnums:
        if a in hubbard_derivs_dict:
            hubbard_str += f"\n    {hubbard_derivs_dict[a]}"
    hubbard_str += "\n  }"
    return hubbard_str


def generate_dftb_input_file(xyz_path: str, qm_method: str, implicit_solvent: Optional[str] = None):
    """Generate DFTB+ instructions"""

    basepath = os.path.dirname(xyz_path)
    json_path = os.path.join(basepath, "chrg_uhf.json")

    with open(json_path, "r") as f:
        chrg_uhf = json.load(f)

    mol = next(readfile("xyz", xyz_path))
    atomicnums = sorted(set([a.atomicnum for a in mol.atoms]))
    max_angular_momentum = "  " + "\n    ".join([get_max_angular_momentum(a) for a in atomicnums])

    if implicit_solvent == "water":
        solvation = "Solvation = GeneralisedBorn {\n  "
        solvation += f'  ParamFile = "{os.path.join(DFTBPLUS_DATA_PATH, "param_gbsa_h2o.txt")}"'
        solvation += "\n  }\n"
    elif implicit_solvent is None:
        solvation = ""

    if chrg_uhf["num_unpaired_electrons"] == 0:
        spin_polarisation = ""
    else:
        spin_polarisation = "SpinPolarisation = Colinear {\n  "
        spin_polarisation += f'  UnpairedElectrons = {chrg_uhf["num_unpaired_electrons"]}'
        spin_polarisation += "\n  }\n"

    if qm_method == "dftb3":
        corrections = "ThirdOrderFull = Yes\n"
        corrections += "  Filling = Fermi {\n    Temperature [K] = 300\n  }\n"
        corrections += "  HubbardDerivs {"
        corrections += get_hubbard_derivs(atomicnums)
        corrections += "\n  HCorrection = Damping { \n    Exponent = 4.00 \n  }"
    else:
        corrections = ""

    cmd_txt = f"""
Geometry = xyzFormat {{
  <<< "{xyz_path}"
}}

Driver = {{}}

Hamiltonian = DFTB {{
  Charge = {chrg_uhf["charge"]}
  Scc = Yes
  MaxSCCIterations = 1000
  {corrections}
  SlaterKosterFiles = Type2FileNames {{
    Prefix = "{os.path.join(DFTBPLUS_DATA_PATH, "recipes/slakos/download/3ob/3ob-3-1/")}" 
    Separator = "-"
    Suffix = ".skf"
  }}
  MaxAngularMomentum {{
  {max_angular_momentum}
  }}
  {solvation}
  {spin_polarisation}
}}

Options {{
    WriteDetailedXML = Yes
    RandomSeed = {SEED}
}}

Analysis {{
    WriteEigenvectors = Yes
}}

ParserOptions {{
    ParserVersion = 7
}}
"""

    dftb_in_hsd = os.path.join(basepath, "dftb_in.hsd")
    with open(dftb_in_hsd, "w") as f:
        f.write(cmd_txt)


def compute_wfn_dftb(xyz_path: str, qm_method: str, implicit_solvent: Optional[str] = None) -> str:
    """Run DFTB+ compute to obtain wavefunction (detailed.xml)

    Parameters
    ----------
    xyz_path : str
        path to XYZ file of protein-ligand complex
    qm_method : str
        which method to use
    implicit_solvent : Optional[str], optional
        which implicit solvent to use, by default None

    Returns
    -------
    str
        wavefunction savepath

    Raises
    ------
    ValueError
        if running the DFTB+ computation failed
    """
    basepath = os.path.dirname(xyz_path)

    generate_dftb_input_file(xyz_path, qm_method, implicit_solvent=implicit_solvent)

    cmd_line_output = os.path.join(basepath, "dftb_cmd_out.log")
    f = open(cmd_line_output, "w+")  # write a new file each time
    cmd = ["dftb+"]
    completed_process = subprocess.run(
        cmd,
        stdout=f,
        stderr=subprocess.STDOUT,
        cwd=basepath,
    )
    return_code = completed_process.returncode

    if return_code != 0:
        raise ValueError(f"{xyz_path} failed with {return_code}")
    f.close()
    wfn_savepath = os.path.join(basepath, "detailed.xml")
    return wfn_savepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz_path", type=str, required=True)
    parser.add_argument("--qm_method", type=str, default="dftb")
    parser.add_argument("--solvent", type=str, default=None)
    args = parser.parse_args()
    wfn_savepath = os.path.join(os.path.dirname(args.xyz_path), "detailed.xml")
    if os.path.exists(wfn_savepath):
        print(f"detailed.xml already exists: {wfn_savepath}", flush=True)
    else:
        wfn_savepath = compute_wfn_dftb(args.xyz_path, args.qm_method, implicit_solvent=args.solvent)
