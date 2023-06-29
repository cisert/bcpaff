"""
© 2023, ETH Zurich
"""

import os

import pandas as pd

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_PATH, "data")
PROCESSED_DATA_PATH = os.path.join(ROOT_PATH, "processed_data")
BASE_OUTPUT_DIR = os.path.join(PROCESSED_DATA_PATH, "model_runs")
REPORT_PATH = os.path.join(PROCESSED_DATA_PATH, "reports")
ANALYSIS_PATH = os.path.join(PROCESSED_DATA_PATH, "analysis")
paths = [DATA_PATH, PROCESSED_DATA_PATH, BASE_OUTPUT_DIR, REPORT_PATH, ANALYSIS_PATH]
for path in paths:
    os.makedirs(path, exist_ok=True)

ELEMENTS = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]

ATOM_NEIGHBOR_IDS = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4, "P": 5, "S": 6, "Cl": 7, "Br": 8, "I": 9, "*": 10}
OTHER = len(ATOM_NEIGHBOR_IDS)  #  no +1 needed since 0-indexed
METALS = [
    "Li",
    "Be",
    "Na",
    "Mg",
    "Al",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
]
ELEMENT_NUMS = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
ATOM_DICT = {
    0: 0,  # BCPs
    1: 1,  # hydrogen
    6: 2,  # carbon
    7: 3,  # nitrogen
    8: 4,  # oxygen
    9: 5,  # fluorine
    15: 6,  # phosphorus
    16: 7,  # sulphure
    17: 8,  # chlorine
    35: 9,  # bromine
    53: 10,  # iodine
}

SEED = 1234
BOHR2ANGSTROM = 0.529177249
ANGSTROM2BOHR = 1 / BOHR2ANGSTROM

DEFAULT_PROPS = [
    "density",
    "laplacian",
    "elf",
    "lol",
    "rdg",
    "sign_lap_rho",
    "gradient_norm",
    "ellipticity",
    "eta_index",
]
DEFAULT_PROPS_CRITIC2 = ["density", "laplacian", "gradient_norm"]
HPARAMS = pd.read_csv(os.path.join(ROOT_PATH, "hparam_files", "hparams_bcp_props.csv"))
DFTBPLUS_DATA_PATH = os.path.join(DATA_PATH, "dftb+")


DATASETS_AND_SPLITS = {
    "pdbbind": ["random"],
    "pde10a": [
        "random",
        "temporal_2011",
        "temporal_2012",
        "temporal_2013",
        "aminohetaryl_c1_amide",
        "c1_hetaryl_alkyl_c2_hetaryl",
        "aryl_c1_amide_c2_hetaryl",
    ],
}
