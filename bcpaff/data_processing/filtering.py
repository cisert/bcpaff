"""
© 2023, ETH Zurich
"""

import os

import pandas as pd

from bcpaff.utils import DATA_PATH

other_binding_sites_txt = os.path.join(DATA_PATH, "pdbbind/verified_other_binding_sites.txt")
OTHER_BINDING_SITES = pd.read_csv(other_binding_sites_txt, sep=";", header=4).pdb_id.tolist()


ALLOSTERIC_BINDING_SITES = []
with open(
    os.path.join(DATA_PATH, "pdbbind/PDBbind_v2019_plain_text_index/plain-text-index/index/INDEX_general_PL.2019"), "r"
) as f:
    for line in f.readlines():
        if "allosteric" in line.lower():
            ALLOSTERIC_BINDING_SITES.append(line.split(" ")[0])
