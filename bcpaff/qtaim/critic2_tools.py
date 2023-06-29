"""
© 2023, ETH Zurich
"""

import os
import subprocess

from bcpaff.qtaim.qtaim_reader import QtaimPropsCritic2
from bcpaff.utils import DATA_PATH, ROOT_PATH

CRITIC2_BINARY = os.path.join(ROOT_PATH, "critic2", "src", "critic2")
CRITIC2_ENV = {"CRITIC_HOME": os.path.join(ROOT_PATH, "critic2")}


def generate_critic2_instructions(basepath):
    """
    Generate instructions for critic2 calculation (search for critical points in DFTB+ wavefunction)
    """
    cmd_str = f"""
molecule {os.path.join(basepath, "pl_complex.xyz")}
zpsp h 1 c 4 n 5 o 6 f 7 p 5 s 6 cl 7 br 7 i 7
load {os.path.join(basepath, "detailed.xml")} {os.path.join(basepath, "eigenvec.bin")} {os.path.join(DATA_PATH, "dftb+/recipes/slakos/download/3ob/3ob-3-1/wfc.3ob-3-1.hsd")} core
auto
cpreport {os.path.join(basepath, "cps.xyz")}
"""
    instructions_file = os.path.join(basepath, "input.cri")
    with open(instructions_file, "w") as f:
        f.write(cmd_str)
    return instructions_file


def run_critic2_analysis(wfn_file):
    """Run critic2 to find critical points (Multiwfn doesn't work for DFTB+ generated wavefunctions.)
    No need to specify charges, this is already read from the DFTB+ wavefunction file (see email 03.02.23)

    Parameters
    ----------
    wfn_file : str
        detailed.xml file from DFTB+

    Returns
    -------
    str
        path to output.cri file with CP information

    Raises
    ------
    ValueError
        in case subprocess returns non-zero return code
    """
    basepath = os.path.dirname(wfn_file)
    output_cri = os.path.join(basepath, "output.cri")
    f_out = open(output_cri, "w+")  # write a new file each time
    instructions_file = generate_critic2_instructions(
        basepath,
    )
    completed_process = subprocess.run(
        [CRITIC2_BINARY, instructions_file], stdout=f_out, stderr=subprocess.STDOUT, cwd=basepath, env=CRITIC2_ENV
    )
    f_out.close()
    return_code = completed_process.returncode
    if return_code != 0:
        raise ValueError(f"{wfn_file} failed with return_code {return_code}")
    return output_cri
