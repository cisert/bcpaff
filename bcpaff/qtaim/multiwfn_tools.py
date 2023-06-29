"""
© 2023, ETH Zurich
"""

import argparse
import os
import subprocess
from sys import platform

from rdkit import Chem

from bcpaff.qtaim import multiwfn_commands as cmds
from bcpaff.utils import ROOT_PATH

if platform == "linux" or platform == "linux2":
    MULTIWFN_BINARY = os.path.join(ROOT_PATH, "multiwfn", "Multiwfn_noGUI")
elif platform == "darwin":
    MULTIWFN_BINARY = os.path.join(ROOT_PATH, "multiwfn", "Multiwfn_noGUI")
MULTIWFN_ENV = {"Multiwfnpath": os.path.dirname(MULTIWFN_BINARY), "OMP_STACKSIZE": "200M"}


def generate_instructions(basepath, only_intermolecular, only_bcps, num_ligand_atoms, include_esp=True):
    outstring = cmds.find_cps_and_paths()
    if only_intermolecular:
        atom_idxs1 = f"1-{num_ligand_atoms}"  # ligand needs to be first in wavefunction geometry
        atom_idxs2 = f"{num_ligand_atoms+1}-99999"  # everything else
        outstring += cmds.keep_only_intermolecular(atom_idxs1, atom_idxs2)
    if only_bcps:
        outstring += cmds.remove_all_but_bcps()
    outstring += cmds.save_paths()
    outstring += cmds.save_cps(include_esp=include_esp)
    outstring += cmds.save_cps_to_pdb()
    outstring += cmds.save_paths_to_pdb()
    outstring += cmds.exit_gracefully()
    outfile = os.path.join(basepath, "multiwfn_instructions.txt")
    with open(outfile, "w") as f:
        f.write(outstring)
    return outfile


def run_multiwfn_analysis(wfn_file, only_intermolecular=False, only_bcps=False, num_ligand_atoms=-1, include_esp=True):
    basepath = os.path.dirname(wfn_file)
    multiwfn_cmd_out = os.path.join(basepath, "multiwfn_cmd_out.log")
    f_out = open(multiwfn_cmd_out, "w+")  # write a new file each time
    instructions = generate_instructions(
        basepath,
        only_intermolecular=only_intermolecular,
        only_bcps=only_bcps,
        num_ligand_atoms=num_ligand_atoms,
        include_esp=include_esp,
    )
    f_in = open(instructions, "r")
    completed_process = subprocess.run(
        [MULTIWFN_BINARY, wfn_file], stdin=f_in, stdout=f_out, stderr=subprocess.STDOUT, cwd=basepath, env=MULTIWFN_ENV
    )
    f_in.close()
    f_out.close()
    return_code = completed_process.returncode
    if return_code != 0:
        raise ValueError(f"{wfn_file} failed with return_code {return_code}")
    cp_file = os.path.join(basepath, "CPs.txt")
    paths_file = os.path.join(basepath, "paths.txt")
    cpprop_file = os.path.join(basepath, "CPprop.txt")
    return cp_file, cpprop_file, paths_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wfn_file", type=str, required=True)
    parser.add_argument("--ligand_sdf", type=str, required=True)
    parser.add_argument("--only_intermolecular", action="store_true", default=False)
    parser.add_argument("--only_bcps", action="store_true", default=False)
    parser.add_argument("--no_esp", action="store_false", default=True, dest="include_esp")
    args = parser.parse_args()
    cp_file, cpprop_file, paths_file = run_multiwfn_analysis(
        args.wfn_file,
        only_intermolecular=args.only_intermolecular,
        only_bcps=args.only_bcps,
        num_ligand_atoms=next(Chem.SDMolSupplier(args.ligand_sdf, removeHs=False, sanitize=False)).GetNumAtoms(),
        include_esp=args.include_esp,
    )
