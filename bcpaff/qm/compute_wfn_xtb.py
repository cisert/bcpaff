"""
© 2023, ETH Zurich
"""

import argparse
import json
import os
import subprocess

from bcpaff.utils import ROOT_PATH

XTB_INPUT_FILE = os.path.join(ROOT_PATH, "bcpaff", "qm", "xtb.inp")
XTB_ENV = {
    "OMP_STACKSIZE": "4G",
    "OMP_NUM_THREADS": "1",
    "OMP_MAX_ACTIVE_LEVELS": "1",
    "MKL_NUM_THREADS": "1",
}
XTB_BINARY = os.path.join(os.environ.get("CONDA_PREFIX"), "bin", "xtb")


def check_xtb_uhf(cmd_line_output):
    with open(cmd_line_output, "r") as f:
        for line in f:
            if line.startswith("          spin                       :"):
                spin = float(line.rstrip("\n").split()[-1])
                break
    if spin != 0.0:
        error_file = os.path.join(os.path.dirname(cmd_line_output), "uhf_error.txt")
        with open(error_file, "w") as f:
            f.write(f"spin = {spin}")


def compute_wfn_xtb(xyz_path, implicit_solvent=None):
    basepath = os.path.dirname(xyz_path)

    json_path = os.path.join(basepath, "chrg_uhf.json")
    with open(json_path, "r") as f:
        chrg_uhf = json.load(f)

    cmd_line_output = os.path.join(basepath, "xtb_cmd_out.log")
    f = open(cmd_line_output, "w+")  # write a new file each time
    cmd = [
        XTB_BINARY,
        xyz_path,
        "--input",
        XTB_INPUT_FILE,
        "--chrg",
        str(chrg_uhf["charge"]),
        "--uhf",
        str(chrg_uhf["num_unpaired_electrons"]),
        "--molden",
        "--iterations",
        "10000",
    ]
    if implicit_solvent is not None:
        cmd += [f"--{implicit_solvent.split('_')[0]}", implicit_solvent.split("_")[1]]
        # e.g. turn "alpb_water" into "--alpb water"
    completed_process = subprocess.run(
        cmd,
        stdout=f,
        stderr=subprocess.STDOUT,
        cwd=basepath,
        env=XTB_ENV,
    )
    return_code = completed_process.returncode
    check_xtb_uhf(cmd_line_output)

    if return_code != 0:
        raise ValueError(f"{xyz_path} failed with {return_code}")
    f.close()
    molden_file = os.path.join(basepath, "molden.input")
    return molden_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz_path", type=str, required=True)
    args = parser.parse_args()
    molden_savepath = os.path.join(os.path.dirname(args.xyz_path), "molden.input")
    if os.path.exists(molden_savepath):
        print(f"molden.input already exists: {molden_savepath}", flush=True)
    else:
        molden_file = compute_wfn_xtb(args.xyz_path)
