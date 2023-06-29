"""
© 2023, ETH Zurich
"""

import argparse
import os
import pickle
import tempfile
from shutil import rmtree

import psi4

PSI4_OPTIONS = {"basis": "def2-svp"}
LEVEL_OF_THEORY = "wb97x-d"


def compute_wfn_psi4(
    psi4_input_pickle, memory=8, num_cores=1, level_of_theory=LEVEL_OF_THEORY, basis_set=PSI4_OPTIONS["basis"]
):
    # read input data & construct molecule
    with open(psi4_input_pickle, "rb") as f:
        psi4_input = pickle.load(f)
    if not all([m == 1 for m in psi4_input["fragment_multiplicities"]]):
        raise ValueError(f"Radical electrons: {psi4_input_pickle}")
    p4mol = psi4.core.Molecule.from_arrays(
        elez=psi4_input["elez"],
        fragment_separators=psi4_input["fragment_separators"],
        fix_com=True,
        fix_orientation=True,
        fix_symmetry="c1",
        fragment_charges=psi4_input["fragment_charges"],
        fragment_multiplicities=psi4_input["fragment_multiplicities"],
        molecular_charge=psi4_input["molecular_charge"],
        molecular_multiplicity=psi4_input["molecular_multiplicity"],
        geom=psi4_input["geom"],
    )

    # set scratch directory
    psi4_io = psi4.core.IOManager.shared_object()
    local_scratch = os.environ.get("TMPDIR")  # local scratch directory on Euler
    if local_scratch:
        os.environ["PSI_SCRATCH"] = local_scratch
        psi4_io.set_default_path(local_scratch)
        print(f"Using local scratch {local_scratch}.", flush=True)
    else:  # if job didn't request a local scratch directory, use global scratch
        tmp_dir = tempfile.mkdtemp()
        os.environ["PSI_SCRATCH"] = tmp_dir
        psi4_io.set_default_path(tmp_dir)
        print(f"Using tempfile scratch {tmp_dir}.", flush=True)
    # define psi4 settings
    psi4.core.clean()
    psi4.set_num_threads(num_cores)
    psi4.set_memory(f"{int(memory)} GB")
    PSI4_OPTIONS["basis"] = basis_set
    psi4.set_options(PSI4_OPTIONS)

    # calculate wavefunction
    E, wfn = psi4.energy(level_of_theory, return_wfn=True, molecule=p4mol)
    psi4.core.clean()
    if not local_scratch:
        rmtree(tmp_dir)
    wfn_file = os.path.join(os.path.dirname(psi4_input_pickle), "wfn.npy")
    fchk_savepath = os.path.join(os.path.dirname(psi4_input_pickle), "wfn.fchk")
    wfn.to_file(wfn_file)
    psi4.fchk(wfn, fchk_savepath)
    return fchk_savepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--psi4_input_pickle", type=str)
    parser.add_argument("--memory", type=int, default=8)
    parser.add_argument("--num_cores", type=int, default=4)
    parser.add_argument("--level_of_theory", type=str, default=LEVEL_OF_THEORY)
    parser.add_argument("--basis_set", type=str, default=PSI4_OPTIONS["basis"])
    args = parser.parse_args()
    fchk_savepath = os.path.join(os.path.dirname(args.psi4_input_pickle), "wfn.fchk")
    npy_savepath = os.path.join(os.path.dirname(args.psi4_input_pickle), "wfn.npy")
    if os.path.exists(fchk_savepath) or os.path.exists(npy_savepath):
        print(f"Wavefunction already exists: {fchk_savepath}", flush=True)
    else:
        fchk_savepath = compute_wfn_psi4(
            args.psi4_input_pickle,
            memory=int(args.memory),
            num_cores=int(args.num_cores),
            level_of_theory=args.level_of_theory,
            basis_set=args.basis_set,
        )
        print(f"Saved wavefunction to {fchk_savepath}")
