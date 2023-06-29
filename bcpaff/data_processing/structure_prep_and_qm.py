"""
© 2023, ETH Zurich
"""

import argparse
import os

from rdkit import Chem

from bcpaff.data_processing.manual_structure_prep import full_structure_prep
from bcpaff.qtaim.critic2_tools import run_critic2_analysis
from bcpaff.qtaim.multiwfn_tools import run_multiwfn_analysis
from bcpaff.utils import DATA_PATH, PROCESSED_DATA_PATH

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("structure_id", type=str)
    parser.add_argument("--dataset", type=str, default="pdbbind")
    parser.add_argument("--cutoff", type=float, default=6)
    parser.add_argument("--qm_method", type=str, default="xtb")
    parser.add_argument("--num_cores", type=int, default=1)
    parser.add_argument("--basepath", type=str, default=None)
    parser.add_argument("--output_basepath", type=str, default=None)
    parser.add_argument("--implicit_solvent", type=str, default=None)
    parser.add_argument("--esp", action="store_true", default=False, dest="esp")
    parser.add_argument("--keep_wfn", action="store_true", default=False, dest="keep_wfn")

    args = parser.parse_args()
    if args.output_basepath is None:
        output_basepath = os.path.join(PROCESSED_DATA_PATH, "prepared_structures", args.dataset)
    else:
        output_basepath = args.output_basepath

    if args.basepath is None:
        if args.dataset == "pdbbind":
            basepath = os.path.join(DATA_PATH, args.dataset, "dataset")
        elif args.dataset == "pde10a":
            basepath = os.path.join(DATA_PATH, args.dataset, "pde-10_pdb_bind_format_blinded")
        elif args.dataset == "d2dr":
            basepath = os.path.join(DATA_PATH, args.dataset, "D2DR_complexes_prepared")
    else:
        basepath = args.basepath

    # 1) structure prep
    full_structure_prep(
        basepath,
        structure_id=args.structure_id,
        output_basepath=output_basepath,
        cutoff=args.cutoff,
        dataset=args.dataset,
    )
    out_folder = os.path.join(output_basepath, args.structure_id)

    # 2) run xTB/Psi4
    if args.qm_method == "psi4":
        from bcpaff.qm.compute_wfn_psi4 import compute_wfn_psi4

        # different environments for psi4 and xTB, so need to import depending on use case

        psi4_input_pickle = os.path.join(out_folder, f"psi4_input.pkl")
        compute_wfn_psi4(psi4_input_pickle, num_cores=args.num_cores, memory=140)
        wfn_file = os.path.join(out_folder, "wfn.fchk")
    elif args.qm_method == "xtb":
        from bcpaff.qm.compute_wfn_xtb import compute_wfn_xtb

        xyz_path = os.path.join(out_folder, f"pl_complex.xyz")
        compute_wfn_xtb(xyz_path, args.implicit_solvent)
        wfn_file = os.path.join(out_folder, "molden.input")
    elif args.qm_method.startswith("dftb"):
        from bcpaff.qm.compute_wfn_dftb import compute_wfn_dftb

        xyz_path = os.path.join(out_folder, f"pl_complex.xyz")
        compute_wfn_dftb(xyz_path, args.qm_method, args.implicit_solvent)
        wfn_file = os.path.join(out_folder, "detailed.xml")
    else:
        raise ValueError("Invalid qm_method")

    # 3) run multiwfn/critic2
    ligand_sdf = os.path.join(out_folder, f"{args.structure_id}_ligand_with_hydrogens.sdf")
    if args.qm_method.startswith("dftb"):  # run critic2
        output_cri = run_critic2_analysis(wfn_file)
    else:  # run multiwfn
        cp_file, cpprop_file, paths_file = run_multiwfn_analysis(
            wfn_file=wfn_file,
            only_intermolecular=False,
            only_bcps=False,
            num_ligand_atoms=next(Chem.SDMolSupplier(ligand_sdf, removeHs=False, sanitize=False)).GetNumAtoms(),
            include_esp=args.esp,
        )

    # 4) potential cleanup
    if not args.keep_wfn:
        os.remove(wfn_file)  # to save space, those files get quite big
