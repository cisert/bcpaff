"""
© 2023, ETH Zurich
"""

import argparse
import glob
import os

from bcpaff.qm.compute_wfn_psi4 import LEVEL_OF_THEORY, PSI4_OPTIONS


def prepare_cluster_job_psi4(folders, cmd_file_out, args):
    python_file = os.path.join(os.path.dirname(__file__), "compute_wfn.py")
    cmd_strs = []
    for folder in folders:
        pdb_id = os.path.basename(folder)
        pocket_sdf = os.path.join(folder, f"{pdb_id}_pocket_with_hydrogens.sdf")
        ligand_sdf = os.path.join(folder, f"{pdb_id}_ligand_with_hydrogens.sdf")
        cmd_str = f"python {python_file} --ligand_sdf {ligand_sdf} --pocket_sdf {pocket_sdf} --memory {args.memory} --num_cores {args.num_cores} --level_of_theory {args.level_of_theory} --basis_set {args.basis_set}\n"
        cmd_strs.append(cmd_str)
    with open(cmd_file_out, "w") as f:
        f.writelines(cmd_strs)
    print(f"Wrote output to {cmd_file_out}")
    memory_per_core_in_mb = int(args.memory / args.num_cores * 1024)
    bsub_cmd = (
        """bsub -W 24:00 -R "rusage[mem="""
        + str(memory_per_core_in_mb)
        + """,scratch=50000]" -n """
        + str(args.num_cores)
    )
    bsub_cmd += (
        """ -J "qm_bcpaff[1-"""
        + str(len(cmd_strs))
        + """]" "awk -v jindex=\$LSB_JOBINDEX 'NR==jindex' """
        + cmd_file_out
        + """ | bash" """
    )
    return bsub_cmd


def prepare_cluster_job_xtb(folders, cmd_file_out, args):
    python_file = os.path.join(os.path.dirname(__file__), "compute_wfn_xtb.py")
    cmd_strs = []
    for folder in folders:
        pdb_id = os.path.basename(folder)
        pocket_sdf = os.path.join(folder, f"{pdb_id}_pocket_with_hydrogens.sdf")
        ligand_sdf = os.path.join(folder, f"{pdb_id}_ligand_with_hydrogens.sdf")
        cmd_str = f"python {python_file} --ligand_sdf {ligand_sdf} --pocket_sdf {pocket_sdf}\n"
        cmd_strs.append(cmd_str)
    with open(cmd_file_out, "w") as f:
        f.writelines(cmd_strs)
    print(f"Wrote output to {cmd_file_out}")
    bsub_cmd = """
        bsub -W 4:00 -R "rusage[mem=10000]"
        """
    bsub_cmd += (
        """ -J "qm_bcpaff[1-"""
        + str(len(cmd_strs))
        + """]" "awk -v jindex=\$LSB_JOBINDEX 'NR==jindex' """
        + cmd_file_out
        + """ | bash" """
    )
    return bsub_cmd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("search_path", type=str)
    parser.add_argument("method", type=str, default="xtb")
    parser.add_argument("--memory", type=int, default=8)
    parser.add_argument("--num_cores", type=int, default=4)
    parser.add_argument("--level_of_theory", type=str, default=LEVEL_OF_THEORY)
    parser.add_argument("--basis_set", type=str, default=PSI4_OPTIONS["basis"])
    args = parser.parse_args()
    folders = sorted(glob.glob(os.path.join(args.search_path, "*" + os.path.sep)))
    cmd_file_out = os.path.join(os.getcwd(), "qm_commands.txt")
    if os.path.exists(cmd_file_out):
        user_input = ""
        while user_input not in ["y", "n"]:
            user_input = input("Output file exists. Overwrite?").lower()
        if user_input == "n":
            raise ValueError(f"Commands file already exists")

    if args.method == "xtb":
        bsub_cmd = prepare_cluster_job_xtb(folders, cmd_file_out, args)
    elif args.method == "psi4":
        bsub_cmd = prepare_cluster_job_psi4(folders, cmd_file_out, args)
    else:
        raise ValueError(f"args.method must be xtb or psi4, you chose {args.method}")

    print("Use this to run your job:\n\n")
    print(f"conda activate bcpaff_{args.method}\n")
    print(bsub_cmd)
