"""
© 2023, ETH Zurich
"""

import os
import subprocess
from typing import List, Optional


def generate_cmds_file(
    cmds_file_path: str,
    config_files: List[str],
    pickle_file: str,
    dataset: str = "pdbbind",
    split_type: str = "random",
    base_output_dir: Optional[str] = None,
    overwrite=False,
    hparam_file: Optional[str] = None,
    num_epochs: int = 300,
) -> int:
    lines = []
    for config_file in config_files:
        python_cmd = "python -m bcpaff.ml.train"
        python_cmd += f" --pickle_file {pickle_file}"
        python_cmd += f" --config_file {config_file}"
        python_cmd += f" --dataset {dataset}"
        python_cmd += f" --split_type {split_type}"
        python_cmd += f" --num_epochs {num_epochs}"
        if base_output_dir is not None:
            python_cmd += f" --base_output_dir {base_output_dir}"
        if overwrite:
            python_cmd += " --overwrite"
        if hparam_file is not None:
            python_cmd += f" --hparam_file {hparam_file}"
        lines.append(python_cmd)
    with open(cmds_file_path, "w") as f:
        f.write("\n".join(lines))
    return len(lines)


def generate_bash_script(
    script_path: str,
    cmds_file_path: str,
    time: str = "04:00:00",
    num_cores: int = 4,
    memory: int = 16,
):
    out_files_folder = os.path.join(os.path.dirname(script_path), "out_files")
    os.makedirs(out_files_folder, exist_ok=True)
    with open(script_path, "w") as f:
        f.write(
            f"""#!/bin/bash

#SBATCH --job-name=bcp_model_run               # Job name
#SBATCH -n {num_cores}                          # Number of CPU cores
#SBATCH --mem-per-cpu={int(memory/num_cores*1024)}   # Memory per CPU in MB
#SBATCH --gpus=1                                # Number of GPUs
#SBATCH --time={time}                         # Maximum execution time (HH:MM:SS)
#SBATCH --tmp=4000                              # Total scratch for job in MB
#SBATCH --output {os.path.join(out_files_folder, "out_%A_%a.out")}                   # Standard output
#SBATCH --error {os.path.join(out_files_folder, "out_%A_%a.out")}   # Standard error


source ~/.bashrc; 
source activate bcpaff; 
export CMDS_FILE_PATH={cmds_file_path}
export cmd=$(head -$SLURM_ARRAY_TASK_ID $CMDS_FILE_PATH|tail -1)
echo "=========SLURM_COMMAND========="
echo $cmd
echo "=========SLURM_COMMAND========="
eval $cmd
"""
        )


def submit_job(script_path: str, num_lines: int = None, cluster_options: Optional[str] = None):
    job_id = None
    if cluster_options == "no_cluster":
        if num_lines is not None:  # job array --> run iteratively
            env = os.environ
            for i in range(num_lines):
                env["SLURM_ARRAY_TASK_ID"] = f"{i + 1}"  # not zero-indexed
                completed_process = subprocess.run(f"bash {script_path}", shell=True, env=env)
        else:  # no job array --> can run directly
            completed_process = subprocess.run(f"bash {script_path}", shell=True, env=env)
    else:
        if num_lines is not None:
            if cluster_options is None:
                cmd_str = f"sbatch --parsable --array=1-{num_lines} < {script_path}"
            else:
                cmd_str = (
                    f"sbatch --dependency=after:{cluster_options}:+5 --parsable --array=1-{num_lines} < {script_path}"
                )
            completed_process = subprocess.run(cmd_str, shell=True, stdout=subprocess.PIPE, universal_newlines=True)
        else:
            if cluster_options is None:
                cmd_str = f"sbatch --parsable < {script_path}"
            else:
                cmd_str = f"sbatch --dependency=after:{cluster_options}:+5 --parsable < {script_path}"
            completed_process = subprocess.run(cmd_str, shell=True, stdout=subprocess.PIPE, universal_newlines=True)
        job_id = completed_process.stdout.rstrip("\n")
    if completed_process.returncode != 0:
        print(completed_process.returncode)
    return job_id


if __name__ == "__main__":
    submit_job(pickle_file="test", config_file="test2", no_cluster=True)
