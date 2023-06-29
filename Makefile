.ONESHELL:
SHELL=/bin/bash
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate
PROC_DATA_DIR:=$(CURDIR)/processed_data
PDE10A_SPLITS = random temporal_2011 temporal_2012 temporal_2013 aminohetaryl_c1_amide c1_hetaryl_alkyl_c2_hetaryl aryl_c1_amide_c2_hetaryl

# Commands
conda=conda
mamba=mamba
python=python

all: env multiwfn download

make with_conda: env_conda multiwfn download

env: 
	${mamba} env create -f env.yml 
	${mamba} env create -f env_psi4.yml 

env_conda: 
	${conda} env create -f env.yml 
	${conda} env create -f env_psi4.yml 

multiwfn:
	wget http://sobereva.com/multiwfn/misc/Multiwfn_3.8_dev_bin_Linux_noGUI.zip
	unzip Multiwfn_3.8_dev_bin_Linux_noGUI.zip
	mv Multiwfn_3.8_dev_bin_Linux_noGUI multiwfn
	chmod 770 ./multiwfn/Multiwfn_noGUI
	rm Multiwfn_3.8_dev_bin_Linux_noGUI.zip

download: 
	source activate bcpaff; ${python} -m bcpaff.data_processing.download
	python -m bcpaff.data_processing.collect_affinity_data

data_processing: 
	source activate bcpaff; ulimit -s unlimited; ${python} -m bcpaff.data_processing.data_processing --action all --test_run --cluster_options no_cluster

data_processing_report: 
	source activate bcpaff; ${python} -m bcpaff.data_processing.data_processing --action report

ml_experiments: 
	source activate bcpaff; ${python} -m bcpaff.ml.run_all_ml_experiments --cluster_options no_cluster
