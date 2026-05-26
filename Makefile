APPTAINER_ENV := \
	APPTAINERENV_LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/.singularity.d/libs" \
	APPTAINERENV_PYTHONPATH=/home/joe/Research/code/equidiff/.container_pkgs:/home/joe/Research/code/equidiff/bimanual_suite:/home/joe/Research/code/equidiff:/opt/repos \
	APPTAINERENV_MUJOCO_GL=osmesa

SIF := equidiff_new.sif

DEMO       ?= 0
HDF5       ?= stacking_easy_1000.hdf5
OUT_DIR    ?= replay_outputs
N_DEMO     ?= 1000

.PHONY: replay_demo train make_dataset

replay_demo:
	mkdir -p $(OUT_DIR)
	$(APPTAINER_ENV) apptainer exec --nv $(SIF) \
		python replay_demo.py \
			--hdf5 $(HDF5) \
			--demo $(DEMO) \
			--out_dir $(OUT_DIR)

train:
	APPTAINERENV_WANDB_MODE=disabled \
	APPTAINERENV_LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/.singularity.d/libs" \
	APPTAINERENV_PYTHONPATH=/home/joe/Research/code/equidiff/.container_pkgs:/home/joe/Research/code/equidiff/bimanual_suite:/home/joe/Research/code/equidiff:/opt/repos \
	APPTAINERENV_MUJOCO_GL=osmesa \
	$(APPTAINER_ENV) apptainer exec --nv \
	--bind /tmp/escnn_cache:/opt/venv/lib/python3.10/site-packages/escnn/group/_cache \
	$(SIF) \
		python train.py \
			--config-name=train_equi_diffusion_unet_bimanual_abs \
			dataset_path=/home/joe/Research/code/equidiff/$(HDF5) \
			n_demo=$(N_DEMO) \
			training.rollout_every=1 \
			logging.mode=disabled

make_dataset:
	python write_stacking_dataset.py "/home/joe/Research/code/bimanual_suite/scripts/data/success/data" --out stacking_easy_1000.hdf5 --n_demos 1000
