================================================================================
COMPLETE EXECUTION GUIDELINES FOR AIC FINAL PROJECT
Group 3
Members: 李尹瑄、曾歆喬、莊蕓安、謝欣陵、徐畹茜、黃襄香
================================================================================

[0] PREREQUISITES
================================================================================
--- Installation ---

1. Docker:
   $ sudo apt-get update
   $ sudo apt-get install -y ca-certificates curl
   $ sudo install -m 0755 -d /etc/apt/keyrings
   $ sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
   $ sudo chmod a+r /etc/apt/keyrings/docker.asc
   $ echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   $ sudo apt-get update
   $ sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

2. Add your user to the docker group (avoids needing sudo for docker commands):
   $ sudo usermod -aG docker $USER
   $ newgrp docker

3. exiftool and ffmpeg:
   $ sudo apt-get install -y libimage-exiftool-perl ffmpeg

4. Verify installations:
   $ docker --version
   $ docker compose version
   $ exiftool -ver
   $ ffmpeg -version

--- Hugging Face Login ---

1. Login to Hugging Face:
   $ hf auth login --token <HF_TOKEN>
   $ export HF_USER=<your-huggingface-username>

2. Verify the authentication:
   $ hf auth whoami
================================================================================

[1] STEP 1: UMI PIPELINE
================================================================================
* Run on local machine (no GPU needed).

1. Build `uv` environment. (This step is for UMI pipeline only):
   $ uv sync --package umi
   $ source .venv/bin/activate

2. Record mapping video, gripper calibration video and demo videos, and place 
   them in the following structure:
   <NAME_OF_YOUR_DIR>/
   |-- raw_videos/
   |   |-- .gitkeep
   |   |-- video1.mp4
   |   `-- ...
   `-- ...

3. Run the verification pipeline:
   $ uv run umi run-slam-pipeline umi_pipeline_configs/verify_pipeline_C6.yaml \
       --session-dir {raw_videos_dir_path}

4. Build the dataset:
   $ uv run umi run-slam-pipeline umi_pipeline_configs/build_dataset_C6.yaml \
       --session-dir {raw_videos_dir_path} \
       --task <kitchen|dining_room|living_room>

5. Upload to Hugging Face:
   $ hf upload ${HF_USER}/<repo_id> data/<demo_directory_name>/demos/mapping/object_poses.json

* Our data: YinXuanLi/aicapstone_course_project
  URL: https://huggingface.co/YinXuanLi/aicapstone_course_project
================================================================================

[2] STEP 2: GENERATE SYNTHETIC DATA IN SIMULATION
================================================================================
* Run on: GPU machine, inside Docker container.

1. Build a docker:
   $ make submodules
   $ uv sync
   $ source .venv/bin/activate
   $ make launch-isaaclab-glowsai-4090

2. Build synthetic dataset (Related file: scripts/datagen/generate_aug_scatter.py):
   $ python scripts/datagen/generate_aug_scatter.py \
       --task HCIS-CupStacking-SingleArm-v0 \
       --num_envs 1 \
       --device cuda \
       --enable_cameras \
       --record \
       --use_lerobot_recorder \
       --lerobot_dataset_repo_id ${HF_USER}/<repo_id> \
       --object_poses data/<demo_directory_name>/object_poses.json \
       --aug_multiplier 10 \
       --aug_pos_noise 0.08 \
       --aug_action_noise 0.005 

3. Upload to Hugging Face:
   $ hf upload ${HF_USER}/<repo_id> ~/.cache/huggingface/lerobot/${HF_USER}/<repo_id>/ --repo-type dataset

4. Build codebase tag:
   $ .venv/bin/python -c "
     from huggingface_hub import HfApi
     HfApi().create_tag('${HF_USER}/<repo_id>', tag='v3.0', repo_type='dataset')
     print('tag v3.0 created')
     "

5. (Optional) Build the distribution map of the cups' initial positions for 
   every episodes (Related file: scripts/rollout_record_scatter.py):
   $ python scripts/rollout_record_scatter.py \
       --json ~/.cache/huggingface/lerobot/${HF_USER}/<repo_id>/scatter_episodes_success.json \
       --annotate

* Our dataset: qiaoceng/AIC-data_augment-v2
  URL: https://huggingface.co/datasets/qiaoceng/AIC-data_augment-v2
================================================================================

[3] STEP 3: TRAIN A POLICY
================================================================================
* Run on GPU machine and host (not inside Docker).

1. Train a policy:
   $ CUDA_VISIBLE_DEVICES=0 lerobot-train \
       --dataset.repo_id=qiaoceng/AIC-data_augment-v2 \
       --dataset.image_transforms.enable=true \
       --policy.type=act \
       --output_dir=outputs/train/act-v3 \
       --job_name=cupstacking \
       --policy.device=cuda \
       --wandb.enable=true \
       --policy.repo_id=qiaoceng/AIC-act-v3-100000 \
       --policy.chunk_size=50 \
       --policy.n_action_steps=50 \
       --batch_size=32 \
       --steps=100000 \
       --save_freq=20000

2. (Optional) Upload the specific checkpoint to Hugging Face:
   $ hf upload qiaoceng/AIC-act-v3-080000 outputs/train/act-v3/checkpoints/080000/pretrained_model --repo-type model

* Our policy: qiaoceng/AIC-act-v3-080000
  URL: https://huggingface.co/qiaoceng/AIC-act-v3-080000
================================================================================

[4] STEP 4: EVALUATE IN SIMULATION (ROLLOUT)
================================================================================
* Run on: GPU machine, inside Docker container.

1. Download policy:
   $ hf download qiaoceng/AIC-act-v3-080000 --local-dir checkpoints/act-v3-080000

2. Evaluate in simulator (Related file: scripts/rollout_record_platform.py):
   $ python scripts/rollout_record_platform.py \
       --task=eval/cup_stacking_eval.py \
       --policy_type=lerobot-act \
       --policy_checkpoint_path=checkpoints/act-v3-080000 \
       --policy_action_horizon=1 \
       --device=cuda \
       --enable_cameras \
       --headless \
       --eval_rounds=50 \
       --episode_length_s=20 

   NOTE: `rollout_record_platform.py` can save the simulation videos and create 
         a .json file to record the initial positions of the blue cups and the 
         pink cups.

3. (Optional) Build the distribution map of the cups' initial positions 
   (Related file: scripts/rollout_record_scatter.py):
   $ python scripts/rollout_record_scatter.py \
       --json [Path to the json file] \
       --annotate
================================================================================