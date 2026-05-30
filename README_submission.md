#  Complete Execution Guidelines

## Prerequisites

### Installation
```bash
# Docker
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add your user to the docker group (avoids needing sudo for docker commands)
sudo usermod -aG docker $USER
newgrp docker

# exiftool and ffmpeg
sudo apt-get install -y libimage-exiftool-perl ffmpeg

# Verify installations:
docker --version
docker compose version
exiftool -ver
ffmpeg -version
```

### Hugging Face login
``` bash
hf auth login --token <YOUR_HF_TOKEN>
export HF_USER=<your-huggingface-username>
```

## UMI Pipeline
Run on local machine (no GPU needed).
1. Build `uv` environment. This step is for UMI pipeline only.
   ```bash
   uv sync --package umi
   source .venv/bin/activate
   ```

2. Then, record **mapping video**, **gripper calibration video** and **demo videos** and place the recorded videos like the below structure:
   ```
   <NAME_OF_YOUR_DIR>/
   ├── raw_videos/
   │   ├── .gitkeep
   │   ├── video1.mp4
   │   └── ...
   └── ...
   ```

3. Run the verification pipeline
    ```bash
    uv run umi run-slam-pipeline umi_pipeline_configs/verify_pipeline_C6.yaml \
        --session-dir {raw_videos_dir_path}
    ```

4. Build the dataset
    ```bash
    uv run umi run-slam-pipeline umi_pipeline_configs/build_dataset_C6.yaml \
        --session-dir {raw_videos_dir_path}\
        --task <kitchen|dining_room|living_room>
    ```

5. Upload to Hugging Face
    ```bash
    hf upload ${HF_USER}/<repo_id> data/<demo_directory_name>/demos/mapping/object_poses.json
    ```
#### Our data: [YinXuanLi/aicapstone_course_project](https://huggingface.co/YinXuanLi/aicapstone_course_project)


## Generate Synthetic Data in Simulation
Run on: GPU machine, inside Docker container.

1. Build a docker
    ```bash
    make submodules
    uv sync
    source .venv/bin/activate
    
    make launch-isaaclab-glowsai-4090
    ```
    
2. Build synthetic dataset
    ```bash
    python scripts/datagen/generate_aug_scatter.py \
        --task HCIS-CupStacking-SingleArm-v0 \
        --num_envs 1 \
        --device cuda \
        --enable_cameras \
        --record \
        --use_lerobot_recorder \
        --lerobot_dataset_repo_id ${HF_USER}/<repo_id> \
        --object_poses data/<demo_directory_name>/object_poses.json
        --aug_multiplier 10 \
        --aug_pos_noise 0.08 \
        --aug_action_noise 0.005 
    ```

3. Upload to Hugging Face
    ```bash
    hf upload ${HF_USER}/<repo_id> ~/.cache/huggingface/lerobot/${HF_USER}/<repo_id>/ --repo-type dataset
    ```

4. Build codebase tag
    ```bash
    .venv/bin/python -c "
    from huggingface_hub import HfApi
    HfApi().create_tag('${HF_USER}/<repo_id>', tag='v3.0', repo_type='dataset')
    print('tag v3.0 created')
    "
    ```

5. (Optional) Build the distribution map of the cups' initial positions for every episodes
    ```bash
    python scripts/rollout_record_scatter.py \
        --json ~/.cache/huggingface/lerobot/${HF_USER}/<repo_id>/scatter_episodes_success.json \
        --annotate
    ```

## Train a Policy
Run on GPU machine and host (not inside Docker).

1. 