import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug Cup Pos")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
from simulator.tasks.external import resolve_task
# Make sure the eval tasks are importable
import leisaac
import simulator.tasks

def main():
    task_id = resolve_task("eval/cup_stacking_eval.py")
    env_cfg = parse_env_cfg(task_id, device="cuda", num_envs=1)
    
    # 關閉超時設定，方便我們觀察
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
        
    print("[Debug] Creating environment...")
    env = gym.make(task_id, cfg=env_cfg).unwrapped

    print("[Debug] Resetting environment...")
    obs_dict, _ = env.reset()
    
    # 等待物體穩定下來或抓取當下數值
    for _ in range(5):
        env.step(torch.zeros(1, env.action_space.shape[1], device=env.device))

    blue_cup = env.scene["blue_cup"]
    pink_cup = env.scene["pink_cup"]

    # 取得當前世界座標
    blue_pos = blue_cup.data.root_pos_w[0].cpu().numpy()
    pink_pos = pink_cup.data.root_pos_w[0].cpu().numpy()

    print("=" * 60)
    print("🔵 藍色杯子 (Blue Cup) 世界座標:")
    print(f"   X: {blue_pos[0]:.4f}, Y: {blue_pos[1]:.4f}, Z: {blue_pos[2]:.4f}")
    print("   (在 eval 預設中心應為 X=0.36, Y=-0.4 附近，Z 約 0.12)")
    print("-" * 60)
    print("💗 粉紅色杯子 (Pink Cup) 世界座標:")
    print(f"   X: {pink_pos[0]:.4f}, Y: {pink_pos[1]:.4f}, Z: {pink_pos[2]:.4f}")
    print("   (在 eval 預設中心應為 X=0.46, Y=-0.4 附近，Z 約 0.12)")
    print("=" * 60)

    # 檢查是否為 (0, 0, 0)
    if sum(abs(blue_pos)) < 0.001 or sum(abs(pink_pos)) < 0.001:
        print("🚨 警告：有杯子的座標接近 (0, 0, 0)！")
        print("這代表杯子的 init_state 沒有成功設定，或是被其他設定（例如異常的 object_pose_cfg 或 UMI JSON）覆蓋了！")
    else:
        print("✅ 座標正常，杯子沒有卡在 (0, 0, 0)。")
        
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
