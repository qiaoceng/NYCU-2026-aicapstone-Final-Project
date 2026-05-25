import argparse
from pxr import Usd, UsdGeom

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd_path", type=str, default="packages/simulator/assets/scenes/kitchen/scene.usd")
    args = parser.parse_args()

    # 開啟 USD 檔案
    stage = Usd.Stage.Open(args.usd_path)
    if not stage:
        print(f"Failed to open USD file: {args.usd_path}")
        return

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default'])

    found = False
    for prim in stage.Traverse():
        # 尋找名稱包含 table, desk 或特定的場景物體
        if "table" in prim.GetName().lower():
            found = True
            print(f"找到桌子 Prim: {prim.GetPath()}")
            bound = bbox_cache.ComputeWorldBound(prim)
            box = bound.ComputeAlignedRange()
            min_pt = box.GetMin()
            max_pt = box.GetMax()
            size = max_pt - min_pt
            
            print(f"  中心點 (Center): [{(min_pt[0]+max_pt[0])/2:.3f}, {(min_pt[1]+max_pt[1])/2:.3f}, {(min_pt[2]+max_pt[2])/2:.3f}]")
            print(f"  尺寸 (Size): X={size[0]:.3f}, Y={size[1]:.3f}, Z={size[2]:.3f}")
            print(f"  範圍 (Bounds):")
            print(f"    X: {min_pt[0]:.3f} ~ {max_pt[0]:.3f}")
            print(f"    Y: {min_pt[1]:.3f} ~ {max_pt[1]:.3f}")
            print(f"    Z: {min_pt[2]:.3f} ~ {max_pt[2]:.3f}")
            print("-" * 40)

    if not found:
        print("在場景中沒有找到名稱包含 'table' 的物件。")

if __name__ == "__main__":
    main()
