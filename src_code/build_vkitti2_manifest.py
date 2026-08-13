import argparse
import re
from pathlib import Path

from depth_manifest import write_manifest


def read_intrinsics(path):
    values = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split()
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            row = dict(zip(header, parts))
            frame = int(row["frame"])
            camera = int(row["cameraID"])
            values[(frame, camera)] = {
                "fx": float(row["K[0,0]"]),
                "fy": float(row["K[1,1]"]),
                "cx": float(row["K[0,2]"]),
                "cy": float(row["K[1,2]"]),
            }
    return values


def build(args):
    rgb_root = Path(args.rgb_root)
    depth_root = Path(args.depth_root)
    textgt_root = Path(args.textgt_root)
    samples = []
    intrinsic_cache = {}

    pattern = f"Scene*/**/frames/rgb/Camera_{args.camera}/rgb_*.jpg"
    rgb_paths = sorted(rgb_root.glob(pattern))

    for rgb_path in rgb_paths:
        rel = rgb_path.relative_to(rgb_root)
        parts = rel.parts
        scene = parts[0]
        variation = parts[1]
        match = re.search(r"rgb_(\d+)\.jpg$", rgb_path.name)
        if not match:
            continue
        frame = int(match.group(1))

        depth_rel = Path(scene) / variation / "frames" / "depth" / f"Camera_{args.camera}" / f"depth_{frame:05d}.png"
        depth_path = depth_root / depth_rel
        intrinsic_path = textgt_root / scene / variation / "intrinsic.txt"
        if not depth_path.exists() or not intrinsic_path.exists():
            continue

        key = str(intrinsic_path)
        if key not in intrinsic_cache:
            intrinsic_cache[key] = read_intrinsics(intrinsic_path)
        intrinsics = intrinsic_cache[key].get((frame, args.camera))
        if intrinsics is None:
            continue

        samples.append(
            {
                "id": f"vkitti2:{scene}:{variation}:cam{args.camera}:{frame:05d}",
                "dataset": "vkitti2",
                "scene": scene,
                "variation": variation,
                "frame_index": frame,
                "image_path": str(rgb_path.resolve()),
                "target_path": str(depth_path.resolve()),
                "target_encoding": "depth_png_cm",
                "target_source": "vkitti2_renderer_depth",
                "max_depth_m": args.max_depth,
                **intrinsics,
            }
        )

    if not samples:
        raise ValueError("No Virtual KITTI 2 pairs found. Check the three extracted archive roots.")

    write_manifest(samples, args.output)
    print(f"Saved {len(samples)} Virtual KITTI 2 samples to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-root", required=True)
    parser.add_argument("--depth-root", required=True)
    parser.add_argument("--textgt-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--max-depth", type=float, default=100.0)
    build(parser.parse_args())
