import argparse

from depth_manifest import read_manifest, write_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    samples = []
    counts = {}
    for path in args.inputs:
        current = read_manifest(path)
        samples.extend(current)
        for sample in current:
            source = sample.get("dataset", "unknown")
            counts[source] = counts.get(source, 0) + 1

    write_manifest(samples, args.output)
    print(f"Saved {len(samples)} concatenated samples to {args.output}")
    for source, count in sorted(counts.items()):
        print(f"  {source}: {count}")
