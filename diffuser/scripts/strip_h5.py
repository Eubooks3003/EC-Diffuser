#!/usr/bin/env python3
"""
Strip a robomimic / MimicGen H5 down to just the pieces needed to boot the
env and replay the first state of each demo:

  - data.attrs  (env_args JSON, mask, etc.)
  - per-demo attrs (model_file XML, num_samples, ...)
  - demo/states[:1]  (or demo/states/states[:1] if that layout)

Drops actions, full state trajectories, obs/*, rewards, dones. A ~5 GB file
typically shrinks to tens of MB.

Usage:
  python strip_h5.py <src.hdf5> <dst.hdf5>
"""
import sys
import h5py


def main(src_path: str, dst_path: str) -> None:
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        data_src = src["data"]
        data_dst = dst.create_group("data")

        for k, v in data_src.attrs.items():
            data_dst.attrs[k] = v

        if "mask" in src:
            src.copy("mask", dst)

        demos = sorted(data_src.keys(), key=lambda s: int(s.split("_")[-1]))
        for name in demos:
            d_src = data_src[name]
            d_dst = data_dst.create_group(name)

            for k, v in d_src.attrs.items():
                d_dst.attrs[k] = v

            if "states" in d_src:
                s = d_src["states"]
                if isinstance(s, h5py.Dataset):
                    d_dst.create_dataset("states", data=s[:1])
                else:
                    inner = d_dst.create_group("states")
                    if "states" in s:
                        inner.create_dataset("states", data=s["states"][:1])

    print(f"wrote {dst_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
