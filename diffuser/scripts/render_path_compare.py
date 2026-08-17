#!/usr/bin/env python
"""
Does a live-rendered frame match the stored training frame, pixel and token wise?

The policy trains on tokens encoded from the preprocessed memmap, but at rollout
it sees tokens encoded from live-rendered frames. If those two render paths
disagree, the policy is fed out-of-distribution observations while training
loss, imagination and the oracle all look healthy.

Three renders of the SAME MuJoCo state are compared:

  stored     the preprocessed memmap frame (what the DLP trained on)
  patched    eval env with EC-Diffuser's vopt/geomgroup monkeypatch
             (eval_utils.py sets geomgroup[0]=0 to hide collision geoms)
  unpatched  eval env without it -- what pi0.5's eval does, and what the
             preprocessing renderer does

Reported per pair: mean/max abs pixel difference, and after encoding through the
task's own DLP, the active-particle count and the token distance. Frame 0 of
each demo is used so this works against stripped HDF5s (which keep states[:1]).

Usage:
    PYTHONPATH=.:.. python scripts/render_path_compare.py \
        --task stack_d1 --n_demos 3 --res 224
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import argparse
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..")),
          os.path.abspath(os.path.join(SCRIPT_DIR, ".."))):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# Defaults point at the machine that holds the RAW 224 image store (the memmaps
# were never copied to matrix -- only the token pkls were). Override per host.
DEF_TOKENS = "/home/ellina/Desktop/data/_matrix_stage/mimicgen_224_wrist_tokens"
DEF_STORE = "/home/ellina/Desktop/data/3D-DLP-mimicgen-data/preprocessed_wrist_224"
DEF_CORE = "/home/ellina/Desktop/data/3D-DLP-mimicgen-data/core"
DEF_LPWM = "/home/ellina/Desktop/lpwm-copy"


def make_env(h5, cams, res, patch):
    """Build the eval env; optionally install the vopt monkeypatch."""
    from robomimic.utils import file_utils as FileUtils, env_utils as EnvUtils
    from robomimic.utils import obs_utils as ObsUtils
    # robomimic keeps the obs-modality registry in module globals; without this
    # an image env raises "argument of type 'NoneType' is not iterable" from
    # get_observation. create_env_for_data_processing does it internally, which
    # is why the preprocessing path never needed it.
    ObsUtils.initialize_obs_utils_with_obs_specs(
        {"obs": {"rgb": [f"{c}_image" for c in cams], "low_dim": []}})
    em = FileUtils.get_env_metadata_from_dataset(h5)
    em.setdefault("env_kwargs", {})
    em["env_kwargs"].update(dict(has_renderer=False, has_offscreen_renderer=True,
                                 use_camera_obs=True, camera_names=list(cams),
                                 camera_heights=res, camera_widths=res))
    env = None
    for extra in ({"use_depth_obs": True}, {"use_obs_depth": True}, {}):
        try:
            env = EnvUtils.create_env_from_metadata(
                env_meta=em, env_name=em.get("env_name"), render=False,
                render_offscreen=True, use_image_obs=True, **extra)
            break
        except TypeError:
            continue
    if env is None:
        raise RuntimeError("could not build env")

    if patch:
        under = env.env if hasattr(env, "env") else env
        if getattr(under, "has_offscreen_renderer", False):
            import mujoco
            orig = under._reset_internal

            def _patched():
                orig()
                ctx = under.sim._render_context_offscreen
                if ctx is not None:
                    o = mujoco.MjvOption()
                    mujoco.mjv_defaultOption(o)
                    o.geomgroup[0] = 0
                    ctx.vopt = o
            under._reset_internal = _patched
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="stack_d1")
    ap.add_argument("--n_demos", type=int, default=3)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--store", default=DEF_STORE)
    ap.add_argument("--core", default=DEF_CORE)
    ap.add_argument("--tokens", default=DEF_TOKENS)
    ap.add_argument("--lpwm", default=DEF_LPWM)
    args = ap.parse_args()
    STORE, CORE, TOKENS = args.store, args.core, args.tokens

    # Load the 2D DLP straight from lpwm-copy so this script needs only torch +
    # robosuite, not the diffuser package (the raw store lives on the render box,
    # which does not have the full EC-Diffuser env).
    for _p in (args.lpwm, os.path.join(args.lpwm, "scripts")):
        if os.path.isdir(_p) and _p not in sys.path:
            sys.path.insert(0, _p)

    import h5py
    meta = json.load(open(f"{STORE}/metadata.json"))
    tm = meta["tasks"][args.task]
    cams = tm["cameras"]
    H, W, _ = meta["image_shape"]
    V = meta["num_views"]
    mm = np.memmap(os.path.join(STORE, tm["memmap_file"]), dtype=np.uint8, mode="r",
                   shape=(tm["total_frames"], V, H, W, 3))
    bnds = tm["demo_boundaries"]
    h5p = f"{CORE}/{args.task}.hdf5"
    print(f"[cfg] task={args.task} cams={cams} res={args.res} store={H}x{W}")

    # DLP for this task (snapshotted next to the token pkl by the preprocessor)
    from dlp_token_common import build_dlp_2d_from_cfg
    dcfg = json.load(open(f"{TOKENS}/{args.task}/dlp_config.json"))
    dlp = build_dlp_2d_from_cfg(dcfg, torch.device(args.device))
    dlp.load_state_dict(torch.load(f"{TOKENS}/{args.task}/dlp_ckpt.pt",
                                   map_location=args.device, weights_only=False))
    dlp.eval()
    img_size = dcfg["image_size"]

    @torch.no_grad()
    def encode(img_hwc):
        t = torch.from_numpy(img_hwc.copy()).float().div(255).permute(2, 0, 1)[None]
        if t.shape[-1] != img_size:
            t = torch.nn.functional.interpolate(t, size=(img_size, img_size),
                                                mode="bilinear", align_corners=False)
        e = dlp.encode_all(t.to(args.device), deterministic=True)
        oo = e.get("z_obj_on", e.get("obj_on"))[:, 0]
        if oo.dim() == 3:
            oo = oo.squeeze(-1)
        z = e["z"][:, 0]
        return oo[0].cpu().numpy(), z[0].cpu().numpy()

    envs = {}
    with h5py.File(h5p, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        for name, patch in (("patched", True), ("unpatched", False)):
            envs[name] = make_env(h5p, cams, args.res, patch)

        print(f"\n{'demo':>6} {'view':>18} | {'pair':>20} "
              f"{'pix mean':>9} {'pix max':>8} | {'act(a)':>7} {'act(b)':>7} {'z dist':>8}")
        print("-" * 104)
        for di in range(min(args.n_demos, len(bnds), len(demos))):
            d = demos[di]
            sk = next((k for k in (f"data/{d}/states/states", f"data/{d}/states") if k in f), None)
            st = np.asarray(f[sk][0], dtype=np.float64)
            start = bnds[di]["start"]

            live = {}
            for name, env in envs.items():
                env.reset()
                obs = env.reset_to({"states": st})
                live[name] = {c: np.asarray(obs[f"{c}_image"]) for c in cams}

            for vi, cam in enumerate(cams):
                stored = np.asarray(mm[start, vi])
                imgs = {"stored": stored, **{k: v[cam] for k, v in live.items()}}
                enc = {k: encode(v) for k, v in imgs.items()}
                for a, b in (("stored", "unpatched"), ("stored", "patched"),
                             ("unpatched", "patched")):
                    dif = np.abs(imgs[a].astype(np.int16) - imgs[b].astype(np.int16))
                    oa, za = enc[a]
                    ob, zb = enc[b]
                    print(f"{di:>6} {cam:>18} | {a+' vs '+b:>20} "
                          f"{dif.mean():9.3f} {dif.max():8.0f} | "
                          f"{int((oa>0.5).sum()):7d} {int((ob>0.5).sum()):7d} "
                          f"{np.abs(za-zb).mean():8.4f}")
            print()


if __name__ == "__main__":
    main()
