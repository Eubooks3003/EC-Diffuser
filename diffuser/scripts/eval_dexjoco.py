"""
Rollout evaluation for DexJoCo EC-Diffuser policies.

Mirrors scripts/eval_paper.py (MimicGen) in structure -- load config + ckpt via
utils.Trainer, wrap the sim in a DLP-encoding wrapper, and run the same
[gripper, bg, obs] conditioning loop -- but talks to the DexJoCo MuJoCo envs
instead of robosuite.

Three things differ from the MimicGen path and each one is a silent-failure
surface, so they are handled explicitly rather than by analogy:

1. RENDER PARITY. The training tokens were NOT produced from live sim frames.
   They came from the released 640x640 mp4s, decoded with PyAV and resized to
   84 via `frame.reformat(width, height, 'rgb24')`, then upsampled to the DLP's
   128 input by `load_view_batch`. A rollout that renders 640 and resizes
   straight to 128 would hand the DLP sharper images than it ever trained on.
   So we reproduce the chain exactly: render 640 -> PyAV reformat to 84 (same
   swscale resampler) -> the shared `load_view_batch` to 128.

2. ACTION FORMAT. The policy emits 22/44-D absolute actions with rotations as
   rotvec; the env consumes 23/46-D with quaternions, and the bimanual layouts
   are ordered differently on each side (policy interleaves per arm, env groups
   poses then hands). `DexJoCoOpenPIEnv._process_action` already encodes that
   conversion, so it is reused verbatim rather than reimplemented.

3. CAMERA SLOTS. The store's generic slots (`base`, `wrist`, ...) map to
   task-specific mp4 basenames (`front`, `ego`, `ego_right`, ...). The same map
   must select env cameras, or view N of the token vector gets filled from the
   wrong camera -- no crash, just a policy that appears not to work.

Usage
-----
Parity check FIRST (no rollout is meaningful until this passes):

    python scripts/eval_dexjoco.py --parity \
        --config config.dexjoco84_single_multitask_semantic_dlp --mode 6C_dlp \
        --ckpt_path .../state_0_step230000.pt --eval_task water_plant

Then rollouts:

    python scripts/eval_dexjoco.py \
        --config config.dexjoco84_single_multitask_semantic_dlp --mode 6C_dlp \
        --ckpt_path .../state_0_step230000.pt \
        --tasks water_plant click_mouse --n_episodes 5 --save_videos
"""
import os
import sys
import json
import copy
import argparse
import importlib
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning)

# lpwm-copy supplies `models.DLP` and the token-packing helpers. Import the
# SAME module the preprocessor used (scripts/dlp_token_common.py) so the eval
# token space cannot drift from the training token space.
_LPWM = os.environ.get("LPWM_DIR", os.path.expanduser("~/Desktop/lpwm-copy"))
for _p in (_LPWM, os.path.join(_LPWM, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import diffuser.utils as utils  # noqa: E402
from dlp_token_common import build_dlp_2d_from_cfg, pack_tokens_2d, load_view_batch  # noqa: E402


# ---------------------------------------------------------------------------
# Camera slots
# ---------------------------------------------------------------------------
# Mirrors _slot_map / _BASE_CAMERA in lpwm-copy/preprocess_dexjoco_multiview.py.
# Kept in sync by construction: a task missing here raises rather than guessing.
_BASE_CAMERA = {"click_mouse": "ego_right"}
_SINGLE_SLOTS = ["base", "wrist"]
_BIMANUAL_SLOTS = ["base", "wrist_left", "wrist_right"]


def is_bimanual(task):
    return task.startswith("bimanual_")


def slot_map(task):
    """Generic slot -> env camera key, for the default (non rand_full) variant."""
    if is_bimanual(task):
        return {"base": "ego", "wrist_left": "wrist_left", "wrist_right": "wrist_right"}
    return {"base": _BASE_CAMERA.get(task, "front"), "wrist": "wrist"}


def slots_for(task):
    return _BIMANUAL_SLOTS if is_bimanual(task) else _SINGLE_SLOTS


# ---------------------------------------------------------------------------
# Render parity
# ---------------------------------------------------------------------------
def resize_like_store(img, store_size):
    """Downsample a rendered frame the way the store was built.

    The memmap was written by PyAV's `frame.reformat(...)`, i.e. libswscale.
    Feeding a live render through the same resampler keeps the DLP's input
    distribution identical; PIL/cv2 would be a different filter and would show
    up as a systematic domain shift that is very hard to attribute later.
    """
    import av

    if img.shape[0] == store_size and img.shape[1] == store_size:
        return np.ascontiguousarray(img)
    frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")
    return frame.reformat(width=store_size, height=store_size,
                          format="rgb24").to_ndarray()


# ---------------------------------------------------------------------------
# Env + DLP wrapper
# ---------------------------------------------------------------------------
class DexJoCoDLPWrapper:
    """DexJoCo sim -> EC-Diffuser particle observation.

    Deliberately thin: env construction and the action conversion are delegated
    to the upstream DexJoCoOpenPIEnv so there is exactly one definition of the
    policy->env action mapping in the codebase.
    """

    def __init__(self, task, dlp_model, device, store_size=84, seed=0,
                 randomize=False, randomize_dynamics=False):
        self.task = task
        self.dual_arm = is_bimanual(task)
        self.slots = slots_for(task)
        self.cam_map = slot_map(task)
        self.dlp = dlp_model
        self.device = device
        self.store_size = int(store_size)
        self.proprio_dim = 46 if self.dual_arm else 23
        self.image_size = int(dlp_model.image_size)

        from dexjoco.tasks import CONFIG_MAPPING

        cfg = CONFIG_MAPPING[task]()
        self.env = cfg.get_environment(
            policy_mode=True, render_mode="rgb_array",
            randomize=randomize, seed=seed,
            randomize_dynamics=randomize_dynamics,
        )

        # Reuse the upstream action conversion rather than reimplementing it:
        # bind the unbound method onto a shim carrying just the flag it reads.
        DexJoCoOpenPIEnv = _load_openpi_env_class()
        self._convert_action = DexJoCoOpenPIEnv._process_action.__get__(
            _ActionShim(self.dual_arm), DexJoCoOpenPIEnv)

        self.last_raw = {}
        self.last_toks = None
        self.last_gripper_state = None
        self.last_bg_features = None

    # -- observation --------------------------------------------------------
    def _encode(self, env_obs):
        """Render all slots, encode with the DLP, pack into the token layout."""
        imgs, raw = [], {}
        for slot in self.slots:
            key = self.cam_map[slot]
            if key not in env_obs:
                raise KeyError(
                    f"{self.task}: camera '{key}' (slot '{slot}') absent from env obs; "
                    f"available: {sorted(k for k in env_obs if k != 'state')}")
            raw[slot] = env_obs[key]
            imgs.append(resize_like_store(env_obs[key], self.store_size))
        self.last_raw = raw

        # (1, V, H, W, 3) so load_view_batch's memmap indexing applies unchanged
        stacked = np.stack(imgs, axis=0)[None]

        toks_per_view, bg_per_view = [], []
        with torch.no_grad():
            for vi in range(len(self.slots)):
                x = load_view_batch(stacked, 0, [0], vi, self.image_size, self.device)
                enc = self.dlp.encode_all(x, deterministic=True)
                t, bg = pack_tokens_2d(enc)
                toks_per_view.append(t[0])
                bg_per_view.append(bg[0])

        toks = torch.cat(toks_per_view, dim=0)          # (K_total, Dtok)
        bg = torch.cat(bg_per_view, dim=0)              # (BG_total,)
        self.last_toks = toks.cpu().numpy()             # kept 2-D for overlays
        self.last_bg_features = bg.cpu().numpy()
        self.last_gripper_state = np.asarray(
            env_obs["state"], dtype=np.float32).flatten()[:self.proprio_dim]
        # The conditioning path concatenates [gripper, bg, obs] as 1-D vectors
        # and the normalizer was fit on the flattened K*Dtok layout, so the
        # observation must be flat here -- matching mimicgen_dlp_wrapper.py:943.
        return self.last_toks.reshape(-1).astype(np.float32)

    def reset(self):
        obs, _ = self.env.reset()
        self._success = False
        return self._encode(obs)

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        # Under a MERGED single+bimanual policy the action head is 44-D. The
        # bimanual layout leads with the right arm, so a single-arm task is
        # exactly the first 22 dims and the rest is the phantom left arm --
        # DexJoCo's own convention (dp_dexjoco_env.py takes action[:22] too).
        native = 44 if self.dual_arm else 22
        if action.shape[0] > native:
            action = action[:native]
        env_action = self._convert_action(action)
        obs, reward, terminated, truncated, info = self.env.step(env_action)
        self._success = bool(info.get("succeed", False))
        obs_vec = self._encode(obs)
        done = bool(terminated) or bool(truncated) or self._success
        return obs_vec, float(reward), done, {"success": self._success, **info}

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


class _ActionShim:
    """Minimal carrier for the one attribute _process_action reads."""

    def __init__(self, dual_arm):
        self.dual_arm = dual_arm


def _load_openpi_env_class():
    """Load DexJoCoOpenPIEnv from its file, bypassing the package __init__.

    `import dexjoco_openpi_client.dexjoco_openpi_env` executes the package
    __init__, which pulls in eval_dexjoco_openpi -> websockets: a policy-server
    client this script never uses. Loading the module file directly keeps the
    one thing we want (the canonical policy->env action conversion) without
    dragging the serving stack into the eval env.
    """
    import importlib.util

    root = os.environ.get("DEXJOCO_ROOT")
    if root is None:
        import dexjoco
        root = os.path.dirname(os.path.dirname(os.path.abspath(dexjoco.__file__)))
    path = os.path.join(root, "dexjoco_openpi_client", "dexjoco_openpi_env.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"dexjoco_openpi_env.py not found at {path}; set DEXJOCO_ROOT to the "
            "directory containing dexjoco/ and dexjoco_openpi_client/")
    spec = importlib.util.spec_from_file_location("_dexjoco_openpi_env", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DexJoCoOpenPIEnv


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_trainer(args):
    """Rebuild dataset/model/trainer from the config, then load the ckpt."""

    class _A:
        pass

    a = _A()
    a.config = args.config
    a.mode = args.mode
    a.device = args.device
    module = importlib.import_module(args.config)
    margs = module.mode_to_args[args.mode]
    base = copy.deepcopy(module.base)

    cfg = _A()
    for k, v in base["diffusion"].items():
        setattr(cfg, k, v)
    for k, v in margs.items():
        setattr(cfg, k, v)
    cfg.device = args.device
    cfg.savepath = os.path.dirname(args.ckpt_path).replace("/ckpt", "")

    dataset_config = utils.Config(
        cfg.loader, savepath=None, horizon=cfg.horizon,
        normalizer=cfg.normalizer, particle_normalizer=cfg.particle_normalizer,
        preprocess_fns=cfg.preprocess_fns, use_padding=cfg.use_padding,
        max_path_length=cfg.max_path_length, obs_only=cfg.obs_only,
        action_only=cfg.action_only, action_z_scale=cfg.action_z_scale,
        use_gripper_obs=cfg.use_gripper_obs, use_bg_obs=cfg.use_bg_obs,
        task_entries=cfg.task_entries, max_demos_per_task=cfg.max_demos_per_task,
        dataset_name=cfg.dataset,
    )
    dataset = dataset_config()

    obs_dim = dataset.observation_dim
    act_dim = dataset.action_dim
    grip_dim = getattr(dataset, "gripper_dim", 0)
    bg_dim = getattr(dataset, "bg_dim", 0)

    model_config = utils.Config(
        cfg.model, savepath=None, features_dim=cfg.features_dim,
        action_dim=act_dim, hidden_dim=cfg.hidden_dim,
        projection_dim=cfg.projection_dim, n_head=cfg.n_heads,
        n_layer=cfg.n_layers, dropout=cfg.dropout,
        # block_size IS the horizon (train.py:289). Left at the module default
        # of 50 it builds a 50-step pos_emb that no checkpoint can load.
        block_size=cfg.horizon,
        positional_bias=cfg.positional_bias, max_particles=cfg.max_particles,
        multiview=cfg.multiview, gripper_dim=grip_dim, bg_dim=bg_dim,
        n_tasks=getattr(cfg, "n_tasks", 1),
        action_token_groups=getattr(cfg, "action_token_groups", None),
        proprio_token_groups=getattr(cfg, "proprio_token_groups", None),
        device=cfg.device,
    )
    model = model_config()

    diffusion_config = utils.Config(
        cfg.diffusion, savepath=None, horizon=cfg.horizon,
        observation_dim=obs_dim, action_dim=act_dim,
        gripper_dim=grip_dim, bg_dim=bg_dim,
        n_timesteps=cfg.n_diffusion_steps, loss_type=cfg.loss_type,
        clip_denoised=cfg.clip_denoised, predict_epsilon=cfg.predict_epsilon,
        action_weight=cfg.action_weight, loss_weights=cfg.loss_weights,
        loss_discount=cfg.loss_discount, device=cfg.device,
    )
    diffusion = diffusion_config(model)

    trainer_config = utils.Config(
        utils.Trainer, savepath=None, train_batch_size=cfg.batch_size,
        train_lr=cfg.learning_rate, gradient_accumulate_every=cfg.gradient_accumulate_every,
        ema_decay=cfg.ema_decay, sample_freq=cfg.sample_freq,
        save_freq=cfg.save_freq, label_freq=int(cfg.n_train_steps // cfg.n_saves),
        save_parallel=cfg.save_parallel, results_folder=cfg.savepath,
        bucket=cfg.bucket, n_reference=cfg.n_reference,
    )
    trainer = trainer_config(diffusion, dataset, None)

    data = torch.load(args.ckpt_path, map_location=cfg.device, weights_only=False)
    trainer.model.load_state_dict(data["model"])
    trainer.ema_model.load_state_dict(data["ema"])
    trainer.step = data.get("step", 0)
    print(f"[eval] loaded step {trainer.step} | act={act_dim} prop={grip_dim} "
          f"bg={bg_dim} obs={obs_dim}")
    return trainer, cfg


def load_dlp(entry, device):
    """Load the per-task DLP snapshot exactly as the preprocessor did.

    Same builder, same bare-state_dict load (ec_diffuser_dexjoco_preprocess.py
    :246-250). Any divergence here would silently produce a different token
    space at eval than the one the policy was trained on.
    """
    with open(entry["dlp_cfg"]) as f:
        dcfg = json.load(f)
    model = build_dlp_2d_from_cfg(dcfg, device)
    model.load_state_dict(
        torch.load(entry["dlp_ckpt"], map_location=device, weights_only=False))
    model.eval()
    return model, dcfg


# ---------------------------------------------------------------------------
# Parity check
# ---------------------------------------------------------------------------
def run_parity(trainer, cfg, task, entry, device, store_size, n_frames=8):
    """Compare live-rendered encodings against the stored training tokens.

    This does NOT check that the same frame reproduces byte-for-byte -- the sim
    is reset to a fresh episode, not to a recorded state, so the scenes differ.
    What it checks is that the two token distributions are compatible: same
    shape, same active-particle count, overlapping per-dimension ranges. A
    rendering mismatch (wrong camera, flipped image, wrong resize) shows up
    here as a gross distribution shift, which is the failure this is for.
    """
    import pickle

    dlp, dcfg = load_dlp(entry, device)
    with open(entry["pkl"], "rb") as f:
        stored = pickle.load(f)
    s_obs = stored["observations"]                     # (E, T, K, Dtok)
    s_len = stored["path_lengths"]
    s_flat = np.concatenate([s_obs[e, :s_len[e]] for e in range(min(5, len(s_len)))], axis=0)

    envw = DexJoCoDLPWrapper(task, dlp, device, store_size=store_size, seed=0)
    envw.reset()
    live = [envw.last_toks.copy()]          # (K, Dtok); the returned vector is flat
    for _ in range(n_frames - 1):
        _, _, done, _ = envw.step(_zero_policy_action(task))
        live.append(envw.last_toks.copy())
        if done:
            break
    envw.close()
    live = np.stack(live, axis=0)           # (n, K, Dtok), comparable to stored

    print(f"\n=== PARITY: {task} ===")
    print(f"  stored tokens : {s_flat.shape}   live tokens: {live.shape}")
    ok = True
    if s_flat.shape[1:] != live.shape[1:]:
        print("  !! SHAPE MISMATCH — token layouts differ")
        ok = False

    on_s = s_flat[..., 5]
    on_l = live[..., 5]
    print(f"  obj_on   stored mean={on_s.mean():.3f}  live mean={on_l.mean():.3f}")
    print(f"  active(>0.5) stored={float((on_s > 0.5).mean()):.3f}  "
          f"live={float((on_l > 0.5).mean()):.3f}")

    names = ["z_y", "z_x", "scale_y", "scale_x", "depth", "obj_on"]
    print(f"  {'dim':10s} {'stored [min, max]':>26s} {'live [min, max]':>26s}  overlap")
    for d in range(min(6, s_flat.shape[-1])):
        a0, a1 = float(s_flat[..., d].min()), float(s_flat[..., d].max())
        b0, b1 = float(live[..., d].min()), float(live[..., d].max())
        inter = max(0.0, min(a1, b1) - max(a0, b0))
        union = max(a1, b1) - min(a0, b0)
        frac = inter / union if union > 0 else 1.0
        flag = "" if frac > 0.3 else "   <-- LOW"
        if frac <= 0.3:
            ok = False
        print(f"  {names[d]:10s} [{a0:9.3f},{a1:9.3f}] [{b0:9.3f},{b1:9.3f}]  "
              f"{frac:6.2f}{flag}")

    print(f"\n  VERDICT: {'PASS' if ok else 'FAIL — do not trust rollouts yet'}")
    return ok


def _zero_policy_action(task):
    """A no-op-ish policy action used only to advance frames during parity."""
    return np.zeros(44 if is_bimanual(task) else 22)


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------
def _fit_width(vec, width):
    """Zero-pad (or trim) a 1-D observation slice to the policy's width.

    A merged single+bimanual policy expects the bimanual widths (proprio 46,
    bg 12, 60 particles). A single-arm rollout produces the narrower vectors,
    and the buffer padded exactly the same way at training time -- with zeros,
    on the right -- so this reproduces the training-time layout. For a
    single-embodiment policy the widths already match and this is a no-op.
    """
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if v.shape[0] == width:
        return v
    if v.shape[0] > width:
        return v[:width]
    return np.concatenate([v, np.zeros(width - v.shape[0], dtype=np.float32)])


def run_rollouts(trainer, cfg, task, task_id, entry, device, n_episodes,
                 max_steps, exe_steps, seed, store_size,
                 save_videos=False, video_dir=None, video_episodes=3):
    dlp, dcfg = load_dlp(entry, device)
    np.random.seed(seed)
    torch.manual_seed(seed)

    a_dim = trainer.dataset.action_dim
    grip_dim = getattr(trainer.dataset, "gripper_dim", 0)
    bg_dim = getattr(trainer.dataset, "bg_dim", 0)
    obs_dim = trainer.dataset.observation_dim
    norm = trainer.dataset.normalizer
    tid = torch.tensor([int(task_id)], dtype=torch.long, device=device)

    envw = DexJoCoDLPWrapper(task, dlp, device, store_size=store_size, seed=seed)
    successes, lengths = [], []

    if save_videos and video_dir:
        os.makedirs(video_dir, exist_ok=True)

    for ep in range(n_episodes):
        obs_vec = envw.reset()
        frames = []
        buf, idx, t, done, info = None, 0, 0, False, {}
        record = save_videos and ep < video_episodes

        while t < max_steps and not done:
            if buf is None or idx >= exe_steps:
                parts = []
                if grip_dim > 0:
                    g = _fit_width(envw.last_gripper_state, grip_dim)
                    parts.append(norm.normalize(g[None], "gripper_state")[0])
                if bg_dim > 0:
                    b = _fit_width(envw.last_bg_features, bg_dim)
                    parts.append(norm.normalize(b[None], "bg_features")[0])
                parts.append(norm.normalize(_fit_width(obs_vec, obs_dim)[None],
                                            "observations")[0])
                cond = {0: torch.from_numpy(np.concatenate(parts)[None]).float().to(device)}

                with torch.no_grad():
                    sample = trainer.ema_model(cond, verbose=False, task_id=tid)
                buf = sample.trajectories[0][:, :a_dim].detach().cpu().numpy()
                idx = 0

            a = norm.unnormalize(buf[idx][None], "actions")[0]
            obs_vec, r, done, info = envw.step(a)

            if record:
                frames.append(np.concatenate(
                    [envw.last_raw[s] for s in envw.slots], axis=1))
            idx += 1
            t += 1

        success = bool(info.get("success", False))
        successes.append(success)
        lengths.append(t)
        print(f"  ep {ep:02d}: {'SUCCESS' if success else 'fail   '} "
              f"steps={t:4d}  running sr={100*np.mean(successes):.0f}%")

        if record and frames and video_dir:
            import imageio
            status = "success" if success else "fail"
            imageio.mimsave(
                os.path.join(video_dir, f"{task}_seed{seed}_ep{ep:02d}_{status}.mp4"),
                frames, fps=30)

    envw.close()
    return successes, lengths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", required=True, help="e.g. 6C_dlp / 5C_dlp")
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="subset of task names; default = all in the config")
    ap.add_argument("--eval_task", default=None, help="alias for a single --tasks entry")
    ap.add_argument("--n_episodes", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=600)
    ap.add_argument("--exe_steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--store_size", type=int, default=84,
                    help="resolution the training memmap was built at")
    ap.add_argument("--save_videos", action="store_true")
    ap.add_argument("--video_episodes", type=int, default=3)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--parity", action="store_true",
                    help="run the render/token parity check and exit")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")

    trainer, cfg = load_trainer(args)
    entries = {e["name"]: e for e in cfg.task_entries}
    tasks = args.tasks or ([args.eval_task] if args.eval_task else list(entries))
    exe_steps = args.exe_steps or getattr(cfg, "exe_steps", 8)

    if args.parity:
        allok = True
        for t in tasks:
            allok &= bool(run_parity(trainer, cfg, t, entries[t], args.device,
                                     args.store_size))
        sys.exit(0 if allok else 1)

    out_dir = args.output_dir or os.path.join(
        os.path.dirname(args.ckpt_path), "eval_results")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_name = os.path.basename(args.ckpt_path).replace(".pt", "")

    results = {}
    for t in tasks:
        e = entries[t]
        print(f"\n=== {t} (task_id {e['task_id']}) — {args.n_episodes} eps, "
              f"seed {args.seed} ===")
        succ, lens = run_rollouts(
            trainer, cfg, t, e["task_id"], e, args.device,
            n_episodes=args.n_episodes, max_steps=args.max_steps,
            exe_steps=exe_steps, seed=args.seed, store_size=args.store_size,
            save_videos=args.save_videos,
            video_dir=os.path.join(out_dir, "videos"),
            video_episodes=args.video_episodes)
        results[t] = {"success_rate": float(np.mean(succ)),
                      "n_episodes": len(succ),
                      "successes": [bool(s) for s in succ],
                      "mean_length": float(np.mean(lens))}
        print(f"  -> {t}: {100*np.mean(succ):.1f}%")

    summary = {
        "ckpt_path": args.ckpt_path, "config": args.config, "mode": args.mode,
        "seed": args.seed, "n_episodes": args.n_episodes,
        "exe_steps": exe_steps, "store_size": args.store_size,
        "per_task": results,
        "mean_success_rate": float(np.mean([r["success_rate"] for r in results.values()])),
    }
    path = os.path.join(out_dir, f"dexjoco_eval_{ckpt_name}_seed{args.seed}.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 56)
    for t, r in sorted(results.items(), key=lambda kv: -kv[1]["success_rate"]):
        print(f"  {t:26s} {100*r['success_rate']:5.1f}%  ({sum(r['successes'])}/{r['n_episodes']})")
    print("-" * 56)
    print(f"  {'MEAN':26s} {100*summary['mean_success_rate']:5.1f}%")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
