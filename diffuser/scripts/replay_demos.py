"""
Replay demonstration actions from a preprocessed RLBench pkl directly in
CoppeliaSim, bypassing the diffusion policy. This verifies whether the demos
in the dataset are actually solving the task.

If a replay succeeds → the demo is a valid solution and the dataset is OK.
If a replay fails → the pkl contains broken / non-solving demos, and no
policy trained on this data can ever succeed.

The replay restores the exact initial scene state of each source demo via
task.reset_to_demo(demo) (not task.reset(), which randomizes the state).
Pkl row i corresponds to <dataset_root>/<task>/all_variations/episodes/episode<i>
— the preprocessing script packs episodes in sorted numeric order.

Usage:
    cd /home/ellina/Desktop/EC-Diffuser
    export PYTHONPATH=/home/ellina/Desktop/lpwm-copy:$(pwd):$(pwd)/diffuser
    xvfb-run -a python diffuser/scripts/replay_demos.py \\
        --pkl /home/ellina/Desktop/data/rlbench_preprocessed_multiview_tokens/rlbench_close_jar/rlbench_close_jar.pkl \\
        --dataset_root /home/ellina/Desktop/data/rlbench/train_data \\
        --task_name close_jar \\
        --episodes 0 1 2 3 4 \\
        --out_dir /tmp/replays
"""
import argparse
import os
import pickle
import sys

import numpy as np

from diffuser.envs.rlbench_dlp_wrapper import (
    _resolve_task_class,
    rot6d_to_quat_xyzw,
)


def _inspect_conditions(task):
    """Return list of (condition_name, met_bool) for each registered success
    condition on the underlying RLBench Task. Lets us see *which* condition
    is failing — e.g. NothingGrasped vs DetectedCondition(lid, sensor) — when
    a replay completes the visual motion but terminal stays False."""
    underlying = getattr(task, "_task", None)
    if underlying is None:
        return [("<no _task attr>", False)]
    conds = getattr(underlying, "_success_conditions", None) or []
    out = []
    for cond in conds:
        try:
            met, _ = cond.condition_met()
        except Exception as e:
            met = f"err({type(e).__name__})"
        out.append((type(cond).__name__, bool(met) if isinstance(met, (bool, np.bool_)) else met))
    return out


def _load_demo_from_disk(dataset_root: str, task_name: str, ep_idx: int):
    """Load a Demo directly from low_dim_obs.pkl, skipping RLBench's
    get_stored_demos() which would fail because this local copy only has
    low_dim_obs.pkl / variation_*.pkl / voxel_cache per episode — the raw
    image folders (left_shoulder_rgb, front_rgb, ...) were not synced down
    from the preprocessing host. get_stored_demos unconditionally listdirs
    every camera folder before it even checks obs_config, so it raises
    FileNotFoundError before returning.

    We only need the Demo so task.reset_to_demo(demo) can call
    demo.restore_state() to reseed numpy before the next task.reset().
    That restores the exact object poses the demo was recorded against.
    """
    ep_dir = os.path.join(
        dataset_root, task_name, "all_variations", "episodes", f"episode{ep_idx}",
    )
    low_dim_path = os.path.join(ep_dir, "low_dim_obs.pkl")
    var_path = os.path.join(ep_dir, "variation_number.pkl")
    if not os.path.isfile(low_dim_path):
        raise FileNotFoundError(f"missing {low_dim_path}")

    with open(low_dim_path, "rb") as f:
        demo = pickle.load(f)
    if os.path.isfile(var_path):
        with open(var_path, "rb") as f:
            demo.variation_number = int(pickle.load(f))
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--dataset_root",
                        default="/home/ellina/Desktop/data/rlbench_rgb",
                        help="Raw RLBench dataset root. Must contain "
                             "<task>/all_variations/episodes/episode<N>/ layout. "
                             "Required so task.reset_to_demo() can restore the "
                             "original initial scene state for each demo.")
    parser.add_argument("--task_name", default="close_jar")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out_dir", default="/home/ellina/Desktop/replays")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no_headless", dest="headless", action="store_false")
    parser.add_argument("--skip_z_clamp", action="store_true",
                        help="Skip the z>=0.760 safety clamp (faithful replay).")
    parser.add_argument("--settle_steps", type=int, default=30,
                        help="After the demo's actions exhaust, repeat the "
                             "last action this many times to let physics "
                             "settle. RLBench evaluates success the moment "
                             "task.step() returns; if the lid is mid-fall at "
                             "the demo's final frame, success won't fire "
                             "without settling steps.")
    parser.add_argument("--force_open_settle", action="store_true",
                        help="During settling, override the gripper component "
                             "to open (gopen=1.0). Diagnostic: if this turns "
                             "a NothingGrasped=False episode into a success, "
                             "the demo's last action was missing the release.")
    parser.add_argument("--force_release", action="store_true",
                        help="After the action sequence ends (and before "
                             "settling), call gripper.release() directly. "
                             "Diagnostic for the RLBench Discrete bug where "
                             "screw-motion physically opens the fingers, so "
                             "the commanded open->open transition is a no-op "
                             "and the lid stays re-parented to the gripper.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[replay] loading {args.pkl}")
    with open(args.pkl, "rb") as f:
        data = pickle.load(f)

    print(f"[replay] pkl keys: {list(data.keys())}")
    actions_all = np.asarray(data["actions"])
    variations = np.asarray(data["variation_number"]) if "variation_number" in data else None
    languages = data.get("language")
    print(f"[replay] actions shape: {actions_all.shape}")
    print(f"[replay] n_episodes: {actions_all.shape[0]}")
    if variations is not None:
        unique_vars = np.unique(variations).tolist()
        print(f"[replay] unique variations: {unique_vars}")

    from rlbench import ObservationConfig, CameraConfig
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment

    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.gripper_pose = True
    obs_config.gripper_open = True
    cc = CameraConfig(image_size=(args.image_size, args.image_size),
                      rgb=True, depth=False, point_cloud=False, mask=False)
    obs_config.front_camera = cc

    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(),
        gripper_action_mode=Discrete(),
    )
    # dataset_root is where _load_demo_from_disk() reads each episode's
    # low_dim_obs.pkl. Passing it to Environment keeps RLBench consistent
    # with where we loaded the demos from, even though we bypass its own
    # get_demos() because the local copy is missing image folders.
    if not os.path.isdir(args.dataset_root):
        raise FileNotFoundError(
            f"--dataset_root does not exist: {args.dataset_root}. "
            f"Point it at the raw RLBench data (contains <task>/all_variations/)."
        )
    env = Environment(
        action_mode=action_mode,
        dataset_root=args.dataset_root,
        obs_config=obs_config,
        headless=args.headless,
    )
    env.launch()
    task_class = _resolve_task_class(args.task_name)
    task = env.get_task(task_class)

    results = []
    for ep_idx in args.episodes:
        if ep_idx >= actions_all.shape[0]:
            print(f"[replay] skipping ep {ep_idx} (only {actions_all.shape[0]} in pkl)")
            continue

        actions = actions_all[ep_idx]  # (T, 10)
        variation = int(variations[ep_idx]) if variations is not None else 0
        lang = ""
        if languages is not None and ep_idx < len(languages):
            lang_list = languages[ep_idx]
            if isinstance(lang_list, (list, tuple)) and len(lang_list) > 0:
                lang = lang_list[0]

        print(f"\n{'='*60}")
        print(f"[replay] ep={ep_idx}  variation={variation}  lang={lang!r}")
        print(f"[replay] action range: pos=[{actions[:,:3].min():.3f},{actions[:,:3].max():.3f}] "
              f"grip_values=set({set(np.round(actions[:,9], 1).tolist())})")

        # Show the binarized gripper action transitions across the trajectory.
        # RLBench's Discrete gripper only fires release()/grasp() on TRANSITION
        # (current_state != commanded_state). If the demo's released-frame is
        # immediately followed by another close, the lid gets re-attached.
        # Find non-padded extent first (drop trailing zero-rows).
        nonzero = np.where(~np.all(actions == 0.0, axis=-1))[0]
        last_nonzero = int(nonzero[-1]) if len(nonzero) else -1
        binarized = (actions[:last_nonzero + 1, 9] >= 0.5).astype(int) if last_nonzero >= 0 else np.array([], dtype=int)
        # Find transition indices.
        trans = []
        for i in range(1, len(binarized)):
            if binarized[i] != binarized[i - 1]:
                trans.append((i, int(binarized[i - 1]), int(binarized[i])))
        print(f"[replay] gripper binarized: len={len(binarized)} "
              f"first5={binarized[:5].tolist()} last10={binarized[-10:].tolist()} "
              f"transitions={trans[:10]}{'...' if len(trans) > 10 else ''}")
        print(f"{'='*60}")

        # Load the Demo directly from low_dim_obs.pkl so we can restore the
        # exact initial scene state. Using task.reset() instead would
        # randomize object poses and the open-loop absolute-pose actions
        # (pkl row is `build_actions(gripper_states[t+1])`) would operate on
        # a different scene — replay would fail even for valid demos.
        try:
            demo = _load_demo_from_disk(args.dataset_root, args.task_name, ep_idx)
        except Exception as e:
            print(f"[replay] ep={ep_idx}: failed to load demo "
                  f"({type(e).__name__}: {str(e)[:200]}); skipping")
            continue

        demo_var = int(getattr(demo, "variation_number", variation))
        if demo_var != int(variation):
            print(f"[replay] WARN pkl variation ({variation}) != demo "
                  f"variation ({demo_var}); using demo's variation_number")
        task.set_variation(demo_var)

        descriptions, obs = task.reset_to_demo(demo)
        print(f"[replay] task descriptions: {descriptions[:3]}")

        frames = [np.asarray(obs.front_rgb)]
        last_reward = 0.0
        max_reward = 0.0  # peak reward across episode; OCGS-style success detection
        last_error = None
        error_count = 0
        terminal = False
        t_end = 0
        last_rlb_action = None  # for settling phase

        for t in range(min(actions.shape[0], args.max_steps)):
            a = actions[t]

            # Stop if action looks like pure padding (all zeros).
            if np.allclose(a, 0.0):
                print(f"[replay]   t={t}: action is zeros — end of valid sequence")
                break

            pos = a[:3].astype(np.float64)
            rot6d = a[3:9].astype(np.float64)
            gopen = 1.0 if float(a[9]) >= 0.5 else 0.0

            if not args.skip_z_clamp:
                pos[2] = max(pos[2], 0.760)

            quat = rot6d_to_quat_xyzw(rot6d)
            # 9D action: [pos(3), quat(4), gripper(1), ignore_collisions(1)].
            # MoveArmThenGripper.action() in RLBench parses action[8:9] as the
            # ignore_collisions flag (action_mode.py:34). Without it the bool of
            # an empty slice is False -> planner runs collision checks -> demos
            # that brush past the jar/lid get rejected or detoured. flag=1
            # tells the planner to ignore collisions for this step (matches
            # OCGS Mover and 3d_diffuser_actor convention).
            ignore_coll = 1.0
            rlb_action = np.concatenate([pos, quat, [gopen], [ignore_coll]]).astype(np.float32)

            try:
                obs, reward, terminal = task.step(rlb_action)
                last_reward = float(reward)
                max_reward = max(max_reward, last_reward)
                last_rlb_action = rlb_action
                frames.append(np.asarray(obs.front_rgb))
                t_end = t
                if terminal:
                    print(f"[replay]   t={t}: terminal! reward={reward:.3f}")
                    break
            except Exception as e:
                error_count += 1
                last_error = f"{type(e).__name__}"
                if error_count <= 5:
                    print(f"[replay]   t={t}: {last_error}: {str(e)[:120]}")
                continue

        # --- Diagnostic: inspect each success condition right after the action
        # sequence ends. Tells us *which* condition is False (e.g. gripper
        # still grasping vs lid not in proximity sensor) when terminal=False.
        pre_conds = _inspect_conditions(task)
        print(f"[replay] post-actions conditions: {pre_conds}")

        # --- Optional manual release: workaround for RLBench Discrete's
        # transition-only release. Discrete computes current_ee from physical
        # finger open_amount > 0.9 and only fires release() when commanded
        # state differs from physical state. Screwing the lid into the jar
        # can physically separate the fingers (open_amount > 0.9) while the
        # commanded state is still "closed", so the next open command sees
        # current_ee == action == 1.0 and skips the release. The lid stays
        # re-parented to the gripper -> NothingGrasped is False forever.
        if args.force_release:
            try:
                gripper = task._task.robot.gripper
                pre_grasped = list(gripper.get_grasped_objects())
                gripper.release()
                post_grasped = list(gripper.get_grasped_objects())
                print(f"[replay]   force_release: grasped {len(pre_grasped)} "
                      f"-> {len(post_grasped)} objects")
            except Exception as e:
                print(f"[replay]   force_release failed: "
                      f"{type(e).__name__}: {str(e)[:120]}")

        # --- Settling phase: replay the last action (typically gripper open
        # over the jar) for a few extra steps so physics can settle. RLBench
        # evaluates success at the moment task.step() returns, so a lid that
        # is mid-fall at the final demo frame won't trip DetectedCondition
        # without these extra steps. With --force_open_settle, override the
        # gripper component to gopen=1.0 to test whether the demo's stored
        # last action was missing the release.
        settled_terminal = False
        settled_steps_taken = 0
        if not terminal and last_rlb_action is not None and args.settle_steps > 0:
            settle_action = last_rlb_action.copy()
            if args.force_open_settle:
                settle_action[-1] = 1.0
                print(f"[replay]   forcing gripper_open=1 during settle "
                      f"(was {last_rlb_action[-1]})")
            for s in range(args.settle_steps):
                try:
                    obs, reward, terminal = task.step(settle_action)
                    last_reward = float(reward)
                    max_reward = max(max_reward, last_reward)
                    frames.append(np.asarray(obs.front_rgb))
                    settled_steps_taken += 1
                    if terminal:
                        settled_terminal = True
                        print(f"[replay]   settle t={s}: terminal! "
                              f"reward={reward:.3f} (success fired during settling)")
                        break
                except Exception as e:
                    last_error = f"settle:{type(e).__name__}"
                    if error_count <= 5:
                        print(f"[replay]   settle t={s}: {last_error}: {str(e)[:120]}")
                    error_count += 1
                    break

        post_conds = _inspect_conditions(task)
        if pre_conds != post_conds or settled_steps_taken > 0:
            print(f"[replay] post-settle conditions ({settled_steps_taken} steps): {post_conds}")

        success = max_reward >= 0.5
        print(f"[replay] ep={ep_idx} steps_executed={t_end+1} "
              f"settled_steps={settled_steps_taken} settled_terminal={settled_terminal} "
              f"success={success} max_reward={max_reward:.3f} last_reward={last_reward:.3f} "
              f"errors={error_count} last_error={last_error}")

        try:
            import imageio.v2 as imageio
            tag = "success" if success else "fail"
            video_path = os.path.join(
                args.out_dir,
                f"replay_ep{ep_idx:03d}_var{variation:02d}_{tag}.mp4",
            )
            imageio.mimsave(video_path, frames, fps=20, macro_block_size=1)
            print(f"[replay] saved video: {video_path}")
        except Exception as e:
            print(f"[replay] video save failed: {e}")

        results.append({
            "ep": ep_idx,
            "variation": variation,
            "lang": lang,
            "steps": t_end + 1,
            "success": success,
            "reward": last_reward,
            "max_reward": max_reward,
            "errors": error_count,
        })

    env.shutdown()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        flag = "✓" if r["success"] else "✗"
        print(f"  {flag}  ep={r['ep']:3d}  var={r['variation']:2d}  "
              f"steps={r['steps']:3d}  max_reward={r['max_reward']:.3f}  "
              f"errors={r['errors']:3d}  lang={r['lang']!r}")
    n = len(results)
    n_ok = sum(1 for r in results if r["success"])
    print(f"\nTotal: {n_ok}/{n} succeeded.")
    if n > 0 and n_ok == 0:
        print("[!] ALL REPLAYS FAILED. The demos in this pkl do not solve the task.")
    elif n > 0 and n_ok == n:
        print("[OK] All replays solved the task. Demos are valid; failure is on the policy side.")
    else:
        print(f"[mixed] {n_ok}/{n} succeeded; inspect videos to diagnose.")


if __name__ == "__main__":
    main()
