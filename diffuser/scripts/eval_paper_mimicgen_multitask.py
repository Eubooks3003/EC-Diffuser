"""
Multitask wrapper around eval_paper.py for MimicGen mimicgen_multitask configs.

Spawns one subprocess per task (12 tasks for the d0 mimicgen multitask config),
evaluating the same multitask checkpoint against each task in turn (or in
parallel up to --concurrency). GPUs are assigned round-robin via
CUDA_VISIBLE_DEVICES.

Each worker re-loads the multitask dataset (for the normalizer) and the
per-task DLP, runs N rollouts × M seeds, and writes a JSON result file
under <output_dir>/<task>/.

Usage:
    cd /home/ellina/Desktop/EC-Diffuser

    python diffuser/scripts/eval_paper_mimicgen_multitask.py \\
        --ckpt_path data/multitask/diffusion/mimicgen_multitask/12C_dlp_adalnpint_relative_H16_T5_seed42/ckpt/state_1000000_step1990000.pt \\
        --n_rollouts 50 --seeds 42,123,456 \\
        --gpus 0,1 --concurrency 12

    # Subset of tasks
    ... --tasks coffee,stack --concurrency 2

    # Dry-run (print planned worker commands and exit)
    ... --dry_run
"""
import argparse
import ast
import glob
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
WORKER_SCRIPT  = os.path.join(_SCRIPT_DIR, "eval_paper.py")
PREPARE_SCRIPT = os.path.join(_SCRIPT_DIR, "prepare_eval_cache_mimicgen.py")
CACHE_FILENAME = "eval_cache.pkl"
DEFAULT_CONFIG = "mimicgen_multitask_dlp"
DEFAULT_MODE   = "12C_dlp"


def _build_args(raw_argv):
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                   help=f"Config module under diffuser/config (default: {DEFAULT_CONFIG})")
    p.add_argument("--mode", type=str, default=DEFAULT_MODE,
                   help=f"Mode key in config.mode_to_args (default: {DEFAULT_MODE})")
    p.add_argument("--ckpt_path", type=str, required=True,
                   help="Path to the multitask checkpoint .pt file")
    p.add_argument("--tasks", type=str, default=None,
                   help="Comma-separated subset of task names (default: all from config.TASK_NAMES)")
    p.add_argument("--n_rollouts", type=int, default=50,
                   help="Rollouts per seed per task (default: 50)")
    p.add_argument("--seeds", type=str, default="42,123,456",
                   help="Comma-separated list of seeds (default: 42,123,456)")
    p.add_argument("--max_steps", type=int, default=None,
                   help="Override max steps per episode (default: from config)")
    p.add_argument("--execute_aux", type=int, default=None,
                   help="Execute auxiliary branch N's decode instead of the primary head")
    p.add_argument("--no_head_delta", action="store_true",
                   help="Disable head-delta recording (on by default for aux models)")
    p.add_argument("--save_videos", action="store_true",
                   help="Forwarded to each worker.")
    p.add_argument("--video_episodes", type=int, default=5,
                   help="Forwarded to each worker (default: 5)")
    p.add_argument("--concurrency", type=int, default=12,
                   help="Max simultaneous workers (default: 12)")
    p.add_argument("--gpus", type=str, default=None,
                   help="Comma-separated CUDA device ids for round-robin "
                        "(e.g. '0,1'). Default: '0'.")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Top-level output dir "
                        "(default: <ckpt_dir>/../paper_eval_multitask/<timestamp>/)")
    p.add_argument("--dry_run", action="store_true",
                   help="Print planned worker commands and exit")
    p.add_argument("--poll_interval", type=float, default=2.0,
                   help="Seconds between subprocess status polls (default: 2.0)")
    p.add_argument("--no_cache", action="store_true",
                   help="Skip the one-time eval_cache.pkl prep; each worker rebuilds the "
                        "multitask dataset from raw pkls (heavy; not recommended for "
                        "concurrency>=6).")
    p.add_argument("--force_prepare", action="store_true",
                   help="Rebuild <savepath>/eval_cache.pkl even if it exists.")
    return p.parse_args(raw_argv)


def _resolve_task_names(args):
    """Extract TASK_NAMES from the config file via AST so we don't pull in
    diffuser.utils (and the accelerate dep chain) just to read a constant."""
    if args.tasks:
        return [t.strip() for t in args.tasks.split(",") if t.strip()]
    cfg_path = os.path.abspath(os.path.join(
        _SCRIPT_DIR, "..", "config", f"{args.config}.py"
    ))
    if not os.path.isfile(cfg_path):
        raise RuntimeError(
            f"Could not find config source at {cfg_path}; pass --tasks explicitly."
        )
    src = open(cfg_path).read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "TASK_NAMES":
                    val = node.value
                    if isinstance(val, (ast.List, ast.Tuple)):
                        names = []
                        for elt in val.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                names.append(elt.value)
                            else:
                                raise RuntimeError(
                                    f"TASK_NAMES in {cfg_path} contains a non-string element; "
                                    f"pass --tasks explicitly."
                                )
                        return names
    raise RuntimeError(
        f"Could not find module-level TASK_NAMES in {cfg_path}; "
        f"pass --tasks explicitly."
    )


def _resolve_gpus(args):
    if args.gpus:
        return [g.strip() for g in args.gpus.split(",") if g.strip()]
    cur = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cur:
        return [g.strip() for g in cur.split(",") if g.strip()]
    return ["0"]


def _build_worker_cmd(args, task, output_subdir):
    cmd = [
        sys.executable, WORKER_SCRIPT,
        "--config",       args.config,
        "--mode",         args.mode,
        "--ckpt_path",    args.ckpt_path,
        "--eval_task",    task,
        "--n_rollouts",   str(args.n_rollouts),
        "--seeds",        args.seeds,
        "--output_dir",   output_subdir,
        # cuda:0 inside the worker maps to whatever CUDA_VISIBLE_DEVICES sets.
        "--device",       "cuda:0",
    ]
    if args.max_steps is not None:
        cmd += ["--max_steps", str(args.max_steps)]
    if args.save_videos:
        cmd += ["--save_videos", "--video_episodes", str(args.video_episodes)]
    if getattr(args, "execute_aux", None) is not None:
        cmd += ["--execute_aux", str(args.execute_aux)]
    if getattr(args, "no_head_delta", False):
        cmd += ["--no_head_delta"]
    return cmd


def _read_worker_result(output_subdir):
    """Locate the most recent eval_*.json the worker wrote."""
    files = sorted(glob.glob(os.path.join(output_subdir, "eval_*.json")),
                   key=os.path.getmtime)
    if not files:
        return None, None
    path = files[-1]
    try:
        with open(path) as f:
            return json.load(f), path
    except Exception as e:
        return {"_parse_error": str(e)}, path


def main(argv):
    args = _build_args(argv)

    task_names = _resolve_task_names(args)
    gpus = _resolve_gpus(args)
    K = max(1, int(args.concurrency))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        top_out = args.output_dir
    else:
        ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt_path))
        run_dir = os.path.dirname(ckpt_dir)  # strip 'ckpt'
        top_out = os.path.join(run_dir, "paper_eval_multitask", timestamp)
    os.makedirs(top_out, exist_ok=True)

    # Tee orchestrator stdout/stderr to <run_dir>/orchestrator.log so the full
    # progress timeline lives alongside per-task logs and summary.json.
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, s):
            for st in self.streams:
                try:
                    st.write(s)
                    st.flush()
                except Exception:
                    pass
        def flush(self):
            for st in self.streams:
                try: st.flush()
                except Exception: pass
        def isatty(self):
            return False

    _orch_log_fh = open(os.path.join(top_out, "orchestrator.log"), "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, _orch_log_fh)
    sys.stderr = _Tee(sys.__stderr__, _orch_log_fh)
    print(f"[multitask_eval] orchestrator.log = {os.path.join(top_out, 'orchestrator.log')}", flush=True)

    print(f"[multitask_eval] ckpt_path     = {args.ckpt_path}", flush=True)
    print(f"[multitask_eval] output_dir    = {top_out}", flush=True)
    print(f"[multitask_eval] tasks ({len(task_names)})  = {task_names}", flush=True)
    print(f"[multitask_eval] concurrency   = {K}", flush=True)
    print(f"[multitask_eval] gpus (round-robin) = {gpus}", flush=True)
    print(f"[multitask_eval] seeds         = {args.seeds}", flush=True)
    print(f"[multitask_eval] n_rollouts/seed = {args.n_rollouts}", flush=True)

    plans = []
    for i, task in enumerate(task_names):
        subdir = os.path.join(top_out, task)
        os.makedirs(subdir, exist_ok=True)
        gpu = gpus[i % len(gpus)]
        cmd = _build_worker_cmd(args, task, subdir)
        log_path = os.path.join(subdir, "log.txt")
        plans.append({
            "idx":      i,
            "task":     task,
            "gpu":      gpu,
            "cmd":      cmd,
            "subdir":   subdir,
            "log_path": log_path,
        })

    print("\n[multitask_eval] planned worker commands:", flush=True)
    for p in plans:
        print(f"  [{p['idx']:2d}] task={p['task']:<24} gpu={p['gpu']}  log={p['log_path']}",
              flush=True)
        print(f"       CUDA_VISIBLE_DEVICES={p['gpu']} \\\n       "
              + " ".join(shlex.quote(x) for x in p["cmd"]), flush=True)
    print("", flush=True)

    if args.dry_run:
        print("[multitask_eval] --dry_run: not launching workers.", flush=True)
        return

    # Build/locate the one-time eval cache before fanning out so 12 workers
    # don't all race on the multi-GB normalizer fit (the original cause of the
    # rc=-15 SIGTERM cascade at concurrency=12). Workers detect the cache
    # automatically via <savepath>/eval_cache.pkl.
    savepath = os.path.dirname(os.path.abspath(args.ckpt_path)).replace("/ckpt", "")
    cache_path = os.path.join(savepath, CACHE_FILENAME)
    if args.no_cache:
        print(f"[multitask_eval] --no_cache set: workers will rebuild dataset from raw pkls",
              flush=True)
    elif os.path.isfile(cache_path) and not args.force_prepare:
        print(f"[multitask_eval] eval cache present: {cache_path}", flush=True)
    else:
        reason = "--force_prepare" if args.force_prepare else "missing cache"
        print(f"[multitask_eval] preparing eval cache ({reason}) -> {cache_path}", flush=True)
        print(f"[multitask_eval] one-time ~3-5 min build; subsequent runs reuse it.", flush=True)
        prep_cmd = [
            sys.executable, PREPARE_SCRIPT,
            "--config",    args.config,
            "--mode",      args.mode,
            "--ckpt_path", args.ckpt_path,
        ]
        if args.force_prepare:
            prep_cmd.append("--force")
        print(f"[multitask_eval] $ {' '.join(shlex.quote(x) for x in prep_cmd)}", flush=True)
        t0 = time.time()
        rc = subprocess.call(prep_cmd, env=os.environ.copy(), cwd=os.getcwd())
        elapsed = time.time() - t0
        if rc != 0:
            raise RuntimeError(
                f"prepare_eval_cache_mimicgen.py failed (rc={rc}, elapsed={elapsed:.1f}s). "
                f"Run it manually to debug, or pass --no_cache to skip."
            )
        if not os.path.isfile(cache_path):
            raise RuntimeError(
                f"prepare_eval_cache_mimicgen.py exited 0 but no cache at {cache_path}."
            )
        print(f"[multitask_eval] eval cache ready in {elapsed:.1f}s", flush=True)

    summary_path = os.path.join(top_out, "summary.json")
    summary = {
        "status":              "running",
        "ckpt_path":           args.ckpt_path,
        "output_dir":          top_out,
        "tasks":               task_names,
        "seeds":               [int(s) for s in args.seeds.split(",") if s.strip()],
        "n_rollouts_per_seed": args.n_rollouts,
        "max_steps":           args.max_steps,
        "config":              args.config,
        "mode":                args.mode,
        "concurrency":         K,
        "gpus":                gpus,
        "timestamp_started":   timestamp,
        "per_task":            [],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    pending = list(plans)
    running = []
    completed = []

    # Build the worker env once; reuse for every launch.
    # Strip any EC-Diffuser-2D / EC-Diffuser-3D-other-checkout entries from
    # PYTHONPATH so the namespace-package `diffuser` resolves only against
    # this repo. Also pin PYTHONPATH so workers don't depend on cwd.
    base_env = os.environ.copy()
    ec_root = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
    diffuser_root = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
    incoming_pp = base_env.get("PYTHONPATH", "")
    kept = []
    for entry in incoming_pp.split(os.pathsep):
        entry = entry.strip()
        if not entry:
            continue
        # Drop other EC-Diffuser checkouts (EC-Diffuser-2D, etc.) to prevent
        # the namespace `diffuser` from picking up the wrong code.
        if "EC-Diffuser-" in entry and ec_root not in entry:
            print(f"[multitask_eval] dropping PYTHONPATH entry from worker env: {entry}",
                  flush=True)
            continue
        kept.append(entry)
    new_pp = os.pathsep.join([ec_root, diffuser_root] + kept)
    base_env["PYTHONPATH"] = new_pp

    def _launch(plan):
        env = base_env.copy()
        env["CUDA_VISIBLE_DEVICES"] = plan["gpu"]
        log_fh = open(plan["log_path"], "w", buffering=1)
        log_fh.write(f"# task={plan['task']} gpu={plan['gpu']} "
                     f"started={datetime.now().isoformat()}\n")
        log_fh.write("# cmd: " + " ".join(shlex.quote(x) for x in plan["cmd"]) + "\n\n")
        log_fh.flush()
        proc = subprocess.Popen(
            plan["cmd"],
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=os.getcwd(),
        )
        running.append({"plan": plan, "proc": proc, "log_fh": log_fh, "t_start": time.time()})
        print(f"[multitask_eval] launched {plan['task']:<24} on GPU {plan['gpu']}  "
              f"(running={len(running)}/{K}, pending={len(pending)}, done={len(completed)})",
              flush=True)

    while pending or running:
        while pending and len(running) < K:
            _launch(pending.pop(0))

        time.sleep(args.poll_interval)
        still_running = []
        for slot in running:
            rc = slot["proc"].poll()
            if rc is None:
                still_running.append(slot)
                continue
            slot["log_fh"].close()
            elapsed = time.time() - slot["t_start"]
            plan = slot["plan"]
            result, json_path = _read_worker_result(plan["subdir"])
            sr_str = "?"
            if result and isinstance(result, dict):
                if "overall_success_rate" in result:
                    sr_str = f"{result['overall_success_rate']*100:.1f}%"
            tag = "OK " if rc == 0 else f"FAIL(rc={rc})"
            print(f"[multitask_eval] {tag} {plan['task']:<24} "
                  f"in {elapsed:6.1f}s  success={sr_str}  log={plan['log_path']}",
                  flush=True)
            completed.append({
                "task":        plan["task"],
                "gpu":         plan["gpu"],
                "exit_code":   rc,
                "elapsed_sec": elapsed,
                "log_path":    plan["log_path"],
                "json_path":   json_path,
                "result":      result,
            })

            summary["per_task"] = completed
            summary["status"] = "running" if (pending or still_running) else "finalizing"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2,
                          default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
        running = still_running

    rates_per_task = []
    for c in completed:
        r = c["result"] or {}
        if isinstance(r, dict) and c["exit_code"] == 0 and "overall_success_rate" in r:
            rates_per_task.append(r["overall_success_rate"])

    # Pivot: {seed -> [per-task success_rate]} so we can show per-seed across-task means.
    per_seed_rates = {}
    per_task_seed_results = {}
    for c in completed:
        r = c["result"] or {}
        if not (c["exit_code"] == 0 and isinstance(r, dict) and "per_seed_results" in r):
            continue
        per_task_seed_results[c["task"]] = r["per_seed_results"]
        for ps in r["per_seed_results"]:
            seed = int(ps["seed"])
            per_seed_rates.setdefault(seed, []).append(float(ps["success_rate"]))

    across_task_per_seed = {
        seed: {
            "mean": float(np.mean(rates)),
            "std":  float(np.std(rates, ddof=0)),
            "n_tasks": len(rates),
        }
        for seed, rates in per_seed_rates.items()
    }

    n_complete = sum(1 for c in completed
                     if c["exit_code"] == 0 and isinstance(c["result"], dict)
                     and "overall_success_rate" in c["result"])
    n_failed = len(completed) - n_complete
    summary["status"] = "complete"
    summary["timestamp_finished"] = datetime.now().isoformat()
    summary["n_tasks_complete"] = n_complete
    summary["n_tasks_failed"] = n_failed
    summary["across_task_mean_success_rate"] = float(np.mean(rates_per_task)) if rates_per_task else 0.0
    summary["across_task_std_success_rate"] = float(np.std(rates_per_task, ddof=0)) if rates_per_task else 0.0
    summary["per_task_seed_results"] = per_task_seed_results
    summary["across_task_per_seed"] = across_task_per_seed
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))

    print("\n" + "=" * 76, flush=True)
    print("[multitask_eval] FINAL", flush=True)
    print("=" * 76, flush=True)
    for c in completed:
        r = c["result"] or {}
        sr = r.get("overall_success_rate", 0.0) if isinstance(r, dict) else 0.0
        ok = (c["exit_code"] == 0 and isinstance(r, dict) and "overall_success_rate" in r)
        flag = "  " if ok else "!!"
        # Per-seed breakdown for this task (mean +/- std across the 3 seeds, plus each seed's rate).
        seed_str = ""
        if isinstance(r, dict) and "per_seed_results" in r:
            psr = r["per_seed_results"]
            rates = [float(ps["success_rate"]) for ps in psr]
            mean_s = float(np.mean(rates)) if rates else 0.0
            std_s = float(np.std(rates, ddof=0)) if rates else 0.0
            per_s = "  ".join(f"s{int(ps['seed'])}={float(ps['success_rate'])*100:5.1f}%"
                              for ps in psr)
            seed_str = (f"  mean={mean_s*100:5.1f}% +/- {std_s*100:4.1f}%  "
                        f"[{per_s}]")
        print(f"  {flag} {c['task']:<24} "
              f"rc={c['exit_code']:>3}  "
              f"sr={sr*100:5.1f}%"
              f"{seed_str}  "
              f"t={c['elapsed_sec']:6.1f}s  "
              f"gpu={c['gpu']}",
              flush=True)
    print(f"  across-task mean: "
          f"{summary['across_task_mean_success_rate']*100:5.1f}% "
          f"+/- {summary['across_task_std_success_rate']*100:.1f}%  "
          f"({n_complete}/{len(completed)} tasks complete)", flush=True)
    if across_task_per_seed:
        per_seed_pretty = "  ".join(
            f"s{seed}={across_task_per_seed[seed]['mean']*100:5.1f}% "
            f"+/- {across_task_per_seed[seed]['std']*100:4.1f}%"
            for seed in sorted(across_task_per_seed.keys())
        )
        print(f"  per-seed across-task: {per_seed_pretty}", flush=True)
    print(f"  summary: {summary_path}", flush=True)
    print("=" * 76, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
