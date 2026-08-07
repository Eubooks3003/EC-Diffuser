"""
Global-queue orchestrator across MULTIPLE MimicGen multitask checkpoints.

Unlike eval_paper_mimicgen_multitask.py (which runs one checkpoint's 12 tasks and
blocks until all finish before you start the next checkpoint), this pools EVERY
(experiment x task) job into a single queue and keeps `--concurrency` workers busy
at all times. As soon as a fast task (e.g. stack) frees a slot, the next queued
job starts -- even if it belongs to a later experiment. No idle GPU while slow
tasks (pick_place, coffee) drain at an experiment boundary.

All eval_cache.pkl files are built up front (one per experiment) so interleaved
workers never race on cache creation. Each experiment still gets its own
summary.json in the same format as the per-checkpoint orchestrator.

Usage (run from EC-Diffuser/diffuser with PYTHONPATH=.:..):
    python scripts/eval_paper_mimicgen_multi_experiment.py \\
        --run randgroupA mimicgen_multitask_randgroupA_dlp /abs/ckptA.pt \\
        --run randgroupB mimicgen_multitask_randgroupB_dlp /abs/ckptB.pt \\
        --run tokenaction_singleprop mimicgen_multitask_tokenaction_singleprop_dlp /abs/ckptC.pt \\
        --mode 12C_dlp --n_rollouts 50 --seeds 42,123,456 \\
        --gpus 0,1 --concurrency 12 --save_videos --video_episodes 5
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


def _build_args(argv):
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--run", action="append", nargs=3, metavar=("NAME", "CONFIG", "CKPT"),
                   required=True, help="Repeatable: experiment name, config module, ckpt .pt path")
    p.add_argument("--run_flags", action="append", nargs=2, metavar=("NAME", "FLAGS"),
                   default=None,
                   help="Repeatable: extra worker flags for one run, e.g. "
                        "--run_flags c6 '--execute_aux 0 --gate_delta 0.0405'. "
                        "Lets two runs share a checkpoint but differ in execution.")
    p.add_argument("--out_prefix", type=str, default=None,
                   help="If set, each run writes to <run_dir>/<out_prefix>_<NAME>/<stamp> "
                        "instead of <run_dir>/paper_eval_multitask/<stamp>. Required when "
                        "two runs share a checkpoint.")
    p.add_argument("--mode", type=str, default="12C_dlp")
    p.add_argument("--tasks", type=str, default=None,
                   help="Comma-separated task subset (default: all TASK_NAMES from each config)")
    p.add_argument("--n_rollouts", type=int, default=50)
    p.add_argument("--seeds", type=str, default="42,123,456")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--save_videos", action="store_true")
    p.add_argument("--video_episodes", type=int, default=5)
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--gpus", type=str, default="0,1")
    p.add_argument("--poll_interval", type=float, default=2.0)
    p.add_argument("--force_prepare", action="store_true")
    p.add_argument("--no_cache", action="store_true")
    return p.parse_args(argv)


def _resolve_task_names(config, tasks_override):
    if tasks_override:
        return [t.strip() for t in tasks_override.split(",") if t.strip()]
    cfg_path = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "config", f"{config}.py"))
    if not os.path.isfile(cfg_path):
        raise RuntimeError(f"Config source not found: {cfg_path}; pass --tasks explicitly.")
    tree = ast.parse(open(cfg_path).read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "TASK_NAMES":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        return [e.value for e in node.value.elts
                                if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    raise RuntimeError(f"TASK_NAMES not found in {cfg_path}; pass --tasks explicitly.")


def _build_worker_cmd(args, config, ckpt, task, output_subdir, extra_flags=""):
    cmd = [sys.executable, WORKER_SCRIPT,
           "--config", config, "--mode", args.mode, "--ckpt_path", ckpt,
           "--eval_task", task, "--n_rollouts", str(args.n_rollouts),
           "--seeds", args.seeds, "--output_dir", output_subdir, "--device", "cuda:0"]
    if args.max_steps is not None:
        cmd += ["--max_steps", str(args.max_steps)]
    if args.save_videos:
        cmd += ["--save_videos", "--video_episodes", str(args.video_episodes)]
    if extra_flags:
        cmd += shlex.split(extra_flags)
    return cmd


def _read_worker_result(output_subdir):
    files = sorted(glob.glob(os.path.join(output_subdir, "eval_*.json")), key=os.path.getmtime)
    if not files:
        return None, None
    try:
        with open(files[-1]) as f:
            return json.load(f), files[-1]
    except Exception as e:
        return {"_parse_error": str(e)}, files[-1]


def _base_env():
    env = os.environ.copy()
    ec_root = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
    diffuser_root = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
    kept = []
    for entry in env.get("PYTHONPATH", "").split(os.pathsep):
        entry = entry.strip()
        if not entry:
            continue
        if "EC-Diffuser-" in entry and ec_root not in entry:
            continue
        kept.append(entry)
    env["PYTHONPATH"] = os.pathsep.join([ec_root, diffuser_root] + kept)
    return env


def _write_summary(run, completed):
    """Write one experiment's summary.json in the per-checkpoint format."""
    rates = [c["result"]["overall_success_rate"] for c in completed
             if c["exit_code"] == 0 and isinstance(c["result"], dict)
             and "overall_success_rate" in c["result"]]
    per_seed_rates, per_task_seed_results = {}, {}
    for c in completed:
        r = c["result"] or {}
        if not (c["exit_code"] == 0 and isinstance(r, dict) and "per_seed_results" in r):
            continue
        per_task_seed_results[c["task"]] = r["per_seed_results"]
        for ps in r["per_seed_results"]:
            per_seed_rates.setdefault(int(ps["seed"]), []).append(float(ps["success_rate"]))
    across_task_per_seed = {s: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=0)),
                                "n_tasks": len(v)} for s, v in per_seed_rates.items()}
    n_complete = len(rates)
    summary = {
        "status": "complete",
        "experiment": run["name"],
        "ckpt_path": run["ckpt"],
        "config": run["config"],
        "output_dir": run["out"],
        "seeds": [int(s) for s in run["seeds"].split(",") if s.strip()],
        "n_rollouts_per_seed": run["n_rollouts"],
        "timestamp_finished": datetime.now().isoformat(),
        "n_tasks_complete": n_complete,
        "n_tasks_failed": len(completed) - n_complete,
        "across_task_mean_success_rate": float(np.mean(rates)) if rates else 0.0,
        "across_task_std_success_rate": float(np.std(rates, ddof=0)) if rates else 0.0,
        "per_task_seed_results": per_task_seed_results,
        "across_task_per_seed": across_task_per_seed,
        "per_task": completed,
    }
    with open(os.path.join(run["out"], "summary.json"), "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
    return summary


def main(argv):
    args = _build_args(argv)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()] or ["0"]
    K = max(1, int(args.concurrency))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Resolve runs: output dir, task list, cache path.
    flags_by_name = {n: f for n, f in (args.run_flags or [])}
    unknown = set(flags_by_name) - {r[0] for r in args.run}
    if unknown:
        raise SystemExit(f"--run_flags names not matched by any --run: {sorted(unknown)}")
    runs = []
    for name, config, ckpt in args.run:
        ckpt = os.path.abspath(ckpt)
        ckpt_dir = os.path.dirname(ckpt)
        run_dir = os.path.dirname(ckpt_dir)  # strip 'ckpt'
        sub = f"{args.out_prefix}_{name}" if args.out_prefix else "paper_eval_multitask"
        out = os.path.join(run_dir, sub, stamp)
        os.makedirs(out, exist_ok=True)
        runs.append({
            "name": name, "config": config, "ckpt": ckpt, "out": out,
            "flags": flags_by_name.get(name, ""),
            "tasks": _resolve_task_names(config, args.tasks),
            "seeds": args.seeds, "n_rollouts": args.n_rollouts,
            "cache": os.path.join(run_dir, CACHE_FILENAME),
        })

    print(f"[multi_exp] stamp={stamp}  concurrency={K}  gpus={gpus}", flush=True)
    for r in runs:
        print(f"[multi_exp] run {r['name']:<24} tasks={len(r['tasks'])} out={r['out']}", flush=True)

    # 1) Build every experiment's eval cache UP FRONT so interleaved workers never
    #    race on cache creation.
    if not args.no_cache:
        for r in runs:
            if os.path.isfile(r["cache"]) and not args.force_prepare:
                print(f"[multi_exp] cache present: {r['cache']}", flush=True)
                continue
            print(f"[multi_exp] building cache for {r['name']} -> {r['cache']}", flush=True)
            prep = [sys.executable, PREPARE_SCRIPT, "--config", r["config"],
                    "--mode", args.mode, "--ckpt_path", r["ckpt"]]
            if args.force_prepare:
                prep.append("--force")
            t0 = time.time()
            rc = subprocess.call(prep, env=os.environ.copy(), cwd=os.getcwd())
            if rc != 0 or not os.path.isfile(r["cache"]):
                raise RuntimeError(f"cache prep failed for {r['name']} (rc={rc})")
            print(f"[multi_exp] cache for {r['name']} ready in {time.time()-t0:.1f}s", flush=True)

    # 2) Global job queue: every (run, task). Ordered run-by-run, but the pool
    #    never waits on a run boundary -- a freed slot pulls the next queued job.
    queue = []
    for r in runs:
        for task in r["tasks"]:
            subdir = os.path.join(r["out"], task)
            os.makedirs(subdir, exist_ok=True)
            queue.append({"run": r, "task": task, "subdir": subdir,
                          "log_path": os.path.join(subdir, "log.txt")})
    total = len(queue)
    print(f"[multi_exp] global queue: {total} jobs across {len(runs)} experiments", flush=True)

    base_env = _base_env()
    running, done_by_run = [], {r["name"]: [] for r in runs}
    launched = 0

    def _launch(job, idx):
        gpu = gpus[idx % len(gpus)]
        cmd = _build_worker_cmd(args, job["run"]["config"], job["run"]["ckpt"],
                                job["task"], job["subdir"], job["run"].get("flags", ""))
        env = base_env.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        fh = open(job["log_path"], "w", buffering=1)
        fh.write("# " + " ".join(shlex.quote(x) for x in cmd) + "\n\n"); fh.flush()
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=os.getcwd())
        running.append({"job": job, "proc": proc, "fh": fh, "gpu": gpu, "t0": time.time()})
        print(f"[multi_exp] launch {job['run']['name']}/{job['task']:<22} gpu={gpu} "
              f"(running={len(running)}/{K} done={sum(len(v) for v in done_by_run.values())}/{total})",
              flush=True)

    qi = 0
    while qi < total or running:
        while qi < total and len(running) < K:
            _launch(queue[qi], launched); launched += 1; qi += 1
        time.sleep(args.poll_interval)
        still = []
        for slot in running:
            rc = slot["proc"].poll()
            if rc is None:
                still.append(slot); continue
            slot["fh"].close()
            job = slot["job"]
            result, jpath = _read_worker_result(job["subdir"])
            sr = "?"
            if isinstance(result, dict) and "overall_success_rate" in result:
                sr = f"{result['overall_success_rate']*100:.1f}%"
            tag = "OK " if rc == 0 else f"FAIL(rc={rc})"
            print(f"[multi_exp] {tag} {job['run']['name']}/{job['task']:<22} "
                  f"{time.time()-slot['t0']:6.1f}s success={sr}", flush=True)
            rec = {"task": job["task"], "gpu": slot["gpu"], "exit_code": rc,
                   "elapsed_sec": time.time()-slot["t0"], "log_path": job["log_path"],
                   "json_path": jpath, "result": result}
            done = done_by_run[job["run"]["name"]]
            done.append(rec)
            # When a run's tasks are all finished, write its summary immediately.
            if len(done) == len(job["run"]["tasks"]):
                s = _write_summary(job["run"], done)
                print(f"[multi_exp] === {job['run']['name']} COMPLETE: "
                      f"across-task {s['across_task_mean_success_rate']*100:.1f}% "
                      f"+/- {s['across_task_std_success_rate']*100:.1f}%  "
                      f"summary={os.path.join(job['run']['out'],'summary.json')}", flush=True)
        running = still

    print("\n" + "=" * 76 + "\n[multi_exp] ALL EXPERIMENTS COMPLETE\n" + "=" * 76, flush=True)
    for r in runs:
        done = done_by_run[r["name"]]
        rates = [c["result"]["overall_success_rate"] for c in done
                 if c["exit_code"] == 0 and isinstance(c["result"], dict)
                 and "overall_success_rate" in c["result"]]
        mean = float(np.mean(rates)) if rates else 0.0
        std = float(np.std(rates, ddof=0)) if rates else 0.0
        print(f"  {r['name']:<24} across-task {mean*100:5.1f}% +/- {std*100:4.1f}%  "
              f"({len(rates)}/{len(r['tasks'])} tasks)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
