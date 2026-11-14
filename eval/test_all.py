"""The final results: every run on the Hub, played on the TEST maps under every mod.

    python -m eval.test_all prepare                      # once: render the test maps' frames (all mods)
    python -m eval.test_all list                         # which runs it found, and which are done
    CUDA_VISIBLE_DEVICES=0 python -m eval.test_all run --lane 0/2     # two GPUs: half the runs each
    CUDA_VISIBLE_DEVICES=1 python -m eval.test_all run --lane 1/2

For each run (a seed folder of a repo, or bc-baseline's root): download it (train/hub.py pull: best.pt
and the small files) into runs/<name>_seed<N>/, run eval.transfer_matrix --split test --mods NM HR DT
HD HRDT --motor human (these results predate CAP as the default) into transfer_test.json there, and upload that one file next to the run's other files on the
Hub. A run whose transfer_test.json is already on the Hub is skipped, so rerunning picks up where it
stopped. Results: python -m eval.compare_runs --split test, python -m eval.report_figures --split test.

the project proposal: test maps are never used for selection; run this once, after every experiment is final.
`prepare` renders first so that parallel lanes never write the same cache file at once.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

REPOS = ["bc-baseline", "bc-framestack-half", "bc-framestack-double", "bc-cnn3d", "bc-cnn-gru",
         "bc-cnn-gru-e40", "bc-baseline-e40", "bc-softargmax"]
MODS = ["NM", "HR", "DT", "HD", "HRDT"]
OUT_NAME = "transfer_test.json"


def run_prefix(files: list[str], seed: int) -> str | None:
    """Where seed `seed` lives in a repo's file list: "seed<N>/", "" (one run at the root, as in
    bc-baseline; its seed is read from config.yaml by the caller), or None."""
    if any(f.startswith(f"seed{seed}/") for f in files):
        return f"seed{seed}/"
    return "" if "config.yaml" in files else None


def seeds_in(files: list[str]) -> list[int]:
    """Seed numbers with a folder seed<N>/ holding a best.pt."""
    out = set()
    for f in files:
        head, _, rest = f.partition("/")
        if head.startswith("seed") and head[4:].isdigit() and rest == "best.pt":
            out.add(int(head[4:]))
    return sorted(out)


def find_runs(api, org: str, repos: list[str]) -> list[dict]:
    """Every run in the repos that exist: {repo, seed, prefix, done}."""
    import yaml
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import RepositoryNotFoundError
    runs = []
    for name in repos:
        repo = f"{org}/{name}"
        try:
            files = api.list_repo_files(repo, repo_type="model")
        except RepositoryNotFoundError:
            continue
        found = [(s, f"seed{s}/") for s in seeds_in(files)]
        if "config.yaml" in files and "best.pt" in files:             # one run at the root
            root_seed = yaml.safe_load(Path(hf_hub_download(repo, "config.yaml")).read_text())["seed"]
            if root_seed not in [s for s, _ in found]:
                found.append((root_seed, ""))
        for seed, prefix in sorted(found):
            runs.append({"repo": repo, "seed": seed, "prefix": prefix, "done": prefix + OUT_NAME in files})
    return runs


def prepare(splits=("test", "val")) -> None:
    """Render and cache every split map under every mod (skips what is cached)."""
    from data.dataset import build_map_frames
    from eval.rollout import split_maps
    for split in splits:
        for mods in MODS:
            for bid, _, pl in split_maps(split, mods):
                build_map_frames(pl, bid)
            print(f"{split} {mods}: cached", flush=True)


def evaluate_run(api, run: dict, device: str) -> None:
    from train.hub import pull
    folder = pull(api, run["repo"], run["seed"])
    out = folder / OUT_NAME
    if not out.exists():
        subprocess.run([sys.executable, "-m", "eval.transfer_matrix", "--checkpoint", str(folder / "best.pt"),
                        "--split", "test", "--mods", *MODS, "--motor", "human",     # the HUMAN results on the Hub
                        "--device", device, "--out", str(out)], check=True)
    api.upload_file(path_or_fileobj=str(out), path_in_repo=run["prefix"] + OUT_NAME, repo_id=run["repo"],
                    repo_type="model", commit_message=f"test results, seed {run['seed']}")
    print(f"== {run['repo']} seed {run['seed']}: uploaded {run['prefix']}{OUT_NAME}", flush=True)


def main():
    import os
    load_dotenv()
    ap = argparse.ArgumentParser(description="Final test-map results for every run on the Hub.")
    ap.add_argument("action", choices=["prepare", "list", "run"])
    ap.add_argument("--org", default=os.environ.get("HF_ORG", "sc4001-project"))
    ap.add_argument("--repos", nargs="+", default=REPOS, help="repo names in the org (missing ones are skipped)")
    ap.add_argument("--lane", default="0/1", help="i/n: take every n-th run starting at i (one lane per GPU)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.action == "prepare":
        prepare()
        return
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    runs = find_runs(api, args.org, args.repos)
    if args.action == "list":
        for r in runs:
            print(f"{r['repo']:42} seed {r['seed']}  {'done' if r['done'] else 'to do'}  ({r['prefix'] or 'root'})")
        print(f"{sum(r['done'] for r in runs)} of {len(runs)} done")
        return
    i, n = (int(v) for v in args.lane.split("/"))
    mine = [r for k, r in enumerate(runs) if k % n == i]
    for r in mine:
        if r["done"]:
            print(f"== {r['repo']} seed {r['seed']}: already done", flush=True)
            continue
        evaluate_run(api, r, args.device)
    print(f"lane {args.lane}: finished {len(mine)} runs")


if __name__ == "__main__":
    main()
