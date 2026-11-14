"""Share training runs (train/bc.py's run folders) through PRIVATE Hugging Face model repos.

    python -m train.hub push runs/cnn3d_seed0             # -> sc4001-project/bc-cnn3d, folder seed0/
    python -m train.hub push runs/cnn3d_seed0 --with-last # also last.pt (to resume training elsewhere)
    python -m train.hub pull sc4001-project/bc-cnn3d --seed 1    # -> runs/cnn3d_seed1/
    python -m train.hub pull sc4001-project/bc-baseline --seed 0 # Astrayt's baseline (files at the root)

Layout: one repo per config name, `<org>/bc-<name>` (underscores become hyphens: cnn_gru ->
bc-cnn-gru), and one folder per seed, `seed<N>/`, holding what sc4001-project/bc-baseline holds:
best.pt, config.yaml, run.json, metrics.jsonl, train.log, val_epoch_*.json and transfer_*.json
(eval.transfer_matrix --out runs/<run>/transfer_val.json). last.pt (40 MB, with optimizer state) only
with --with-last. bc-baseline itself keeps its one seed at the repo root; pull reads that layout too.

The org comes from --org or env HF_ORG (default sc4001-project); the token from `hf auth login` or
env HF_TOKEN (.env is read; a write token is needed to push). Like data/hub.py, nothing here
creates or uploads to a public repo.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import yaml
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

ORG = "sc4001-project"
RUNS = Path("runs")
FILES = ["best.pt", "config.yaml", "run.json", "metrics.jsonl", "train.log", "val_epoch_*.json", "transfer_*.json"]
CARD = """\
# {repo} (private)

Behaviour-cloning policies for osu!, from the SC4001 (NTU) group project osu-mod-transfer. For
coursework only: do not redistribute or make public. The training data are players' replays and
mappers' beatmaps (see the dataset sc4001-project/osu-mod-transfer-raw).

Config `{name}` (configs/arch/ or configs/ in the project repo). One folder per seed, each a run
folder of train/bc.py: `best.pt` (the epoch that played the NoMod val maps best), `config.yaml`,
`run.json` (git revision, parameter count), `metrics.jsonl`, `train.log`, `val_epoch_*.json`, and
`transfer_val.json` (eval.transfer_matrix on the val maps) when it was run.

    python -m train.hub pull {repo} --seed 0
    python -m eval.play --checkpoint runs/{name}_seed0/best.pt --beatmap 4912305 --mods HR
"""


def repo_for(name: str, org: str = ORG) -> str:
    """Config name -> its model repo: "cnn_gru" -> "sc4001-project/bc-cnn-gru"."""
    return f"{org}/bc-{name.replace('_', '-')}"


def run_files(run: Path, with_last: bool = False) -> list[Path]:
    """The files of a run folder that go to the Hub."""
    files = sorted({p for pattern in FILES for p in run.glob(pattern)})
    if with_last and (run / "last.pt").exists():
        files.append(run / "last.pt")
    return files


def _git_short(run: Path) -> str:
    try:
        return json.loads((run / "run.json").read_text())["git_revision"][:7]
    except (OSError, KeyError, ValueError):
        return "unknown"


def push(api, run: Path, repo: str | None = None, org: str = ORG, with_last: bool = False) -> str:
    """Upload a run folder to <repo>/seed<N>/. Returns the repo id."""
    if not (run / "config.yaml").exists() or not (run / "best.pt").exists():
        raise SystemExit(f"{run} is not a finished run folder (no config.yaml / best.pt)")
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    name, seed = cfg["name"], cfg["seed"]
    repo = repo or repo_for(name, org)
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    # exist_ok leaves an existing repo's visibility alone, so check it: never upload to a public repo
    if not api.repo_info(repo, repo_type="model").private:
        raise SystemExit(f"{repo} is public; refusing to upload. Make it private on the Hub first.")
    if "README.md" not in api.list_repo_files(repo, repo_type="model"):
        api.upload_file(path_or_fileobj=CARD.format(repo=repo, name=name).encode(), path_in_repo="README.md",
                        repo_id=repo, repo_type="model", commit_message="Model card")
    files = run_files(run, with_last)
    if not (run / "transfer_val.json").exists():
        print(f"note: no transfer_val.json in {run} (python -m eval.transfer_matrix --checkpoint "
              f"{run}/best.pt --split val --mods NM HR DT HD HRDT --out {run}/transfer_val.json)")
    api.upload_folder(repo_id=repo, repo_type="model", folder_path=run, path_in_repo=f"seed{seed}",
                      allow_patterns=[p.name for p in files],
                      commit_message=f"{name}, seed {seed} (git {_git_short(run)})")
    print(f"pushed {len(files)} files of {run} to https://huggingface.co/{repo}/tree/main/seed{seed} (private)")
    return repo


def pull(api, repo: str, seed: int, out: Path | None = None, token=None) -> Path:
    """Download seed<N>/ of a repo (or, if the repo has no seed folders, its root: bc-baseline) into
    runs/<name>_seed<N>/ (or `out`). Returns the folder."""
    remote = api.list_repo_files(repo, repo_type="model")
    prefix = f"seed{seed}/"
    if any(f.startswith(prefix) for f in remote):
        wanted = [f for f in remote if f.startswith(prefix)]
    elif "config.yaml" in remote:                       # one run at the root
        wanted = [f for f in remote if "/" not in f and f not in ("README.md", ".gitattributes")]
        root_seed = yaml.safe_load(Path(hf_hub_download(repo, "config.yaml", token=token)).read_text())["seed"]
        if root_seed != seed:
            raise SystemExit(f"{repo} holds seed {root_seed} at its root, not seed {seed}")
        prefix = ""
    else:
        raise SystemExit(f"{repo} has no seed{seed}/ folder")
    cfg_file = Path(hf_hub_download(repo, prefix + "config.yaml", token=token))
    name = yaml.safe_load(cfg_file.read_text())["name"]
    out = out or RUNS / f"{name}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    for f in wanted:
        shutil.copyfile(hf_hub_download(repo, f, token=token), out / f[len(prefix):])
    print(f"pulled {len(wanted)} files of {repo} ({prefix or 'root'}) into {out}")
    return out


def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description="Push or pull training runs to private HF model repos.")
    sub = ap.add_subparsers(dest="action", required=True)
    p = sub.add_parser("push", help="upload a run folder")
    p.add_argument("run", type=Path, help="e.g. runs/cnn3d_seed0")
    p.add_argument("--repo", help="default <org>/bc-<config name>")
    p.add_argument("--with-last", action="store_true", help="also upload last.pt")
    q = sub.add_parser("pull", help="download one seed of a repo")
    q.add_argument("repo", help="e.g. sc4001-project/bc-cnn3d")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--out", type=Path, help="default runs/<config name>_seed<seed>")
    for s in (p, q):
        s.add_argument("--org", default=os.environ.get("HF_ORG", ORG))
    args = ap.parse_args()
    # huggingface_hub reads HF_TOKEN when imported, before load_dotenv(): pass a .env token explicitly
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    if args.action == "push":
        push(api, args.run, args.repo, args.org, args.with_last)
    else:
        pull(api, args.repo, args.seed, args.out, token)


if __name__ == "__main__":
    main()
