"""Share data/raw/ through a PRIVATE Hugging Face dataset, so everyone trains on the same files.

    python -m data.hub push [--repo USER/NAME]   # create the private repo if needed, upload data/raw/
    python -m data.hub pull [--repo USER/NAME]   # download the dataset into data/raw/

The repo id comes from --repo or env HF_DATASET (.env is read). The token comes from
`hf auth login` or env HF_TOKEN. Nothing here creates or makes anything public.
"""
import argparse
import csv
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError

RAW = Path("data/raw")
CARD = """\
# osu-mod-transfer raw data (private)

Private dataset for the SC4001 (NTU) group project osu-mod-transfer. For coursework only: do not
redistribute it or make it public.

Replays (.osr) belong to the players who set them; beatmaps (.osu) belong to their mappers and to
ppy (osu!). They were fetched through the osu! API v2 and https://osu.ppy.sh/osu/{beatmap_id}.

## Contents

- `manifest.csv`: one row per replay (score, map, player, mods, accuracy, judgement counts, era,
  file paths relative to this folder). `mods` is the normalised set of studied mods (HD, HR, DT;
  NC counts as DT, SD/PF/NF are dropped); `mods_raw` is what the API returned. `style` and
  `split` are filled by later steps.
- `replays/{score_id}.osr`, `beatmaps/{beatmap_id}.osu`
- `failures.csv`: scores that could not be downloaded or did not parse.

## Rebuilding from manifest.csv

From the project repo, with osu! OAuth credentials (a supporter account, for mod-filtered
leaderboards) in `.env`:

    python -m data.download --from-manifest path/to/manifest.csv

This downloads exactly the listed scores and their beatmaps, and lists any replay that is no
longer available (players can delete scores, and old replays can be dropped).
"""


def merge_rows(local_path: Path, remote_path: Path, key: str = "score_id") -> int:
    """Append to local_path the rows of remote_path whose `key` it lacks; return how many.

    Each machine's manifest.csv only knows its own downloads, and upload_folder replaces the Hub's
    file, so pushing without this drops other machines' rows (it happened: the laptop's first 21 rows).
    """
    remote = list(csv.DictReader(remote_path.open(newline="")))
    if not local_path.exists():
        local_path.write_bytes(remote_path.read_bytes())
        return len(remote)
    with local_path.open(newline="") as f:
        reader = csv.DictReader(f)
        columns, local = list(reader.fieldnames or []), list(reader)
    have = {r[key] for r in local}
    extra = [r for r in remote if r[key] not in have]
    if extra:
        columns += [c for r in extra for c in r if c not in columns]   # keep any column only the Hub has
        with local_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(columns)), restval="")
            writer.writeheader()
            writer.writerows(local + extra)
    return len(extra)


def push(api, repo):
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    # exist_ok leaves an existing repo's visibility alone, so check it: never upload to a public repo
    info = api.repo_info(repo, repo_type="dataset")
    if not info.private:
        raise SystemExit(f"{repo} is public; refusing to upload. Make it private on the Hub first.")
    # Merge with the Hub's manifest as of one revision (info.sha), and upload on top of that same
    # revision: if a teammate pushed in between, the upload fails instead of dropping their rows.
    for name in ("manifest.csv", "failures.csv"):
        try:
            remote = hf_hub_download(repo, name, repo_type="dataset", token=api.token, revision=info.sha)
        except EntryNotFoundError:
            continue
        added = merge_rows(RAW / name, Path(remote))
        if added:
            print(f"merged {added} rows from the Hub's {name} (run `pull` to fetch their files)")
    (RAW / "README.md").write_text(CARD)
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=RAW,
                      # half-written downloads; data/select_maps.py's API cache; judge validation replays
                      ignore_patterns=["*.part", "cache/*", "judge_validation/*"],
                      commit_message="Update raw data", parent_commit=info.sha)
    print(f"pushed {RAW} to https://huggingface.co/datasets/{repo} (private)")


def pull(repo, token=None):
    # Files already in data/raw/ that differ from the Hub copy are overwritten; extra files are kept
    snapshot_download(repo, repo_type="dataset", local_dir=RAW, token=token)
    print(f"pulled {repo} into {RAW}")


def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description="Push or pull data/raw/ to a private HF dataset.")
    ap.add_argument("action", choices=["push", "pull"])
    ap.add_argument("--repo", default=os.environ.get("HF_DATASET"), help="USER/NAME (or env HF_DATASET)")
    args = ap.parse_args()
    if not args.repo:
        ap.error("give --repo USER/NAME or set HF_DATASET")
    # huggingface_hub reads HF_TOKEN from the environment when it is imported, before load_dotenv()
    # runs, so a token in .env must be passed explicitly. None falls back to `hf auth login`.
    token = os.environ.get("HF_TOKEN")
    if args.action == "push":
        push(HfApi(token=token), args.repo)
    else:
        pull(args.repo, token)


if __name__ == "__main__":
    main()
