"""train/hub.py without the network: runs go to private repos only, one folder per seed, and pull
reads both that layout and bc-baseline's (one run at the root)."""
from types import SimpleNamespace as NS

import pytest
import yaml

from train import hub


class FakeApi:
    def __init__(self, private=True, files=()):
        self.private, self.files, self.uploads = private, list(files), []

    def create_repo(self, repo, repo_type, private, exist_ok):
        assert (repo_type, private) == ("model", True)

    def repo_info(self, repo, repo_type):
        return NS(private=self.private)

    def list_repo_files(self, repo, repo_type):
        return self.files

    def upload_file(self, **kwargs):
        self.uploads.append(kwargs)

    def upload_folder(self, **kwargs):
        self.uploads.append(kwargs)


def make_run(path, name="cnn_gru", seed=2):
    path.mkdir(parents=True)
    (path / "config.yaml").write_text(yaml.safe_dump({"name": name, "seed": seed}))
    for f in ("best.pt", "last.pt", "run.json", "metrics.jsonl", "train.log", "val_epoch_000.json",
              "transfer_val.json"):
        (path / f).write_text("{}")
    return path


def test_repo_names_follow_bc_baseline():
    assert hub.repo_for("baseline") == "sc4001-project/bc-baseline"
    assert hub.repo_for("cnn_gru") == "sc4001-project/bc-cnn-gru"


def test_push_refuses_a_public_repo(tmp_path):
    api = FakeApi(private=False)
    with pytest.raises(SystemExit):
        hub.push(api, make_run(tmp_path / "run"))
    assert not api.uploads


def test_push_uploads_the_run_into_its_seed_folder_without_last(tmp_path):
    api = FakeApi()
    repo = hub.push(api, make_run(tmp_path / "run"))
    assert repo == "sc4001-project/bc-cnn-gru"
    card, folder = api.uploads
    assert card["path_in_repo"] == "README.md"
    assert folder["path_in_repo"] == "seed2"
    assert "best.pt" in folder["allow_patterns"] and "transfer_val.json" in folder["allow_patterns"]
    assert "last.pt" not in folder["allow_patterns"]
    assert hub.push(FakeApi(files=["README.md"]), tmp_path / "run", with_last=True)   # card kept, last.pt opt-in


def test_push_needs_a_finished_run(tmp_path):
    (tmp_path / "run").mkdir()
    with pytest.raises(SystemExit):
        hub.push(FakeApi(), tmp_path / "run")


def fake_download(remote: dict):
    def download(repo, filename, token=None):
        path = remote["dir"] / filename.replace("/", "__")
        path.write_text(remote["files"][filename])
        return str(path)
    return download


@pytest.mark.parametrize("layout", ["seed folder", "root"])
def test_pull_reads_both_layouts(tmp_path, monkeypatch, layout):
    prefix = "seed0/" if layout == "seed folder" else ""
    files = {prefix + "config.yaml": yaml.safe_dump({"name": "baseline", "seed": 0}), prefix + "best.pt": "w"}
    if not prefix:
        files["README.md"] = "card"
    (tmp_path / "hub").mkdir()
    monkeypatch.setattr(hub, "hf_hub_download", fake_download({"dir": tmp_path / "hub", "files": files}))
    out = hub.pull(FakeApi(files=list(files)), "sc4001-project/bc-baseline", 0, out=tmp_path / "runs" / "b")
    assert sorted(p.name for p in out.iterdir()) == ["best.pt", "config.yaml"]


def test_pull_refuses_a_seed_the_repo_does_not_have(tmp_path, monkeypatch):
    files = {"config.yaml": yaml.safe_dump({"name": "baseline", "seed": 0}), "best.pt": "w"}
    (tmp_path / "hub").mkdir()
    monkeypatch.setattr(hub, "hf_hub_download", fake_download({"dir": tmp_path / "hub", "files": files}))
    with pytest.raises(SystemExit):
        hub.pull(FakeApi(files=list(files)), "sc4001-project/bc-baseline", 1, out=tmp_path / "x")
