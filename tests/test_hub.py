"""data/hub.py without the network: push must never upload to a public repo, and must never drop
rows another machine pushed."""
import csv
from types import SimpleNamespace as NS

import pytest

from data import hub


class FakeApi:
    token = None

    def __init__(self, private):
        self.private, self.uploaded = private, False

    def create_repo(self, repo, repo_type, private, exist_ok):
        assert (repo_type, private) == ("dataset", True)

    def repo_info(self, repo, repo_type):
        return NS(private=self.private,sha="remote-base")

    def upload_folder(self, **kwargs):
        assert kwargs['parent_commit'] == 'remote-base'
        self.uploaded = True


class Missing(hub.EntryNotFoundError):
    def __init__(self):
        Exception.__init__(self, "not on the Hub")


def no_remote_files(*args, **kwargs):
    raise Missing()


def write_csv(path, rows, columns):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    return list(csv.DictReader(path.open(newline="")))


def test_push_refuses_public_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "RAW", tmp_path)
    monkeypatch.setattr(hub, "hf_hub_download", no_remote_files)
    api = FakeApi(private=False)
    with pytest.raises(SystemExit):
        hub.push(api, "someone/data")
    assert not api.uploaded


def test_push_private_repo_writes_card(tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "RAW", tmp_path)
    monkeypatch.setattr(hub, "hf_hub_download", no_remote_files)
    api = FakeApi(private=True)
    hub.push(api, "someone/data")
    assert api.uploaded
    assert "coursework only" in (tmp_path / "README.md").read_text()


def test_merge_keeps_rows_only_the_hub_has(tmp_path):
    local, remote = tmp_path / "local.csv", tmp_path / "remote.csv"
    write_csv(local, [{"score_id": "1", "mods": "NM"}, {"score_id": "2", "mods": "HD"}], ["score_id", "mods"])
    write_csv(remote, [{"score_id": "2", "mods": "HD"}, {"score_id": "3", "mods": "DT"}], ["score_id", "mods"])
    assert hub.merge_rows(local, remote) == 1
    assert [r["score_id"] for r in read_csv(local)] == ["1", "2", "3"]
    assert hub.merge_rows(local, remote) == 0                 # merging again changes nothing


def test_merge_keeps_extra_columns_and_creates_missing_local(tmp_path):
    local, remote = tmp_path / "local.csv", tmp_path / "remote.csv"
    write_csv(local, [{"score_id": "1", "mods": "NM", "mods_raw": "NM"}], ["score_id", "mods", "mods_raw"])
    write_csv(remote, [{"score_id": "9", "mods": "HR", "old_col": "x"}], ["score_id", "mods", "old_col"])
    hub.merge_rows(local, remote)
    rows = read_csv(local)
    assert rows[1] == {"score_id": "9", "mods": "HR", "mods_raw": "", "old_col": "x"}
    fresh = tmp_path / "none.csv"
    assert hub.merge_rows(fresh, remote) == 1 and read_csv(fresh)[0]["score_id"] == "9"


def test_push_merges_the_hub_manifest_first(tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "RAW", tmp_path)
    write_csv(tmp_path / "manifest.csv", [{"score_id": "1"}], ["score_id"])
    hub_copy = tmp_path / "hub_manifest.csv"
    write_csv(hub_copy, [{"score_id": "7"}], ["score_id"])

    def fake_download(repo, name, **kwargs):
        assert kwargs['revision'] == 'remote-base'
        if name == "manifest.csv":
            return str(hub_copy)
        raise Missing()

    monkeypatch.setattr(hub, "hf_hub_download", fake_download)
    hub.push(FakeApi(private=True), "someone/data")
    assert [r["score_id"] for r in read_csv(tmp_path / "manifest.csv")] == ["1", "7"]
