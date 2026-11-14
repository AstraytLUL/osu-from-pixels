"""data/download.py without the network: score filtering, caps and the manifest, on fake scores."""
from collections import Counter
from types import SimpleNamespace as NS

import pytest
from ossapi import Mod

from data import download as dl

CFG = {"accuracy_min": 0.98, "per_map": 10, "per_player": 3, "cap_groups": [["NM", "HR"]]}


def lazer(id, user=1, mods=(), acc=0.99, replay=True, legacy_id=None, ruleset=0, settings=None):
    """A score in the lazer shape: list of mods, great/ok/meh/miss, has_replay."""
    return NS(id=id, user_id=user, accuracy=acc, has_replay=replay, replay=replay,
              ruleset_id=ruleset, legacy_score_id=legacy_id,
              mods=[NS(acronym=a, settings=settings or {}) for a in mods],
              statistics=NS(great=500, ok=5, meh=None, miss=None))


def legacy(id, user=1, mods="NM", acc=0.99, replay=True, mode_int=0):
    """A score in the legacy shape: one Mod bitfield, count_*, replay, and a mode attribute."""
    return NS(id=id, user_id=user, accuracy=acc, replay=replay, mode="osu", mode_int=mode_int,
              mods=Mod(mods), statistics=NS(count_300=400, count_100=3, count_50=1, count_miss=0))


def test_mod_str_strips_cl_in_both_shapes():
    assert dl.mod_str(lazer(1, mods=["CL"])) == "NM"
    assert dl.mod_str(lazer(1, mods=["HD", "HR", "CL"])) == "HDHR"
    assert dl.mod_str(legacy(1, mods="NM")) == "NM"
    assert dl.mod_str(legacy(1, mods="HDHR")) == "HDHR"


@pytest.mark.parametrize("acronyms, mods", [
    (["SD"], "NM"), (["PF"], "NM"), (["NF"], "NM"), (["CL"], "NM"),   # ignored
    (["HD", "PF", "SD"], "HD"),
    (["NC"], "DT"), (["DT", "NC"], "DT"),                           # NC is DT (legacy NC sets both bits)
    (["DT", "HR", "CL", "HD"], "HDHRDT"),                           # canonical order
])
def test_normalise(acronyms, mods):
    assert dl.rejected_mods(acronyms) == []
    assert dl.normalise(acronyms) == mods


@pytest.mark.parametrize("mod", ["TD", "FL", "EZ", "HT", "SO", "RX", "AP", "DA", "MR", "TC", "WG"])
def test_rejected_mods(mod):                                    # listed ones, then lazer-only ones
    assert dl.rejected_mods(["HD", mod, "CL"]) == [mod]
    assert dl.reject_reason(lazer(1, mods=["HD", mod]), "HD", 0.98) == f"mod {mod}"


@pytest.mark.parametrize("mod", ["TD", "FL", "EZ", "HT", "SO", "RX", "AP"])
def test_rejected_mods_legacy_shape(mod):
    assert dl.reject_reason(legacy(1, mods="HD" + mod), "HD", 0.98) == f"mod {mod}"


def test_normalised_mods_match_the_board_in_both_shapes():
    assert dl.reject_reason(lazer(1, mods=["NC"]), "DT", 0.98) is None
    assert dl.reject_reason(legacy(1, mods="NC"), "DT", 0.98) is None       # bitfield DT + NC
    assert dl.reject_reason(legacy(1, mods="PF"), "NM", 0.98) is None       # bitfield SD + PF
    assert dl.reject_reason(lazer(1, mods=["HR", "SD", "CL"]), "HR", 0.98) is None


def test_settings_only_matter_on_kept_mods():
    sd_restart = lazer(1, mods=["HD", "SD"])
    sd_restart.mods[1].settings = {"restart": True}
    assert dl.reject_reason(sd_restart, "HD", 0.98) is None
    assert dl.reject_reason(lazer(1, mods=["NC"], settings={"speed_change": 1.2}), "DT", 0.98) \
        == "custom mod settings"


def test_era_and_legacy_score_id():
    assert (dl.era(lazer(1, legacy_id=55)), dl.legacy_score_id(lazer(1, legacy_id=55))) == ("stable", 55)
    assert (dl.era(lazer(1)), dl.legacy_score_id(lazer(1))) == ("lazer", None)
    assert (dl.era(legacy(7)), dl.legacy_score_id(legacy(7))) == ("stable", 7)


def test_counts_both_shapes():
    assert dl.counts(lazer(1)) == (500, 5, 0, 0)            # None judgements count as 0
    assert dl.counts(legacy(1)) == (400, 3, 1, 0)


@pytest.mark.parametrize("sc, mod_set, reason", [
    (lazer(1, mods=["CL"]), "NM", None),
    (legacy(1), "NM", None),
    (lazer(1, mods=["HR", "HD"]), "HDHR", None),             # acronym order doesn't matter
    (legacy(1, mods="HDHR"), "HDHR", None),
    (lazer(1, replay=False), "NM", "no replay"),
    (legacy(1, replay=False), "NM", "no replay"),
    (lazer(1, acc=0.975), "NM", "low accuracy"),
    (legacy(1, acc=0.975), "NM", "low accuracy"),
    (lazer(1, mods=["HD", "HR"]), "HD", "mods HDHR"),
    (legacy(1, mods="HD"), "NM", "mods HD"),
    (lazer(1, mods=["DT"], settings={"speed_change": 1.3}), "DT", "custom mod settings"),
    (lazer(1, ruleset=3), "NM", "not osu!standard"),
    (legacy(1, mode_int=1), "NM", "not osu!standard"),
])
def test_reject_reason(sc, mod_set, reason):
    assert dl.reject_reason(sc, mod_set, 0.98) == reason


def test_select_caps_per_player_and_skips_seen():
    scores = [lazer(i, user=1) for i in range(5)] + [legacy(10, user=2), lazer(11, user=3)]
    kept, skipped = dl.select(scores, "NM", CFG, seen={11}, per_player=Counter(), have=0)
    assert [sc.id for sc in kept] == [0, 1, 2, 10]
    assert [(sc.id, r) for sc, r in skipped] == [(3, "player cap"), (4, "player cap"), (11, "already seen")]


def test_player_cap_counts_across_maps():
    # one player tops the NM board of 5 maps: only their first 3 are kept
    per_player, kept_ids = Counter(), []
    for bid in range(5):
        kept, skipped = dl.select([lazer(100 + bid, user=1)], "NM", CFG, set(), per_player, have=0)
        kept_ids += [sc.id for sc in kept]
    assert kept_ids == [100, 101, 102]
    assert skipped[0][1] == "player cap"


def test_player_cap_counts_manifest_rows_when_resuming():
    per_player = Counter({(1, dl.cap_key("NM", CFG["cap_groups"])): 3})   # from earlier runs
    kept, _ = dl.select([lazer(1, user=1)], "NM", CFG, set(), per_player, have=0)
    assert kept == []


def test_cap_shared_by_nm_and_hr_but_not_hd():
    per_player = Counter()
    dl.select([lazer(1, user=1), lazer(2, user=1)], "NM", CFG, set(), per_player, have=0)
    kept, skipped = dl.select([lazer(3, user=1, mods=["HR"]), lazer(4, user=1, mods=["HR"])],
                              "HR", CFG, set(), per_player, have=0)
    assert [sc.id for sc in kept] == [3] and skipped[0][1] == "player cap"   # 2 NM + 1 HR = 3
    kept, _ = dl.select([lazer(5, user=1, mods=["HD"])], "HD", CFG, set(), per_player, have=0)
    assert [sc.id for sc in kept] == [5]                   # HD has its own cap


def test_cap_key():
    groups = [["NM", "HR"]]
    assert dl.cap_key("NM", groups) == dl.cap_key("HR", groups)
    assert dl.cap_key("HD", groups) != dl.cap_key("NM", groups)
    assert dl.cap_key("HDHR", groups) == dl.cap_key("HRHD", groups) != dl.cap_key("HR", groups)


def test_select_stops_when_map_is_full():
    scores = [lazer(i, user=i) for i in range(5)]
    kept, skipped = dl.select(scores, "NM", CFG, set(), Counter(), have=8)
    assert [sc.id for sc in kept] == [0, 1]
    assert {r for _, r in skipped} == {"map full"}


def test_manifest_row_and_round_trip(tmp_path):
    sc = lazer(42, user=9, mods=["HD", "CL"], legacy_id=4242)
    row = dl.manifest_row(sc, 123, 8, "replays/42.osr", "beatmaps/123.osu")
    assert list(row) == dl.MANIFEST_COLUMNS
    assert (row["mods"], row["legacy_mods"], row["era"], row["legacy_score_id"]) == ("HD", 8, "stable", 4242)
    assert row["mods_raw"] == "HDCL"                         # as the API gave them
    assert (row["count_300"], row["count_100"], row["count_50"], row["count_miss"]) == (500, 5, 0, 0)
    assert row["style"] == row["split"] == ""

    path = tmp_path / "manifest.csv"
    dl.append_row(path, row, dl.MANIFEST_COLUMNS)
    dl.append_row(path, dl.manifest_row(legacy(43), 123, 0, "replays/43.osr", "beatmaps/123.osu"),
                  dl.MANIFEST_COLUMNS)
    rows = dl.read_rows(path)
    assert [r["score_id"] for r in rows] == ["42", "43"]   # one header, rows in order
    assert rows[1]["era"] == "stable" and rows[1]["mods"] == rows[1]["mods_raw"] == "NM"
    assert dl.read_rows(tmp_path / "missing.csv") == []


def test_read_map_ids_text_and_yaml(tmp_path):
    path = tmp_path / "maps.txt"
    path.write_text("# jump maps\n814293\n816600  # stream\n\n- 2935781\n814293\n")
    assert dl.read_map_ids(path) == [814293, 816600, 2935781]


def test_with_backoff_treats_html_bytes_as_rate_limit(monkeypatch):
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)
    answers = iter([b"<!DOCTYPE html>429", b"\x00replay"])
    assert dl.with_backoff(lambda: next(answers), "test") == b"\x00replay"
    with pytest.raises(RuntimeError):
        dl.with_backoff(lambda: b"<html>", "test", tries=2)


class FakeApi:
    """download_score / download_score_mode that log the call; ids in `gone` answer with an API error."""
    def __init__(self, gone=()):
        self.gone, self.calls = set(gone), []

    def download_score(self, score_id, raw):
        return self._get("new", score_id)

    def download_score_mode(self, mode, score_id, raw):
        return self._get("old", score_id)

    def _get(self, endpoint, score_id):
        self.calls.append((endpoint, score_id))
        if score_id in self.gone:
            raise ValueError("api returned an error of `Not Found`")
        return b"\x00fake osr"


@pytest.fixture
def raw(tmp_path, monkeypatch):
    """data/raw/ in a temp dir, with no sleeping, no .osu download and a stub replay parser."""
    monkeypatch.setattr(dl, "RAW", tmp_path)
    monkeypatch.setattr(dl, "MANIFEST", tmp_path / "manifest.csv")
    monkeypatch.setattr(dl, "FAILURES", tmp_path / "failures.csv")
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)
    monkeypatch.setattr(dl, "fetch_beatmap", lambda bid, path: None)
    monkeypatch.setattr(dl, "load_replay", lambda path: NS(t=[0, 16], mods=8))
    (tmp_path / "replays").mkdir()
    return tmp_path


def test_rebuild_downloads_exactly_listed_scores(raw, tmp_path_factory):
    src = tmp_path_factory.mktemp("other") / "manifest.csv"
    for sc in [lazer(1, legacy_id=99), lazer(2), legacy(3), lazer(4)]:
        dl.append_row(src, dl.manifest_row(sc, 555, 0, f"replays/{sc.id}.osr", "beatmaps/555.osu"),
                      dl.MANIFEST_COLUMNS)
    (raw / "replays" / "4.osr").write_bytes(b"\x00")         # left by an interrupted run
    api = FakeApi(gone={2})
    dl.rebuild(api, dl.read_rows(src), seen=set(), download_sleep=0, dry_run=False)

    # lazer ids use download_score, a legacy-shape id the per-mode endpoint, 4 is already on disk
    assert api.calls == [("new", 1), ("new", 2), ("old", 3)]
    rows = dl.read_rows(dl.MANIFEST)
    assert [r["score_id"] for r in rows] == ["1", "3", "4"]
    assert rows[0]["legacy_mods"] == "8" and rows[0]["legacy_score_id"] == "99"
    assert [r["score_id"] for r in dl.read_rows(dl.FAILURES)] == ["2"]
    assert sorted(p.name for p in (raw / "replays").iterdir()) == ["1.osr", "3.osr", "4.osr"]

    api = FakeApi()                                          # a second run asks for nothing
    seen = {int(r["score_id"]) for r in rows + dl.read_rows(dl.FAILURES)}
    dl.rebuild(api, dl.read_rows(src), seen, download_sleep=0, dry_run=False)
    assert api.calls == []


def test_collect_warns_on_shortfall_and_keeps_the_map(raw, capsys):
    class Board:                                             # NM full board, DT board with 1 replay
        def beatmap_scores(self, bid, mode, mods, limit):
            n = 10 if mods == "NM" else 1
            return NS(scores=[lazer(bid * 100 + i, user=i, mods=[] if mods == "NM" else ["DT"])
                              for i in range(n)])
    cfg = CFG | {"mod_sets": ["NM", "DT"], "leaderboard_limit": 100, "leaderboard_sleep": 0}
    dl.collect(Board(), [7], cfg, rows=[], seen=set(), dry_run=True)
    out = capsys.readouterr().out
    assert "would download 11 replays: NM 10, DT 1" in out
    assert "WARNING: 7 DT short, 1 of 10 (DT is test-only" in out
    assert "WARNING: 7 NM" not in out
