"""The quickstart's two promises: a fresh clone reproduces the published G1c trials bit for bit, and it refuses a
kit that is not the one it pins."""
import json
import shutil

import pytest

from fishbrain import demo


def test_replay_matches_the_published_record(capsys):
    assert demo.main([]) == 0
    out = capsys.readouterr().out
    assert out.count("MATCH") == 2 and "DIFFERENT" not in out


def test_published_hashes_are_the_ones_checked():
    record = json.loads(demo.RECORD.read_text())["trials"]["intact"]
    for name, seed in demo.REPLAY:
        assert any(r["seed"] == seed and len(r["spike_hash"]) == 64 for r in record[name])


def test_a_tampered_kit_is_refused(tmp_path, monkeypatch):
    kit = tmp_path / "kit"
    shutil.copytree(demo.KIT, kit)
    blob = bytearray((kit / "topology_intact.npz").read_bytes())
    blob[-1] ^= 1
    (kit / "topology_intact.npz").write_bytes(bytes(blob))
    monkeypatch.setattr(demo, "KIT", kit)
    with pytest.raises(SystemExit, match="does not match its sha256"):
        demo.ensure_kit()


def test_removing_the_crossing_pathway_silences_the_escape():
    demo.ensure_kit()
    spec = demo.BR.load_spec()
    cond = demo.published_conditions(spec)["loom_az-45_lv120"]
    intact = demo.build_fish(1, 4, [])
    retina = demo.BR.GateRetina(intact.stim_ids, cache=False)
    assert demo.run_trial(intact, retina, cond, "G1-seed-0")["score"]["escape"] is True
    ablated = demo.build_fish(1, 4, ["c"])
    assert demo.run_trial(ablated, retina, cond, "G1-seed-0")["score"]["escape"] is False
