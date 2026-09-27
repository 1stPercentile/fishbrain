"""Tests for fishbrain.cells: retinotopic tectum drive and the identified-cells file."""
import os

import numpy as np
import pytest

from fishbrain import cells as C

AXES = {"midline": {"a": 32800.0, "b": 0.0}}


def synthetic_cells():
    """Two tectal lobes on a regular grid: x 40k-64k, |y - midline| 1k-24k, z 1k-8k.

    Dorsal cells are placed medially (as in Fish1, where corr(z, lateral) is ~0.75).
    """
    tect, seg = [], 1
    for side, sgn in (("left", 1), ("right", -1)):
        for x in np.linspace(40000, 64000, 13):
            for k, z in enumerate(np.linspace(1000, 8000, 8)):
                lat = 1000 + k * 3000
                tect.append({"seg_id": seg, "side": side, "pos": [float(x), 32800 + sgn * lat, float(z)]})
                seg += 1
    return {"axes": AXES, "tectum": tect,
            "exclusions": {"seg_ids": [999999]}}


def uv_of(cells):
    m = C.TectumMap(cells)
    return m, m.u, m.v


def test_left_eye_drives_only_right_lobe_anterior_dorsal():
    cells = synthetic_cells()
    m, u, v = uv_of(cells)
    d = C.tectum_drive({"left": [(0.0, 30.0, 1.0)]}, cells)
    assert d[m.side == "left"].max() == 0.0
    assert d[m.side == "right"].max() > 0.5
    top = np.argmax(d)
    assert m.side[top] == "right"
    assert u[top] < 1 / 3, "frontal field must land in the anterior third"
    assert v[top] < 0.5, "upper field must land dorsomedially"


def test_rear_field_maps_posterior():
    cells = synthetic_cells()
    m, u, v = uv_of(cells)
    d = C.tectum_drive({"left": [(-150.0, 0.0, 1.0)]}, cells)
    top = np.argmax(d)
    assert m.side[top] == "right" and u[top] > 2 / 3


def test_right_eye_lower_lateral_field_maps_left_lobe_ventrolateral():
    cells = synthetic_cells()
    m, u, v = uv_of(cells)
    d = C.tectum_drive({"right": [(90.0, -40.0, 1.0)]}, cells)
    assert d[m.side == "right"].max() == 0.0
    top = np.argmax(d)
    assert m.side[top] == "left" and v[top] > 0.5 and 0.2 < u[top] < 0.8


def test_outside_field_is_ignored_and_contrast_sign_ignored():
    cells = synthetic_cells()
    # 60 deg into the right hemifield is outside the left eye's field (theta_min = -20)
    assert C.tectum_drive({"left": [(60.0, 0.0, 1.0)]}, cells).sum() == 0.0
    a = C.tectum_drive({"left": [(-60.0, 0.0, 0.8)]}, cells)
    b = C.tectum_drive({"left": [(-60.0, 0.0, -0.8)]}, cells)
    assert np.allclose(a, b) and a.sum() > 0


def test_unit_response_at_preferred_location():
    cells = synthetic_cells()
    m = C.TectumMap(cells)
    i = int(np.nonzero(m.side == "right")[0][5])
    # right lobe is driven by the left eye; left-eye theta = -azimuth
    d = C.tectum_drive({"left": [(-m.theta[i], m.elev[i], 1.0)]}, cells)
    assert d[i] == pytest.approx(1.0, abs=1e-9)


def test_looming_disk_recruits_more_drive():
    cells = synthetic_cells()
    small = C.tectum_drive({"left": C.disk(-45.0, 10.0, 3.0)}, cells).sum()
    big = C.tectum_drive({"left": C.disk(-45.0, 10.0, 30.0)}, cells).sum()
    assert big > 5 * small > 0


def test_bad_eye_rejected():
    with pytest.raises(ValueError):
        C.tectum_drive({"middle": [(0, 0, 1)]}, synthetic_cells())


REAL = os.path.exists(C.CELLS_JSON)


@pytest.mark.skipif(not REAL, reason="data/cells_identified.json not built")
def test_real_file_invariants():
    cells = C.load()
    tect = cells["tectum"]
    ids = [c["seg_id"] for c in tect]
    assert len(ids) == len(set(ids)) and 0 not in ids
    never = C.never_stimulate(cells)
    assert not (set(ids) & never), "a readout / reticulospinal cell is in the visual input set"
    ml, mr = cells["mauthner"]["left"], cells["mauthner"]["right"]
    assert ml["seg_id"] in never and mr["seg_id"] in never
    assert ml["seg_id"] != mr["seg_id"]
    # the Mauthner pair sits on opposite sides of the fitted midline
    ax = cells["axes"]
    assert C.side_of(ml["pos"], ax)[0] == "left" and C.side_of(mr["pos"], ax)[0] == "right"
    assert len(cells["spn"]) == 50
    assert all(s["seg_id"] in never for s in cells["spn"] if s["seg_id"])
    # left eye drives only right-lobe tectum in the real map too
    d = C.tectum_drive({"left": C.disk(-30.0, 20.0, 10.0)}, cells)
    side = np.array([c["side"] for c in tect])
    assert d[side == "left"].max() == 0.0 and d[side == "right"].max() > 0.5
