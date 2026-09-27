"""Visual stimuli for FISHBRAIN's eyes (G1, task B2).

Each stimulus is a sequence of frames. Frame i covers t in [i*dt_ms, (i+1)*dt_ms) and is

    {"left": [(azimuth_deg, elevation_deg, contrast), ...],
     "right": [(azimuth_deg, elevation_deg, contrast), ...]}

which is the input cells.tectum_drive converts into tectal drive. Frames are computed on demand
(a whole-field dimming lists thousands of points per eye per frame), and `stim.arrays(i)` gives
the same frame as two (n, 3) float64 arrays for speed.

Conventions
-----------
* Fish-centred, head-fixed directions. azimuth_deg: 0 = straight ahead, positive = the fish's
  right, negative = its left, in (-180, 180]. elevation_deg: 0 = horizon, +90 = straight up.
  A point is listed under every eye whose visual field contains it, so points in the frontal
  binocular zone appear in both lists. Each eye's field is a cone of FOV_DEG = 163 degrees (the
  functional retina's angular subtense, Easter & Nicola 1996, as used by the BadenLab
  Zebrafish-visual-space-model) around the eye's optical axis. The optical axis sits at azimuth
  +/-(90 - eye_angle_deg), elevation 0, where eye_angle_deg is the angle between the eye's long
  axis and the body axis: 18.5 at rest (BadenLab model parameter; Bianco et al. 2011 measure a
  vergence of 36.0 deg at rest and 66.9-76.4 deg while hunting). The axis derivation, the cone
  shape and the zero eye tilt are our choices. Pass another EyeModel when the body converges
  the eyes.
* The visual sphere is sampled on a fixed near-equal-area lattice with spacing res_deg (default
  2 deg): one ring per elevation band, round(360 cos(el) / res) points per ring. Only points that
  differ from the background are listed (except in dimming, where every point differs).
* contrast is signed: -1 = black on the background (OFF), +1 = the brightest spot (ON), 0 = the
  background. A disc's edge is antialiased: a lattice point at angular distance d from the centre
  of a disc of radius r gets contrast * clip((r + res/2 - d) / res, 0, 1), so a dot smaller than
  the lattice spacing still appears, dimmer, and the summed |contrast| tracks the disc's area.
* Output values are rounded to 1e-6, so ulp-level differences between platforms' trig functions
  only matter if they straddle a rounding boundary. They are not eliminated.

Stimuli and their sources
-------------------------
looming   dark disc, angular size theta(t) = 2 atan(l/v / (t_c - t)), the classic l/v time course
          (Gabbiani et al.). Defaults l/|v| = 240 ms, 4 -> 140 deg, dark on bright: Fotowat &
          Engert 2023, eLife 12:e82916 (PMC10014075), methods.
receding  the looming time course reversed in time (140 -> 4 deg): the standard control.
dimming   uniform whole-field darkening with no motion. By default its contrast follows the
          looming disc's covered fraction of one eye's field, -Omega_disc(t) / Omega_eye, which
          matches the loom's luminance loss in the eye that sees it (the dimming component
          Fotowat & Engert 2023 separate out). The matching rule is our choice.
prey_dot  a small bright spot (default 3 deg, contrast +1) sweeping back and forth around a given
          azimuth at 30 deg/s. Bianco, Kampff & Engert 2011 (Front Syst Neurosci 5:101): bright
          white spots at maximum contrast; 1-5 deg spots evoke prey-like orienting, 10 deg evokes
          avoidance; ~30 deg/s in the restrained assay (30 and 60 deg/s free-swimming); 3 s trials.
          The 20 deg sweep is our choice.
"""
from __future__ import annotations

import collections.abc
import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

STIMULI_VERSION = "fishbrain-stimuli-v1"
RES_DEG = 2.0
FOV_DEG = 163.0
EYE_ANGLE_REST_DEG = 18.5
EYES = ("left", "right")


@dataclass(frozen=True)
class EyeModel:
    fov_deg: float = FOV_DEG
    eye_angle_deg: float = EYE_ANGLE_REST_DEG
    eye_elev_deg: float = 0.0

    def axis_az(self, eye: str) -> float:
        az = 90.0 - self.eye_angle_deg
        return az if eye == "right" else -az

    def solid_angle_sr(self) -> float:
        return 2.0 * math.pi * (1.0 - math.cos(math.radians(self.fov_deg / 2.0)))


def unit_vectors(az_deg, el_deg) -> np.ndarray:
    """x forward, y right, z up."""
    az = np.radians(np.asarray(az_deg, np.float64))
    el = np.radians(np.asarray(el_deg, np.float64))
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)


def angle_between_deg(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    # elementwise, not `@`: BLAS may fuse or reorder the sum differently on another machine
    dot = v1[..., 0] * v2[..., 0] + v1[..., 1] * v2[..., 1] + v1[..., 2] * v2[..., 2]
    return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))


@lru_cache(maxsize=16)
def lattice(res_deg: float = RES_DEG) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(az_deg, el_deg, unit_xyz) of the sampling lattice; deterministic, rounded to 1e-6."""
    n_rings = int(round(180.0 / res_deg))
    az_l, el_l = [], []
    for i in range(n_rings):
        el = -90.0 + (i + 0.5) * 180.0 / n_rings
        n_az = max(1, int(round(360.0 * math.cos(math.radians(el)) / res_deg)))
        az = -180.0 + (np.arange(n_az) + 0.5) * 360.0 / n_az
        az_l.append(az)
        el_l.append(np.full(n_az, el))
    az = np.round(np.concatenate(az_l), 6)
    el = np.round(np.concatenate(el_l), 6)
    xyz = unit_vectors(az, el)
    for a in (az, el, xyz):
        a.setflags(write=False)
    return az, el, xyz


@lru_cache(maxsize=16)
def eye_masks(res_deg: float, eye: EyeModel) -> Dict[str, np.ndarray]:
    """Boolean mask over the lattice for each eye's field of view."""
    _, _, xyz = lattice(res_deg)
    out = {}
    for e in EYES:
        axis = unit_vectors(eye.axis_az(e), eye.eye_elev_deg)
        m = angle_between_deg(xyz, axis) <= eye.fov_deg / 2.0
        m.setflags(write=False)
        out[e] = m
    return out


def in_eye_field(az_deg, el_deg, eye: str, model: EyeModel = EyeModel()) -> np.ndarray:
    axis = unit_vectors(model.axis_az(eye), model.eye_elev_deg)
    return angle_between_deg(unit_vectors(az_deg, el_deg), axis) <= model.fov_deg / 2.0


def disc_solid_angle_sr(diameter_deg) -> np.ndarray:
    return 2.0 * np.pi * (1.0 - np.cos(np.radians(np.asarray(diameter_deg, np.float64) / 2.0)))


# ---------------------------------------------------------------------------------------------

class Stimulus(collections.abc.Sequence):
    """A lazily rendered frame sequence. `trace` holds the per-frame description:

    t_ms, and for disc stimuli az_deg, el_deg, size_deg (angular diameter), contrast, visible;
    for uniform stimuli contrast and the eyes it applies to.
    """

    def __init__(self, kind: str, dt_ms: float, trace: Dict[str, np.ndarray], params: dict,
                 eye: EyeModel = EyeModel(), res_deg: float = RES_DEG, uniform_eyes=EYES):
        self.kind, self.dt_ms, self.trace, self.params = kind, float(dt_ms), trace, dict(params)
        self.eye, self.res_deg = eye, float(res_deg)
        self.uniform_eyes = tuple(uniform_eyes)
        self._n = len(trace["t_ms"])

    def __len__(self) -> int:
        return self._n

    @property
    def duration_ms(self) -> float:
        return self._n * self.dt_ms

    def time_ms(self, i: int) -> float:
        return float(self.trace["t_ms"][i])

    def frame_at(self, t_ms: float) -> int:
        return int(min(max(math.floor(t_ms / self.dt_ms + 1e-9), 0), self._n - 1))

    def arrays(self, i: int) -> Dict[str, np.ndarray]:
        if i < 0:
            i += self._n
        if not 0 <= i < self._n:
            raise IndexError(i)
        az, el, xyz = lattice(self.res_deg)
        masks = eye_masks(self.res_deg, self.eye)
        tr = self.trace
        out = {}
        if self.kind == "dimming":
            c = float(tr["contrast"][i])
            for e in EYES:
                if e in self.uniform_eyes and c != 0.0:
                    m = masks[e]
                    out[e] = np.stack([az[m], el[m], np.full(int(m.sum()), round(c, 6))], axis=1)
                else:
                    out[e] = np.zeros((0, 3))
            return out
        if not tr["visible"][i]:
            return {e: np.zeros((0, 3)) for e in EYES}
        centre = unit_vectors(tr["az_deg"][i], tr["el_deg"][i])
        r = float(tr["size_deg"][i]) / 2.0
        d = angle_between_deg(xyz, centre)
        cov = np.clip((r + self.res_deg / 2.0 - d) / self.res_deg, 0.0, 1.0)
        sel = cov > 0.0
        c = np.round(float(tr["contrast"][i]) * cov, 6)
        for e in EYES:
            m = sel & masks[e]
            out[e] = np.stack([az[m], el[m], c[m]], axis=1)
        return out

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(self._n))]
        a = self.arrays(i)
        return {e: [(float(x), float(y), float(z)) for x, y, z in a[e]] for e in EYES}

    def describe(self) -> dict:
        return {"version": STIMULI_VERSION, "kind": self.kind, "dt_ms": self.dt_ms, "n_frames": self._n,
                "res_deg": self.res_deg, "eye": dataclasses.asdict(self.eye), "params": self.params,
                "uniform_eyes": list(self.uniform_eyes)}

    def digest(self) -> str:
        """sha256 of the stimulus description (the rendering code is versioned by STIMULI_VERSION)."""
        return hashlib.sha256(json.dumps(self.describe(), sort_keys=True, default=repr).encode()).hexdigest()


def _times(n: int, dt_ms: float) -> np.ndarray:
    return np.arange(n, dtype=np.float64) * dt_ms


def _n_frames(total_ms: float, dt_ms: float) -> int:
    return max(1, int(math.ceil(total_ms / dt_ms - 1e-9)))


def loom_schedule(l_over_v_ms: float = 240.0, start_deg: float = 4.0, end_deg: float = 140.0):
    """(t_collision_ms, t_end_ms) measured from onset (theta = start_deg at t = 0)."""
    tc = l_over_v_ms / math.tan(math.radians(start_deg) / 2.0)
    t_end = tc - l_over_v_ms / math.tan(math.radians(end_deg) / 2.0)
    return tc, t_end


def loom_angle_deg(t_ms, l_over_v_ms: float = 240.0, start_deg: float = 4.0, end_deg: float = 140.0) -> np.ndarray:
    """theta(t) = 2 atan(l/v / (t_c - t)) for t in [0, t_end], held at end_deg afterwards."""
    tc, t_end = loom_schedule(l_over_v_ms, start_deg, end_deg)
    t = np.minimum(np.asarray(t_ms, np.float64), t_end)
    return np.degrees(2.0 * np.arctan(l_over_v_ms / (tc - t)))


def _disc_trace(t, az, el, size, contrast, visible):
    n = len(t)
    return {"t_ms": t, "az_deg": np.broadcast_to(np.asarray(az, np.float64), (n,)).copy(),
            "el_deg": np.broadcast_to(np.asarray(el, np.float64), (n,)).copy(),
            "size_deg": np.asarray(size, np.float64), "contrast": np.full(n, float(contrast)),
            "visible": np.asarray(visible, bool)}


def looming(azimuth_deg: float = 90.0, elevation_deg: float = 0.0, l_over_v_ms: float = 240.0,
            start_deg: float = 4.0, end_deg: float = 140.0, contrast: float = -1.0,
            pre_ms: float = 0.0, hold_ms: float = 0.0, dt_ms: float = 1.0, res_deg: float = RES_DEG,
            eye: EyeModel = EyeModel()) -> Stimulus:
    """A dark disc expanding from start_deg to end_deg along theta(t) = 2 atan(l/v / (t_c - t)),
    centred at (azimuth, elevation), after pre_ms of blank and held at end_deg for hold_ms."""
    _, t_end = loom_schedule(l_over_v_ms, start_deg, end_deg)
    n = _n_frames(pre_ms + t_end + hold_ms, dt_ms)
    t = _times(n, dt_ms)
    rel = t - pre_ms
    size = np.where(rel >= 0, loom_angle_deg(np.maximum(rel, 0.0), l_over_v_ms, start_deg, end_deg), 0.0)
    params = dict(azimuth_deg=azimuth_deg, elevation_deg=elevation_deg, l_over_v_ms=l_over_v_ms,
                  start_deg=start_deg, end_deg=end_deg, contrast=contrast, pre_ms=pre_ms, hold_ms=hold_ms,
                  expansion_ms=t_end)
    return Stimulus("looming", dt_ms, _disc_trace(t, azimuth_deg, elevation_deg, size, contrast, rel >= 0),
                    params, eye, res_deg)


def receding(azimuth_deg: float = 90.0, elevation_deg: float = 0.0, l_over_v_ms: float = 240.0,
             start_deg: float = 4.0, end_deg: float = 140.0, contrast: float = -1.0,
             pre_ms: float = 0.0, hold_ms: float = 0.0, dt_ms: float = 1.0, res_deg: float = RES_DEG,
             eye: EyeModel = EyeModel()) -> Stimulus:
    """The looming time course reversed: end_deg shrinking to start_deg, then held at start_deg."""
    _, t_end = loom_schedule(l_over_v_ms, start_deg, end_deg)
    n = _n_frames(pre_ms + t_end + hold_ms, dt_ms)
    t = _times(n, dt_ms)
    rel = t - pre_ms
    size = np.where(rel >= 0, loom_angle_deg(np.maximum(t_end - rel, 0.0), l_over_v_ms, start_deg, end_deg), 0.0)
    size = np.where(rel > t_end, start_deg, size)
    params = dict(azimuth_deg=azimuth_deg, elevation_deg=elevation_deg, l_over_v_ms=l_over_v_ms,
                  start_deg=start_deg, end_deg=end_deg, contrast=contrast, pre_ms=pre_ms, hold_ms=hold_ms,
                  expansion_ms=t_end)
    return Stimulus("receding", dt_ms, _disc_trace(t, azimuth_deg, elevation_deg, size, contrast, rel >= 0),
                    params, eye, res_deg)


def dimming(l_over_v_ms: float = 240.0, start_deg: float = 4.0, end_deg: float = 140.0,
            eyes: str = "both", mode: str = "matched", final_contrast: Optional[float] = None,
            pre_ms: float = 0.0, hold_ms: float = 0.0, dt_ms: float = 1.0, res_deg: float = RES_DEG,
            eye: EyeModel = EyeModel()) -> Stimulus:
    """Uniform darkening of the whole field of `eyes` ('both', 'left' or 'right').

    mode='matched': contrast(t) = -Omega_disc(theta(t)) / Omega_eye for the loom with the same
    l/v, start and end, so the eye loses the luminance the loom would take from it.
    mode='linear': contrast ramps linearly from 0 to final_contrast over the same duration.
    """
    _, t_end = loom_schedule(l_over_v_ms, start_deg, end_deg)
    n = _n_frames(pre_ms + t_end + hold_ms, dt_ms)
    t = _times(n, dt_ms)
    rel = t - pre_ms
    if mode == "matched":
        frac = disc_solid_angle_sr(loom_angle_deg(np.maximum(rel, 0.0), l_over_v_ms, start_deg, end_deg)) / eye.solid_angle_sr()
        c = -np.clip(frac, 0.0, 1.0)
    elif mode == "linear":
        fc = -0.5 if final_contrast is None else float(final_contrast)
        c = fc * np.clip(rel / t_end, 0.0, 1.0)
    else:
        raise ValueError("mode must be 'matched' or 'linear'")
    c = np.where(rel >= 0, c, 0.0)
    ue = EYES if eyes == "both" else (eyes,)
    if not set(ue) <= set(EYES):
        raise ValueError("eyes must be 'both', 'left' or 'right'")
    trace = {"t_ms": t, "contrast": c}
    params = dict(l_over_v_ms=l_over_v_ms, start_deg=start_deg, end_deg=end_deg, eyes=eyes, mode=mode,
                  final_contrast=final_contrast, pre_ms=pre_ms, hold_ms=hold_ms, expansion_ms=t_end)
    return Stimulus("dimming", dt_ms, trace, params, eye, res_deg, uniform_eyes=ue)


def prey_dot(azimuth_deg: float, elevation_deg: float = 0.0, size_deg: float = 3.0,
             speed_deg_s: float = 30.0, sweep_deg: float = 20.0, contrast: float = 1.0,
             duration_ms: float = 3000.0, direction: int = 1, pre_ms: float = 0.0,
             dt_ms: float = 1.0, res_deg: float = RES_DEG, eye: EyeModel = EyeModel()) -> Stimulus:
    """A small spot moving back and forth along the horizon band around `azimuth_deg`.

    Azimuth(t) = azimuth_deg + direction * tri(speed * t), a triangle wave of amplitude sweep/2
    that starts at the centre heading in `direction` (+1 = towards the fish's right).
    """
    if size_deg <= 0:
        raise ValueError("size_deg must be positive")
    n = _n_frames(pre_ms + duration_ms, dt_ms)
    t = _times(n, dt_ms)
    rel = t - pre_ms
    A = sweep_deg / 2.0
    x = speed_deg_s * np.maximum(rel, 0.0) / 1000.0
    if A > 0:
        tri = A - np.abs(np.mod(x + A, 4.0 * A) - 2.0 * A)
    else:
        tri = np.zeros_like(x)
    az = azimuth_deg + (1 if direction >= 0 else -1) * tri
    az = (az + 180.0) % 360.0 - 180.0
    visible = (rel >= 0) & (rel < duration_ms)
    params = dict(azimuth_deg=azimuth_deg, elevation_deg=elevation_deg, size_deg=size_deg,
                  speed_deg_s=speed_deg_s, sweep_deg=sweep_deg, contrast=contrast, duration_ms=duration_ms,
                  direction=direction, pre_ms=pre_ms)
    return Stimulus("prey_dot", dt_ms, _disc_trace(t, az, elevation_deg, np.full(n, float(size_deg)), contrast, visible),
                    params, eye, res_deg)


def blank(duration_ms: float, dt_ms: float = 1.0, res_deg: float = RES_DEG, eye: EyeModel = EyeModel()) -> Stimulus:
    """Nothing but background."""
    n = _n_frames(duration_ms, dt_ms)
    t = _times(n, dt_ms)
    return Stimulus("blank", dt_ms, _disc_trace(t, 0.0, 0.0, np.zeros(n), 0.0, np.zeros(n, bool)),
                    {"duration_ms": duration_ms}, eye, res_deg)
