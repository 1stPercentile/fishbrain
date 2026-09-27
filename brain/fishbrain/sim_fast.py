"""Activity-gated simulator: the same arithmetic as sim.Simulator, applied only where it can change.

A neuron whose u and h are both +0.0 stays +0.0 under the integrate step (0*a + 0 = 0, 0*b = 0) and
cannot cross threshold, so updating it is wasted work. ActiveSimulator keeps a sorted index of the
neurons that can change: any with a nonzero (or -0.0) u or h, anything touched this step by a
synapse, a Poisson kick or a current. It applies the identical numpy operations to that subset, in
the same order. Spikes, rasters, stats and state digests are bit-identical to Simulator. That is
tested (tests/test_sim_fast.py), including against the golden hashes pinned for the reference
simulator.

On Fish1's brain of record most of the 13.46M segments never receive input, so the per-step cost
falls from O(all neurons) to O(active neurons).
"""
import numpy as np

from .sim import Simulator

_ZERO_BITS = np.uint32(0)


class ActiveSimulator(Simulator):
    """Drop-in replacement for Simulator.run(); everything else (inputs, snapshot/restore,
    digests) is inherited unchanged."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        # Gating assumes a neuron at u == h == +0.0 cannot fire, i.e. threshold above rest (theta > 0).
        # With theta <= 0 an untouched neuron would cross threshold, so gating would change results.
        if not float(self._theta) > 0.0:
            raise ValueError("ActiveSimulator requires v_thresh > v_rest (theta > 0); use Simulator")
        self._active = np.zeros(0, np.int64)      # unique, unordered (only elementwise ops use it)
        self._in = np.zeros(self.net.n, bool)     # membership mask for _active

    # the set is derived from state, so restore() stays correct: rebuild it from u and h
    def restore(self, snap: dict):
        super().restore(snap)
        self._in[:] = False
        self._active = np.zeros(0, np.int64)
        self._touch(self._nonzero_state())

    def _nonzero_state(self) -> np.ndarray:
        # +0.0 bits only; -0.0 stays active because the reference update turns it into +0.0
        live = (self.u.view(np.uint32) != _ZERO_BITS) | (self.h.view(np.uint32) != _ZERO_BITS)
        return np.flatnonzero(live).astype(np.int64)

    def _touch(self, idx):
        if len(idx):
            idx = np.asarray(idx, np.int64)
            new = idx[~self._in[idx]]
            if len(new):
                new = np.unique(new)
                self._in[new] = True
                self._active = np.concatenate([self._active, new])

    def _deliver(self, src: np.ndarray):
        indptr = self._indptr
        starts = indptr[src]
        cnt = indptr[src + 1] - starts
        if int(cnt.sum()):
            if len(src) == 1:
                s, e = indptr[src[0]], indptr[src[0] + 1]
                self._touch(self._indices[s:e])
            else:
                tot = int(cnt.sum())
                csum = np.cumsum(cnt)
                g = np.arange(tot, dtype=np.int64) + np.repeat(starts - (csum - cnt), cnt)
                self._touch(self._indices[g])
        super()._deliver(src)

    def run(self, duration_ms=None, n_steps=None):
        if n_steps is None:
            n_steps = self.p.steps(duration_ms)
        # O(n) once per call, so state edited between runs (tests, restore, manual kicks) is covered
        self._touch(self._nonzero_state())
        u, h, last = self.u, self.h, self.last_spike
        a, b, theta, u_reset = self._a, self._b, self._theta, self._u_reset
        n_ref, n_delay, slots, ref = self.n_ref, self.n_delay, self._slots, self._ref
        dead, any_dead = self._dead, self._any_dead
        nslot = n_delay + 1
        w0_steps = len(self._rec_steps)
        first = self.step
        currents = [s for s in self.sources if s.kind == "current"]
        poissons = [s for s in self.sources if s.kind == "poisson"]
        for _ in range(n_steps):
            k = self.step
            act = self._active
            # 1. integrate the active subset exactly as Simulator integrates everything
            if ref:
                R = np.concatenate(ref)
                uR, hR = u[R], h[R]
            else:
                R = None
            if len(act):
                ua = u[act]
                np.multiply(ua, a, out=ua)
                np.add(ua, h[act], out=ua)
                u[act] = ua
                ha = h[act]
                np.multiply(ha, b, out=ha)
                h[act] = ha
            for s in currents:
                if s.active(k):
                    s.add(k, u)
                    self._touch(s.targets)
            if R is not None and len(R):
                u[R] = uR
                h[R] = hR
            # 2. threshold: an inactive neuron has u == +0.0 < theta, so only the active can fire.
            # Sorted, so the order matches np.flatnonzero in Simulator (it sets delivery order).
            act = self._active
            fired = np.sort(act[u[act] > theta]) if len(act) else np.zeros(0, np.int64)
            if len(fired):
                fired = fired[last[fired] <= k - n_ref]
                if any_dead and len(fired):
                    fired = fired[~dead[fired]]
            # 3. synapses, then Poisson kicks
            slots[k % nslot] = fired
            arriving = slots[(k - n_delay) % nslot]
            if len(arriving):
                self._deliver(arriving)
            for s in poissons:
                if s.active(k):
                    hit = s.hits(k)
                    if hit is not None:
                        u[hit] += s.weight
                        self.stats["poisson_events"] += len(hit)
                        self._touch(hit)
            # 4. reset
            if len(fired):
                u[fired] = u_reset
                h[fired] = 0.0
                last[fired] = k
                self._rec_steps.append(k)
                self._rec_idx.append(fired)
                self.stats["spikes"] += len(fired)
            if ref is not None:
                ref.append(fired)
            # 5. drop neurons that are exactly +0.0 in both u and h (they cannot change on their own)
            act = self._active
            if len(act):
                keep = (u[act].view(np.uint32) != _ZERO_BITS) | (h[act].view(np.uint32) != _ZERO_BITS)
                if not keep.all():
                    self._in[act[~keep]] = False
                    self._active = act[keep]
            self.step += 1
        steps = self._rec_steps[w0_steps:]
        idx = self._rec_idx[w0_steps:]
        return self._raster(steps, idx, first, n_steps)


# Measured 2026-09-26 on the Mac (loaded, one core), loom window, bit-identical outputs:
#   full brain 13,458,709 segments: Simulator RTF 0.011, ActiveSimulator 0.273 (24.8x faster)
#   gate network 8,654 neurons:     Simulator RTF 2.73,  ActiveSimulator 2.05 (0.7x)
ACTIVE_MIN_NEURONS = 200_000


def make_simulator(network, seed, params=None, **kw):
    """The faster of the two for this network size. Both give identical spikes and state."""
    from .sim import LIFParams
    params = LIFParams() if params is None else params
    theta_ok = float(params.derived()["theta"]) > 0.0
    cls = ActiveSimulator if (network.n >= ACTIVE_MIN_NEURONS and theta_ok) else Simulator
    return cls(network, seed=seed, params=params, **kw)
