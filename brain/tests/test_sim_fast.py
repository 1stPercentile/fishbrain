"""ActiveSimulator must be bit-identical to Simulator: spikes, stats, and every state digest."""
import numpy as np
import pytest

from fishbrain import sim
from fishbrain.sim import LIFParams, Network, Simulator, spike_hash
from fishbrain.sim_fast import ActiveSimulator

SEEDS = ["fast-0", "fast-1", "0xabc"]


def _graph(n=3000, syn=60_000, seed="fast-graph", frac_type1=0.3):
    return sim.synthetic_graph(n, syn, seed=seed, frac_type1=frac_type1)


def _drive(s, n, rng_name):
    # a mix of every input kind: scalar Poisson, per-neuron Poisson frames, and currents
    s.add_poisson(np.arange(0, n, 7), 40.0, start_ms=0, stop_ms=120, name=f"{rng_name}-a")
    rates = np.linspace(0, 200, 5 * 50).reshape(5, 50)
    s.add_poisson(np.arange(100, 150), rates, frame_ms=20.0, name=f"{rng_name}-b")
    s.add_current(np.arange(200, 230), 9.0, start_ms=30, stop_ms=90, name=f"{rng_name}-c")


def _pair(net, seed, params=LIFParams()):
    ref, fast = Simulator(net, seed=seed, params=params), ActiveSimulator(net, seed=seed, params=params)
    for s in (ref, fast):
        _drive(s, net.n, "in")
    return ref, fast


@pytest.mark.parametrize("seed", SEEDS)
def test_identical_spikes_state_and_stats(seed):
    net = _graph()
    ref, fast = _pair(net, seed)
    for chunk in (37.3, 50.0, 112.7):              # several run() calls, uneven lengths
        r1, r2 = ref.run(chunk), fast.run(chunk)
        assert spike_hash(r1) == spike_hash(r2)
        assert ref.state_digest() == fast.state_digest()
    assert ref.stats == fast.stats
    assert ref.stats["spikes"] > 100, "positive control: the network must actually be active"


def test_identical_after_snapshot_restore_and_manual_kick():
    net = _graph(seed="fast-graph-2")
    ref, fast = _pair(net, "fast-restore")
    ref.run(60.0), fast.run(60.0)
    snap_r, snap_f = ref.snapshot(), fast.snapshot()
    for s in (ref, fast):
        s.u[5:9] += np.float32(3.5)                  # state edited between runs
    assert spike_hash(ref.run(40.0)) == spike_hash(fast.run(40.0))
    assert ref.state_digest() == fast.state_digest()
    ref.restore(snap_r), fast.restore(snap_f)
    assert spike_hash(ref.run(80.0)) == spike_hash(fast.run(80.0))
    assert ref.state_digest() == fast.state_digest()


def test_identical_with_lesion_and_dt_0_2():
    net = _graph(seed="fast-graph-3").lesion_indices(np.arange(0, 3000, 11))
    ref, fast = _pair(net, "fast-lesion", LIFParams(dt_ms=0.2))
    assert spike_hash(ref.run(200.0)) == spike_hash(fast.run(200.0))
    assert ref.state_digest() == fast.state_digest()


def test_mutation_is_caught():
    """Positive control on the comparison itself: perturbing the fast kernel's state must show."""
    net = _graph(seed="fast-graph-4")
    ref, fast = _pair(net, "fast-mut")
    ref.run(50.0), fast.run(50.0)
    fast.h[int(np.flatnonzero(fast.h)[0])] += np.float32(1e-3)
    assert ref.state_digest() != fast.state_digest()


def test_sparse_activity_is_faster():
    """On a large, mostly silent network, the active kernel should beat the full update."""
    import time
    n = 400_000
    net = sim.synthetic_graph(n, 400_000, seed="fast-sparse")
    stats = {}
    for cls in (Simulator, ActiveSimulator):
        s = cls(net, seed="fast-sparse")
        s.add_poisson(np.arange(0, 2000), 80.0, name="drive")
        t = time.process_time()
        r = s.run(100.0)
        stats[cls.__name__] = (time.process_time() - t, spike_hash(r), s.state_digest())
    assert stats["Simulator"][1:] == stats["ActiveSimulator"][1:]
    assert stats["ActiveSimulator"][0] < stats["Simulator"][0], stats


def test_make_simulator_picks_by_size():
    from fishbrain.sim_fast import make_simulator, ACTIVE_MIN_NEURONS
    small = _graph(n=1000, syn=5000, seed="pick-small")
    assert type(make_simulator(small, "s")).__name__ == "Simulator"
    big = sim.synthetic_graph(ACTIVE_MIN_NEURONS, 1000, seed="pick-big")
    assert type(make_simulator(big, "s")).__name__ == "ActiveSimulator"


def test_threshold_below_rest_is_refused_not_silently_wrong():
    """Found by the JS-kernel exactness review: with v_thresh < v_rest an untouched neuron can fire,
    so gating would diverge. ActiveSimulator must refuse, and make_simulator must fall back."""
    from fishbrain.sim_fast import make_simulator, ACTIVE_MIN_NEURONS
    low = LIFParams(v_rest_mV=-40.0, v_thresh_mV=-45.0, v_reset_mV=-52.0)
    net = _graph(n=500, syn=2000, seed="theta-low")
    with pytest.raises(ValueError):
        ActiveSimulator(net, seed="t", params=low)
    big = sim.synthetic_graph(ACTIVE_MIN_NEURONS, 1000, seed="theta-low-big")
    assert type(make_simulator(big, "t", params=low)).__name__ == "Simulator"
