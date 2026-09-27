// Run one exported case (js/export_case.py) through the JS kernel and print what it computed.
//
//   node js/run_case.mjs CASE.json [--gated] [--use-exported-weights] [--kernel PATH] [--repeat N]
//
// Prints one JSON object: per-run spike_hash / state_digest / stats, the whole raster's hash, the
// derived constants' bits, params and network digests, the weight hash, timing, and "mismatches":
// every field that differs from the case's "expected" block (the Python engine's answers).
// Exit code: 0 when nothing differs, 1 when something does, 2 on an error.
// --kernel loads a different copy of fishbrain-sim.mjs (the parity test's mutation controls).
// --use-exported-weights runs on the float32 weight bits stored in the case instead of deriving them.

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const args = process.argv.slice(2);
const flag = (f) => args.includes(f);
const opt = (f, d) => { const i = args.indexOf(f); return i >= 0 ? args[i + 1] : d; };
const casePath = args.find((a, i) => !a.startsWith("--") && !["--kernel", "--repeat"].includes(args[i - 1]));
if (!casePath) { console.error("usage: node run_case.mjs CASE.json [--gated] [--use-exported-weights] [--kernel PATH]"); process.exit(2); }

const here = path.dirname(fileURLToPath(import.meta.url));
const kernelPath = path.resolve(opt("--kernel", path.join(here, "fishbrain-sim.mjs")));
const F = await import(pathToFileURL(kernelPath).href);

try {
  const c = JSON.parse(fs.readFileSync(casePath, "utf8"));
  const E = c.expected || {};
  const gated = flag("--gated");
  const useExported = flag("--use-exported-weights");
  const repeat = Number(opt("--repeat", "1"));

  let best = null, out = null;
  for (let rep = 0; rep < repeat; rep++) {
    const t0 = performance.now();
    const { sim, net, params } = F.simulatorFromCase(c, { gated, useExportedWeights: useExported });
    const tBuild = performance.now() - t0;
    const runs = [];
    let simSteps = 0;
    const t1 = performance.now();
    for (const r of c.runs) {
      const ras = r.n_steps !== undefined && r.n_steps !== null ? sim.run(null, r.n_steps) : sim.run(r.duration_ms);
      simSteps += ras.nSteps;
      runs.push({ spike_hash: F.spikeHash(ras), state_digest: sim.stateDigest(), n_spikes: ras.neurons.length,
        n_steps: ras.nSteps, stats: { ...sim.stats } });
    }
    const tRun = performance.now() - t1;       // includes the per-run digests, as a Verify would
    if (best === null || tRun < best) best = tRun;
    if (rep === 0) {
      const d = sim.d;
      const wb = new Uint8Array(sim.w.buffer, sim.w.byteOffset, sim.w.byteLength);
      let weightsMatch = null;
      if (c.weights_f32_bits) {
        const wbits = new Uint32Array(sim.w.buffer, sim.w.byteOffset, sim.w.length);
        weightsMatch = wbits.length === c.weights_f32_bits.length && wbits.every((x, i) => x === c.weights_f32_bits[i]);
      }
      const poissonSeeds = {};
      for (const s of sim.sources) if (s.kind === "poisson") poissonSeeds[s.name] = F.deriveSeed(c.seed, "poisson", s.name).toString();
      const paramsRepr = {};
      for (const k of F.PARAM_FIELDS) paramsRepr[k] = F.pyFloatRepr(params[k]);
      out = {
        case: c.name, kernel: path.basename(kernelPath), gated: sim.gated, exported_weights: useExported,
        runs,
        raster_hash: F.spikeHash(sim.raster()),
        n_spikes: sim.raster().neurons.length,
        params_repr: paramsRepr,
        params_digest: params.digest(),
        network_digest: net.digest(),
        weights_sha256: F.sha256Hex(wb),
        weights_match_exported: weightsMatch,
        derived: { a: F.f32Hex(d.a), b: F.f32Hex(d.b), c: F.f32Hex(d.c), k_i: F.f32Hex(d.k_i), theta: F.f32Hex(d.theta),
          u_reset: F.f32Hex(d.u_reset), c64: F.f64Hex(d.c64), k_i64: F.f64Hex(d.k_i64), n_ref: d.n_ref, n_delay: d.n_delay },
        poisson_seeds: poissonSeeds,
        timing: { build_ms: +tBuild.toFixed(2), sim_steps: simSteps, sim_ms: +(simSteps * params.dt_ms).toFixed(6) },
      };
    }
  }
  out.timing.run_wall_ms_best = +best.toFixed(2);
  out.timing.rtf_best = +(out.timing.sim_ms / best).toFixed(4);     // simulated ms per wall ms
  out.timing.repeats = repeat;
  out.timing.loadavg = os.loadavg().map((x) => +x.toFixed(2));
  out.timing.node = process.version;

  // compare with the Python answers
  const mismatches = [];
  const same = (k, a, b) => { if (JSON.stringify(a) !== JSON.stringify(b)) mismatches.push(k); };
  if (E.runs) {
    same("n_runs", out.runs.length, E.runs.length);
    E.runs.forEach((er, i) => {
      const jr = out.runs[i] || {};
      for (const k of ["spike_hash", "state_digest", "n_spikes", "n_steps", "stats"]) same(`runs[${i}].${k}`, jr[k], er[k]);
    });
  }
  for (const k of ["raster_hash", "n_spikes", "params_repr", "params_digest", "network_digest", "derived", "poisson_seeds"]) {
    if (k in E) same(k, out[k], E[k]);
  }
  if ("weights_sha256" in E && !useExported) same("weights_sha256", out.weights_sha256, E.weights_sha256);
  out.mismatches = mismatches;
  console.log(JSON.stringify(out));
  process.exit(mismatches.length ? 1 : 0);
} catch (err) {
  console.log(JSON.stringify({ error: String(err && err.stack || err) }));
  process.exit(2);
}
