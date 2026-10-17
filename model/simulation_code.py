import csv, simpy, numpy as np, random, json, os

QUICK = os.environ.get('ZTA_QUICK') == '1'
SIM_DURATION = 300 if QUICK else 7200
N_RUNS       = 3 if QUICK else 100
SEEDS        = list(range(N_RUNS))
EVSE_COUNTS  = [10, 20] if QUICK else [50, 100, 500, 1000]
N_EVSE_DEF   = 50

# Assumed session-establishment security-processing delays (s)
MTLS_MEAN, MTLS_STD = 0.150, 0.030     
PE_MEAN             = 0.005           
OCSP_MEAN, OCSP_STD = 0.020, 0.005    
OCSP_CAPACITY       = 8              
CAP_GRID            = [2, 4, 8, 16, 32, 64, 128]

MESSAGE_TIMEOUT    = 30               
SESSION_RECONNECT_MIN, SESSION_RECONNECT_MAX = 30, 300
INITIAL_START_SPREAD = 300

# Session security-processing and shared OCSP service model
def run_overhead(n_evse, seed, mtls_mean=None, pe_mean=None, ocsp_mean=None):
    mtls_mean = MTLS_MEAN if mtls_mean is None else mtls_mean
    pe_mean = PE_MEAN if pe_mean is None else pe_mean
    ocsp_mean = OCSP_MEAN if ocsp_mean is None else ocsp_mean
    np.random.seed(seed); random.seed(seed)
    env = simpy.Environment()
    ocsp = simpy.Resource(env, capacity=OCSP_CAPACITY)
    over = []
    def session(env, initial_delay):
        yield env.timeout(initial_delay)
        while True:
            mtls = max(0.01, np.random.normal(mtls_mean, MTLS_STD))
            yield env.timeout(mtls)
            pe = np.random.exponential(pe_mean)
            yield env.timeout(pe)
            t0 = env.now
            with ocsp.request() as req:
                yield req
                svc = max(0.001, np.random.normal(ocsp_mean, OCSP_STD))
                yield env.timeout(svc)
            ocsp_total = env.now - t0
            over.append(mtls + pe + ocsp_total)
            yield env.timeout(random.uniform(SESSION_RECONNECT_MIN, SESSION_RECONNECT_MAX))
    for i in range(n_evse):
        env.process(session(env, random.uniform(0, INITIAL_START_SPREAD)))
    env.run(until=SIM_DURATION)
    return over

def run_burst(n_evse, seed, frac=0.5, cap=OCSP_CAPACITY):
    np.random.seed(seed); random.seed(seed)
    env = simpy.Environment()
    ocsp = simpy.Resource(env, capacity=cap)
    latencies = []
    def burst_session(env):
        mtls = max(0.01, np.random.normal(MTLS_MEAN, MTLS_STD))
        yield env.timeout(mtls)
        yield env.timeout(np.random.exponential(PE_MEAN))
        t0 = env.now
        with ocsp.request() as req:
            yield req
            yield env.timeout(max(0.001, np.random.normal(OCSP_MEAN, OCSP_STD)))
        latencies.append(env.now - t0)        
    n_burst = int(n_evse * frac)
    for i in range(n_burst):
        env.process(burst_session(env))
    env.run()
    return latencies

def stats(a):
    a = np.asarray(a, float)
    return {'mean': float(a.mean()), 'sd': float(a.std(ddof=1)),
            'p50': float(np.percentile(a, 50)), 'p95': float(np.percentile(a, 95)),
            'p99': float(np.percentile(a, 99)), 'max': float(a.max())}
def boot_ci(a, nboot=2000, seed=12345):
    a = np.asarray(a, float); rng = np.random.default_rng(seed)
    m = [rng.choice(a, a.size, replace=True).mean() for _ in range(nboot)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

print(f"=== Monte Carlo ({N_RUNS} runs, seeds 0-{N_RUNS-1}) ===", flush=True)
raw_rows = []

# S1 security-processing delay at default fleet + tail latency
over_all = []; over_by_seed = {}
for s in SEEDS:
    seed_values = np.asarray(run_overhead(N_EVSE_DEF, s), float) * 1000.0
    over_by_seed[s] = seed_values
    over_all.extend(seed_values.tolist())
    raw_rows.append({
        'seed': s, 'scenario': 'S1', 'configuration': f'EVSE={N_EVSE_DEF}',
        'metric': 'mean_session_security_processing_delay',
        'value': float(seed_values.mean()), 'unit': 'ms',
        'n_observations': int(seed_values.size),
    })
over_all = np.asarray(over_all, float)
o = stats(over_all)
over_seed_means = np.asarray([over_by_seed[s].mean() for s in SEEDS], float)
o_ci = boot_ci(over_seed_means)
print(f"[S1] security-processing mean={o['mean']:.2f} ms | p50={o['p50']:.2f} p95={o['p95']:.2f} "
      f"p99={o['p99']:.2f} max={o['max']:.2f} | 95%CI {o_ci[0]:.2f}-{o_ci[1]:.2f}")
print(f"     = {o['mean']/1000/MESSAGE_TIMEOUT*100:.3f}% of OCPP MessageTimeout ({MESSAGE_TIMEOUT}s)")

# Fleet-size sensitivity
evse_rows = {}
for n in EVSE_COUNTS:
    per_run = []
    for s in SEEDS:
        vals = np.asarray(run_overhead(n, s), float) * 1000.0
        per_run.append(float(vals.mean()))
        raw_rows.append({
            'seed': s, 'scenario': 'S1-sensitivity', 'configuration': f'EVSE={n}',
            'metric': 'mean_session_security_processing_delay',
            'value': float(vals.mean()), 'unit': 'ms',
            'n_observations': int(vals.size),
        })
    evse_rows[n] = (float(np.mean(per_run)), float(np.std(per_run, ddof=1)))
    print(f"[S1-sens] EVSE={n:4d}: security-processing {evse_rows[n][0]:.2f} +/- {evse_rows[n][1]:.2f} ms")

# burst stress (OCSP request latency) at each fleet size
burst_rows = {}
for n in EVSE_COUNTS:
    w = []
    for s in SEEDS:
        seed_w = np.asarray(run_burst(n, s, frac=0.5), float) * 1000.0
        w.extend(seed_w.tolist())
        for metric, value in [
            ('mean_ocsp_request_latency', seed_w.mean()),
            ('p95_ocsp_request_latency', np.percentile(seed_w, 95)),
        ]:
            raw_rows.append({
                'seed': s, 'scenario': 'S1-burst', 'configuration': f'EVSE={n};fraction=0.5',
                'metric': metric, 'value': float(value), 'unit': 'ms',
                'n_observations': int(seed_w.size),
            })
    w = np.asarray(w, float)
    burst_rows[n] = (float(w.mean()), float(np.percentile(w, 95)))
    print(f"[burst] EVSE={n:4d}: OCSP request latency mean={burst_rows[n][0]:.2f} ms "
          f"p95={burst_rows[n][1]:.2f} ms")


# Shared OCSP responder-capacity sweep
cap_rows, cap_summary = [], {}
for cap in CAP_GRID:
    for n in EVSE_COUNTS:
        w = []
        for s in SEEDS:
            seed_w = np.asarray(run_burst(n, s, frac=0.5, cap=cap), float) * 1000.0
            w.extend(seed_w.tolist())
            cap_rows.append({
                'seed': s, 'n_evse': n, 'ocsp_capacity': cap,
                'metric': 'mean_ocsp_request_latency',
                'value': float(seed_w.mean()), 'unit': 'ms',
                'n_observations': int(seed_w.size),
            })
        w = np.asarray(w, float)
        cap_summary[f"EVSE={n};cap={cap}"] = {
            'mean_ms': float(w.mean()),
            'p95_ms': float(np.percentile(w, 95)),
            'queueing_share_pct': float(100.0 * (w.mean() - OCSP_MEAN * 1000.0) / w.mean()),
            'concurrent_requests_per_server': round(int(n * 0.5) / cap, 1),
        }
    if cap == OCSP_CAPACITY:
        for n in EVSE_COUNTS:
            assert abs(cap_summary[f"EVSE={n};cap={cap}"]['mean_ms'] - burst_rows[n][0]) < 1e-9, n

print(f"[cap-sweep] {len(CAP_GRID)} capacities x {len(EVSE_COUNTS)} fleet sizes")
n_max = EVSE_COUNTS[-1]
for cap in CAP_GRID:
    r = cap_summary[f"EVSE={n_max};cap={cap}"]
    print(f"    cap={cap:4d}: EVSE={n_max} mean={r['mean_ms']:8.2f} ms  "
          f"p95={r['p95_ms']:9.2f} ms  "
          f"queueing={r['queueing_share_pct']:5.1f}%  ratio=1:{r['concurrent_requests_per_server']:.1f}")

# Sensitivity to assumed processing means
SENS_FACTORS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
SENS_INPUTS = [
    ("mTLS handshake mean", "mtls_mean", MTLS_MEAN),
    ("OCSP check mean",     "ocsp_mean", OCSP_MEAN),
    ("PE decision mean",    "pe_mean",   PE_MEAN),
]
input_sensitivity = []
for label, kwarg, base_value in SENS_INPUTS:
    for factor in SENS_FACTORS:
        value = base_value * factor
        pooled, seed_means = [], []
        for sd in SEEDS:
            vals = np.asarray(
                run_overhead(N_EVSE_DEF, sd, **{kwarg: value}), float
            ) * 1000.0
            pooled.extend(vals.tolist())
            seed_means.append(float(vals.mean()))
        mean_ms = float(np.asarray(pooled, float).mean())
        lo, hi = boot_ci(np.asarray(seed_means, float))
        input_sensitivity.append({
            "input": label,
            "factor": factor,
            "value_ms": round(value * 1000.0, 2),
            "s1_mean_ms": round(mean_ms, 2),
            "ci_low_ms": round(lo, 2),
            "ci_high_ms": round(hi, 2),
            "pct_message_timeout": round(mean_ms / 1000.0 / MESSAGE_TIMEOUT * 100.0, 3),
        })
        if factor == 1.0:
            assert abs(mean_ms - o['mean']) < 1e-9, (label, mean_ms, o['mean'])
worst = max(input_sensitivity, key=lambda r: r["pct_message_timeout"])
print(f"[sens] {len(input_sensitivity)} configs | worst {worst['input']} "
      f"{worst['factor']:.1f}x = {worst['s1_mean_ms']:.2f} ms ({worst['pct_message_timeout']:.3f}%)")

# Saved numerical outputs
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("ZTA_RESULTS_DIR") or os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
summary = {
    "n_runs": N_RUNS,
    "interpretation": {
        "S1": "Modeled session-establishment security-processing delay under stated input distributions and distributed first-session starts; not incremental overhead versus an OCPP baseline.",
        "S1_initialization": "Each EVSE first session is distributed Uniform(0, 300 s); simultaneous re-establishment is evaluated only in S1-burst.",
        "S1_burst": "OCSP request latency from request submission through service completion (service plus queueing) in the separate 50% simultaneous re-establishment stress test.",
    },
    "S1_security_processing_ms": o, "S1_seed_mean_ci": o_ci,
    "S1_ci_resampling_unit": "seed-level mean",
    "S1_pct_message_timeout_context": o['mean']/1000/MESSAGE_TIMEOUT*100,
    "EVSE_sensitivity": {str(k): v for k, v in evse_rows.items()},
    "burst_ocsp_request_latency": {str(k): v for k, v in burst_rows.items()},
    "burst_ocsp_capacity_sweep": cap_summary,
    "S1_input_sensitivity": input_sensitivity,
}
with open(os.path.join(RES, "simulation_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
with open(os.path.join(RES, "per_seed_results.csv"), "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=['seed', 'scenario', 'configuration', 'metric', 'value', 'unit', 'n_observations'],
        lineterminator='\n',
    )
    writer.writeheader(); writer.writerows(raw_rows)
with open(os.path.join(RES, "ocsp_capacity_sweep.csv"), "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=['seed', 'n_evse', 'ocsp_capacity', 'metric', 'value', 'unit', 'n_observations'],
        lineterminator='\n',
    )
    writer.writeheader(); writer.writerows(cap_rows)
with open(os.path.join(RES, "input_sensitivity.csv"), "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=['input', 'factor', 'value_ms', 's1_mean_ms', 'ci_low_ms',
                    'ci_high_ms', 'pct_message_timeout'],
        lineterminator='\n',
    )
    writer.writeheader(); writer.writerows(input_sensitivity)
print("\nSaved:", os.path.join(RES, "simulation_summary.json"))
print("Saved:", os.path.join(RES, "per_seed_results.csv"))
print("Saved:", os.path.join(RES, "ocsp_capacity_sweep.csv"))
print("Saved:", os.path.join(RES, "input_sensitivity.csv"))

if os.environ.get("ZTA_NO_FIGURES") != "1":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DPI = 600
    COL_WIDTH = 3.35
    H_ONE_PANEL = 2.50
    H_TWO_PANEL = 4.20

    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 3.5,
        "savefig.dpi": FIG_DPI, "figure.dpi": FIG_DPI,
    })

    FIGDIR = RES

    def export(name):
        plt.tight_layout(pad=0.4)
        plt.savefig(os.path.join(FIGDIR, name + ".png"), dpi=FIG_DPI)
        plt.close()

    C_ZTA, C_BASE = "#2b7bba", "#e8734a"
    BLUES = ["#c9d9e8", "#9cc0dc", "#6ba3d0", "#3d7cb8", "#1f4e79"]
    ns = EVSE_COUNTS

    fig, ax = plt.subplots(2, 1, figsize=(COL_WIDTH, H_TWO_PANEL))
    ax[0].hist(over_all, bins=60, color=C_ZTA, alpha=0.8)
    for pc, lab in [(o['p50'], 'p50'), (o['p95'], 'p95'), (o['p99'], 'p99')]:
        ax[0].axvline(pc, color='k', ls='--', lw=0.7)
        ax[0].text(pc, ax[0].get_ylim()[1]*0.88, f" {lab}\n {pc:.0f}", fontsize=6)
    ax[0].set_xlabel("Modeled delay (ms)\n(a)"); ax[0].set_ylabel("Count")
    labs = ['mean', 'p50', 'p95', 'p99', 'max']
    vals = [o['mean'], o['p50'], o['p95'], o['p99'], o['max']]
    b = ax[1].bar(labs, vals, color=BLUES)
    ax[1].bar_label(b, fmt='%.0f', fontsize=6)
    ax[1].set_ylabel("Delay (ms)"); ax[1].set_xlabel("(b)")
    export("overhead")

    fig, ax = plt.subplots(figsize=(COL_WIDTH, H_ONE_PANEL))
    ms = [evse_rows[n][0] for n in ns]; es = [evse_rows[n][1] for n in ns]
    ax.errorbar([str(n) for n in ns], ms, yerr=es, marker='o', color=C_ZTA, capsize=3)
    ax.set_xlabel("Number of EVSEs"); ax.set_ylabel("Modeled delay (ms)")
    for x, y in zip([str(n) for n in ns], ms):
        ax.text(x, y+0.08, f"{y:.1f}", ha='center', fontsize=6)
    export("evse_sensitivity")

    fig, ax = plt.subplots(figsize=(COL_WIDTH, H_ONE_PANEL))
    bm = [burst_rows[n][0] for n in ns]; bp = [burst_rows[n][1] for n in ns]
    ax.plot([str(n) for n in ns], bm, marker='o', color=C_ZTA, label='mean request latency')
    ax.plot([str(n) for n in ns], bp, marker='s', color=C_BASE, ls='--', label='p95 request latency')
    ax.set_xlabel("Number of EVSEs (50% simultaneous re-establishment)")
    ax.set_ylabel("OCSP request latency (ms)"); ax.legend()
    for x0, y in zip([str(n) for n in ns], bm):
        ax.text(x0, y+15, f"{y:.0f}", ha='center', fontsize=6)
    export("burst")

    print(f"Saved figures at {FIG_DPI} dpi")