"""Monte Carlo for the multinomial range, % of mean bucket size."""
import numpy as np, json
rng = np.random.default_rng(20260429)
N_VALUES = [50, 100, 200, 500, 1000, 2000]
K_VALUES = [7, 12, 24]
TRIALS = 20000
results = []
print(f"{'N':>6} {'K':>4} {'mu':>8} {'mean%':>8} {'p95%':>8} {'P(>30%)':>9} {'approx%':>9}")
for K in K_VALUES:
    for N in N_VALUES:
        mu = N / K
        draws = rng.multinomial(N, [1.0/K]*K, size=TRIALS)
        ranges = draws.max(axis=1) - draws.min(axis=1)
        pct = 100.0 * ranges / mu
        m, p95 = float(pct.mean()), float(np.percentile(pct, 95))
        p30 = float((pct > 30).mean())
        approx = 100.0 * 2 * np.sqrt(2 * mu * np.log(K)) / mu
        results.append({"N":N,"K":K,"mu":mu,"mean_pct":m,"p95_pct":p95,"p_over_30":p30,"approx_pct":approx})
        print(f"{N:>6} {K:>4} {mu:>8.2f} {m:>8.1f} {p95:>8.1f} {p30:>9.3f} {approx:>9.1f}")
with open("/home/claude/mc_results.json","w") as f:
    json.dump(results, f, indent=2)
