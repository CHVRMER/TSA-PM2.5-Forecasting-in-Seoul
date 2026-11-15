"""Team-format figure: SARIMAX RMSE across M1-M4 input conditions (2023 validation).
Style copied from the team DCRNN figure (solid lines, hollow markers, same shapes/colors, numeric x at 3/6/12/24,
legend below). Points stay at their true x and y. Only the y axis differs: one stretched band per forecast horizon
(the RMSE range of that horizon, fine ticks), with thin shaded bands marking the compressed ranges between them.
Values = results/score_M/metrics.csv. Writes figures_en/fig_A_M1_M4_rmse.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "Arial", "font.size": 11, "axes.unicode_minus": False})
H = [3, 6, 12, 24]
m = pd.read_csv(HERE / "results" / "score_M" / "metrics.csv")
series = [("M1", "M1: Domestic", "#c0392b", "o"), ("M2", "M2: +Nearby China", "#e0a030", "^"),
          ("M3", "M3: +All China", "#3a9a3a", "s"), ("M4", "M4: +Japan / Mongolia", "#3f6fc4", "D")]
vals = {k: m[m.model == f"A_SARIMAX__{k}"].set_index("horizon").loc[H, "RMSE"] for k, *_ in series}

# --- piecewise y scale: one band per horizon; band height so the closest two points sit >= MIN_SEP markers apart ---
PAD, MIN_SEP_U, GAP_U, UNIT_IN = 0.01, 0.7, 0.8, 0.07          # 1 display unit = 0.07 in (about one 5-pt marker)
bands = []
for h in H:
    c = np.sort([vals[k][h] for k in vals])
    lo, hi = c[0] - PAD, c[-1] + PAD
    bands.append((lo, hi, max(9.0, (hi - lo) / np.diff(c).min() * MIN_SEP_U)))
data_pts, disp_pts, d = [], [], 0.0
for i, (lo, hi, u) in enumerate(bands):
    if i:
        d += GAP_U
    data_pts += [lo, hi]
    disp_pts += [d, d + u]
    d += u
data_pts, disp_pts = np.array(data_pts), np.array(disp_pts)
fwd = lambda y: np.interp(y, data_pts, disp_pts)
inv = lambda z: np.interp(z, disp_pts, data_pts)

fig, ax = plt.subplots(figsize=(10.5, 2.4 + d * UNIT_IN), dpi=300)
ax.set_yscale("function", functions=(fwd, inv))
rows = []
for k, lab, col, mk in series:
    s = vals[k]
    rows += [{"condition": k, "horizon": h, "RMSE": y} for h, y in s.items()]
    ax.plot(H, s.values, color=col, lw=1.4, marker=mk, ms=5, mfc="white", mec=col, mew=1.0, label=lab, zorder=3)
ax.set_ylim(data_pts[0], data_pts[-1])

ticks = []
for lo, hi, u in bands:
    n_target = max(4, int(u / 2.5))                            # roughly one tick per 2.5 units of band height
    step = next(s for s in (0.01, 0.02, 0.05, 0.1, 0.2) if (hi - lo) / s <= n_target)
    ticks += list(np.round(np.arange(np.ceil(lo / step) * step, hi + 1e-9, step), 4))
ax.set_yticks(ticks)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:.2f}"))
ax.tick_params(axis="y", labelsize=9)
ax.grid(axis="y", color="#e6e6e6", lw=0.8, zorder=0)
for i in range(1, len(bands)):                                  # compressed ranges between horizons
    ax.axhspan(bands[i - 1][1], bands[i][0], color="#e9e9e9", lw=0, zorder=0)

ax.set_xticks(H)
ax.set_xlim(1.8, 25.2)
ax.set_xlabel("Forecast horizon (hours)")
ax.set_ylabel("RMSE (µg/m³)")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, columnspacing=3, fontsize=9.5)
fig.text(0.01, 0.005, "A SARIMAX, 2023 validation. RMSE (µg/m³) across M1–M4 input conditions. "
         "Shaded bands mark compressed RMSE intervals.", ha="left", fontsize=9, color="#444444")
fig.tight_layout(rect=(0, 0.02, 1, 1))
out = HERE / "figures_en"
fig.savefig(out / "fig_A_M1_M4_rmse.png", facecolor="white")
pd.DataFrame(rows).to_csv(out / "values_fig_A_M1_M4.csv", index=False)
print("bands", [(round(a, 3), round(b, 3), round(u, 1)) for a, b, u in bands], "height_in", round(2.4 + d * UNIT_IN, 1))
print(pd.DataFrame(rows).pivot(index="condition", columns="horizon", values="RMSE").round(3).to_string())
