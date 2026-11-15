"""Red heat table (team style): SARIMAX RMSE change (%) when inputs are added, 2023 validation.
Rows = the stacking comparisons fixed before scoring (run_score_m.py): M2 vs M1, M3 vs M2, M4 vs M3.
Cell = 100 * (RMSE_new - RMSE_ref) / RMSE_ref (negative = improvement). * = team improvement rule
(bootstrap 95% CI upper < 0 and Holm p < 0.05) from results/score_M/tests.csv.
Writes figures_en/fig_A_M_added_inputs.png and values_fig_A_M_added_inputs.csv.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "Arial", "font.size": 11})
H = [3, 6, 12, 24]
m = pd.read_csv(HERE / "results" / "score_M" / "metrics.csv")
t = pd.read_csv(HERE / "results" / "score_M" / "tests.csv")
r = lambda k, h: float(m[(m.model == f"A_SARIMAX__{k}") & (m.horizon == h)].RMSE.iloc[0])
rows = [("Nearby China\nM2 vs M1", "M2", "M1"), ("All China\nM3 vs M2", "M3", "M2"),
        ("Japan + Mongolia\nM4 vs M3", "M4", "M3")]
val, star, out_rows = np.zeros((len(rows), len(H))), np.zeros((len(rows), len(H)), bool), []
for i, (lab, new, ref) in enumerate(rows):
    for j, h in enumerate(H):
        val[i, j] = 100 * (r(new, h) - r(ref, h)) / r(ref, h)
        tt = t[(t.run_id.str.startswith(f"A_SARIMAX__{new}|")) & (t.versus_run_id.str.startswith(f"A_SARIMAX__{ref}|"))
               & (t.horizon == h)]
        assert len(tt) == 1
        star[i, j] = bool(tt.improved.iloc[0])
        out_rows.append({"comparison": f"{new} vs {ref}", "horizon": h, "RMSE_new": r(new, h), "RMSE_ref": r(ref, h),
                         "RMSE_change_pct": val[i, j], "improved": star[i, j], "p_holm": float(tt.p_holm.iloc[0])})

gain = np.clip(-val, 0, None)                                     # colour = size of the RMSE reduction
vmax = np.ceil(gain.max())
fig, ax = plt.subplots(figsize=(9.0, 4.2), dpi=300)
im = ax.imshow(gain, cmap="Reds", vmin=0, vmax=vmax, aspect="auto")
for i in range(len(rows)):
    for j in range(len(H)):
        txt = f"{val[i, j]:+.2f}" + ("*" if star[i, j] else "")
        ax.text(j, i, txt, ha="center", va="center", fontsize=11,
                color="white" if gain[i, j] > 0.6 * vmax else "#222222")
ax.set_xticks(range(len(H)), [f"{h}h" for h in H])
ax.xaxis.tick_top()
ax.set_yticks(range(len(rows)), [lab for lab, *_ in rows], fontsize=10)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xticks(np.arange(-0.5, len(H)), minor=True)
ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
ax.grid(which="minor", color="white", lw=2)
ax.tick_params(which="both", length=0)
ax.set_title("Added inputs: RMSE change (%)", fontsize=12, fontweight="bold", pad=28)
cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.06, pad=0.08)
cb.set_label("Deeper red = larger RMSE reduction (%)", fontsize=9)
cb.ax.tick_params(labelsize=8)
fig.text(0.01, 0.01, "A SARIMAX, 2023 validation. RMSE change after adding the named inputs; "
         "negative = improvement. * significant (95% CI upper < 0, Holm p < 0.05).", fontsize=8, color="#444444")
fig.tight_layout(rect=(0, 0.04, 1, 1))
out = HERE / "figures_en"
fig.savefig(out / "fig_A_M_added_inputs.png", facecolor="white")
pd.DataFrame(out_rows).to_csv(out / "values_fig_A_M_added_inputs.csv", index=False)
print(pd.DataFrame(out_rows).round(3).to_string(index=False))
