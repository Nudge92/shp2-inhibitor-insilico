import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

bonds = {"carbox_1":8,"carbox_2":7,"carbox_3":9,"hydrox_1":6,"hydrox_2":8,
         "hydrox_3":8,"sulfon_1":13,"sulfon_2":13,"sulfon_3":16}
data = json.load(open("md_runs_4class/summary.json"))

x, y, labels, colors = [], [], [], []
for r in data:
    n = r["name"]
    if n not in bonds: continue
    occ, end = r["occupancy_pct"], r["r32_dist_end"]
    x.append(bonds[n]); y.append(occ); labels.append(n)
    # 판정 = end 거리 기준 (실제 유지 여부)
    colors.append("#0F6E56" if end <= 5 else "#A32D2D")

fig, ax = plt.subplots(figsize=(7.5, 5.2))
ax.scatter(x, y, c=colors, s=90, zorder=3, edgecolors="white", linewidths=0.5)
for xi, yi, l in zip(x, y, labels):
    ax.annotate(l, (xi, yi), fontsize=8, xytext=(6, 3), textcoords="offset points")

# 추세선
import numpy as np
z = np.polyfit(x, y, 1)
xs = np.linspace(min(x), max(x), 50)
ax.plot(xs, z[0]*xs + z[1], color="#888780", ls="--", lw=1.2, zorder=1)

ax.set_xlabel("number of docked-pose interactions (PLIP)")
ax.set_ylabel("MD occupancy (%)")
ax.set_title("More interactions did NOT mean more stable (r = -0.70)")
# 범례
from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([0],[0],marker='o',color='w',markerfacecolor="#0F6E56",markersize=9,label='retained (end ≤ 5Å)'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor="#A32D2D",markersize=9,label='escaped (end > 5Å)'),
], fontsize=9, loc="upper right")
ax.grid(alpha=0.3)
ax.set_ylim(-5, 100)
plt.tight_layout()
plt.savefig("bonds_vs_stability_v2.png", dpi=150)
print("saved bonds_vs_stability_v2.png")
