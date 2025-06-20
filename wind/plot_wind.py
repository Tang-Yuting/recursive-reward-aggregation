import numpy as np
import matplotlib.pyplot as plt
from wind_env import wind_fn, reward_fn
from pathlib import Path

fig_path = Path("figures")
fig_path.mkdir(exist_ok=True)

w = 2.0
x, y = np.meshgrid(np.linspace(-w, w, 101), np.linspace(-w, w, 101))
u, v = wind_fn(x, y)
r = reward_fn(x, y)

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-w, w)
ax.set_ylim(-w, w)
# ax.set_xticks([])
# ax.set_yticks([])
ax.set_xlabel("x")
ax.xaxis.label.set_alpha(0.0)
ax.set_aspect("equal")


ax.contourf(x, y, r, levels=100, cmap="RdYlGn")
ax.streamplot(x, y, u, v, density=1, color="k", linewidth=1)

fig.tight_layout()
fig.savefig(fig_path / "wind_env.pdf")