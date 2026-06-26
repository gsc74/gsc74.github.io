import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
})


def box(ax, x, y, w, h, text, fc="#f8fafc", ec="#334155", lw=1.8, fs=12, weight="bold"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight, color="#1f2937")
    return patch


def arrow(ax, x1, y1, x2, y2, color="#1f2937", lw=1.8, style="-|>", ls="-", ms=14):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        mutation_scale=ms,
        linewidth=lw,
        linestyle=ls,
        color=color,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arr)
    return arr


fig, ax = plt.subplots(figsize=(13.5, 16.0), dpi=180)
ax.set_xlim(0, 10)
ax.set_ylim(0, 14.5)
ax.axis("off")

blue = "#0f6fb3"
light_blue = "#e7f4fd"
orange = "#d97706"
light_orange = "#fff7e6"
green = "#059669"
light_green = "#ecfdf5"
gray = "#64748b"
light_gray = "#f8fafc"
dark = "#1f2937"

ax.text(5, 14.08, "MyLLM decoder-only Transformer", ha="center", va="center",
        fontsize=25, fontweight="bold", color=dark)
ax.text(5, 13.78, "standard causal LM layout • 1,055,231,744 trainable parameters",
        ha="center", va="center", fontsize=13, color=gray)

# Top level flow.
box(ax, 3.0, 13.0, 4.0, 0.55, r"Token IDs  $x_{1:T}$", fc=light_gray, ec=gray, fs=13)
arrow(ax, 5.0, 13.0, 5.0, 12.55)
box(ax, 2.25, 11.9, 5.5, 0.65,
    "Token embedding lookup\n"
    r"$X^{(0)}=E[x_{1:T}],\quad E\in\mathbb{R}^{65{,}536\times1{,}792}$",
    fc=light_blue, ec=blue, fs=12)
arrow(ax, 5.0, 11.9, 5.0, 11.45)

# Decoder stack outer container.
stack = FancyBboxPatch(
    (0.75, 5.0), 8.5, 6.25,
    boxstyle="round,pad=0.035,rounding_size=0.06",
    linewidth=2.2,
    edgecolor=dark,
    facecolor="#ffffff",
)
ax.add_patch(stack)
ax.text(1.05, 10.93, "Repeated decoder block  × 28",
        ha="left", va="center", fontsize=17, fontweight="bold", color=dark)
ax.text(8.85, 10.9, r"$D=1{,}792$", ha="right", va="center", fontsize=12, color=gray)

# Residual stream spine.
ax.plot([1.55, 1.55], [5.45, 10.4], color=green, lw=3)
# ax.text(1.25, 7.93, "residual stream\n" + r"$X^{(\ell)}$",
#         ha="center", va="center", rotation=90, fontsize=12, color=green)
# arrow(ax, 1.55, 10.35, 2.5, 10.05, color=green, lw=2.0)
# arrow(ax, 1.55, 7.75, 2.5, 7.45, color=green, lw=2.0)

# Inside one pre-norm block.
box(ax, 2.45, 9.75, 5.45, 0.52, "RMSNorm", fc=light_gray, ec=gray, fs=12, weight="normal")
arrow(ax, 5.18, 9.75, 5.18, 9.25)
box(ax, 1.85, 8.55, 6.65, 0.7,
    "Causal grouped-query attention\n" + r"$h_q=14,\;h_{kv}=2,\;d=128,\;T\leq8{,}192$",
    fc=light_blue, ec=blue, fs=12)
box(ax, 8.2, 8.68, 1.1, 0.44, "RoPE\n" + r"$\theta=500{,}000$", fc=light_orange, ec=orange, fs=10)
arrow(ax, 5.18, 8.55, 5.18, 8.02)
box(ax, 2.25, 7.55, 5.9, 0.55,
    "Residual addition\n" + r"$U^{(\ell)}=X^{(\ell)}+\mathrm{Attn}(\mathrm{RMSNorm}(X^{(\ell)}))$",
    fc=light_green, ec=green, fs=11)
arrow(ax, 5.18, 7.55, 5.18, 7.02)
box(ax, 2.45, 6.72, 5.45, 0.52, "RMSNorm", fc=light_gray, ec=gray, fs=12, weight="normal")
arrow(ax, 5.18, 6.72, 5.18, 6.22)
box(ax, 1.85, 5.52, 6.65, 0.7,
    "SwiGLU feed-forward network\n" + r"$1{,}792\rightarrow4{,}864\rightarrow1{,}792$",
    fc=light_orange, ec=orange, fs=12)
arrow(ax, 5.18, 5.52, 5.18, 5.12)
box(ax, 2.25, 4.92, 5.9, 0.55,
    "Residual addition\n" + r"$X^{(\ell+1)}=U^{(\ell)}+\mathrm{FFN}(\mathrm{RMSNorm}(U^{(\ell)}))$",
    fc=light_green, ec=green, fs=11)

# Side note clarifying residual.
# ax.text(9.45, 8.0,
#         "Residual stream means\naddition around a sublayer,\nnot a layer named “skip”.",
#         ha="left", va="center", fontsize=11, color=green)
# arrow(ax, 9.25, 7.98, 8.15, 7.82, color=green, lw=1.6)

# Stack continuation and output.
arrow(ax, 5.0, 5.0, 5.0, 4.55)
box(ax, 2.5, 3.85, 5.0, 0.58, "Final RMSNorm", fc=light_gray, ec=gray, fs=13)
arrow(ax, 5.0, 3.85, 5.0, 3.38)
box(ax, 2.15, 2.55, 5.7, 0.72,
    "Tied language-model head\n" + r"logits $Z=X^{(L)}E^\top\in\mathbb{R}^{T\times65{,}536}$",
    fc=light_blue, ec=blue, fs=12)
arrow(ax, 5.0, 2.55, 5.0, 2.08)
box(ax, 3.0, 1.3, 4.0, 0.62, "Softmax gives next-token distribution", fc=light_green, ec=green, fs=12)

# Tied weight arrow outside the decoder stack.
arrow(ax, 7.9, 2.92, 8.9, 2.92, color=blue, lw=1.8, ls="--", ms=12)
ax.plot([8.9, 8.9, 7.75], [2.92, 12.22, 12.22], color=blue, lw=1.8, linestyle="--")
arrow(ax, 7.75, 12.22, 7.52, 12.22, color=blue, lw=1.8, ls="--", ms=12)
ax.text(9.03, 7.45, "same matrix $E$\nused as $E^\\top$\nfor logits",
        ha="left", va="center", fontsize=11, color=blue)

ax.text(0.42, 8.1, r"Sequence length $T\leq8{,}192$", rotation=90,
        ha="center", va="center", fontsize=12, color=gray)

fig.tight_layout()
fig.savefig("HTML/training/architecture.png", bbox_inches="tight", facecolor="white")
