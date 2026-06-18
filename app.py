"""
Peaked CDFs: Visualize Data Without Losing Information
An interactive pedagogical Streamlit app.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Peaked CDFs: Beyond Histograms",
    page_icon="📊",
    layout="wide",
)

# ── Core math ──────────────────────────────────────────────────────────────────

def ecdf(data):
    """Return (x, y) for the empirical CDF, prepending a zero."""
    x = np.sort(data)
    n = len(x)
    y = np.arange(1, n + 1) / n
    return np.concatenate([[x[0]], x]), np.concatenate([[0.0], y])


def pcdf_from_cdf(y):
    """Fold a CDF array into a peaked CDF: min(F, 1-F)."""
    return np.minimum(y, 1.0 - y)


def make_data(dist_name, params, n, rng):
    """Generate samples for the chosen distribution."""
    if dist_name == "Normal":
        return rng.normal(params["mu"], params["sigma"], n)
    elif dist_name == "Bimodal Normal":
        mask = rng.random(n) < params["mix"]
        return np.where(
            mask,
            rng.normal(params["mu1"], params["sigma1"], n),
            rng.normal(params["mu2"], params["sigma2"], n),
        )
    elif dist_name == "Log-Normal (right-skewed)":
        return rng.lognormal(params["mu"], params["sigma"], n)
    elif dist_name == "Student-t (heavy tails)":
        return rng.standard_t(params["df"], n)
    elif dist_name == "Uniform":
        return rng.uniform(params["a"], params["b"], n)
    elif dist_name == "Exponential":
        return rng.exponential(params["scale"], n)
    else:  # Gamma
        return rng.gamma(params["shape"], params["scale"], n)


# ── Shared color palette ───────────────────────────────────────────────────────
C_HIST  = "#60a5fa"   # blue
C_CDF   = "#a78bfa"   # violet
C_PCDF  = "#34d399"   # emerald
C_FOLD  = "#f59e0b"   # amber (1-CDF line)


# ── Helpers for common figure settings ────────────────────────────────────────

def base_layout(title, xlab, ylab, height=340, log_y=False):
    return dict(
        title=dict(text=title, font_size=14),
        xaxis_title=xlab,
        yaxis_title=ylab,
        yaxis_type="log" if log_y else "linear",
        height=height,
        showlegend=False,
        margin=dict(t=50, b=40, l=60, r=20),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
        xaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
        yaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title("📊  Peaked CDFs: Visualize Data Without Losing Information")
st.markdown(
    """
Histograms are the go-to tool for exploring data distributions — but they carry
a hidden cost: you must choose a bin width. Too few bins and peaks merge; too many
and the signal drowns in noise. **Peaked CDFs** give you the same intuitive visual
as a histogram — peaks where data concentrates, valleys in sparse regions —
while preserving every single data point.

Work through the sections below to see why.
"""
)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – The Binning Problem
# ══════════════════════════════════════════════════════════════════════════════
st.header("① The Binning Problem")
st.markdown(
    """
Below is the **same dataset** — 600 samples from a bimodal distribution — displayed
as a histogram.  Drag the slider to change the number of bins.
"""
)

rng_fixed = np.random.default_rng(7)
demo_data = np.concatenate([
    rng_fixed.normal(-2.0, 0.6, 300),
    rng_fixed.normal( 2.0, 0.9, 300),
])

n_bins_s1 = st.slider(
    "Number of bins", min_value=4, max_value=200, value=25, step=1, key="s1_bins"
)

fig_s1 = go.Figure(go.Histogram(
    x=demo_data, nbinsx=n_bins_s1,
    histnorm="probability density",
    marker_color=C_HIST, opacity=0.85,
))
fig_s1.update_layout(**base_layout(
    f"Histogram — {n_bins_s1} bins (same data each time)",
    "x", "Density", height=320,
))
st.plotly_chart(fig_s1, use_container_width=True)

st.info(
    "**Try this:** Set bins to 6 — the two peaks vanish into one.  "
    "Set to 180 — the peaks fragment into noisy spikes.  "
    "The \"right\" bin count is subjective and changes the story you tell."
)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – The CDF Never Changes
# ══════════════════════════════════════════════════════════════════════════════
st.header("② The CDF Never Changes")
st.markdown(
    """
A **Cumulative Distribution Function** (CDF) at point *x* answers:
*"What fraction of my data is ≤ x?"*

It requires **no bin width** — it uses every data point exactly once.
Keep moving the slider and watch what doesn't change.
"""
)

n_bins_s2 = st.slider(
    "Number of bins", min_value=4, max_value=200, value=25, step=1, key="s2_bins"
)
xe, ye = ecdf(demo_data)

col2a, col2b = st.columns(2, gap="medium")

with col2a:
    fig_hist2 = go.Figure(go.Histogram(
        x=demo_data, nbinsx=n_bins_s2,
        histnorm="probability density",
        marker_color=C_HIST, opacity=0.85,
    ))
    fig_hist2.update_layout(**base_layout(
        f"Histogram — {n_bins_s2} bins", "x", "Density"
    ))
    st.plotly_chart(fig_hist2, use_container_width=True)

with col2b:
    fig_cdf2 = go.Figure(go.Scatter(
        x=xe, y=ye, mode="lines",
        line=dict(color=C_CDF, width=2.5),
    ))
    fig_cdf2.update_layout(**base_layout(
        "CDF — unchanged regardless of bins", "x", "Fraction of data ≤ x"
    ))
    st.plotly_chart(fig_cdf2, use_container_width=True)

st.warning(
    "**The catch:** the CDF is completely stable, but its S-shape is hard to read. "
    "Where are the peaks?  You can see the CDF rises faster between –3 and –1, "
    "and again between 1 and 3, but you have to work for it.  "
    "Histograms were invented because humans find peaked shapes much easier to parse."
)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – From CDF to Peaked CDF
# ══════════════════════════════════════════════════════════════════════════════
st.header("③ The Peaked CDF — Fold the CDF in Half")
st.markdown(
    r"""
The trick is one line of math:

$$\text{pCDF}(x) \;=\; \min\!\bigl(\,F(x),\; 1 - F(x)\,\bigr)$$

We take the CDF $F(x)$ and its mirror $1-F(x)$, then keep whichever is smaller
at each point.  This **folds** the CDF around $y = 0.5$:

- Where $F$ rises **steeply** (many data points nearby) → both $F$ and $1-F$
  approach 0.5 together → the minimum stays **high** → a **peak**.
- In **sparse tails** → $F$ barely moves → the minimum is **low** → a **valley**.

The three panels below show the transformation step by step.
"""
)

xp = xe
yp = pcdf_from_cdf(ye)

col3a, col3b, col3c = st.columns(3, gap="small")

with col3a:
    fig3a = go.Figure()
    fig3a.add_trace(go.Scatter(
        x=xe, y=ye, mode="lines",
        line=dict(color=C_CDF, width=2.5), name="F(x)",
    ))
    fig3a.add_hline(y=0.5, line_dash="dot", line_color="#475569", line_width=1.5,
                    annotation_text="y = 0.5", annotation_font_size=10,
                    annotation_font_color="#94a3b8")
    fig3a.update_layout(**base_layout("① Start: the CDF", "x", "F(x)"))
    st.plotly_chart(fig3a, use_container_width=True)

with col3b:
    fig3b = go.Figure()
    fig3b.add_trace(go.Scatter(
        x=xe, y=ye, mode="lines",
        line=dict(color=C_CDF, width=2, dash="dash"), name="F(x)", opacity=0.55,
    ))
    fig3b.add_trace(go.Scatter(
        x=xe, y=1 - ye, mode="lines",
        line=dict(color=C_FOLD, width=2, dash="dash"), name="1−F(x)", opacity=0.55,
    ))
    fig3b.add_trace(go.Scatter(
        x=xp, y=yp, mode="lines",
        line=dict(color=C_PCDF, width=3), name="min(F, 1−F)",
    ))
    fig3b.add_hline(y=0.5, line_dash="dot", line_color="#475569", line_width=1.5)
    fig3b.update_layout(**base_layout(
        "② Fold: take min(F, 1−F)", "x", "pCDF(x)",
        height=340,
    ))
    fig3b.update_layout(showlegend=True, legend=dict(
        x=0.02, y=0.98, font_size=11,
        bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
    ))
    st.plotly_chart(fig3b, use_container_width=True)

with col3c:
    fig3c = go.Figure()
    fig3c.add_trace(go.Scatter(
        x=xp, y=yp, mode="lines",
        line=dict(color=C_PCDF, width=2.5),
    ))
    fig3c.update_layout(**base_layout("③ Result: the Peaked CDF", "x", "pCDF(x)"))
    st.plotly_chart(fig3c, use_container_width=True)

st.success(
    "The two peaks in the peaked CDF correspond exactly to the two modes of the "
    "bimodal distribution — same information as the histogram, zero arbitrary choices."
)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – Log Scale Reveals the Tails
# ══════════════════════════════════════════════════════════════════════════════
st.header("④ Log Scale: Inspect the Tails")
st.markdown(
    """
Log-scale histograms are a common trick for seeing rare events.
The peaked CDF does the same on log scale — but **without the empty bins**
that appear when tail counts are too low.  Toggle the log scale and compare.
"""
)

log_s4 = st.toggle("Log scale on y-axis", value=True, key="s4_log")

col4a, col4b = st.columns(2, gap="medium")

with col4a:
    fig4a = go.Figure(go.Histogram(
        x=demo_data, nbinsx=40,
        histnorm="probability density",
        marker_color=C_HIST, opacity=0.85,
    ))
    if log_s4:
        fig4a.update_yaxes(range=[-4, 1])  # 10^-4 to 10^1 on log scale
    fig4a.update_layout(**base_layout(
        "Histogram — 40 fixed bins", "x", "Density", log_y=log_s4
    ))
    st.plotly_chart(fig4a, use_container_width=True)

with col4b:
    # Avoid log(0): clip peaked CDF at a small floor
    yp_safe = np.clip(yp, 1e-6, None)
    fig4b = go.Figure(go.Scatter(
        x=xp, y=yp_safe if log_s4 else yp,
        mode="lines", line=dict(color=C_PCDF, width=2.5),
    ))
    if log_s4:
        fig4b.update_yaxes(range=[-6, np.log10(0.5) + 0.1])
    fig4b.update_layout(**base_layout(
        "Peaked CDF — no bins needed", "x", "pCDF(x)", log_y=log_s4
    ))
    st.plotly_chart(fig4b, use_container_width=True)

st.markdown(
    """
On log scale the histogram develops **gaps and spikes** in the tails —
an artifact of sparse bins.  The peaked CDF remains **smooth all the way to
the edge of your data**.

The y-axis of the peaked CDF has a direct statistical meaning:
a value of 0.1 means 10% of the data lies below (or above) that x value.
"""
)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – Interactive Playground
# ══════════════════════════════════════════════════════════════════════════════
st.header("⑤ Interactive Playground")
st.markdown(
    "Choose a distribution, adjust its parameters, and watch all three views update "
    "simultaneously.  Try very small sample sizes to see how the peaked CDF stays "
    "informative where the histogram gives up."
)

# ── Controls ──────────────────────────────────────────────────────────────────
ctrl1, ctrl2, ctrl3 = st.columns([1.4, 1.4, 1], gap="medium")

with ctrl1:
    dist_name = st.selectbox("Distribution", [
        "Normal",
        "Bimodal Normal",
        "Log-Normal (right-skewed)",
        "Student-t (heavy tails)",
        "Uniform",
        "Exponential",
        "Gamma",
    ], key="pg_dist")

    n_pg = st.slider("Sample size  n", 20, 5000, 400, step=10, key="pg_n")
    seed_pg = st.number_input("Random seed", value=42, step=1, key="pg_seed")
    rng_pg = np.random.default_rng(int(seed_pg))

with ctrl2:
    params = {}
    if dist_name == "Normal":
        params["mu"]    = st.slider("Mean μ", -5.0, 5.0, 0.0, 0.1, key="pg_mu")
        params["sigma"] = st.slider("Std dev σ", 0.1, 4.0, 1.0, 0.1, key="pg_sig")
        desc = f"Normal(μ={params['mu']:.1f}, σ={params['sigma']:.1f})"

    elif dist_name == "Bimodal Normal":
        params["mu1"]    = st.slider("Peak 1 mean", -5.0, 0.0, -2.0, 0.1, key="pg_m1")
        params["mu2"]    = st.slider("Peak 2 mean",  0.0, 5.0,  2.0, 0.1, key="pg_m2")
        params["sigma1"] = st.slider("Peak 1 std",   0.1, 2.0,  0.6, 0.1, key="pg_s1")
        params["sigma2"] = st.slider("Peak 2 std",   0.1, 2.0,  0.9, 0.1, key="pg_s2")
        params["mix"]    = st.slider("Fraction in peak 1", 0.05, 0.95, 0.5, 0.05, key="pg_mix")
        desc = (
            f"Bimodal(μ₁={params['mu1']:.1f}, σ₁={params['sigma1']:.1f}, "
            f"μ₂={params['mu2']:.1f}, σ₂={params['sigma2']:.1f})"
        )

    elif dist_name == "Log-Normal (right-skewed)":
        params["mu"]    = st.slider("Log-mean μ", -1.0, 2.0,  0.0, 0.1, key="pg_lmu")
        params["sigma"] = st.slider("Log-std σ",   0.1, 1.5,  0.5, 0.05, key="pg_lsig")
        desc = f"LogNormal(μ={params['mu']:.1f}, σ={params['sigma']:.2f})"

    elif dist_name == "Student-t (heavy tails)":
        params["df"] = st.slider("Degrees of freedom ν", 1, 30, 3, 1, key="pg_df")
        desc = f"Student-t(ν={params['df']})"

    elif dist_name == "Uniform":
        params["a"] = st.slider("Lower bound a", -5.0, 0.0, -1.0, 0.1, key="pg_ua")
        params["b"] = st.slider("Upper bound b",  0.1, 5.0,  1.0, 0.1, key="pg_ub")
        desc = f"Uniform({params['a']:.1f}, {params['b']:.1f})"

    elif dist_name == "Exponential":
        params["scale"] = st.slider("Scale 1/λ", 0.1, 4.0, 1.0, 0.1, key="pg_esc")
        desc = f"Exponential(scale={params['scale']:.1f})"

    else:  # Gamma
        params["shape"] = st.slider("Shape k", 0.5, 8.0, 2.0, 0.1, key="pg_gsh")
        params["scale"] = st.slider("Scale θ", 0.1, 3.0, 1.0, 0.1, key="pg_gsc")
        desc = f"Gamma(k={params['shape']:.1f}, θ={params['scale']:.1f})"

with ctrl3:
    n_bins_pg = st.slider("Histogram bins", 4, 120, 30, 2, key="pg_bins")
    log_pg    = st.toggle("Log scale y-axis", value=False, key="pg_log")
    show_fold = st.toggle("Show CDF & 1−CDF on pCDF panel", value=False, key="pg_fold")

# ── Generate data ──────────────────────────────────────────────────────────────
pg_data = make_data(dist_name, params, n_pg, rng_pg)
xep, yep = ecdf(pg_data)
ypp = pcdf_from_cdf(yep)
ypp_safe = np.clip(ypp, 1e-7, None)

# ── Three-panel figure ─────────────────────────────────────────────────────────
fig_pg = make_subplots(
    rows=1, cols=3,
    subplot_titles=["Histogram", "CDF", "Peaked CDF"],
    horizontal_spacing=0.08,
)

# Histogram
fig_pg.add_trace(go.Histogram(
    x=pg_data, nbinsx=n_bins_pg,
    histnorm="probability density",
    marker_color=C_HIST, opacity=0.8,
    name="Histogram",
), row=1, col=1)

# CDF
fig_pg.add_trace(go.Scatter(
    x=xep, y=yep, mode="lines",
    line=dict(color=C_CDF, width=2.5),
    name="CDF",
), row=1, col=2)

# (optionally) CDF & 1-CDF reference on the pCDF panel
if show_fold:
    fig_pg.add_trace(go.Scatter(
        x=xep, y=yep, mode="lines",
        line=dict(color=C_CDF, width=1.5, dash="dot"), opacity=0.45,
        name="F(x)",
    ), row=1, col=3)
    fig_pg.add_trace(go.Scatter(
        x=xep, y=1 - yep, mode="lines",
        line=dict(color=C_FOLD, width=1.5, dash="dot"), opacity=0.45,
        name="1−F(x)",
    ), row=1, col=3)

# Peaked CDF
fig_pg.add_trace(go.Scatter(
    x=xep, y=ypp_safe if log_pg else ypp,
    mode="lines", line=dict(color=C_PCDF, width=2.5),
    name="pCDF",
), row=1, col=3)

y_type = "log" if log_pg else "linear"
fig_pg.update_yaxes(type=y_type)
fig_pg.update_layout(
    title=dict(
        text=f"{desc} — n={n_pg} samples",
        font_size=13,
    ),
    height=400,
    showlegend=show_fold,
    margin=dict(t=70, b=40, l=50, r=20),
    plot_bgcolor="#0f172a",
    paper_bgcolor="#0f172a",
    font_color="#e2e8f0",
    legend=dict(x=0.68, y=0.98, font_size=11, bgcolor="rgba(0,0,0,0)"),
)
for ax in ["xaxis", "xaxis2", "xaxis3", "yaxis", "yaxis2", "yaxis3"]:
    fig_pg.update_layout(**{ax: dict(gridcolor="#1e293b", zerolinecolor="#334155")})

fig_pg.update_xaxes(title_text="x",        row=1, col=1)
fig_pg.update_xaxes(title_text="x",        row=1, col=2)
fig_pg.update_xaxes(title_text="x",        row=1, col=3)
fig_pg.update_yaxes(title_text="Density",  row=1, col=1)
fig_pg.update_yaxes(title_text="F(x)",     row=1, col=2)
fig_pg.update_yaxes(title_text="pCDF(x)",  row=1, col=3)

st.plotly_chart(fig_pg, use_container_width=True)

st.markdown(
    "**Suggested experiments:**\n"
    "- Set n=20 — histogram is barely informative; peaked CDF still shows shape\n"
    "- Try Student-t with ν=2 — log-scale peaked CDF shows the heavy power-law tails\n"
    "- Bimodal with closely spaced peaks — see how many bins you need vs pCDF"
)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – Key Takeaways
# ══════════════════════════════════════════════════════════════════════════════
st.header("⑥ Key Takeaways")

col6a, col6b, col6c = st.columns(3, gap="medium")

with col6a:
    st.markdown(
        """
### 📊  Histogram
| | |
|---|---|
| ✅ | Intuitive peaks |
| ❌ | Arbitrary bin width |
| ❌ | Loses data-point positions |
| ❌ | Empty bins in tails |
| ❌ | Misleading for small n |
"""
    )

with col6b:
    st.markdown(
        """
### 📈  CDF
| | |
|---|---|
| ✅ | Uses all data exactly |
| ✅ | No bin width needed |
| ✅ | Stable and unambiguous |
| ❌ | Monotone S-shape is hard to read |
| ❌ | Peaks and modes not obvious |
"""
    )

with col6c:
    st.markdown(
        """
### ✨  Peaked CDF
| | |
|---|---|
| ✅ | Uses all data exactly |
| ✅ | No bin width needed |
| ✅ | Peaks = data concentration |
| ✅ | Log scale reveals tails smoothly |
| ✅ | Works well even for small n |
"""
    )

st.markdown(
    r"""
---

**The formula in one line:**

$$\text{pCDF}(x) = \min\!\bigl(F(x),\; 1-F(x)\bigr)$$

**Further reading:**
Berg & Harris (2008) — *Normalizing the height distribution of a stellar population*,
[arXiv:0712.3852](https://arxiv.org/abs/0712.3852)

**Python package used here:**
[`SEdist`](https://github.com/yipihey/SEDist) — smooth empirical distributions
with peaked CDF support via `scipy.stats`.
"""
)
