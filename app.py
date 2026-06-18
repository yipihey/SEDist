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
Histograms require choosing a bin width — a free parameter that affects the visual
representation without changing the underlying data.  Choose too few bins and distinct
features merge; choose too many and statistical noise dominates.

The **peaked CDF** is a binning-free alternative.  It is derived directly from the
empirical CDF, so it uses every data point exactly once, and its y-axis carries a
direct probability interpretation: pCDF(*x*) is the fraction of the data that lies
on the minority side of *x*.

The sections below build up the concept step by step.
"""
)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – The Binning Problem
# ══════════════════════════════════════════════════════════════════════════════
st.header("① Effect of Bin Width")
st.markdown(
    """
The dataset below is fixed: 600 samples from a bimodal distribution.
Only the histogram bin count changes.
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
    "At 6 bins the two modes merge into one broad hump.  "
    "At 180 bins the signal is buried in sampling noise.  "
    "The data have not changed; only the bin width has."
)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – The CDF Never Changes
# ══════════════════════════════════════════════════════════════════════════════
st.header("② The CDF Requires No Bin Width")
st.markdown(
    """
The **Cumulative Distribution Function** F(*x*) is the fraction of data points
with values ≤ *x*.  It is determined entirely by the data; bin width does not enter.

Move the slider and observe that the CDF on the right does not change.
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
    "The CDF is stable and unambiguous, but its monotone S-shape makes it hard to "
    "identify structure visually.  The two modes of this dataset appear as regions "
    "where the slope is steeper, but that is not immediately obvious from inspection.  "
    "Histograms became the standard tool partly because peaked shapes are easier to parse."
)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – From CDF to Peaked CDF
# ══════════════════════════════════════════════════════════════════════════════
st.header("③ From CDF to Peaked CDF")
st.markdown(
    r"""
Define the peaked CDF by folding $F$ around $y = 0.5$:

$$\text{pCDF}(x) \;=\; \min\!\bigl(\,F(x),\; 1 - F(x)\,\bigr)$$

**Reading the y-axis directly:**
- pCDF(*x*) is the fraction of data on the *minority* side of *x*
- To the left of the median, pCDF(*x*) = F(*x*) = fraction of data below *x*
- To the right of the median, pCDF(*x*) = 1−F(*x*) = fraction of data above *x*
- The maximum value is always 0.5, reached at the median

**Shape and density:**
Where F rises steeply — i.e., where data is densely concentrated — pCDF also
changes rapidly (steep slope upward on the left of the median, steep slope
downward on the right).  In sparse regions F is nearly flat, so pCDF is also flat.

For a unimodal distribution this produces a single peak at the median.
For a bimodal distribution with roughly equal weight in each mode, the two dense
regions produce steep sides, with a **plateau near 0.5** between them where
data is sparse.  The plateau is *not* two peaks — it reflects the gap between modes.

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

st.info(
    "For this symmetric bimodal distribution the peaked CDF shows a broad plateau "
    "near 0.5 — the sparse gap between the two modes — flanked by steep slopes where "
    "each mode concentrates the data.  Compare with the histogram: the steep slopes in "
    "the peaked CDF correspond to the tall bars in the histogram, with no bin width choice required."
)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – Log Scale Reveals the Tails
# ══════════════════════════════════════════════════════════════════════════════
st.header("④ Log Scale and the Probability Interpretation")
st.markdown(
    """
A log-scale y-axis is standard practice when inspecting distribution tails.
Toggle it on and compare the two representations.
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
    r"""
On log scale the histogram shows gaps and large statistical fluctuations in the tails
where bins contain few counts.  The peaked CDF is smooth down to the outermost data
point because it places one step per observation rather than aggregating into bins.

**Reading probabilities directly off the peaked CDF:**

| pCDF value | Meaning |
|---|---|
| 0.5 | This is the median; half the data lie on each side |
| 0.1 | 10% of the data lie below this *x* (left of median) or above it (right of median) |
| 0.01 | 1% of the data are more extreme than this *x* on the minority side |

Because pCDF(*x*) = min(F(*x*), 1−F(*x*)), it is always the fraction of data on
the *smaller* side of *x*.  On a log scale, reading a tail probability at any *x*
is a direct read from the y-axis — no integration or table lookup needed.
"""
)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – Interactive Playground
# ══════════════════════════════════════════════════════════════════════════════
st.header("⑤ Interactive Comparison")
st.markdown(
    "Select a distribution and adjust parameters.  All three representations update "
    "from the same random sample.  Small sample sizes (n ≲ 50) are informative: "
    "compare how the peaked CDF and histogram each represent the same sparse data."
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
    "- n = 20: with few data points the histogram is strongly bin-width dependent; "
    "the peaked CDF places one step per observation\n"
    "- Student-t, ν = 2, log scale: the power-law tail is visible as a straight line "
    "in the peaked CDF; empty histogram bins appear at the same location\n"
    "- Bimodal, adjust peak separation: narrow the gap and note when the histogram "
    "requires many bins to resolve both modes vs. what the peaked CDF shows"
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
### Histogram
| | |
|---|---|
| ✅ | Peaked shape is easy to read |
| ❌ | Requires a bin width choice |
| ❌ | Aggregates data into bins |
| ❌ | Empty bins in sparse tails |
| ❌ | Sensitive to n in each bin |
"""
    )

with col6b:
    st.markdown(
        """
### CDF
| | |
|---|---|
| ✅ | Uses all data exactly |
| ✅ | No bin width needed |
| ✅ | y-axis is cumulative fraction |
| ❌ | Monotone — structure not obvious |
| ❌ | Modes appear only as slope changes |
"""
    )

with col6c:
    st.markdown(
        """
### Peaked CDF
| | |
|---|---|
| ✅ | Uses all data exactly |
| ✅ | No bin width needed |
| ✅ | Steep slopes = data concentration |
| ✅ | y-axis is a direct probability |
| ✅ | Log scale gives smooth tails |
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
