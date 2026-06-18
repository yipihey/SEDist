"""
Peaked CDFs: Visualize Data Without Losing Information
Streamlit app — SEDist repository.
"""

import os, sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter
import streamlit as st

# ── SEdist (same repo, src/) ───────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
try:
    from SEdist import SE_distribution
    SEDIST_AVAILABLE = True
except ImportError:
    SEDIST_AVAILABLE = False

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Peaked CDFs — SEdist",
    page_icon="📊",
    layout="wide",
)

# ══ Core math ═══════════════════════════════════════════════════════════════════

def ecdf(data):
    x = np.sort(data)
    n = len(x)
    y = np.arange(1, n + 1) / n
    return np.concatenate([[x[0]], x]), np.concatenate([[0.0], y])


def pcdf_from_cdf(y):
    return np.minimum(y, 1.0 - y)


def make_data(dist_name, params, n, rng):
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
    else:
        return rng.gamma(params["shape"], params["scale"], n)


# ── Demo image ─────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading demo image…")
def load_demo_image(size: int = 350):
    """Crop the scipy raccoon face to a square, or return a synthetic fallback."""
    try:
        from scipy.datasets import face
        img = face(gray=True).astype(float) / 255.0
        h, w = img.shape
        cy, cx = h // 2, w // 2
        half = min(size // 2, cy, cx)
        return img[cy - half : cy + half, cx - half : cx + half], "scipy sample image (raccoon)"
    except Exception:
        pass
    # Synthetic: large-scale blobs + fine texture
    rng = np.random.default_rng(42)
    x = np.linspace(-3, 3, size)
    xx, yy = np.meshgrid(x, x)
    structure = (
        np.exp(-0.5 * (xx ** 2 + yy ** 2))
        + 0.65 * np.exp(-0.5 * ((xx - 1.6) ** 2 + (yy + 1.2) ** 2))
        + 0.45 * np.exp(-0.5 * ((xx + 1.1) ** 2 + (yy - 1.0) ** 2))
    )
    texture = gaussian_filter(rng.normal(0, 1, (size, size)), sigma=3)
    img = structure + 0.15 * texture
    img = (img - img.min()) / (img.max() - img.min())
    return img, "synthetic test image (use Upload to try your own)"


# ── SE_distribution CDF curves (cached to avoid re-sorting large arrays) ───────

@st.cache_data(show_spinner="Computing distributions…")
def compute_se_curves(dist_name: str, n: int, nfit: int, seed: int = 17):
    """Return pCDF arrays for full, linear, and log-compressed SE_distribution."""
    rng = np.random.default_rng(seed)
    if dist_name == "Normal":
        data = rng.normal(0, 1, n)
    elif dist_name == "Student-t (ν=3)":
        data = rng.standard_t(3, n)
    elif dist_name == "Log-Normal":
        data = rng.lognormal(0, 0.8, n)
    else:
        data = rng.exponential(1.0, n)

    lo, hi = np.percentile(data, 0.05), np.percentile(data, 99.95)
    x_eval = np.linspace(lo, hi, 3000)

    d_full = SE_distribution(data)
    d_lin  = SE_distribution(data, compress="linear", Ninterpolants=nfit)
    d_log  = SE_distribution(data, compress="log",    Ninterpolants=nfit)

    return {
        "x":      x_eval,
        "full":   {"pcdf": d_full.pcdf(x_eval), "nfit": d_full.Nfit},
        "linear": {"pcdf": d_lin.pcdf(x_eval),  "nfit": d_lin.Nfit},
        "log":    {"pcdf": d_log.pcdf(x_eval),   "nfit": d_log.Nfit},
        "median": float(d_full.ppf(0.5)),
    }


# ── Colors ─────────────────────────────────────────────────────────────────────
C_HIST = "#60a5fa"
C_CDF  = "#a78bfa"
C_PCDF = "#34d399"
C_FOLD = "#f59e0b"
PALETTE = ["#34d399", "#60a5fa", "#f59e0b", "#f87171", "#a78bfa", "#fb923c", "#38bdf8"]

DARK = dict(
    plot_bgcolor  = "#0f172a",
    paper_bgcolor = "#0f172a",
    font_color    = "#e2e8f0",
)
GRID = dict(gridcolor="#1e293b", zerolinecolor="#334155")

# Log-scale y-axis bounds used on every peaked-CDF plot:
#   lower = 1e-3  (0.1th percentile — one tenth of a percent in the tail)
#   upper = just above 0.5 (the pCDF maximum, reached at the median)
PCDF_LOG_RANGE = (-3, round(np.log10(0.5) + 0.1, 3))


def base_layout(title, xlab, ylab, height=340, log_y=False):
    return dict(
        title       = dict(text=title, font_size=14),
        xaxis_title = xlab,
        yaxis_title = ylab,
        yaxis_type  = "log" if log_y else "linear",
        height      = height,
        margin      = dict(t=50, b=40, l=60, r=20),
        xaxis       = GRID,
        yaxis       = GRID,
        **DARK,
    )


# ══ Page header ════════════════════════════════════════════════════════════════

st.title("📊  Peaked CDFs: Visualize Data Without Losing Information")
st.markdown(
    """
    Histograms require choosing a bin width — a free parameter that affects the visual
    representation without changing the underlying data.  The **peaked CDF** is a
    binning-free alternative that uses every data point exactly once.  Its y-axis carries
    a direct probability interpretation: pCDF(*x*) is the fraction of data on the
    minority side of *x*.
    """
)

tab1, tab2, tab3 = st.tabs([
    "📐  Tutorial",
    "🖼️  Image Example",
    "📦  SEdist Package",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Tutorial
# ══════════════════════════════════════════════════════════════════════════════

with tab1:

    rng_fixed = np.random.default_rng(7)
    demo_data = np.concatenate([
        rng_fixed.normal(-2.0, 0.6, 300),
        rng_fixed.normal( 2.0, 0.9, 300),
    ])

    # ── §1 ─────────────────────────────────────────────────────────────────────
    st.header("① Effect of Bin Width")
    st.markdown(
        "The dataset below is fixed: 600 samples from a bimodal distribution.  "
        "Only the histogram bin count changes."
    )
    n_bins_s1 = st.slider("Number of bins", 4, 200, 25, 1, key="s1_bins")
    fig_s1 = go.Figure(go.Histogram(
        x=demo_data, nbinsx=n_bins_s1,
        histnorm="probability density",
        marker_color=C_HIST, opacity=0.85,
    ))
    fig_s1.update_layout(**base_layout(
        f"Histogram — {n_bins_s1} bins (same data each time)", "x", "Density", height=320,
    ))
    st.plotly_chart(fig_s1, use_container_width=True)
    st.info(
        "At 6 bins the two modes merge into one broad hump.  "
        "At 180 bins the signal is buried in sampling noise.  "
        "The data have not changed; only the bin width has."
    )
    st.divider()

    # ── §2 ─────────────────────────────────────────────────────────────────────
    st.header("② The CDF Requires No Bin Width")
    st.markdown(
        "The **Cumulative Distribution Function** F(*x*) is the fraction of data points "
        "with values ≤ *x*.  It is determined entirely by the data; bin width does not enter.  "
        "Move the slider and observe that the CDF on the right does not change."
    )
    n_bins_s2 = st.slider("Number of bins", 4, 200, 25, 1, key="s2_bins")
    xe, ye = ecdf(demo_data)

    col2a, col2b = st.columns(2, gap="medium")
    with col2a:
        fig2h = go.Figure(go.Histogram(
            x=demo_data, nbinsx=n_bins_s2,
            histnorm="probability density",
            marker_color=C_HIST, opacity=0.85,
        ))
        fig2h.update_layout(**base_layout(f"Histogram — {n_bins_s2} bins", "x", "Density"))
        st.plotly_chart(fig2h, use_container_width=True)
    with col2b:
        fig2c = go.Figure(go.Scatter(
            x=xe, y=ye, mode="lines", line=dict(color=C_CDF, width=2.5),
        ))
        fig2c.update_layout(**base_layout(
            "CDF — unchanged regardless of bins", "x", "Fraction of data ≤ x",
        ))
        st.plotly_chart(fig2c, use_container_width=True)

    st.warning(
        "The CDF is stable and unambiguous, but its monotone S-shape makes it hard "
        "to identify structure visually.  The two modes appear as regions where the "
        "slope is steeper, but that is not immediately obvious from inspection."
    )
    st.divider()

    # ── §3 ─────────────────────────────────────────────────────────────────────
    st.header("③ From CDF to Peaked CDF")
    st.markdown(
        r"""
        Define the peaked CDF by folding $F$ around $y = 0.5$:
        $$\text{pCDF}(x) = \min\!\bigl(\,F(x),\; 1 - F(x)\,\bigr)$$

        **Reading the y-axis directly:**
        - pCDF(*x*) is the fraction of data on the *minority* side of *x*
        - Left of the median: pCDF(*x*) = F(*x*) = fraction of data below *x*
        - Right of the median: pCDF(*x*) = 1 − F(*x*) = fraction of data above *x*
        - Maximum of 0.5 is always at the median

        **Shape and density:**  where F rises steeply (high data density), pCDF also
        changes rapidly — steep slope upward to the left of the median, steep downward
        to the right.  In sparse regions F is nearly flat, so pCDF is also flat.

        For a unimodal distribution this produces a single peak at the median.  For a
        symmetric bimodal distribution, each mode produces a steep slope, separated by a
        **plateau near 0.5** spanning the sparse gap between them.  The plateau is not
        two peaks; it reflects the gap.
        """
    )

    xp = xe
    yp = pcdf_from_cdf(ye)

    col3a, col3b, col3c = st.columns(3, gap="small")
    with col3a:
        f3a = go.Figure()
        f3a.add_trace(go.Scatter(x=xe, y=ye, mode="lines", line=dict(color=C_CDF, width=2.5)))
        f3a.add_hline(y=0.5, line_dash="dot", line_color="#475569", line_width=1.5,
                      annotation_text="y = 0.5", annotation_font_size=10,
                      annotation_font_color="#94a3b8")
        f3a.update_layout(**base_layout("① Start: the CDF", "x", "F(x)"))
        st.plotly_chart(f3a, use_container_width=True)

    with col3b:
        f3b = go.Figure()
        f3b.add_trace(go.Scatter(x=xe, y=ye, mode="lines",
                                 line=dict(color=C_CDF, width=2, dash="dash"),
                                 name="F(x)", opacity=0.55))
        f3b.add_trace(go.Scatter(x=xe, y=1 - ye, mode="lines",
                                 line=dict(color=C_FOLD, width=2, dash="dash"),
                                 name="1−F(x)", opacity=0.55))
        f3b.add_trace(go.Scatter(x=xp, y=yp, mode="lines",
                                 line=dict(color=C_PCDF, width=3),
                                 name="min(F, 1−F)"))
        f3b.add_hline(y=0.5, line_dash="dot", line_color="#475569", line_width=1.5)
        f3b.update_layout(**base_layout("② Fold: take min(F, 1−F)", "x", "pCDF(x)", height=340))
        f3b.update_layout(showlegend=True, legend=dict(
            x=0.02, y=0.98, font_size=11, bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
        ))
        st.plotly_chart(f3b, use_container_width=True)

    with col3c:
        f3c = go.Figure()
        f3c.add_trace(go.Scatter(x=xp, y=yp, mode="lines",
                                 line=dict(color=C_PCDF, width=2.5)))
        f3c.update_layout(**base_layout("③ Result: the Peaked CDF", "x", "pCDF(x)"))
        st.plotly_chart(f3c, use_container_width=True)

    st.info(
        "For this symmetric bimodal distribution the peaked CDF shows a broad plateau "
        "near 0.5 — the sparse gap between the two modes — flanked by steep slopes "
        "where each mode concentrates data.  The steep slopes correspond to the tall "
        "bars in the histogram, with no bin width choice required."
    )
    st.divider()

    # ── §4 ─────────────────────────────────────────────────────────────────────
    st.header("④ Log Scale and the Probability Interpretation")
    st.markdown(
        "A log-scale y-axis is standard practice when inspecting distribution tails.  "
        "Toggle it on and compare the two representations."
    )
    log_s4 = st.toggle("Log scale on y-axis", value=True, key="s4_log")

    col4a, col4b = st.columns(2, gap="medium")
    yp_safe = np.clip(yp, 1e-6, None)
    with col4a:
        f4h = go.Figure(go.Histogram(
            x=demo_data, nbinsx=40,
            histnorm="probability density",
            marker_color=C_HIST, opacity=0.85,
        ))
        if log_s4:
            f4h.update_yaxes(range=[-3, 1])
        f4h.update_layout(**base_layout("Histogram — 40 fixed bins", "x", "Density", log_y=log_s4))
        st.plotly_chart(f4h, use_container_width=True)

    with col4b:
        f4p = go.Figure(go.Scatter(
            x=xp, y=yp_safe if log_s4 else yp,
            mode="lines", line=dict(color=C_PCDF, width=2.5),
        ))
        if log_s4:
            f4p.update_yaxes(range=list(PCDF_LOG_RANGE))
        f4p.update_layout(**base_layout("Peaked CDF — no bins needed", "x", "pCDF(x)", log_y=log_s4))
        st.plotly_chart(f4p, use_container_width=True)

    st.markdown(
        r"""
        On log scale the histogram shows gaps and fluctuations in the tails where bins
        contain few counts.  The peaked CDF is smooth down to the outermost data point
        because it places one step per observation rather than aggregating into bins.

        **Reading probabilities directly:**

        | pCDF value | Meaning |
        |---|---|
        | 0.5 | The median: half the data lie on each side |
        | 0.1 | 10 % of the data are more extreme than *x* on the minority side |
        | 0.01 | 1 % of the data are more extreme — a direct tail probability |

        Because pCDF(*x*) = min(F(*x*), 1−F(*x*)), it always equals the fraction of data on
        the *smaller* side of *x*.  On log scale, a tail probability at any *x* is a
        direct read from the y-axis — no integration or table lookup needed.
        """
    )
    st.divider()

    # ── §5 ─────────────────────────────────────────────────────────────────────
    st.header("⑤ Interactive Comparison")
    st.markdown(
        "Select a distribution and adjust parameters.  All three representations update "
        "from the same random sample.  Small sample sizes (n ≲ 50) are informative: "
        "compare how the peaked CDF and histogram each represent the same sparse data."
    )

    ctrl1, ctrl2, ctrl3 = st.columns([1.4, 1.4, 1], gap="medium")
    with ctrl1:
        dist_name = st.selectbox("Distribution", [
            "Normal", "Bimodal Normal", "Log-Normal (right-skewed)",
            "Student-t (heavy tails)", "Uniform", "Exponential", "Gamma",
        ], key="pg_dist")
        n_pg   = st.slider("Sample size  n", 20, 5000, 400, 10, key="pg_n")
        seed_pg = st.number_input("Random seed", value=42, step=1, key="pg_seed")
        rng_pg = np.random.default_rng(int(seed_pg))

    with ctrl2:
        params = {}
        if dist_name == "Normal":
            params["mu"]    = st.slider("Mean μ",    -5.0, 5.0, 0.0, 0.1, key="pg_mu")
            params["sigma"] = st.slider("Std dev σ",  0.1, 4.0, 1.0, 0.1, key="pg_sig")
            desc = f"Normal(μ={params['mu']:.1f}, σ={params['sigma']:.1f})"
        elif dist_name == "Bimodal Normal":
            params["mu1"]    = st.slider("Peak 1 mean", -5.0, 0.0, -2.0, 0.1, key="pg_m1")
            params["mu2"]    = st.slider("Peak 2 mean",  0.0, 5.0,  2.0, 0.1, key="pg_m2")
            params["sigma1"] = st.slider("Peak 1 std",   0.1, 2.0,  0.6, 0.1, key="pg_s1")
            params["sigma2"] = st.slider("Peak 2 std",   0.1, 2.0,  0.9, 0.1, key="pg_s2")
            params["mix"]    = st.slider("Fraction in peak 1", 0.05, 0.95, 0.5, 0.05, key="pg_mix")
            desc = (f"Bimodal(μ₁={params['mu1']:.1f}, σ₁={params['sigma1']:.1f}, "
                    f"μ₂={params['mu2']:.1f}, σ₂={params['sigma2']:.1f})")
        elif dist_name == "Log-Normal (right-skewed)":
            params["mu"]    = st.slider("Log-mean μ", -1.0, 2.0, 0.0, 0.1,  key="pg_lmu")
            params["sigma"] = st.slider("Log-std σ",   0.1, 1.5, 0.5, 0.05, key="pg_lsig")
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
        else:
            params["shape"] = st.slider("Shape k",  0.5, 8.0, 2.0, 0.1, key="pg_gsh")
            params["scale"] = st.slider("Scale θ",  0.1, 3.0, 1.0, 0.1, key="pg_gsc")
            desc = f"Gamma(k={params['shape']:.1f}, θ={params['scale']:.1f})"

    with ctrl3:
        n_bins_pg = st.slider("Histogram bins", 4, 120, 30, 2, key="pg_bins")
        log_pg    = st.toggle("Log scale y-axis", value=False, key="pg_log")
        show_fold = st.toggle("Show F and 1−F on pCDF panel", value=False, key="pg_fold")

    pg_data = make_data(dist_name, params, n_pg, rng_pg)
    xep, yep = ecdf(pg_data)
    ypp = pcdf_from_cdf(yep)

    fig_pg = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Histogram", "CDF", "Peaked CDF"],
        horizontal_spacing=0.08,
    )
    fig_pg.add_trace(go.Histogram(
        x=pg_data, nbinsx=n_bins_pg, histnorm="probability density",
        marker_color=C_HIST, opacity=0.8, name="Histogram",
    ), row=1, col=1)
    fig_pg.add_trace(go.Scatter(
        x=xep, y=yep, mode="lines", line=dict(color=C_CDF, width=2.5), name="CDF",
    ), row=1, col=2)
    if show_fold:
        fig_pg.add_trace(go.Scatter(
            x=xep, y=yep, mode="lines",
            line=dict(color=C_CDF, width=1.5, dash="dot"), opacity=0.45, name="F(x)",
        ), row=1, col=3)
        fig_pg.add_trace(go.Scatter(
            x=xep, y=1 - yep, mode="lines",
            line=dict(color=C_FOLD, width=1.5, dash="dot"), opacity=0.45, name="1−F(x)",
        ), row=1, col=3)
    fig_pg.add_trace(go.Scatter(
        x=xep,
        y=np.clip(ypp, 1e-7, None) if log_pg else ypp,
        mode="lines", line=dict(color=C_PCDF, width=2.5), name="pCDF",
    ), row=1, col=3)

    fig_pg.update_yaxes(type="log" if log_pg else "linear")
    if log_pg:
        fig_pg.update_yaxes(range=list(PCDF_LOG_RANGE), row=1, col=3)
    fig_pg.update_layout(
        title=dict(text=f"{desc} — n={n_pg} samples", font_size=13),
        height=400, showlegend=show_fold,
        margin=dict(t=70, b=40, l=50, r=20),
        legend=dict(x=0.68, y=0.98, font_size=11, bgcolor="rgba(0,0,0,0)"),
        **DARK,
    )
    for ax in ["xaxis", "xaxis2", "xaxis3", "yaxis", "yaxis2", "yaxis3"]:
        fig_pg.update_layout(**{ax: GRID})
    fig_pg.update_xaxes(title_text="x", row=1, col=1)
    fig_pg.update_xaxes(title_text="x", row=1, col=2)
    fig_pg.update_xaxes(title_text="x", row=1, col=3)
    fig_pg.update_yaxes(title_text="Density",  row=1, col=1)
    fig_pg.update_yaxes(title_text="F(x)",     row=1, col=2)
    fig_pg.update_yaxes(title_text="pCDF(x)",  row=1, col=3)
    st.plotly_chart(fig_pg, use_container_width=True)

    st.markdown(
        "**Suggested experiments:**\n"
        "- n = 20: the histogram is strongly bin-width dependent; "
        "the peaked CDF places one step per observation\n"
        "- Student-t, ν = 2, log scale: the power-law tail appears as a straight line "
        "in the peaked CDF; empty histogram bins appear at the same location\n"
        "- Bimodal, narrow the gap: note how many bins are needed before the histogram "
        "resolves both modes compared with the peaked CDF"
    )
    st.divider()

    # ── Key properties ─────────────────────────────────────────────────────────
    st.header("Key Properties")
    col6a, col6b, col6c = st.columns(3, gap="medium")
    with col6a:
        st.markdown(
            "### Histogram\n"
            "| | |\n|---|---|\n"
            "| ✅ | Peaked shape is easy to read |\n"
            "| ❌ | Requires a bin width choice |\n"
            "| ❌ | Aggregates data into bins |\n"
            "| ❌ | Empty bins in sparse tails |\n"
            "| ❌ | Sensitive to count per bin |\n"
        )
    with col6b:
        st.markdown(
            "### CDF\n"
            "| | |\n|---|---|\n"
            "| ✅ | Uses all data exactly |\n"
            "| ✅ | No bin width needed |\n"
            "| ✅ | y-axis is cumulative fraction |\n"
            "| ❌ | Monotone — structure not obvious |\n"
            "| ❌ | Modes appear only as slope changes |\n"
        )
    with col6c:
        st.markdown(
            "### Peaked CDF\n"
            "| | |\n|---|---|\n"
            "| ✅ | Uses all data exactly |\n"
            "| ✅ | No bin width needed |\n"
            "| ✅ | Steep slopes = data concentration |\n"
            "| ✅ | y-axis is a direct probability |\n"
            "| ✅ | Log scale gives smooth tails |\n"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Image Example
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Image Pixel Distributions Across Smoothing Scales")
    st.markdown(
        """
        This example follows the [original notebook](https://github.com/yipihey/SEDist/blob/main/docs/UseInsteadOfHistogram.ipynb)
        in this repository.  An image is smoothed with Gaussian filters of increasing
        width σ (pixels).  As σ grows, fine-grained texture is averaged out and the
        pixel value distribution narrows toward the image mean.

        The peaked CDF traces this narrowing continuously: each curve is computed
        directly from the pixel values at that smoothing scale, with no bin width
        choice, using the `SE_distribution` class from the SEdist package with log
        compression to preserve tail accuracy.
        """
    )

    ctrl_img, disp_img = st.columns([1, 2], gap="medium")

    with ctrl_img:
        uploaded = st.file_uploader(
            "Upload image (optional)", type=["png", "jpg", "jpeg"],
            help="Any grayscale or RGB image; converted to grayscale internally.",
        )
        all_sigmas = [0, 1, 2, 4, 8, 16, 32, 64, 128]
        sigmas = st.multiselect(
            "Smoothing scales σ (pixels)",
            options=all_sigmas,
            default=[0, 1, 4, 16, 64],
        )
        if not sigmas:
            sigmas = [0, 4, 16]
        sigmas = sorted(sigmas)
        log_img  = st.toggle("Log scale on y-axis", value=True, key="img_log")
        show_hist_bg = st.toggle("Show histogram in background", value=True, key="img_hist")

    with disp_img:
        if uploaded is not None:
            from PIL import Image as PILImage
            pil_img  = PILImage.open(uploaded).convert("L")
            base_img = np.array(pil_img, dtype=float) / 255.0
            img_label = uploaded.name
        else:
            base_img, img_label = load_demo_image(size=350)

        base_img = (base_img - base_img.min()) / (base_img.max() - base_img.min() + 1e-12)
        st.image(base_img, caption=f"Input: {img_label}", use_container_width=True, clamp=True)

    # Build smoothed images
    smoothed = {s: gaussian_filter(base_img, sigma=s) if s > 0 else base_img.copy()
                for s in sigmas}

    # Grid of smoothed images
    st.subheader("Smoothed images")
    n_show = min(len(sigmas), 6)
    img_cols = st.columns(n_show)
    for i, (s, im) in enumerate(list(smoothed.items())[:n_show]):
        with img_cols[i]:
            lbl = f"σ = {s} px" if s > 0 else "Original"
            st.image(np.clip(im, 0, 1), caption=f"{lbl}\nstd = {im.std():.3f}", use_container_width=True, clamp=True)

    # Peaked CDF plot — using SE_distribution with log compression if available
    st.subheader("Peaked CDF of pixel values")

    fig_img = go.Figure()
    for i, (s, im) in enumerate(smoothed.items()):
        color = PALETTE[i % len(PALETTE)]
        pixels = im.flatten()
        label  = f"σ = {s}" if s > 0 else "Original"

        # Background histograms
        if show_hist_bg:
            # subsample for histogram speed
            px_sub = pixels[::max(1, len(pixels) // 5000)]
            fig_img.add_trace(go.Histogram(
                x=px_sub, nbinsx=80, histnorm="probability density",
                marker_color=color, opacity=0.10,
                showlegend=False,
            ))

        # Peaked CDF — use SEdist if available for smooth log-scale tails
        if SEDIST_AVAILABLE:
            d = SE_distribution(pixels, compress="log", Ninterpolants=600)
            x_grid = np.linspace(max(0, pixels.min()), min(1, pixels.max()), 1000)
            yp_vals = np.clip(d.pcdf(x_grid), 1e-6, None) if log_img else d.pcdf(x_grid)
            fig_img.add_trace(go.Scatter(
                x=x_grid, y=yp_vals, mode="lines", name=label,
                line=dict(color=color, width=2.5),
            ))
        else:
            px_sub = pixels[::max(1, len(pixels) // 4000)]
            xe_i, ye_i = ecdf(px_sub)
            yp_i = pcdf_from_cdf(ye_i)
            y_plot = np.clip(yp_i, 1e-6, None) if log_img else yp_i
            fig_img.add_trace(go.Scatter(
                x=xe_i, y=y_plot, mode="lines", name=label,
                line=dict(color=color, width=2.5),
            ))

    fig_img.update_layout(
        **base_layout(
            "Peaked CDF of pixel values (log-compressed, one curve per smoothing scale)",
            "Pixel value", "pCDF", height=430, log_y=log_img,
        ),
        showlegend=True,
        xaxis_range=[0, 1],
        legend=dict(x=0.01, y=0.99, font_size=12, bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0"),
    )
    if log_img:
        fig_img.update_yaxes(range=list(PCDF_LOG_RANGE))
    st.plotly_chart(fig_img, use_container_width=True)

    st.markdown(
        """
        Each curve uses the same pixel values as one of the smoothed images above.
        As σ increases the distribution narrows: the curve shifts toward the mean
        pixel value and the tails on log scale become steeper.  No bin width was
        chosen at any point — the log-compressed `SE_distribution` places interpolation
        knots densely in the tails where the log scale expands the axis.
        """
    )

    # Std dev vs smoothing scale
    if len(sigmas) >= 3:
        st.subheader("Pixel standard deviation vs smoothing scale")
        pairs = [(s, smoothed[s].std()) for s in sigmas if s > 0]
        if len(pairs) >= 2:
            sx, sy = zip(*pairs)
            fig_std = go.Figure(go.Scatter(
                x=sx, y=sy, mode="lines+markers",
                line=dict(color=C_PCDF, width=2), marker=dict(size=8),
            ))
            fig_std.update_layout(
                **base_layout(
                    "Pixel std deviation vs Gaussian smoothing scale",
                    "Smoothing scale σ (pixels)", "Pixel std deviation",
                    height=280, log_y=True,
                ),
                xaxis_type="log",
            )
            st.plotly_chart(fig_std, use_container_width=True)
            st.markdown(
                "The standard deviation decreases monotonically with smoothing scale.  "
                "The slope on a log–log plot characterizes the spatial frequency content "
                "of the image: a steeper slope indicates that power falls off faster with "
                "spatial scale (smoother underlying scene)."
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SEdist Package
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("The SEdist Package")
    st.markdown(
        """
        [`SEdist`](https://github.com/yipihey/SEDist) is the Python package this app
        is part of.  It wraps any 1-D array of observations in a
        `scipy.stats.rv_continuous`-compatible object, exposing the full scipy
        distribution interface — `cdf`, `ppf`, `sf`, `pdf`, `rvs` — alongside
        `pcdf` and `logpcdf` for the peaked CDF and its logarithm.

        The main use case is working with large empirical datasets where storing
        all N sample points as CDF interpolation knots is wasteful.  The
        *compression* modes reduce the knot count while preserving accuracy where
        it matters most.
        """
    )
    st.divider()

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.subheader("Installation")
        st.code(
            "pip install git+https://github.com/yipihey/SEDist",
            language="bash",
        )

        st.subheader("Basic usage")
        st.code(
            """
import numpy as np
from SEdist import SE_distribution

data = np.random.normal(0, 1, 10_000)
dist = SE_distribution(data)

# scipy interface
print(dist.mean(), dist.std())
print(dist.ppf([0.1, 0.5, 0.9]))   # quantiles

# Peaked CDF
x = np.linspace(-4, 4, 200)
p  = dist.pcdf(x)       # values in [0, 0.5]
lp = dist.logpcdf(x)    # log of the above

# Convenience plotters (matplotlib)
dist.plotlogpcdf(label="log pCDF")
dist.plotcdf(color="gray")
            """,
            language="python",
        )

    with col_right:
        st.subheader("Compression modes")
        st.markdown(
            """
            For a dataset of N observations, the uncompressed `SE_distribution`
            stores all N values as CDF interpolation knots.  For N = 10⁶ this is
            both memory-intensive and slower to evaluate.

            Two compression strategies reduce the knot count to `Ninterpolants`
            without meaningful accuracy loss:

            | `compress=` | Knot placement |
            |---|---|
            | *(default)* | All N data points |
            | `"linear"` | `Ninterpolants` evenly-spaced quantile points |
            | `"log"` | Log-spaced from each tail toward the center |

            **Log compression** is designed for peaked CDFs on a log y-axis.
            It places the most knots where the log scale expands the axis — in
            the tails — and fewer near the center where density is already well
            resolved.  This preserves tail accuracy with a fraction of the
            original knots.
            """
        )
        st.code(
            """
# Default: all N knots
d_full = SE_distribution(data)

# 500 evenly-spaced quantile knots
d_lin  = SE_distribution(data, compress="linear",
                         Ninterpolants=500)

# 500 knots, dense at tails
d_log  = SE_distribution(data, compress="log",
                         Ninterpolants=500)

print(d_full.N,    d_full.Nfit)   # N,  N
print(d_lin.N,     d_lin.Nfit)    # N,  500
print(d_log.N,     d_log.Nfit)    # N,  ≤ 500
            """,
            language="python",
        )

    st.divider()

    # ── Interactive compression comparison ─────────────────────────────────────
    st.subheader("Compression Mode Comparison")

    if not SEDIST_AVAILABLE:
        st.info(
            "SEdist could not be imported from `src/`.  "
            "Run `pip install -e .` from the repo root to enable this demo."
        )
    else:
        cc1, cc2 = st.columns([1, 2.2], gap="large")
        with cc1:
            comp_dist  = st.selectbox(
                "Distribution",
                ["Normal", "Student-t (ν=3)", "Log-Normal", "Exponential"],
                key="comp_dist",
            )
            comp_n     = st.select_slider(
                "Dataset size N",
                [1_000, 5_000, 10_000, 50_000, 100_000],
                value=10_000,
                key="comp_n",
            )
            comp_nfit  = st.slider("Ninterpolants", 50, 2000, 300, 50, key="comp_nfit")
            comp_log   = st.toggle("Log scale", value=True, key="comp_log")
            comp_zoom  = st.toggle("Zoom into tail (x > median)", value=False, key="comp_zoom")

        with cc2:
            curves = compute_se_curves(comp_dist, comp_n, comp_nfit)
            x_all  = curves["x"]
            median = curves["median"]

            if comp_zoom:
                mask  = x_all > median
                x_plot = x_all[mask]
            else:
                mask  = np.ones(len(x_all), dtype=bool)
                x_plot = x_all

            fig_comp = go.Figure()
            styles = [
                ("full",   f"Full empirical ({curves['full']['nfit']:,} knots)",   "#f87171", "dash",  1.5),
                ("linear", f"Linear ({curves['linear']['nfit']:,} knots)",          "#60a5fa", "solid", 2.5),
                ("log",    f"Log compression ({curves['log']['nfit']:,} knots)",    "#34d399", "solid", 2.5),
            ]
            for key, label, color, dash, width in styles:
                y_vals = curves[key]["pcdf"][mask]
                if comp_log:
                    y_vals = np.clip(y_vals, 1e-7, None)
                fig_comp.add_trace(go.Scatter(
                    x=x_plot, y=y_vals,
                    mode="lines", name=label,
                    line=dict(color=color, dash=dash, width=width),
                ))

            fig_comp.update_layout(
                **base_layout(
                    "pCDF: full empirical vs compressed",
                    "x", "pCDF(x)", height=390, log_y=comp_log,
                ),
                showlegend=True,
                legend=dict(x=0.01, y=0.99, font_size=11,
                            bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0"),
            )
            if comp_log:
                fig_comp.update_yaxes(range=list(PCDF_LOG_RANGE))
            st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown(
            """
            **What to look for:**
            - On a linear y-scale all three modes are visually indistinguishable.
            - Enable log scale and zoom into the tail.  Linear compression distributes
              knots evenly in quantile space; near the tails — where quantiles are
              widely spaced — the interpolation shows staircase artifacts.
            - Log compression concentrates knots in the tails, eliminating the
              staircase while using the same or fewer total knots.
            - For heavy-tailed distributions (Student-t) the difference is most
              pronounced; for light-tailed distributions (Normal) both modes agree
              well even at moderate Ninterpolants.
            """
        )

    st.divider()

    # ── Pyodide ────────────────────────────────────────────────────────────────
    st.subheader("Pyodide / Browser Compatibility")
    st.markdown(
        """
        SEdist depends only on `numpy` and `scipy.stats` / `scipy.interpolate`,
        both of which are available in [Pyodide](https://pyodide.org) — the
        WebAssembly port of CPython.  The package contains no compiled extensions,
        so it installs cleanly via `micropip` in browser-based Python environments.

        This means the app can be deployed with
        [**stlite**](https://github.com/whitphx/stlite) (Streamlit on Pyodide),
        running entirely in the visitor's browser with no server required.  The
        `requirements.txt` in this repository (`numpy`, `scipy`, `plotly`,
        `streamlit`, `pooch`) are all present in the Pyodide package index.

        The image smoothing example requires `scipy.ndimage.gaussian_filter`, also
        available in Pyodide.  The only potential constraint for a fully offline
        stlite deployment is `scipy.datasets.face`, which downloads the sample
        image on first use; a user-uploaded image bypasses that entirely.
        """
    )
    st.divider()

    # ── Reference ──────────────────────────────────────────────────────────────
    st.subheader("Reference")
    st.markdown(
        r"""
        > Berg & Harris (2008) — *Normalizing the Height Distribution of a
        > Stellar Population*, [arXiv:0712.3852](https://arxiv.org/abs/0712.3852)

        The peaked CDF was introduced in an astronomical context but applies
        wherever a histogram would otherwise be used.

        The key formula:
        $$\text{pCDF}(x) = \min\!\bigl(F(x),\; 1-F(x)\bigr)$$

        `SE_distribution` extends `scipy.stats.rv_continuous` to wrap empirical
        data, providing `pcdf`, `logpcdf`, and the compressed interpolation modes
        (`compress="linear"` and `compress="log"`) alongside the full scipy
        distribution interface.
        """
    )
