"""مساعدات مشتركة للعرض والتنسيق (RTL، بطاقات المفاهيم، ألوان الرسوم)."""

from __future__ import annotations

import re

import numpy as np
import streamlit as st

AUTHOR = "د. مروان رودان  ·  Dr Merwan Roudane"

# لوحة ألوان فاتحة موحّدة لكل الرسوم
PALETTE = {
    "indigo": "#5B6FD6",
    "teal": "#2FA98C",
    "orange": "#EF9036",
    "pink": "#D7568C",
    "violet": "#8A6FD1",
    "sky": "#4ABEDC",
    "lime": "#9CCB5A",
    "red": "#E05C6E",
    "yellow": "#E8C04B",
    "ink": "#1F2A44",
    "muted": "#6B7694",
    "grid": "#E3E9F6",
    "paper": "#FBFCFF",
    "card": "#F3F6FD",
}

SEQ = [
    "#F6F9FE", "#E4ECFB", "#CFDDF7", "#B5CAF1", "#97B4E9",
    "#7A9CE0", "#6183D4", "#4C6AC2", "#3A52A5", "#2A3B7E",
]

_CSS = """
<style>
/* ===== اتجاه الصفحة من اليمين إلى اليسار + السايد بار على اليمين ===== */
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
.stApp,
[data-testid="stSidebar"],
[data-testid="stHeader"],
[data-testid="stSidebarContent"],
[data-testid="stMain"],
[data-testid="stBottomBlockContainer"] {
    direction: rtl;
}

[data-testid="stAppViewContainer"] > section:first-child,
[data-testid="stAppViewContainer"] { text-align: right; }

/* ملاحظة: direction:rtl وحدها تقلب ترتيب الـ flex فينتقل السايد بار إلى اليمين.
   إضافة row-reverse هنا تلغي المفعول، لذا لا تُستعمل. */

/* مقبض تصغير السايد بار */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"] { direction: ltr; }

/* ===== استثناءات تبقى من اليسار إلى اليمين ===== */
code, pre, .stCode, [data-testid="stCode"], [data-testid="stCodeBlock"],
[data-testid="stJson"], .katex, .katex-display,
.js-plotly-plot, .plot-container, [data-testid="stPlotlyChart"],
[data-testid="stDataFrame"], [data-testid="stTable"],
[data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
    direction: ltr;
}
code, pre, [data-testid="stCode"] { text-align: left; }

/* ===== عناوين وأقسام ===== */
h1, h2, h3, h4, h5, h6 { letter-spacing: 0 !important; line-height: 1.55 !important; }
p, li { line-height: 2.0 !important; }
li { margin-bottom: .25rem; }

/* شريط ملوّن لطيف أعلى الصفحة */
.mr-hero {
    background: linear-gradient(110deg, #EEF3FF 0%, #F4EEFF 45%, #E9FAF6 100%);
    border: 1px solid #DDE5F7;
    border-radius: 18px;
    padding: 22px 26px 18px 26px;
    margin-bottom: 14px;
}
.mr-hero h1 { margin: 0 0 6px 0; font-size: 30px; color: #24305A; }
.mr-hero p  { margin: 0; color: #4A5578; font-size: 17px; line-height: 1.9 !important; }
.mr-kicker {
    display: inline-block; background: #FFFFFFCC; border: 1px solid #DDE5F7;
    color: #5B6FD6; font-weight: 700; font-size: 13px;
    padding: 3px 12px; border-radius: 999px; margin-bottom: 10px;
}

/* بطاقة مفهوم */
.mr-card {
    background: #FFFFFF; border: 1px solid #E1E8F7; border-right: 5px solid #5B6FD6;
    border-radius: 14px; padding: 14px 18px; margin: 10px 0;
}
.mr-card .t { font-weight: 700; color: #24305A; font-size: 18px; margin-bottom: 4px; }
.mr-card .b { color: #46506E; font-size: 16px; line-height: 1.95; }
.mr-card.teal   { border-right-color: #2FA98C; }
.mr-card.orange { border-right-color: #EF9036; }
.mr-card.pink   { border-right-color: #D7568C; }
.mr-card.violet { border-right-color: #8A6FD1; }

/* صندوق صيغة رياضية */
.mr-eq {
    background: #F7F9FF; border: 1px dashed #C9D6F2; border-radius: 14px;
    padding: 6px 14px 2px 14px; margin: 8px 0 12px 0;
}

/* مصطلح إنجليزي */
.mr-en {
    font-family: 'JetBrains Mono', monospace; direction: ltr; display: inline-block;
    background: #EEF2FC; color: #3A4B86; border: 1px solid #DCE4F6;
    border-radius: 8px; padding: 0 7px; font-size: 14px; font-weight: 500;
}

/* شريط خطوات */
.mr-steps { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 4px 0; }
.mr-step {
    background: #FFFFFF; border: 1px solid #E1E8F7; border-radius: 999px;
    padding: 5px 14px; font-size: 14px; color: #46506E; font-weight: 600;
}
.mr-step.on { background: #5B6FD6; color: #FFFFFF; border-color: #5B6FD6; }

/* التذييل */
.mr-foot {
    text-align: center; color: #8A93AC; font-size: 14px;
    border-top: 1px solid #E6EBF7; margin-top: 26px; padding-top: 12px;
}

/* تبويبات أوضح */
[data-testid="stTabs"] button p { font-size: 16px; font-weight: 600; }
</style>
"""


def setup(title: str, icon: str = ":material/school:") -> None:
    """تهيئة الصفحة + حقن تنسيقات RTL. تُنادى في بداية كل صفحة."""
    if not st.session_state.get("_page_cfg"):
        st.set_page_config(
            page_title="منصة تعلّم الآلة — Machine Learning من الصفر",
            page_icon=":material/model_training:",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        st.session_state["_page_cfg"] = True
    st.html(_CSS)


def hero(kicker: str, title: str, subtitle: str) -> None:
    subtitle = _md_code_to_html(subtitle)
    st.html(
        f"""<div class="mr-hero">
              <span class="mr-kicker">{kicker}</span>
              <h1>{title}</h1>
              <p>{subtitle}</p>
            </div>"""
    )


def _md_code_to_html(text: str) -> str:
    """تحويل ماركداون مبسّط (`code` و **bold**) إلى HTML داخل البطاقات."""
    text = re.sub(r"`([^`]+)`", r'<span class="mr-en"></span>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b></b>", text)
    return text


def card(title: str, body: str, color: str = "") -> None:
    body = _md_code_to_html(body)
    st.html(
        f"""<div class="mr-card {color}">
              <div class="t">{title}</div>
              <div class="b">{body}</div>
            </div>"""
    )


def en(term: str) -> str:
    """مصطلح إنجليزي داخل نص عربي.

    يُرجع صيغة ماركداون (كود سطري) فتعمل في كل عناصر Streamlit
    بما فيها st.info / st.warning التي لا تقبل HTML.
    داخل بطاقات HTML تُحوَّل تلقائيًا عبر ``_md_code_to_html``.
    """
    return f"`{term}`"


def eq(latex: str, caption: str = "") -> None:
    """صيغة رياضية داخل صندوق بحدود."""
    with st.container(border=True):
        st.latex(latex)
        if caption:
            st.caption(caption)


def steps_bar(labels: list[str], active: int = -1) -> None:
    chips = "".join(
        f'<span class="mr-step{" on" if i == active else ""}">{i + 1}. {t}</span>'
        for i, t in enumerate(labels)
    )
    st.html(f'<div class="mr-steps">{chips}</div>')


def footer() -> None:
    st.html(
        f'<div class="mr-foot">منصة تعلّم الآلة التفاعلية &nbsp;·&nbsp; إعداد {AUTHOR}</div>'
    )


# ----------------------------------------------------------------------------
# إعدادات موحّدة لرسوم plotly بألوان فاتحة
# ----------------------------------------------------------------------------
def style_fig(fig, height: int = 420, legend_top: bool = True, **kwargs):
    fig.update_layout(
        height=height,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FDFEFF",
        font=dict(family="Cairo, sans-serif", size=14, color=PALETTE["ink"]),
        margin=dict(l=55, r=25, t=50, b=50),
        colorway=[PALETTE[k] for k in ("indigo", "teal", "orange", "pink", "violet", "sky", "lime")],
        hoverlabel=dict(font=dict(family="Cairo, sans-serif", size=13), bgcolor="white"),
        **kwargs,
    )
    if legend_top:
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        bgcolor="rgba(255,255,255,0.6)", bordercolor="#E3E9F6", borderwidth=1)
        )
    fig.update_xaxes(gridcolor=PALETTE["grid"], zerolinecolor="#CBD5EC", linecolor="#DDE4F3")
    fig.update_yaxes(gridcolor=PALETTE["grid"], zerolinecolor="#CBD5EC", linecolor="#DDE4F3")
    return fig


def play_buttons(fig, frame_ms: int = 300, label_play: str = "تشغيل", label_pause: str = "إيقاف"):
    """إضافة أزرار تشغيل/إيقاف لرسم متحرّك."""
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                showactive=False,
                x=0.0, y=1.18, xanchor="left", yanchor="top",
                pad=dict(r=6, t=4),
                bgcolor="#FFFFFF",
                bordercolor="#D6DEF2",
                font=dict(family="Cairo, sans-serif", size=13, color="#3A4B86"),
                buttons=[
                    dict(
                        label="▶  " + label_play,
                        method="animate",
                        args=[None, dict(frame=dict(duration=frame_ms, redraw=True),
                                         fromcurrent=True, transition=dict(duration=0))],
                    ),
                    dict(
                        label="❙❙  " + label_pause,
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False),
                                           mode="immediate", transition=dict(duration=0))],
                    ),
                ],
            )
        ]
    )
    return fig


# ----------------------------------------------------------------------------
# بيانات اصطناعية مشتركة
# ----------------------------------------------------------------------------
@st.cache_data
def toy_regression(n: int = 60, noise: float = 2.0, seed: int = 7):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 10, n))
    y = 2.3 * x + 4.0 + rng.normal(0, noise, n)
    return x, y


@st.cache_data
def toy_curve(n: int = 40, noise: float = 0.25, seed: int = 3):
    """دالة غير خطية لتوضيح الـ overfitting."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-3, 3, n))
    y = np.sin(x) + 0.15 * x**2 + rng.normal(0, noise, n)
    return x, y


@st.cache_data
def toy_classification(n: int = 200, sep: float = 1.6, seed: int = 11):
    rng = np.random.default_rng(seed)
    n0 = n // 2
    a = rng.normal([-sep, -sep * 0.6], [1.0, 1.0], size=(n0, 2))
    b = rng.normal([sep, sep * 0.6], [1.0, 1.0], size=(n - n0, 2))
    X = np.vstack([a, b])
    y = np.r_[np.zeros(n0), np.ones(n - n0)].astype(int)
    return X, y
