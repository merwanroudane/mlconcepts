import numpy as np
import plotly.graph_objects as go
import streamlit as st

from lib.ui import PALETTE, card, en, eq, footer, hero, setup, style_fig, toy_curve

setup("regularization")

hero(
    "الصفحة 13 من 18",
    "التنظيم  Regularization",
    "كيف نمنع الموديل من الحفظ دون أن نغيّر الخوارزمية؟ الجواب: نضيف **عقوبة** على التعقيد "
    "داخل دالة التكلفة نفسها، فيصبح التعقيد مكلفًا ولا يختاره الموديل إلا إذا كان يستحقّ فعلًا.",
)

# ---------------------------------------------------------------------------
st.header("1) الفكرة الأساسية")

card(
    "التعديل على دالة التكلفة",
    "بدل أن نطلب من الموديل «صغّر الخطأ فقط»، نطلب منه: "
    "«<b>صغّر الخطأ + عقوبة على كِبَر المعاملات</b>».<br><br>"
    "لماذا كِبَر المعاملات؟ لأن الموديل المُفرِط يحتاج معاملات ضخمة ليلتوي ويمرّ بكل نقطة. "
    "فإذا منعنا المعاملات من الكِبَر، منعنا الالتواء المفرط تلقائيًا.",
)

eq(r"J_{\text{reg}}(\boldsymbol{\theta}) \;=\; "
   r"\underbrace{\frac{1}{n}\sum_{i=1}^{n} L(y_i, \hat{y}_i)}_{\text{الخطأ على البيانات}} \;+\; "
   r"\underbrace{\lambda \cdot \Omega(\boldsymbol{\theta})}_{\text{عقوبة التعقيد}}")

st.markdown(
    "- `Ω(θ)` = مقياس «حجم» المعاملات (العقوبة).\n"
    "- `λ` (لامدا) = **قوّة التنظيم** — هايبرباراميتر تختاره أنت. "
    f"في `scikit-learn` يُسمّى غالبًا {en('alpha')}.\n\n"
    "**ملاحظة في غاية الأهمية:** العقوبة تُطبَّق على الأوزان `w` فقط، **وليس** على الثابت `b`، "
    "لأن الثابت مجرّد إزاحة عمودية ولا علاقة له بتعقيد الشكل."
)

# ---------------------------------------------------------------------------
st.header("2) النوعان الأساسيان: Ridge و Lasso")

r1, r2 = st.columns(2, gap="medium")
with r1:
    with st.container(border=True):
        st.markdown("##### :material/water_drop: Ridge — تنظيم L2")
        st.latex(r"\Omega = \sum_{j=1}^{p} w_j^{2} \;=\; \|\mathbf{w}\|_2^2")
        st.markdown(
            "- **يُصغّر** كل المعاملات نحو الصفر بشكل متناسب.\n"
            "- **لا يصل بها إلى الصفر تمامًا** أبدًا.\n"
            "- يحتفظ بكل المتغيّرات في الموديل.\n"
            "- ممتاز عند وجود ارتباط بين المتغيّرات (`Multicollinearity`): "
            "يوزّع التأثير بينها بدل أن يتذبذب.\n\n"
            "`Ridge(alpha=1.0)`")
with r2:
    with st.container(border=True):
        st.markdown("##### :material/content_cut: Lasso — تنظيم L1")
        st.latex(r"\Omega = \sum_{j=1}^{p} \big|w_j\big| \;=\; \|\mathbf{w}\|_1")
        st.markdown(
            "- **يدفع** المعاملات الضعيفة إلى **الصفر تمامًا**.\n"
            "- يقوم بـ **اختيار آلي للمتغيّرات** (`Feature selection`).\n"
            "- ينتج موديلًا **متناثرًا** (`Sparse`) وأسهل تفسيرًا.\n"
            "- مع متغيّرات مترابطة يختار واحدًا منها عشوائيًا ويلغي الباقي.\n\n"
            "`Lasso(alpha=0.1)`")

st.markdown("**وهناك ثالث يجمع بينهما:**")
eq(r"\Omega_{\text{ElasticNet}} = \alpha\Big[\, r\sum_j |w_j| \;+\; "
   r"\tfrac{1-r}{2}\sum_j w_j^2 \,\Big]",
   "r = l1_ratio يتحكّم في المزج: r=1 يعطي Lasso و r=0 يعطي Ridge")

# ---------------------------------------------------------------------------
st.header("3) لماذا يُصفّر Lasso ولا يُصفّر Ridge؟ — الشرح الهندسي")

st.markdown(
    "هذا سؤال يُربك كثيرين، والجواب جميل وبصري. "
    "القيود الهندسية للعقوبتين لهما **شكلان مختلفان**:"
)

g1, g2 = st.columns(2, gap="medium")
th = np.linspace(0, 2 * np.pi, 300)
for cont, title, shape_x, shape_y, note, col in [
    (g1, "Ridge (L2): دائرة", np.cos(th), np.sin(th),
     "محيط الدائرة **ناعم** بلا زوايا. نقطة التماس غالبًا في مكان لا يكون فيه أي معامل = 0.",
     PALETTE["indigo"]),
    (g2, "Lasso (L1): مُعيَّن", np.r_[1, 0, -1, 0, 1], np.r_[0, 1, 0, -1, 0],
     "للمُعيَّن **زوايا حادّة على المحاور**. نقطة التماس تقع غالبًا على زاوية — "
     "وعندها أحد المعاملين = 0 بالضبط!", PALETTE["pink"]),
]:
    with cont:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=shape_x, y=shape_y, mode="lines", name="منطقة القيد",
                                 fill="toself", fillcolor="rgba(91,111,214,0.12)",
                                 line=dict(color=col, width=3)))
        c0 = (1.75, 1.35)
        for r in (0.75, 1.1, 1.45):
            fig.add_trace(go.Scatter(x=c0[0] + r * np.cos(th) * 1.25,
                                     y=c0[1] + r * np.sin(th) * 0.8,
                                     mode="lines", showlegend=False,
                                     line=dict(color="#D9A6BE", width=1.6)))
        fig.add_trace(go.Scatter(x=[c0[0]], y=[c0[1]], mode="markers+text",
                                 text=["الحل بلا تنظيم (OLS)"], textposition="top center",
                                 name="OLS", marker=dict(size=11, color=PALETTE["orange"])))
        tp = (0.0, 1.0) if "Lasso" in title else (0.62, 0.78)
        fig.add_trace(go.Scatter(x=[tp[0]], y=[tp[1]], mode="markers+text",
                                 text=["★ الحل المنظَّم"], textposition="bottom center",
                                 name="الحل", marker=dict(size=15, color=PALETTE["red"],
                                                          symbol="star")))
        style_fig(fig, height=360, legend_top=False, title=title,
                  xaxis_title="w₁", yaxis_title="w₂")
        fig.update_xaxes(range=[-1.6, 3.2], zeroline=True, zerolinewidth=2)
        fig.update_yaxes(range=[-1.6, 2.6], zeroline=True, zerolinewidth=2, scaleanchor="x")
        st.plotly_chart(fig, width="stretch")
        st.caption(note)

st.info(
    "**الخلاصة:** الحل المنظَّم هو أول نقطة تلامس فيها حلقات الخطأ منطقةَ القيد. "
    "زوايا المُعيَّن تقع على المحاور، والمحور يعني «هذا المعامل يساوي صفرًا» — "
    "ولهذا يُصفّر Lasso معاملات، بينما دائرة Ridge الملساء لا تفعل.",
    icon=":material/lightbulb:",
)

# ---------------------------------------------------------------------------
st.header("4) تجربة: أثر λ على الموديل")

x, y = toy_curve(n=28, noise=0.4, seed=3)
from sklearn.linear_model import Lasso, LinearRegression, Ridge  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # noqa: E402

cA, cB = st.columns([1, 2], gap="medium")
with cA:
    kind = st.segmented_control("نوع التنظيم", ["بلا تنظيم", "Ridge (L2)", "Lasso (L1)"],
                                default="بلا تنظيم")
    log_lam = st.slider("قوّة التنظيم  log₁₀(λ)", -4.0, 2.0, -2.0, 0.25,
                        disabled=(kind == "بلا تنظيم"))
    lam = 10 ** log_lam
    deg = st.slider("درجة كثير الحدود", 3, 15, 12)
    if kind != "بلا تنظيم":
        st.metric("λ (alpha)", f"{lam:.5g}")
    st.caption(
        "λ صغير جدًا ⟵ كأنه بلا تنظيم (إفراط).\n\n"
        "λ كبير جدًا ⟵ كل المعاملات ≈ 0 والموديل يصبح خطًا أفقيًا (تقصير شديد)."
    )

est = {"بلا تنظيم": LinearRegression(),
       "Ridge (L2)": Ridge(alpha=lam),
       "Lasso (L1)": Lasso(alpha=lam, max_iter=20000)}[kind]
pipe = make_pipeline(PolynomialFeatures(deg), StandardScaler(), est)
pipe.fit(x.reshape(-1, 1), y)
xs = np.linspace(x.min() - .3, x.max() + .3, 300)

with cB:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="البيانات",
                             marker=dict(size=10, color=PALETTE["indigo"],
                                         line=dict(width=1.5, color="white"))))
    fig.add_trace(go.Scatter(x=xs, y=np.sin(xs) + 0.15 * xs**2, mode="lines",
                             name="الحقيقة f", line=dict(color=PALETTE["muted"],
                                                          width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=xs, y=pipe.predict(xs.reshape(-1, 1)), mode="lines",
                             name=f"الموديل ({kind})",
                             line=dict(color=PALETTE["orange"], width=3.5)))
    style_fig(fig, height=400, xaxis_title="x", yaxis_title="y")
    fig.update_yaxes(range=[min(y) - 2, max(y) + 2])
    st.plotly_chart(fig, width="stretch")

coefs = pipe[-1].coef_.ravel()
nz = int(np.sum(np.abs(coefs) > 1e-6))
k1, k2, k3 = st.columns(3)
k1.metric("أكبر معامل |w|", f"{np.abs(coefs).max():.3g}")
k2.metric("مجموع |w|", f"{np.abs(coefs).sum():.3g}")
k3.metric("معاملات غير صفرية", f"{nz} من {len(coefs)}")

fig = go.Figure(go.Bar(x=[f"w{i}" for i in range(len(coefs))], y=coefs,
                       marker_color=[PALETTE["pink"] if abs(c) < 1e-6 else PALETTE["indigo"]
                                     for c in coefs]))
style_fig(fig, height=280, legend_top=False, xaxis_title="المعامل",
          yaxis_title="القيمة", title="قيم المعاملات المتعلَّمة")
st.plotly_chart(fig, width="stretch")
if kind == "Lasso (L1)":
    st.caption("الأعمدة الوردية = معاملات صُفِّرت تمامًا بواسطة Lasso (حُذف المتغيّر فعليًا).")

# ---------------------------------------------------------------------------
st.header("5) مسار المعاملات — Regularization Path")

st.markdown(
    "رسم يبيّن كيف تنكمش المعاملات كلّما زاد λ. "
    "هو أداة تشخيص ممتازة لاختيار λ ولفهم أي المتغيّرات أكثر صمودًا."
)

rng = np.random.default_rng(3)
Xm = rng.normal(0, 1, (120, 8))
true_w = np.array([3.0, -2.2, 1.5, 0.0, 0.0, 0.8, 0.0, -0.4])
ym = Xm @ true_w + rng.normal(0, 1.0, 120)

lams = np.logspace(-3, 2.2, 60)
path_kind = st.segmented_control("الخوارزمية", ["Ridge", "Lasso"], default="Lasso",
                                 key="pathkind")
paths = []
for L in lams:
    m = (Ridge(alpha=L) if path_kind == "Ridge" else Lasso(alpha=L, max_iter=20000)).fit(Xm, ym)
    paths.append(m.coef_)
paths = np.array(paths)

fig = go.Figure()
cols = [PALETTE[k] for k in ("indigo", "pink", "teal", "orange", "violet",
                             "sky", "lime", "red")]
for j in range(paths.shape[1]):
    fig.add_trace(go.Scatter(x=lams, y=paths[:, j], mode="lines",
                             name=f"w{j + 1} (الحقيقي {true_w[j]:+.1f})",
                             line=dict(color=cols[j], width=3)))
fig.add_hline(y=0, line=dict(color=PALETTE["ink"], width=1.5))
style_fig(fig, height=430, xaxis_title="λ (مقياس لوغاريتمي) →",
          yaxis_title="قيمة المعامل")
fig.update_xaxes(type="log")
st.plotly_chart(fig, width="stretch")

st.markdown(
    f"**اقرأ الرسم:** على اليسار (λ صغير) المعاملات قريبة من قيمها الحقيقية. "
    f"كلّما تحرّكنا يمينًا تنكمش كلها نحو الصفر. "
    + ("مع **Lasso** لاحظ أن المعاملات تصل إلى الصفر **بالضبط** وتبقى هناك — "
       "وأول من يصل هي المعاملات الحقيقية الصفرية (w4, w5, w7). "
       "هذا هو الاختيار الآلي للمتغيّرات."
       if path_kind == "Lasso" else
       "مع **Ridge** تقترب المعاملات من الصفر بسلاسة لكن **لا تصل إليه أبدًا**.")
)

# ---------------------------------------------------------------------------
st.header("6) أشكال أخرى من التنظيم")

o1, o2, o3 = st.columns(3, gap="medium")
with o1:
    with st.container(border=True):
        st.markdown("##### :material/timer_off: Early stopping")
        st.markdown(
            "نوقف التدريب عند أدنى نقطة في خسارة التحقّق. "
            "تنظيم «مجاني» بلا أي صيغة رياضية إضافية، ومستعمل في كل التعلّم العميق.")
    with st.container(border=True):
        st.markdown("##### :material/content_copy: Data augmentation")
        st.markdown(
            "توليد عيّنات جديدة بتحويلات على الموجودة (تدوير صورة، إضافة ضوضاء). "
            "يزيد البيانات فعليًا وهو أقوى علاج للإفراط.")
with o2:
    with st.container(border=True):
        st.markdown("##### :material/grid_off: Dropout")
        st.markdown(
            "في الشبكات العصبية: نُطفئ عشوائيًا نسبة من الخلايا في كل خطوة تدريب "
            "(مثلًا 50%)، فلا تستطيع أي خلية الاعتماد على غيرها، وتتعلّم الشبكة تمثيلات متينة.")
    with st.container(border=True):
        st.markdown("##### :material/park: تقليم الأشجار Pruning")
        st.markdown(
            "تحديد `max_depth` أو `min_samples_leaf` أو `ccp_alpha` "
            "يمنع الشجرة من النمو حتى حفظ كل عيّنة.")
with o3:
    with st.container(border=True):
        st.markdown("##### :material/forest: التجميع Ensembling")
        st.markdown(
            "متوسط عدّة موديلات يقلّل التباين (Bagging، Random Forest). "
            "وهو شكل غير مباشر لكنه فعّال جدًا من التنظيم.")
    with st.container(border=True):
        st.markdown("##### :material/layers: Batch Normalization")
        st.markdown(
            "يعيّر مخرجات الطبقات أثناء التدريب، فيستقرّ التدريب "
            "ويضيف أثرًا تنظيميًا خفيفًا.")

st.code(
    """from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
import numpy as np

# اختيار λ آليًا بالتحقّق المتقاطع — الطريقة الصحيحة
ridge = RidgeCV(alphas=np.logspace(-4, 2, 50), cv=5).fit(X_train, y_train)
print("أفضل alpha:", ridge.alpha_)

lasso = LassoCV(alphas=np.logspace(-4, 1, 50), cv=5, max_iter=10000).fit(X_train, y_train)
print("المتغيّرات المختارة:", np.sum(lasso.coef_ != 0), "من", X_train.shape[1])""",
    language="python",
)

st.error(
    "**لا تنسَ التقييس!** التنظيم يعاقب المعاملات حسب حجمها، "
    "والمعامل يكون كبيرًا إذا كان متغيّره بوحدة صغيرة (مثلًا بالأمتار مقابل الكيلومترات). "
    "بدون `StandardScaler` تكون العقوبة **غير عادلة** بين المتغيّرات، والنتيجة خاطئة. "
    "القاعدة: التنظيم ← يسبقه تقييس، دائمًا.",
    icon=":material/straighten:",
)

st.divider()
st.markdown(
    "#### :material/arrow_circle_left: الخطوة التالية\n"
    "ندخل قسم **التقييم**: كيف نقيس الأداء بالأرقام، بدءًا بمعايير الانحدار."
)
footer()
