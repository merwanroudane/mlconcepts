import numpy as np
import plotly.graph_objects as go
import streamlit as st

from lib.ui import (PALETTE, card, en, eq, footer, hero, setup, style_fig,
                    toy_classification, toy_regression)

setup("model")

hero(
    "الصفحة 7 من 18",
    "الموديل والخوارزمية والمعاملات  Model, Algorithm & Parameters",
    "ما هو «الموديل» رياضيًا؟ ولماذا نقول إن التعلّم هو «البحث في فضاء الفرضيات»؟ "
    "هذه الصفحة تضع الأساس الذي تقوم عليه كل الصفحات التالية.",
)

# ---------------------------------------------------------------------------
st.header("1) الموديل = دالة")

card(
    "التعريف الرياضي",
    "الموديل هو ببساطة <b>دالة</b> تأخذ المتغيّرات وتُخرِج تنبؤًا. "
    "كل ما يفعله التدريب هو اختيار أرقام هذه الدالة.",
)

eq(r"\hat{y} \;=\; f_{\boldsymbol{\theta}}(\mathbf{x})",
   "x = متّجه المتغيّرات لعيّنة واحدة،  θ (ثيتا) = مجموعة المعاملات،  ŷ = التنبؤ")

st.markdown(
    "الرمز `θ` (ثيتا) هو الاصطلاح العام لكل معاملات الموديل مجتمعة. "
    f"في الانحدار الخطي {en('θ = (w₁, w₂, …, w_p, b)')}، "
    "وفي شبكة عصبية قد يكون ملايين الأوزان، وفي شجرة قرار هو العتبات والتقسيمات."
)

st.subheader("أبسط مثال: الانحدار الخطي البسيط")
eq(r"\hat{y} = w\,x + b")
st.markdown(
    "- `w` (الوزن / الميل — `weight`, `slope`): كم تتغيّر `ŷ` عند زيادة `x` بوحدة واحدة.\n"
    "- `b` (الثابت / التقاطع — `bias`, `intercept`): قيمة `ŷ` عندما `x = 0`.\n\n"
    "هذان رقمان فقط، لكن كل مفاهيم التعلّم تظهر فيهما."
)

st.subheader("التعميم: الانحدار الخطي المتعدّد")
eq(r"\hat{y} = w_1x_1 + w_2x_2 + \dots + w_px_p + b "
   r"\;=\; \sum_{j=1}^{p} w_j x_j + b \;=\; \mathbf{w}^{\top}\mathbf{x} + b")
st.markdown(
    "الصيغ الثلاث متكافئة تمامًا: الأولى مفصّلة، والثانية بالمجموع، والثالثة بلغة المتّجهات "
    f"({en('vector notation')}) وهي التي تجدها في الكتب والأوراق العلمية. "
    "لا تخف من الرمز الأخير — إنه نفس المعنى باختصار."
)

with st.expander("تذكير سريع: ما معنى wᵀx؟", icon=":material/function:"):
    st.markdown(
        "`wᵀx` يسمّى **الجداء السُّلَّمي** (`dot product`): اضرب كل وزن في متغيّره ثم اجمع.\n\n"
        "مثال: `w = (2, 5, -1)` و `x = (10, 3, 4)` ⟵ "
        "`wᵀx = 2×10 + 5×3 + (−1)×4 = 20 + 15 − 4 = 31`.\n\n"
        "في بايثون: `np.dot(w, x)` أو `w @ x`."
    )

# ---------------------------------------------------------------------------
st.header("2) فضاء الفرضيات — The Hypothesis Space")

card(
    "ما معنى «التعلّم» بدقّة؟",
    "اختيار الخوارزمية = تحديد <b>شكل</b> الدالة المسموح به (كل الخطوط المستقيمة مثلًا). "
    "هذه المجموعة اللانهائية من الدوال الممكنة تُسمّى <b>فضاء الفرضيات</b> "
    "(Hypothesis space, ℋ).<br><br>"
    "والتدريب = <b>البحث داخل هذا الفضاء</b> عن الدالة الوحيدة التي تصغّر الخطأ.",
    "teal",
)

st.markdown("كل خط رمادي أدناه فرضية ممكنة. التدريب يختار واحدًا منها — البرتقالي.")

x, y = toy_regression(n=40, noise=2.5)
rng = np.random.default_rng(4)
fig = go.Figure()
xs = np.linspace(x.min(), x.max(), 30)
for _ in range(28):
    ww, bb = rng.uniform(-0.5, 5), rng.uniform(-8, 18)
    fig.add_trace(go.Scatter(x=xs, y=ww * xs + bb, mode="lines",
                             line=dict(color="rgba(150,160,190,0.35)", width=1.3),
                             showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="البيانات",
                         marker=dict(size=9, color=PALETTE["indigo"],
                                     line=dict(width=1, color="white"))))
cw, cb = np.polyfit(x, y, 1)
fig.add_trace(go.Scatter(x=xs, y=cw * xs + cb, mode="lines", name="الفرضية المختارة",
                         line=dict(color=PALETTE["orange"], width=4)))
style_fig(fig, height=400, xaxis_title="x", yaxis_title="y")
st.plotly_chart(fig, width="stretch")

st.info(
    "**نتيجة مهمة:** إذا كانت العلاقة الحقيقية منحنية، فلن تجدها أبدًا داخل فضاء الخطوط المستقيمة "
    "مهما درّبت. هذا بالضبط هو **التحيّز** (`Bias`) الذي سنفصّله في صفحة مقايضة التحيّز والتباين.",
    icon=":material/search:",
)

# ---------------------------------------------------------------------------
st.header("3) الخوارزمية ≠ الموديل ≠ المعاملات")

st.table({
    "**المفهوم**": ["**الخوارزمية** — Algorithm", "**فضاء الفرضيات** — ℋ",
                    "**المعاملات** — Parameters θ", "**الهايبرباراميترز** — Hyperparameters",
                    "**الموديل المدرَّب** — Fitted model"],
    "**ما هو؟**": ["وصفة البحث عن أفضل معاملات", "كل الدوال الممكنة ضمن الشكل المختار",
                   "أرقام يتعلّمها الموديل من البيانات", "أرقام تختارها أنت قبل التدريب",
                   "الخوارزمية + المعاملات المتعلَّمة"],
    "**مثال**": ["`LinearRegression()`", "كل الخطوط `y = wx + b`", "`w = 2.31`, `b = 4.07`",
                 "`alpha=0.1`, `max_depth=5`", "`y = 2.31x + 4.07`"],
    "**من يحدّده؟**": [":blue-badge[أنت]", ":blue-badge[أنت ضمنيًا]", ":green-badge[البيانات]",
                       ":blue-badge[أنت]", ":green-badge[نتيجة التدريب]"],
}, border="horizontal")

st.code(
    """from sklearn.linear_model import Ridge

# 1) تختار الخوارزمية وتضبط الهايبرباراميترز
model = Ridge(alpha=1.0)        # alpha هايبرباراميتر — أنت حدّدته

# 2) التدريب: الخوارزمية تبحث عن المعاملات
model.fit(X_train, y_train)

# 3) المعاملات المتعلَّمة — لم تكتبها أنت
print(model.coef_)        # الأوزان w
print(model.intercept_)   # الثابت b""",
    language="python",
)

# ---------------------------------------------------------------------------
st.header("4) من الانحدار إلى التصنيف — دالة سيجمويد")

st.markdown(
    "في التصنيف نريد **احتمالًا** بين 0 و1، لكن `wᵀx + b` قد يعطي أي رقم "
    "(‎−∞‎ إلى ‎+∞‎). الحل: نمرّره على دالة تضغطه داخل المجال [0, 1]، "
    f"هي الدالة اللوجستية أو {en('Sigmoid')}:"
)
eq(r"\sigma(z) = \frac{1}{1 + e^{-z}} \qquad \text{حيث} \quad z = \mathbf{w}^{\top}\mathbf{x} + b")
eq(r"P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^{\top}\mathbf{x} + b)")

g1, g2 = st.columns([1, 1.6], gap="medium")
with g1:
    thr = st.slider("عتبة القرار Threshold", 0.05, 0.95, 0.5, 0.05)
    st.markdown(
        f"- `σ(0) = 0.5` دائمًا — نقطة الحياد.\n"
        f"- `z` كبير موجب ⟵ احتمال قريب من 1.\n"
        f"- `z` كبير سالب ⟵ احتمال قريب من 0.\n\n"
        f"القرار النهائي: نُصنّف الفئة 1 إذا كان الاحتمال ≥ **{thr}**.\n\n"
        "العتبة **ليست** جزءًا من الموديل، بل قرار تتخذه أنت حسب تكلفة الخطأ — "
        "وسنفصّلها في صفحة معايير التصنيف."
    )
with g2:
    z = np.linspace(-8, 8, 300)
    sg = 1 / (1 + np.exp(-z))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=z, y=sg, mode="lines", name="σ(z)",
                             line=dict(color=PALETTE["indigo"], width=4)))
    fig.add_hline(y=thr, line=dict(color=PALETTE["orange"], dash="dash", width=2))
    zt = -np.log(1 / thr - 1)
    fig.add_vline(x=zt, line=dict(color=PALETTE["orange"], dash="dot", width=2))
    fig.add_trace(go.Scatter(x=[zt], y=[thr], mode="markers", name="نقطة القرار",
                             marker=dict(size=14, color=PALETTE["orange"],
                                         line=dict(width=2, color="white"))))
    style_fig(fig, height=350, xaxis_title="z = wᵀx + b", yaxis_title="الاحتمال P(y=1)")
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
st.header("5) عائلات الموديلات — كيف تختلف فضاءات الفرضيات؟")

st.markdown("نفس البيانات، أربعة فضاءات فرضيات مختلفة، أربعة حدود قرار مختلفة تمامًا:")

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

Xc, yc = toy_classification(n=240, sep=1.3, seed=9)


@st.cache_data
def boundary(name: str):
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=0),
        "KNN (k=15)": KNeighborsClassifier(15),
        "Random Forest": RandomForestClassifier(n_estimators=120, random_state=0),
    }
    m = models[name].fit(Xc, yc)
    gx = np.linspace(Xc[:, 0].min() - .6, Xc[:, 0].max() + .6, 140)
    gy = np.linspace(Xc[:, 1].min() - .6, Xc[:, 1].max() + .6, 140)
    GX, GY = np.meshgrid(gx, gy)
    Z = m.predict_proba(np.c_[GX.ravel(), GY.ravel()])[:, 1].reshape(GX.shape)
    return gx, gy, Z


algo = st.segmented_control(
    "اختر الخوارزمية",
    ["Logistic Regression", "Decision Tree", "KNN (k=15)", "Random Forest"],
    default="Logistic Regression",
)
gx, gy, Z = boundary(algo)
fig = go.Figure()
fig.add_trace(go.Heatmap(x=gx, y=gy, z=Z, zmin=0, zmax=1, showscale=False, opacity=0.45,
                         colorscale=[[0, "#C9D4F5"], [0.5, "#FFFFFF"], [1, "#F5C5DC"]]))
fig.add_trace(go.Contour(x=gx, y=gy, z=Z, contours=dict(start=.5, end=.5, size=.1,
                                                        coloring="none"),
                         line=dict(color=PALETTE["ink"], width=2.5), showscale=False,
                         name="حدّ القرار"))
for k, (nm, col) in enumerate([("الفئة 0", PALETTE["indigo"]), ("الفئة 1", PALETTE["pink"])]):
    m = yc == k
    fig.add_trace(go.Scatter(x=Xc[m, 0], y=Xc[m, 1], mode="markers", name=nm,
                             marker=dict(size=8, color=col, line=dict(width=1, color="white"))))
style_fig(fig, height=430, xaxis_title="المتغيّر 1", yaxis_title="المتغيّر 2")
st.plotly_chart(fig, width="stretch")

d = {
    "Logistic Regression": "فضاء الفرضيات = **المستويات الفائقة** (خطوط مستقيمة). "
                           "بسيط ومفسَّر جدًا، لكنه عاجز عن الحدود المنحنية.",
    "Decision Tree": "فضاء الفرضيات = **المستطيلات المتعامدة**. الحدّ دائمًا «درجات سلّم» "
                     "لأنها تقسّم على عمود واحد في كل مرة.",
    "KNN (k=15)": "لا يوجد تدريب حقيقي! يحفظ البيانات، وعند التنبؤ يصوّت أقرب 15 جار. "
                  "الحدّ مرن جدًا ويتبع شكل البيانات.",
    "Random Forest": "مئات الأشجار تصوّت معًا. الحدّ أنعم من شجرة واحدة وأكثر استقرارًا، "
                     "وهو الخيار الافتراضي القوي للبيانات الجدولية.",
}
st.info(d[algo], icon=":material/insights:")

st.success(
    "**الخلاصة التي تربط كل شيء:** اختيار الخوارزمية = اختيار **نوع الأشكال** التي "
    "يُسمح للموديل برسمها. التدريب = اختيار **الشكل الأفضل** ضمنها. "
    "والسؤال «أي شكل أفضل؟» يحتاج مقياسًا رقميًا للخطأ — وهذا بالضبط موضوع الصفحة التالية.",
    icon=":material/check_circle:",
)

st.divider()
st.markdown(
    "#### :material/arrow_circle_left: الخطوة التالية\n"
    "**دالة الخسارة ودالة التكلفة**: كيف نحوّل «جودة الموديل» إلى رقم واحد نستطيع تصغيره."
)
footer()
