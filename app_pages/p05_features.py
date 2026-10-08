import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from lib.ui import PALETTE, card, en, eq, footer, hero, setup, style_fig

setup("features")

hero(
    "الصفحة 5 من 18",
    "هندسة المتغيّرات والتقييس  Feature Engineering & Scaling",
    "الموديل لا يفهم إلا الأرقام، ولا يفهمها إلا إذا كانت على مقياس متقارب. "
    "هنا نتعلّم كيف نحوّل الفئات إلى أرقام، وكيف ولماذا نقيّس الأعمدة، ومتى نشتقّ متغيّرات جديدة.",
)

# ---------------------------------------------------------------------------
st.header("1) ترميز المتغيّرات الفئوية — Encoding")

e1, e2 = st.tabs([":material/grid_on: One-Hot Encoding", ":material/sort: Ordinal Encoding"])

with e1:
    st.markdown(
        "لكل فئة عمود جديد يأخذ 0 أو 1. يُستعمل للمتغيّرات **الاسمية** (بلا ترتيب)."
    )
    a, b = st.columns(2, gap="medium")
    with a:
        st.markdown("**قبل**")
        st.dataframe(pd.DataFrame({"المدينة": ["الجزائر", "وهران", "عنابة", "الجزائر"]}),
                     hide_index=True, width="stretch")
    with b:
        st.markdown("**بعد**")
        st.dataframe(pd.DataFrame({
            "مدينة_الجزائر": [1, 0, 0, 1],
            "مدينة_وهران": [0, 1, 0, 0],
            "مدينة_عنابة": [0, 0, 1, 0],
        }), hide_index=True, width="stretch")
    st.code(
        """from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(
    drop="first",            # نحذف عمودًا لتفادي فخّ المتغيّر الوهمي
    handle_unknown="ignore", # فئة جديدة في الاختبار؟ لا تنهار
    sparse_output=False,
)
enc.fit(X_train[["المدينة"]])""",
        language="python",
    )
    st.warning(
        f"**فخّ المتغيّر الوهمي** ({en('Dummy variable trap')}): إذا احتفظت بكل الأعمدة "
        "فمجموعها دائمًا 1، فيصبح أحدها مُستنتَجًا من البقية وتنهار استقرارية الانحدار الخطي. "
        "لذلك `drop='first'` مع الموديلات الخطية. (الأشجار لا تتأثر).",
        icon=":material/warning:",
    )
    st.error(
        "**مشكلة الكاردينالية العالية:** عمود فيه 5000 فئة (رقم المنتج مثلًا) سيُنتج 5000 عمود! "
        "البدائل: تجميع الفئات النادرة في «أخرى»، أو `Target Encoding` "
        "(استبدال الفئة بمتوسط الهدف فيها — بحذر شديد من التسريب)، أو `Embeddings`.",
        icon=":material/expand:",
    )

with e2:
    st.markdown("نُعطي أرقامًا **تحترم الترتيب الطبيعي**. يُستعمل للمتغيّرات الترتيبية فقط.")
    st.dataframe(pd.DataFrame({
        "الحالة": ["متوسط", "جيد", "ممتاز"], "الترميز": [0, 1, 2],
    }), hide_index=True, width="stretch")
    st.code(
        """from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder(categories=[["متوسط", "جيد", "ممتاز"]])  # الترتيب صريح!""",
        language="python",
    )
    st.caption("تحديد `categories` يدويًا مهم، وإلا رتّبها أبجديًا وأفسد المعنى.")

# ---------------------------------------------------------------------------
st.header("2) التقييس — Feature Scaling")

card(
    "لماذا؟ المشكلة في سطر",
    "تخيّل عمودين: المساحة بالمتر (40 إلى 400) والغرف (1 إلى 8). "
    "أي خوارزمية تحسب <b>مسافات</b> أو تتحرّك بخطوات متساوية في كل الاتجاهات "
    "ستعتبر المساحة أهم بخمسين مرّة لمجرّد أن أرقامها أكبر — وهذا خطأ محض في وحدة القياس وليس في المعنى.",
    "orange",
)

st.subheader("الطريقتان الأساسيتان")

s1, s2 = st.columns(2, gap="medium")
with s1:
    with st.container(border=True):
        st.markdown("##### Standardization — التقييس المعياري")
        st.latex(r"z = \frac{x - \mu}{\sigma}")
        st.markdown(
            "- النتيجة: متوسط = 0، انحراف معياري = 1.\n"
            "- المدى **غير محدود** (قد تخرج قيم ‎±4‎).\n"
            "- أقوى أمام القيم الشاذّة من MinMax.\n"
            "- **الخيار الافتراضي** في معظم الحالات.\n\n"
            "`StandardScaler()`"
        )
with s2:
    with st.container(border=True):
        st.markdown("##### Normalization (Min-Max) — التطبيع")
        st.latex(r"x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}")
        st.markdown(
            "- النتيجة: كل القيم بين 0 و1 بالضبط.\n"
            "- يحافظ على شكل التوزيع الأصلي.\n"
            "- **حسّاس جدًا** للقيم الشاذّة (قيمة واحدة ضخمة تسحق البقية).\n"
            "- مفيد في الشبكات العصبية ومعالجة الصور.\n\n"
            "`MinMaxScaler()`"
        )

st.markdown(
    f"وهناك ثالث مفيد عند وجود شواذّ: {en('RobustScaler')} الذي يستعمل الوسيط والمدى الربيعي "
    "بدل المتوسط والانحراف، فلا تؤثّر فيه القيم المتطرّفة."
)
eq(r"x' = \frac{x - \text{median}(x)}{Q_3 - Q_1}")

st.subheader("شاهد الفرق بنفسك")

rng = np.random.default_rng(1)
raw = pd.DataFrame({
    "المساحة": rng.normal(140, 45, 400).clip(40, 400),
    "الغرف": rng.integers(1, 9, 400).astype(float),
    "عمر_البناء": rng.integers(0, 45, 400).astype(float),
})
choice = st.segmented_control("طريقة التقييس",
                              ["بلا تقييس", "Standard", "MinMax", "Robust"],
                              default="بلا تقييس")
out = raw.copy()
if choice == "Standard":
    out = (raw - raw.mean()) / raw.std()
elif choice == "MinMax":
    out = (raw - raw.min()) / (raw.max() - raw.min())
elif choice == "Robust":
    q1, q3 = raw.quantile(0.25), raw.quantile(0.75)
    out = (raw - raw.median()) / (q3 - q1)

fig = go.Figure()
for i, c in enumerate(out.columns):
    fig.add_trace(go.Box(y=out[c], name=c,
                         marker_color=[PALETTE["indigo"], PALETTE["teal"],
                                       PALETTE["orange"]][i]))
style_fig(fig, height=360, legend_top=False, yaxis_title="القيمة")
st.plotly_chart(fig, width="stretch")
st.caption(
    "«بلا تقييس»: المساحة تبتلع الرسم كله ولا ترى الأعمدة الأخرى. "
    "بعد التقييس تصبح الأعمدة الثلاثة على نفس المقياس وقابلة للمقارنة."
)

st.subheader("أي خوارزمية تحتاج التقييس؟")

st.table(pd.DataFrame({
    "الخوارزمية": ["KNN", "SVM", "K-Means / PCA", "Logistic / Linear Regression",
                   "الشبكات العصبية", "Decision Tree", "Random Forest",
                   "Gradient Boosting (XGBoost…)"],
    "يحتاج تقييس؟": [":red-badge[ضروري]", ":red-badge[ضروري]", ":red-badge[ضروري]",
                     ":orange-badge[مستحسن]", ":red-badge[ضروري]",
                     ":green-badge[لا]", ":green-badge[لا]", ":green-badge[لا]"],
    "السبب": ["تعتمد على المسافة", "تعتمد على المسافة والهوامش",
              "تعتمد على المسافة والتباين",
              "يسرّع الانحدار التدريجي ويجعل معامل التنظيم عادلًا",
              "يستقرّ التدريب ويتجنّب تشبّع دوال التفعيل",
              "تقسّم على عتبات لكل عمود منفردًا", "مجموعة أشجار", "مجموعة أشجار"],
}), border="horizontal")

st.error(
    "**القاعدة الذهبية التي لا يجوز كسرها أبدًا:**\n\n"
    "```python\n"
    "scaler.fit(X_train)            # نتعلّم μ و σ من التدريب فقط\n"
    "X_train = scaler.transform(X_train)\n"
    "X_test  = scaler.transform(X_test)   # transform فقط!\n"
    "```\n"
    "استعمال `fit_transform` على بيانات الاختبار = تسريب بيانات.",
    icon=":material/gpp_bad:",
)

# ---------------------------------------------------------------------------
st.header("3) معالجة الالتواء — Skewness")

st.markdown(
    "كثير من الأعمدة (الدخل، السعر، عدد الزيارات) لها توزيع ملتوٍ لليمين: "
    "أغلبية صغيرة وذيل طويل من القيم الكبيرة. بعض الموديلات (الخطية خصوصًا) تعمل أفضل "
    "عندما يكون التوزيع قريبًا من الطبيعي. الحل: تحويل لوغاريتمي."
)
eq(r"x' = \log(1 + x)", "نستعمل 1+x لتفادي لوغاريتم الصفر")

vals = rng.lognormal(3, 0.9, 1500)
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=("قبل: ملتوٍ لليمين", "بعد: log(1+x)"))
fig.add_trace(go.Histogram(x=vals, nbinsx=50, marker_color=PALETTE["orange"],
                           showlegend=False), row=1, col=1)
fig.add_trace(go.Histogram(x=np.log1p(vals), nbinsx=50, marker_color=PALETTE["teal"],
                           showlegend=False), row=1, col=2)
style_fig(fig, height=320, legend_top=False)
st.plotly_chart(fig, width="stretch")
st.caption(f"معامل الالتواء قبل = {pd.Series(vals).skew():.2f} ، "
           f"بعد = {pd.Series(np.log1p(vals)).skew():.2f}  (كلما اقترب من 0 كان أفضل)")

# ---------------------------------------------------------------------------
st.header("4) اشتقاق متغيّرات جديدة — Feature Engineering")

card(
    "الفكرة",
    "أحيانًا المعلومة موجودة في البيانات لكنها <b>مخفيّة</b>، وعمل الباحث هو إظهارها. "
    "متغيّر واحد ذكي مشتقّ قد يرفع الأداء أكثر من تبديل الخوارزمية بالكامل.",
    "teal",
)

st.markdown(
    """
| البيانات الخام | المتغيّر المشتقّ | لماذا يفيد |
|---|---|---|
| تاريخ ووقت | اليوم من الأسبوع، الشهر، عطلة/لا، الساعة | السلوك موسمي وليس خطيًا مع الزمن |
| السعر والمساحة | السعر للمتر المربع | يقارن بين عقارات مختلفة الأحجام |
| الطول والوزن | مؤشّر كتلة الجسم BMI | متغيّر طبي ذو معنى حقيقي |
| تاريخ الميلاد | العمر | الرقم الخام بلا معنى للموديل |
| نص | عدد الكلمات، وجود كلمات مفتاحية، TF-IDF | يحوّل النص إلى أرقام |
| إحداثيات | المسافة إلى مركز المدينة | يلخّص بعدين في بُعد ذي معنى |
| عمودان | حاصل ضربهما (تفاعل Interaction) | يلتقط تأثيرًا مشتركًا لا يراه الموديل الخطي |
"""
)

st.subheader("تجربة: متغيّر مشتقّ واحد يغيّر كل شيء")

n = 300
x1 = rng.uniform(-3, 3, n)
x2 = rng.uniform(-3, 3, n)
y = (x1**2 + x2**2 < 3.2).astype(int)

o1, o2 = st.columns(2, gap="medium")
with o1:
    fig = go.Figure()
    for k, (nm, col) in enumerate([("الفئة 0", PALETTE["indigo"]), ("الفئة 1", PALETTE["pink"])]):
        m = y == k
        fig.add_trace(go.Scatter(x=x1[m], y=x2[m], mode="markers", name=nm,
                                 marker=dict(size=8, color=col, line=dict(width=1, color="white"))))
    style_fig(fig, height=360, xaxis_title="x₁", yaxis_title="x₂",
              title="الأصلي: مستحيل فصلهما بخط مستقيم")
    st.plotly_chart(fig, width="stretch")
with o2:
    fig = go.Figure()
    r2 = x1**2 + x2**2
    for k, (nm, col) in enumerate([("الفئة 0", PALETTE["indigo"]), ("الفئة 1", PALETTE["pink"])]):
        m = y == k
        fig.add_trace(go.Scatter(x=r2[m], y=rng.normal(0, .3, m.sum()), mode="markers",
                                 name=nm, marker=dict(size=8, color=col,
                                                      line=dict(width=1, color="white"))))
    fig.add_vline(x=3.2, line=dict(color=PALETTE["teal"], width=3, dash="dash"))
    style_fig(fig, height=360, xaxis_title="المتغيّر الجديد:  r² = x₁² + x₂²",
              yaxis_title="(إزاحة عشوائية للعرض)",
              title="بعد الاشتقاق: خط عمودي واحد يفصلهما!")
    st.plotly_chart(fig, width="stretch")

st.success(
    "هذه هي قوّة هندسة المتغيّرات: نفس البيانات، لكن بتمثيل أذكى أصبحت المسألة تافهة. "
    "وهذا أيضًا سرّ عمل نواة SVM (`Kernel trick`) والشبكات العصبية: كلاهما يبحث آليًا عن تمثيل أفضل.",
    icon=":material/auto_awesome:",
)

st.divider()
st.markdown(
    "#### :material/arrow_circle_left: الخطوة التالية\n"
    "**تقسيم البيانات والتحقّق المتقاطع**: لماذا لا يجوز تقييم الموديل على البيانات التي تدرّب عليها."
)
footer()
