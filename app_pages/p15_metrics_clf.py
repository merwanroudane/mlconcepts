import numpy as np
import plotly.graph_objects as go
import streamlit as st

from lib.ui import PALETTE, card, en, eq, footer, hero, setup, style_fig

setup("metrics_clf")

hero(
    "الصفحة 15 من 18",
    "معايير تقييم التصنيف  Classification Metrics",
    "لماذا قد تكون «الدقّة 99%» أسوأ نتيجة في مشروعك؟ وما الفرق بين Precision و Recall؟ "
    "وكيف تختار العتبة؟ هذه الصفحة هي الأهم عمليًا في قسم التقييم.",
)

# ---------------------------------------------------------------------------
st.header("1) مصفوفة الالتباس — Confusion Matrix")

card(
    "الأساس الذي يُبنى عليه كل شيء",
    "في التصنيف الثنائي هناك أربع نتائج ممكنة فقط لكل تنبؤ. "
    "كل معيار تقييم في هذه الصفحة ليس إلا نسبة بين أرقام هذه المصفوفة الأربعة.",
)

st.markdown(
    """
|  | **التنبؤ: إيجابي** | **التنبؤ: سلبي** |
|---|---|---|
| **الحقيقة: إيجابي** | :green-badge[TP] True Positive — إصابة صحيحة | :red-badge[FN] False Negative — **فوّتناه** (خطأ من النوع الثاني) |
| **الحقيقة: سلبي** | :orange-badge[FP] False Positive — **إنذار كاذب** (خطأ من النوع الأول) | :green-badge[TN] True Negative — رفض صحيح |
"""
)

st.info(
    "**حيلة لحفظها بلا عناء:** الكلمة الأولى (True/False) تعني «هل كان التنبؤ صحيحًا؟»، "
    "والثانية (Positive/Negative) تعني «ماذا تنبّأ الموديل؟». "
    "فـ `False Negative` = تنبّأ بـ «سلبي» وكان مخطئًا ⟵ الحالة كانت إيجابية وفوّتناها.",
    icon=":material/psychology:",
)

# ---------------------------------------------------------------------------
st.header("2) المعايير الخمسة ومتى يُستعمل كلّ منها")

q1, q2 = st.columns(2, gap="medium")
with q1:
    with st.container(border=True):
        st.markdown("##### Accuracy — الدقّة الكلية")
        st.latex(r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}")
        st.markdown(
            "نسبة التنبؤات الصحيحة من الكل.\n\n"
            ":red-badge[خطر كبير] **مضلّلة تمامًا** مع البيانات غير المتوازنة.")
    with st.container(border=True):
        st.markdown("##### Precision — الدقّة الموجبة")
        st.latex(r"\text{Precision} = \frac{TP}{TP + FP}")
        st.markdown(
            "**من بين كل من قلنا عنهم «إيجابي»، كم كان إيجابيًا فعلًا؟**\n\n"
            ":blue-badge[استعملها حين] الإنذار الكاذب مكلف: "
            "تشخيص سرطان يؤدي لعملية جراحية، أو حظر حساب بريء.")
    with st.container(border=True):
        st.markdown("##### Recall / Sensitivity — الاستدعاء")
        st.latex(r"\text{Recall} = \frac{TP}{TP + FN}")
        st.markdown(
            "**من بين كل الحالات الإيجابية الحقيقية، كم أمسكنا منها؟**\n\n"
            ":blue-badge[استعملها حين] التفويت كارثي: "
            "كشف مرض خطير، كشف احتيال، كشف عطل في طائرة.")
with q2:
    with st.container(border=True):
        st.markdown("##### F1-Score — المتوسط التوافقي")
        st.latex(r"F_1 = 2\cdot\frac{\text{Precision}\times\text{Recall}}"
                 r"{\text{Precision} + \text{Recall}}")
        st.markdown(
            "يوازن بين الاثنين. يستعمل المتوسط **التوافقي** وليس الحسابي، "
            "لأنه يعاقب بشدّة إذا كان أحدهما منخفضًا جدًا.\n\n"
            "مثال: P=1.0 و R=0.02 ⟵ المتوسط الحسابي 0.51 (مضلّل!) بينما F1 = 0.039 (صادق).")
    with st.container(border=True):
        st.markdown("##### Specificity — النوعية")
        st.latex(r"\text{Specificity} = \frac{TN}{TN + FP}")
        st.markdown("من بين كل السلبيين الحقيقيين، كم صنّفناهم سلبيين صحيحًا؟ "
                    "مهمّ في الاختبارات الطبية (معدّل الرفض الصحيح).")
    with st.container(border=True):
        st.markdown("##### F-beta — عندما تريد ترجيحًا")
        st.latex(r"F_\beta = (1+\beta^2)\cdot\frac{P \cdot R}{\beta^2 P + R}")
        st.markdown("`β = 2` يرجّح Recall، و `β = 0.5` يرجّح Precision. "
                    "استعمله عندما يكون أحد الخطأين أكلف من الآخر بوضوح.")

st.error(
    "#### :material/warning: مفارقة الدقّة — Accuracy Paradox\n"
    "مرض نادر يصيب 1 من كل 1000 شخص. موديل يقول «سليم» لكل الناس دائمًا، بلا أي تعلّم.\n\n"
    "**دقّته = 99.9%** — رقم مبهر في أي عرض تقديمي.\n\n"
    "لكن: `Recall = 0%` — لم يكتشف **ولا مريضًا واحدًا**. الموديل عديم الفائدة تمامًا.\n\n"
    "**هذا هو السبب الذي يجعل الدقّة وحدها معيارًا خطيرًا.** "
    "مع بيانات غير متوازنة، استعمل F1 أو Recall أو PR-AUC دائمًا.",
    icon=":material/dangerous:",
)

# ---------------------------------------------------------------------------
st.header("3) مختبر تفاعلي: العتبة تغيّر كل شيء")

st.markdown(
    "الموديل لا يُخرج فئة، بل **احتمالًا**. أنت من يحوّله إلى قرار باختيار **عتبة**. "
    "العتبة الافتراضية 0.5 ليست مقدّسة إطلاقًا — بل غالبًا ليست الأفضل."
)

rng = np.random.default_rng(13)


@st.cache_data
def gen(n: int, pos_rate: float, sep: float):
    n_pos = int(n * pos_rate)
    n_neg = n - n_pos
    s_neg = rng.beta(2, 2 + sep * 2, n_neg)
    s_pos = rng.beta(2 + sep * 2, 2, n_pos)
    y = np.r_[np.zeros(n_neg), np.ones(n_pos)].astype(int)
    s = np.r_[s_neg, s_pos]
    return y, s


c1, c2 = st.columns([1, 2.1], gap="medium")
with c1:
    pos_rate = st.slider("نسبة الفئة الإيجابية", 0.02, 0.50, 0.30, 0.02,
                         help="اجعلها 0.05 لترى مفارقة الدقّة بنفسك")
    sep = st.slider("جودة الموديل (قدرة الفصل)", 0.2, 3.0, 1.4, 0.1)
    thr = st.slider("عتبة القرار Threshold", 0.05, 0.95, 0.50, 0.01)

y, s = gen(1200, pos_rate, sep)
pred = (s >= thr).astype(int)
TP = int(((pred == 1) & (y == 1)).sum())
FP = int(((pred == 1) & (y == 0)).sum())
FN = int(((pred == 0) & (y == 1)).sum())
TN = int(((pred == 0) & (y == 0)).sum())
acc = (TP + TN) / len(y)
prec = TP / (TP + FP) if TP + FP else 0.0
rec = TP / (TP + FN) if TP + FN else 0.0
f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
spec = TN / (TN + FP) if TN + FP else 0.0

with c1:
    st.metric("Accuracy", f"{acc:.1%}")
    st.metric("Precision", f"{prec:.1%}")
    st.metric("Recall", f"{rec:.1%}")
    st.metric("F1-Score", f"{f1:.3f}")
    st.metric("Specificity", f"{spec:.1%}")

with c2:
    z = [[TN, FP], [FN, TP]]
    fig = go.Figure(go.Heatmap(
        z=z, x=["تنبؤ: سلبي", "تنبؤ: إيجابي"], y=["حقيقة: سلبي", "حقيقة: إيجابي"],
        colorscale=[[0, "#F6F9FE"], [1, "#5B6FD6"]], showscale=False,
        text=[[f"TN<br>{TN}", f"FP<br>{FP}"], [f"FN<br>{FN}", f"TP<br>{TP}"]],
        texttemplate="%{text}", textfont=dict(size=20)))
    style_fig(fig, height=300, legend_top=False, title="مصفوفة الالتباس")
    st.plotly_chart(fig, width="stretch")

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=s[y == 0], name="الفئة السلبية", nbinsx=42,
                               marker_color=PALETTE["indigo"], opacity=.62))
    fig.add_trace(go.Histogram(x=s[y == 1], name="الفئة الإيجابية", nbinsx=42,
                               marker_color=PALETTE["pink"], opacity=.62))
    fig.add_vline(x=thr, line=dict(color=PALETTE["red"], width=3))
    fig.add_annotation(x=thr, y=1, yref="paper", text="العتبة", showarrow=False,
                       font=dict(color=PALETTE["red"], size=14))
    fig.update_layout(barmode="overlay")
    style_fig(fig, height=300, xaxis_title="الاحتمال المتنبَّأ به",
              yaxis_title="عدد العيّنات")
    st.plotly_chart(fig, width="stretch")

st.markdown(
    "**جرّب هذا الآن:** حرّك العتبة إلى اليسار (0.2) ثم إلى اليمين (0.8) وراقب:\n"
    "- عتبة **منخفضة** ⟵ نصنّف الكثيرين كإيجابيين ⟵ **Recall يرتفع** و **Precision ينخفض**.\n"
    "- عتبة **مرتفعة** ⟵ نتحفّظ ولا نصنّف إيجابيًا إلا بثقة ⟵ العكس تمامًا.\n\n"
    "هذه هي **مقايضة Precision–Recall**: لا يمكن تحسين الاثنين معًا بتحريك العتبة. "
    "لتحسين الاثنين معًا يجب تحسين **الموديل نفسه** (حرّك مؤشّر جودة الموديل لترى الفرق)."
)

# ---------------------------------------------------------------------------
st.header("4) منحنى ROC و AUC")

st.markdown(
    "بدل اختيار عتبة واحدة، لماذا لا نقيّم الموديل عند **كل العتبات الممكنة**؟ "
    "هذا بالضبط ما يفعله منحنى ROC."
)

e1, e2 = st.columns(2)
with e1:
    eq(r"\text{TPR} = \text{Recall} = \frac{TP}{TP+FN}", "محور y — معدّل الإصابات الصحيحة")
with e2:
    eq(r"\text{FPR} = 1 - \text{Specificity} = \frac{FP}{FP+TN}",
       "محور x — معدّل الإنذارات الكاذبة")

order = np.argsort(-s)
ys = y[order]
tpr = np.cumsum(ys) / max(ys.sum(), 1)
fpr = np.cumsum(1 - ys) / max((1 - ys).sum(), 1)
tpr = np.r_[0, tpr, 1]
fpr = np.r_[0, fpr, 1]
auc = float(np.trapezoid(tpr, fpr))

cur_fpr = FP / (FP + TN) if FP + TN else 0
cur_tpr = rec

# منحنى Precision-Recall
rec_c, prec_c = [], []
for t in np.linspace(0.01, 0.99, 120):
    p_ = (s >= t).astype(int)
    tp_ = ((p_ == 1) & (y == 1)).sum()
    fp_ = ((p_ == 1) & (y == 0)).sum()
    fn_ = ((p_ == 0) & (y == 1)).sum()
    rec_c.append(tp_ / (tp_ + fn_) if tp_ + fn_ else 0)
    prec_c.append(tp_ / (tp_ + fp_) if tp_ + fp_ else 1)

t1, t2 = st.columns(2, gap="medium")
with t1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="التخمين العشوائي",
                             line=dict(color=PALETTE["muted"], width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {auc:.3f})",
                             line=dict(color=PALETTE["indigo"], width=4),
                             fill="tozeroy", fillcolor="rgba(91,111,214,0.12)"))
    fig.add_trace(go.Scatter(x=[cur_fpr], y=[cur_tpr], mode="markers",
                             name=f"عتبتك ({thr:.2f})",
                             marker=dict(size=17, color=PALETTE["red"], symbol="star",
                                         line=dict(width=1.5, color="white"))))
    style_fig(fig, height=390, xaxis_title="FPR — معدّل الإنذار الكاذب",
              yaxis_title="TPR — معدّل الاستدعاء", title="منحنى ROC")
    st.plotly_chart(fig, width="stretch")
with t2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rec_c, y=prec_c, mode="lines", name="PR curve",
                             line=dict(color=PALETTE["teal"], width=4),
                             fill="tozeroy", fillcolor="rgba(47,169,140,0.12)"))
    fig.add_hline(y=float(y.mean()), line=dict(color=PALETTE["muted"], dash="dash"))
    fig.add_annotation(x=0.5, y=float(y.mean()), text="خط الأساس = نسبة الفئة الإيجابية",
                       showarrow=False, yshift=12, font=dict(size=12, color=PALETTE["muted"]))
    fig.add_trace(go.Scatter(x=[rec], y=[prec], mode="markers", name=f"عتبتك ({thr:.2f})",
                             marker=dict(size=17, color=PALETTE["red"], symbol="star",
                                         line=dict(width=1.5, color="white"))))
    style_fig(fig, height=390, xaxis_title="Recall", yaxis_title="Precision",
              title="منحنى Precision–Recall")
    st.plotly_chart(fig, width="stretch")

st.markdown("**كيف تقرأ AUC (المساحة تحت منحنى ROC)؟**")
st.table({
    "**قيمة AUC**": ["1.00", "0.90 – 0.99", "0.80 – 0.89", "0.70 – 0.79",
                     "0.50", "< 0.50"],
    "**التفسير**": ["فصل تامّ (اشكك في وجود تسريب!)", "ممتاز", "جيّد", "مقبول",
                    "تخمين عشوائي — بلا قيمة", "أسوأ من العشوائي (اقلب التنبؤات!)"],
})

st.success(
    "**المعنى الاحتمالي الأنيق لـ AUC:** هو احتمال أن يعطي الموديل عيّنة إيجابية عشوائية "
    "درجةً **أعلى** من عيّنة سلبية عشوائية.\n\n"
    "وميزته الكبرى: **لا يعتمد على العتبة إطلاقًا** — فهو يقيس جودة **الترتيب** الذي ينتجه الموديل.",
    icon=":material/insights:",
)

st.warning(
    "**متى تفضّل منحنى PR على ROC؟** عندما تكون الفئة الإيجابية **نادرة جدًا** (أقل من 10%). "
    "في هذه الحالة يبدو ROC متفائلًا بشكل مخادع لأن عدد السلبيين الضخم يجعل FPR صغيرًا دائمًا، "
    f"بينما يكشف منحنى PR الحقيقة. جرّب: اضبط نسبة الفئة الإيجابية على 0.02 وقارن المنحنيين.",
    icon=":material/balance:",
)

# ---------------------------------------------------------------------------
st.header("5) البيانات غير المتوازنة — Imbalanced data")

i1, i2 = st.columns(2, gap="medium")
with i1:
    st.markdown("##### :material/tune: على مستوى البيانات")
    st.markdown(
        "- **Oversampling** للفئة النادرة — تكرار عيّناتها أو توليد عيّنات صناعية بـ `SMOTE`.\n"
        "- **Undersampling** للفئة الكثيرة — حذف جزء منها (خسارة معلومات).\n"
        "- **جمع بيانات أكثر** عن الفئة النادرة — الأفضل دائمًا إن أمكن.")
with i2:
    st.markdown("##### :material/settings: على مستوى الخوارزمية والتقييم")
    st.markdown(
        "- `class_weight='balanced'` — يزيد عقوبة الخطأ في الفئة النادرة تلقائيًا.\n"
        "- **اضبط العتبة** بناءً على تكلفة كل خطأ بدل 0.5.\n"
        "- **استعمل المعيار الصحيح**: F1 أو Recall أو PR-AUC، وليس Accuracy أبدًا.\n"
        "- `StratifiedKFold` في التحقّق المتقاطع.")

st.code(
    """from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_recall_curve)
from sklearn.linear_model import LogisticRegression
import numpy as np

model = LogisticRegression(class_weight="balanced").fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, proba >= 0.5, digits=3))
print("ROC-AUC:", roc_auc_score(y_test, proba))

# اختيار العتبة التي تعظّم F1 — على بيانات التحقّق وليس الاختبار!
p, r, th = precision_recall_curve(y_val, model.predict_proba(X_val)[:, 1])
f1s = 2 * p * r / (p + r + 1e-12)
best_threshold = th[np.argmax(f1s[:-1])]
print("أفضل عتبة:", best_threshold)""",
    language="python",
)

st.divider()
st.markdown(
    "#### :material/arrow_circle_left: الخطوة التالية\n"
    "**ضبط الهايبرباراميترز**: كيف نبحث عن أفضل إعدادات بشكل منهجي بدل التخمين."
)
footer()
