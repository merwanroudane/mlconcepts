import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.ui import PALETTE, card, en, eq, footer, hero, setup, style_fig

setup("split")

hero(
    "الصفحة 6 من 18",
    "تقسيم البيانات والتحقّق المتقاطع  Train / Test Split & Cross-Validation",
    "القاعدة التي تفصل بين باحث جادّ وهاوٍ: لا تُقيَّم الآلة أبدًا على الأسئلة التي تدرّبت عليها. "
    "هنا نرى لماذا، وكيف نقسّم بشكل صحيح، وما هو التحقّق المتقاطع ولماذا هو أقوى.",
)

# ---------------------------------------------------------------------------
st.header("1) لماذا نقسّم أصلًا؟")

card(
    "مثال الطالب",
    "لو أعطيت طالبًا 100 مسألة محلولة ليدرسها، ثم امتحنته بنفس الـ 100 مسألة، "
    "فدرجته العالية لا تدلّ على <b>فهم</b> بل قد تدلّ على <b>حفظ</b> فقط. "
    "لمعرفة ما إذا كان قد فهم حقًا، يجب أن تمتحنه بمسائل <b>جديدة لم يرَها</b>.<br><br>"
    "الموديل بالضبط كذلك. هدفنا ليس أن يحفظ بيانات التدريب، بل أن "
    "<b>يُعمّم</b> (Generalize) على بيانات جديدة.",
)

st.markdown(
    "لهذا نميّز بين خطأين مختلفين تمامًا، ولا تخلط بينهما أبدًا:"
)
e1, e2 = st.columns(2, gap="medium")
with e1:
    with st.container(border=True):
        st.markdown("##### خطأ التدريب — Training error")
        st.latex(r"E_{\text{train}} = \frac{1}{n_{tr}}\sum_{i \in \text{train}} L(y_i, \hat{y}_i)")
        st.markdown("يقيس **الحفظ**. يمكن تصفيره دائمًا بموديل معقّد كفاية. لا يدلّ على شيء وحده.")
with e2:
    with st.container(border=True):
        st.markdown("##### خطأ التعميم — Generalization error")
        st.latex(r"E_{\text{test}} = \mathbb{E}_{(x,y)\sim P}\big[L(y, \hat{f}(x))\big]")
        st.markdown("يقيس **الفهم** على بيانات لم تُرَ. هذا هو ما يهمّنا فعلًا، ونقدّره ببيانات الاختبار.")

# ---------------------------------------------------------------------------
st.header("2) التقسيم الثلاثي — Train / Validation / Test")

c1, c2 = st.columns([1.1, 1.6], gap="medium")
with c1:
    tr = st.slider("نسبة التدريب Train %", 40, 85, 70, 5)
    remaining = 100 - tr
    va = st.slider("نسبة التحقّق Validation %", 0, remaining, min(15, remaining), 5)
    te = 100 - tr - va
    st.metric("نسبة الاختبار Test %", f"{te}%")
    st.caption("النسب الشائعة: 70/15/15 أو 60/20/20. مع بيانات ضخمة جدًا يكفي 98/1/1.")
with c2:
    fig = go.Figure()
    parts = [("Train — تدريب", tr, PALETTE["indigo"]),
             ("Validation — تحقّق", va, PALETTE["teal"]),
             ("Test — اختبار", te, PALETTE["orange"])]
    for nm, v, col in parts:
        if v > 0:
            fig.add_trace(go.Bar(x=[v], y=["البيانات"], orientation="h", name=f"{nm} ({v}%)",
                                 marker=dict(color=col, line=dict(color="white", width=2)),
                                 text=f"{v}%", textposition="inside",
                                 textfont=dict(color="white", size=16)))
    fig.update_layout(barmode="stack")
    style_fig(fig, height=210, xaxis_title="", yaxis_title="")
    fig.update_xaxes(showticklabels=False, showgrid=False)
    st.plotly_chart(fig, width="stretch")

d1, d2, d3 = st.columns(3, gap="medium")
with d1:
    with st.container(border=True):
        st.markdown("##### :material/school: Train — التدريب")
        st.markdown(
            "الموديل **يراها ويتعلّم منها معاملاته**.\n\n"
            "كلّما زادت، تعلّم أفضل.\n\n"
            ":blue-badge[يُستعمل في] `model.fit()`")
with d2:
    with st.container(border=True):
        st.markdown("##### :material/tune: Validation — التحقّق")
        st.markdown(
            "الموديل **لا يتعلّم منها معاملات**، لكنّك **أنت** تستعملها لاختيار "
            "الهايبرباراميترز والخوارزمية.\n\n"
            ":orange-badge[تحذير] بكثرة الاستعمال تتلوّث تدريجيًا وتصبح متفائلة.")
with d3:
    with st.container(border=True):
        st.markdown("##### :material/lock: Test — الاختبار")
        st.markdown(
            "**خزنة مقفلة.** لا تُفتح إلا **مرة واحدة** في نهاية المشروع "
            "لتقدير الأداء الحقيقي.\n\n"
            ":red-badge[ممنوع] اتخاذ أي قرار بناءً عليها.")

st.error(
    "**السؤال الذي يحيّر الجميع: لماذا ثلاثة وليس اثنين؟**\n\n"
    "لو جرّبت 50 تركيبة هايبرباراميترات واخترت الأفضل على بيانات الاختبار، فأنت "
    "«تُحسّن على الاختبار»، وستبدو النتيجة أفضل من الواقع بسبب الصدفة وحدها. "
    "بيانات التحقّق تمتصّ هذا التفاؤل، وتبقى الاختبار نظيفة تمامًا كحكم محايد نهائي.",
    icon=":material/gavel:",
)

st.code(
    """from sklearn.model_selection import train_test_split

# أولًا نعزل الاختبار ونقفله
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ثم نقسّم الباقي إلى تدريب وتحقّق
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1875, random_state=42, stratify=y_temp
)
# النتيجة النهائية ≈ 65% / 15% / 20%""",
    language="python",
)

with st.expander("ما هو stratify ولماذا هو مهم جدًا؟", icon=":material/balance:"):
    st.markdown(
        "في مسائل التصنيف، إذا كانت الفئة النادرة تمثّل 5% فقط من البيانات، "
        "فقد يُعطيك التقسيم العشوائي بيانات اختبار فيها 1% أو 9% بمحض الصدفة — "
        "فيصبح التقييم غير عادل.\n\n"
        f"{en('stratify=y')} يضمن أن **نسبة كل فئة** تبقى كما هي في كل جزء. "
        "استعمله دائمًا في التصنيف، خصوصًا مع البيانات غير المتوازنة."
    )
    a, b = st.columns(2)
    rng = np.random.default_rng(0)
    yy = (rng.random(400) < 0.12).astype(int)
    with a:
        st.markdown("**بدون stratify** — النسب تتذبذب")
        rows = []
        for s in range(4):
            idx = np.random.default_rng(s).permutation(400)[:80]
            rows.append({"محاولة": s + 1, "نسبة الفئة النادرة": f"{yy[idx].mean():.1%}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with b:
        st.markdown("**مع stratify** — النسب ثابتة")
        st.dataframe(pd.DataFrame({
            "محاولة": [1, 2, 3, 4],
            "نسبة الفئة النادرة": [f"{yy.mean():.1%}"] * 4,
        }), hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
st.header("3) التحقّق المتقاطع — K-Fold Cross-Validation")

card(
    "المشكلة التي يحلّها",
    "التقسيم الواحد يعتمد على <b>الحظ</b>: ربما وقعت الحالات الصعبة كلّها في الاختبار صدفةً، "
    "فتبدو النتيجة سيّئة. ومع البيانات القليلة تصبح المشكلة خطيرة.<br>"
    "<b>الحل:</b> كرّر التقسيم K مرّات بحيث يلعب كل جزء دور الاختبار مرّة واحدة، ثم خُذ المتوسط.",
    "teal",
)

K = st.slider("عدد الطيّات K", 3, 10, 5)
fold = st.slider("أي طيّة نعرض الآن؟", 1, K, 1) - 1

fig = go.Figure()
for i in range(K):
    for j in range(K):
        is_val = (j == i)
        active = (i == fold)
        fig.add_shape(type="rect", x0=j, x1=j + 0.94, y0=-i - 0.9, y1=-i - 0.08,
                      fillcolor=(PALETTE["orange"] if is_val else PALETTE["indigo"])
                      if active else ("#F5C89C" if is_val else "#C7D0EE"),
                      line=dict(color="white", width=2))
    fig.add_annotation(x=-0.35, y=-i - 0.5, text=f"طيّة {i + 1}", showarrow=False,
                       xanchor="right",
                       font=dict(size=14, color=PALETTE["ink"] if i == fold else PALETTE["muted"]))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="تدريب Train",
                         marker=dict(size=14, color=PALETTE["indigo"], symbol="square")))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="تحقّق Validation",
                         marker=dict(size=14, color=PALETTE["orange"], symbol="square")))
style_fig(fig, height=90 + 46 * K)
fig.update_xaxes(range=[-1.6, K], showticklabels=False, showgrid=False, zeroline=False)
fig.update_yaxes(range=[-K - 0.2, 0.5], showticklabels=False, showgrid=False, zeroline=False)
st.plotly_chart(fig, width="stretch")

st.markdown(
    f"في كل طيّة: ندرّب على {K - 1} أجزاء ونقيّم على الجزء المتبقّي. "
    f"في النهاية عندنا {K} نتائج، نأخذ متوسّطها وانحرافها المعياري."
)
eq(r"\text{CV score} = \frac{1}{K}\sum_{k=1}^{K} \text{score}_k "
   r"\qquad\pm\qquad \sigma = \sqrt{\frac{1}{K}\sum_{k=1}^{K}(\text{score}_k - \overline{\text{score}})^2}",
   "المتوسط يقدّر الأداء، والانحراف المعياري يقدّر مدى استقراره — وكلاهما مهم للتقرير")

st.code(
    """from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")

print(f"{scores.mean():.3f} ± {scores.std():.3f}")""",
    language="python",
)

st.subheader("أنواع التحقّق المتقاطع — اختر الصحيح لبياناتك")
st.table(pd.DataFrame({
    "النوع": ["KFold", "StratifiedKFold", "GroupKFold",
              "TimeSeriesSplit", "LeaveOneOut (LOO)", "RepeatedKFold"],
    "متى يُستعمل": ["انحدار، بيانات متجانسة", "تصنيف (الخيار الافتراضي)",
                    "صفوف تتبع نفس الفرد/المريض/المتجر",
                    "بيانات زمنية", "بيانات قليلة جدًا (< 50 صف)",
                    "نريد تقديرًا أدقّ ونملك وقتًا"],
    "ملاحظة": ["بسيط", "يحافظ على نسب الفئات", "يمنع تسرّب نفس المجموعة بين التدريب والاختبار",
               ":red-badge[إلزامي] لا يُسمح بالتدريب على المستقبل",
               "مكلف جدًا (n موديل) وتباينه عالٍ", "يكرّر KFold بترتيبات مختلفة"],
}), border="horizontal")

# ---------------------------------------------------------------------------
st.header("4) حالة خاصة حرجة: البيانات الزمنية")

st.error(
    "**لا تخلط البيانات الزمنية أبدًا!** `shuffle=True` مع بيانات السلاسل الزمنية "
    "يعني أنك تدرّب على 2024 وتختبر على 2022 — أي أنك تتنبأ بالماضي معتمدًا على المستقبل. "
    "النتيجة ستبدو ممتازة ثم تنهار في الواقع.",
    icon=":material/update_disabled:",
)

n_splits = 5
fig = go.Figure()
total = 100
for i in range(n_splits):
    tr_end = int(total * (i + 1) / (n_splits + 1))
    te_end = int(total * (i + 2) / (n_splits + 1))
    fig.add_shape(type="rect", x0=0, x1=tr_end, y0=-i - 0.85, y1=-i - 0.15,
                  fillcolor=PALETTE["indigo"], line=dict(color="white", width=2))
    fig.add_shape(type="rect", x0=tr_end, x1=te_end, y0=-i - 0.85, y1=-i - 0.15,
                  fillcolor=PALETTE["orange"], line=dict(color="white", width=2))
    fig.add_annotation(x=-2, y=-i - 0.5, text=f"تقسيم {i + 1}", showarrow=False,
                       xanchor="right", font=dict(size=13, color=PALETTE["muted"]))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="تدريب (الماضي)",
                         marker=dict(size=14, color=PALETTE["indigo"], symbol="square")))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="اختبار (المستقبل)",
                         marker=dict(size=14, color=PALETTE["orange"], symbol="square")))
style_fig(fig, height=320, xaxis_title="الزمن ←")
fig.update_xaxes(range=[-18, 101], showticklabels=False, showgrid=False)
fig.update_yaxes(range=[-n_splits - 0.1, 0.6], showticklabels=False, showgrid=False, zeroline=False)
st.plotly_chart(fig, width="stretch")
st.caption("TimeSeriesSplit: نافذة التدريب تكبر دائمًا، والاختبار دائمًا بعدها زمنيًا.")

st.divider()
st.markdown(
    "#### :material/arrow_circle_left: الخطوة التالية\n"
    "ندخل **قلب التعلّم**: ما هو الموديل رياضيًا، وما الفرق بينه وبين الخوارزمية والمعاملات."
)
footer()
