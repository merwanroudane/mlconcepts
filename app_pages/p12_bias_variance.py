import numpy as np
import plotly.graph_objects as go
import streamlit as st

from lib.ui import PALETTE, card, en, eq, footer, hero, setup, style_fig

setup("bias_variance")

hero(
    "الصفحة 12 من 18",
    "مقايضة التحيّز والتباين  Bias–Variance Tradeoff",
    "هذه هي النظرية التي تفسّر **لماذا** يحدث الإفراط والتقصير. "
    "سنفكّك الخطأ إلى ثلاثة مكوّنات، ونرى أن تصغير أحدها يكبّر الآخر — "
    "فلا يوجد موديل مثالي، بل توازن أمثل.",
)

# ---------------------------------------------------------------------------
st.header("1) الحدس: رامي السهام")

st.markdown(
    "تخيّل أربعة رماة، كل واحد يرمي 20 سهمًا على نفس الهدف. المركز = الحقيقة، والسهام = تنبؤات "
    "موديلات مدرَّبة على عيّنات مختلفة من نفس المجتمع."
)

rng = np.random.default_rng(7)
cases = [
    ("تحيّز منخفض · تباين منخفض", (0, 0), 0.22, PALETTE["teal"], "الحالة المثالية — دقيق ومستقرّ"),
    ("تحيّز منخفض · تباين عالٍ", (0, 0), 0.95, PALETTE["orange"],
     "في المتوسط صحيح، لكن كل محاولة بعيدة — **إفراط**"),
    ("تحيّز عالٍ · تباين منخفض", (1.35, 1.1), 0.22, PALETTE["indigo"],
     "مستقرّ جدًا لكنه مخطئ دائمًا بنفس الطريقة — **تقصير**"),
    ("تحيّز عالٍ · تباين عالٍ", (1.35, 1.1), 0.95, PALETTE["red"], "أسوأ حالة ممكنة"),
]
cols = st.columns(4, gap="small")
for col, (nm, c, sd, color, note) in zip(cols, cases):
    with col:
        fig = go.Figure()
        for r, shade in [(1.5, "#F7FAFF"), (1.0, "#EAF0FD"), (0.5, "#D9E3F9")]:
            fig.add_shape(type="circle", x0=-r, y0=-r, x1=r, y1=r,
                          fillcolor=shade, line=dict(color="#C9D6F2", width=1), layer="below")
        pts = rng.normal(c, sd, size=(20, 2))
        fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="markers", showlegend=False,
                                 marker=dict(size=9, color=color, opacity=.85,
                                             line=dict(width=1, color="white"))))
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", showlegend=False,
                                 marker=dict(size=13, color=PALETTE["ink"], symbol="x")))
        style_fig(fig, height=230, legend_top=False, title=nm)
        fig.update_xaxes(range=[-2.6, 2.6], showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(range=[-2.6, 2.6], showticklabels=False, showgrid=False,
                         zeroline=False, scaleanchor="x")
        fig.update_layout(margin=dict(l=4, r=4, t=46, b=4),
                          title=dict(font=dict(size=13)))
        st.plotly_chart(fig, width="stretch")
        st.caption(note)

b1, b2 = st.columns(2, gap="medium")
with b1:
    card("التحيّز — Bias",
         "مدى <b>ابتعاد متوسط</b> تنبؤاتنا عن الحقيقة. "
         "خطأ منهجي ثابت ناتج عن <b>افتراضات مبسِّطة</b> في الموديل.<br>"
         "مثال: استعمال خط مستقيم لعلاقة منحنية — سيُخطئ دائمًا مهما زادت البيانات.",
         "teal")
with b2:
    card("التباين — Variance",
         "مدى <b>تذبذب</b> الموديل لو درّبناه على عيّنة مختلفة قليلًا. "
         "موديل عالي التباين يتعلّم <b>ضوضاء</b> هذه العيّنة بالذات.<br>"
         "مثال: كثير حدود درجة 15 يتغيّر شكله جذريًا بتغيير نقطتين.",
         "pink")

# ---------------------------------------------------------------------------
st.header("2) التفكيك الرياضي — Bias-Variance Decomposition")

st.markdown("يمكن إثبات أن الخطأ المتوقّع عند أي نقطة x ينقسم إلى ثلاثة حدود بالضبط:")
eq(r"\mathbb{E}\Big[\big(y - \hat{f}(x)\big)^2\Big] \;=\; "
   r"\underbrace{\Big(\mathbb{E}[\hat{f}(x)] - f(x)\Big)^{2}}_{\text{Bias}^2} \;+\; "
   r"\underbrace{\mathbb{E}\Big[\big(\hat{f}(x) - \mathbb{E}[\hat{f}(x)]\big)^2\Big]}_{\text{Variance}} \;+\; "
   r"\underbrace{\sigma^2_{\varepsilon}}_{\text{Irreducible}}")

st.markdown("**ترجمة كل حدّ إلى العربية البسيطة:**")
d1, d2, d3 = st.columns(3, gap="medium")
with d1:
    with st.container(border=True):
        st.markdown("##### Bias²  — التحيّز المربّع")
        st.latex(r"\big(\mathbb{E}[\hat{f}(x)] - f(x)\big)^2")
        st.markdown(
            "«لو درّبت الموديل على ألف عيّنة مختلفة وأخذت متوسط تنبؤاته، "
            "كم يبعد هذا المتوسط عن الحقيقة؟»\n\n"
            ":blue-badge[يُخفَّض بـ] موديل أكثر تعقيدًا، متغيّرات أفضل.")
with d2:
    with st.container(border=True):
        st.markdown("##### Variance — التباين")
        st.latex(r"\mathbb{E}\big[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2\big]")
        st.markdown(
            "«كم تتبعثر تنبؤات الموديل حول متوسّطها عبر العيّنات المختلفة؟»\n\n"
            ":blue-badge[يُخفَّض بـ] موديل أبسط، تنظيم، بيانات أكثر، تجميع.")
with d3:
    with st.container(border=True):
        st.markdown("##### σ²  — الخطأ غير القابل للتخفيض")
        st.latex(r"\sigma^2_{\varepsilon}")
        st.markdown(
            "ضوضاء الواقع نفسه. لا يمكن تخفيضها بأي موديل أو بأي كمّية بيانات.\n\n"
            ":red-badge[حدّ أدنى] أي ادّعاء بخطأ أقلّ منها = تسريب بيانات أو خطأ قياس.")

st.info(
    "**نتيجة عملية مهمة:** هذا التفكيك يعني أن هناك **حدًا أدنى نظريًا** للخطأ لا يمكن تجاوزه. "
    "إذا وصلت إليه، فقد انتهى عملك ولا جدوى من تعقيد الموديل أكثر.",
    icon=":material/horizontal_rule:",
)

# ---------------------------------------------------------------------------
st.header("3) المقايضة مرسومة")

st.markdown(
    "لماذا «مقايضة» (`Tradeoff`)؟ لأن الحدّين الأولين يتحرّكان في **اتجاهين متعاكسين** مع التعقيد."
)

cplx = np.linspace(0.4, 10, 220)
noise_lvl = st.slider("مستوى الضوضاء في البيانات σ²", 0.0, 1.6, 0.5, 0.1)
bias2 = 7.0 / (cplx ** 1.45)
var = 0.055 * cplx ** 1.9
total = bias2 + var + noise_lvl
opt = float(cplx[int(np.argmin(total))])

fig = go.Figure()
fig.add_trace(go.Scatter(x=cplx, y=bias2, mode="lines", name="Bias²  — التحيّز",
                         line=dict(color=PALETTE["indigo"], width=3.5)))
fig.add_trace(go.Scatter(x=cplx, y=var, mode="lines", name="Variance — التباين",
                         line=dict(color=PALETTE["pink"], width=3.5)))
fig.add_trace(go.Scatter(x=cplx, y=np.full_like(cplx, noise_lvl), mode="lines",
                         name="σ² — غير قابل للتخفيض",
                         line=dict(color=PALETTE["muted"], width=2.5, dash="dot")))
fig.add_trace(go.Scatter(x=cplx, y=total, mode="lines", name="الخطأ الكلّي",
                         line=dict(color=PALETTE["orange"], width=5)))
fig.add_vline(x=opt, line=dict(color=PALETTE["teal"], width=2.5, dash="dash"))
fig.add_annotation(x=opt, y=float(total.max()) * 0.9, text="★ التعقيد الأمثل",
                   showarrow=False, font=dict(color=PALETTE["teal"], size=15))
fig.add_annotation(x=1.2, y=float(total.max()) * 0.55, text="تقصير<br>Underfitting",
                   showarrow=False, font=dict(color=PALETTE["indigo"], size=13))
fig.add_annotation(x=8.8, y=float(total.max()) * 0.55, text="إفراط<br>Overfitting",
                   showarrow=False, font=dict(color=PALETTE["pink"], size=13))
style_fig(fig, height=450, xaxis_title="تعقيد الموديل →", yaxis_title="الخطأ المتوقّع")
fig.update_yaxes(range=[0, float(total.max()) * 1.05])
st.plotly_chart(fig, width="stretch")

st.markdown(
    f"**اقرأ الرسم:** مع زيادة التعقيد ينخفض التحيّز (الموديل يستطيع تمثيل علاقات أعقد) "
    "لكن يرتفع التباين (يصبح حسّاسًا لتفاصيل العيّنة). مجموعهما + الضوضاء يرسم حرف U، "
    "وأدنى نقطة فيه هي التعقيد الأمثل.\n\n"
    f"جرّب رفع σ²: لاحظ أن المنحنى كلّه يرتفع، وأن التعقيد الأمثل **يصغر** — "
    "فمع بيانات أكثر ضوضاءً يجب أن تكون أكثر تحفّظًا."
)

# ---------------------------------------------------------------------------
st.header("4) أين تقع الخوارزميات على هذا المحور؟")

algos = [
    ("الانحدار الخطي", 1.5, 0.6, PALETTE["indigo"]),
    ("Ridge / Lasso", 1.9, 0.4, PALETTE["sky"]),
    ("الانحدار اللوجستي", 1.7, 0.7, PALETTE["teal"]),
    ("KNN (k كبير)", 2.2, 0.9, PALETTE["lime"]),
    ("Random Forest", 1.1, 1.5, PALETTE["orange"]),
    ("Gradient Boosting", 0.8, 2.0, PALETTE["pink"]),
    ("شجرة قرار عميقة", 0.5, 3.2, PALETTE["red"]),
    ("KNN (k = 1)", 0.4, 3.4, PALETTE["violet"]),
]
fig = go.Figure()
for nm, bi, va, col in algos:
    fig.add_trace(go.Scatter(x=[bi], y=[va], mode="markers+text", text=[nm],
                             textposition="top center", name=nm, showlegend=False,
                             marker=dict(size=19, color=col, opacity=.85,
                                         line=dict(width=2, color="white")),
                             textfont=dict(size=12)))
fig.add_annotation(x=0.5, y=0.45, text="المنطقة المثالية", showarrow=False,
                   font=dict(size=13, color=PALETTE["teal"]))
fig.add_shape(type="rect", x0=0.2, x1=1.0, y0=0.2, y1=0.9, line=dict(color=PALETTE["teal"],
                                                                     dash="dash", width=2),
              fillcolor="rgba(47,169,140,0.06)")
style_fig(fig, height=430, xaxis_title="التحيّز Bias →", yaxis_title="التباين Variance →")
fig.update_xaxes(range=[0, 2.7])
fig.update_yaxes(range=[0, 4.0])
st.plotly_chart(fig, width="stretch")

st.markdown(
    "**لاحظ النمط:** الموديلات البسيطة والمقيّدة (يسار-أسفل) تحيّزها عالٍ وتباينها منخفض. "
    "الموديلات المرنة جدًا (شجرة عميقة، KNN بجار واحد) العكس. "
    "والحيلة الذكية في `Random Forest` هي أنها تأخذ موديلات عالية التباين (أشجارًا عميقة) "
    "وتُخفّض تباينها بالمتوسّط دون أن ترفع تحيّزها كثيرًا."
)

with st.expander("لماذا يخفّض التجميع (Bagging) التباين؟ — البرهان في سطرين",
                 icon=":material/forest:"):
    st.markdown("إذا كان عندنا B موديلًا **مستقلًا** تباين كل منها σ²، فتباين متوسّطها:")
    st.latex(r"\mathrm{Var}\!\left(\frac{1}{B}\sum_{j=1}^{B} \hat{f}_j\right) = \frac{\sigma^2}{B}")
    st.markdown(
        "أي أن التباين ينقسم على عدد الموديلات! وفي الواقع الأشجار ليست مستقلة تمامًا، "
        "فتكون الصيغة الأدقّ:"
    )
    st.latex(r"\rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2")
    st.markdown(
        "حيث ρ هو الارتباط بين الأشجار. ولهذا بالضبط تختار `Random Forest` "
        "مجموعة **عشوائية** من المتغيّرات في كل تقسيم: لتقليل ρ وبالتالي تقليل التباين أكثر."
    )

# ---------------------------------------------------------------------------
st.header("5) ملخّص: جدول القرار السريع")

st.table({
    "**الملاحظة**": ["خطأ التدريب عالٍ، خطأ الاختبار عالٍ، الفجوة صغيرة",
                     "خطأ التدريب منخفض جدًا، خطأ الاختبار عالٍ، الفجوة كبيرة",
                     "الاثنان منخفضان والفجوة صغيرة",
                     "خطأ الاختبار لا ينزل مهما فعلت"],
    "**التشخيص**": [":orange-badge[تحيّز عالٍ]", ":red-badge[تباين عالٍ]",
                    ":green-badge[توازن جيّد]", ":gray-badge[بلغنا σ²]"],
    "**المصطلح**": ["Underfitting", "Overfitting", "Good fit", "Irreducible error"],
    "**الإجراء**": ["عقّد الموديل · أضف متغيّرات · قلّل التنظيم",
                    "بسّط · أضف تنظيمًا · زد البيانات · Early stopping",
                    "ثبّت واحفظ الموديل",
                    "توقّف — أو ابحث عن مصادر بيانات جديدة تمامًا"],
}, border="horizontal")

st.success(
    "**العبارة التي تختصر الصفحة:** هدفك ليس تصفير الخطأ، بل **إيجاد نقطة التوازن**. "
    "كل ما تفعله في بقية المشروع — التنظيم، اختيار الهايبرباراميترز، التحقّق المتقاطع — "
    "هو بحث منهجي عن هذه النقطة.",
    icon=":material/balance:",
)

st.divider()
st.markdown(
    "#### :material/arrow_circle_left: الخطوة التالية\n"
    "**التنظيم Regularization**: الأداة العملية الأقوى للتحكّم في التباين."
)
footer()
