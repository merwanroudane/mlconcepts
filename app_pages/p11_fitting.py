import numpy as np
import plotly.graph_objects as go
import streamlit as st

from lib.ui import PALETTE, card, en, eq, footer, hero, play_buttons, setup, style_fig, toy_curve

setup("fitting")

hero(
    "الصفحة 11 من 18",
    "الإفراط والتقصير في التعلّم  Overfitting & Underfitting",
    "المشكلتان اللتان ستقضي معهما معظم وقتك كباحث. الأولى: الموديل غبي جدًا فلم يتعلّم. "
    "الثانية — وهي الأخطر والأكثر خداعًا: الموديل **حفظ** بدل أن **يفهم**.",
)

# ---------------------------------------------------------------------------
st.header("1) المفهومان في مثال واحد")

st.markdown(
    "لنأخذ طالبًا يستعدّ لامتحان. ثلاثة سيناريوهات:"
)
c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    card("التقصير — Underfitting",
         "طالب لم يدرس أصلًا. يُخطئ في التمارين <b>وفي</b> الامتحان.<br><br>"
         "<b>الأعراض:</b> خطأ عالٍ في التدريب <b>وفي</b> الاختبار معًا.", "orange")
with c2:
    card("التوازن — Good fit",
         "طالب فهم القاعدة. يحلّ التمارين جيدًا <b>والامتحان</b> جيدًا.<br><br>"
         "<b>الأعراض:</b> خطأ منخفض في الاثنين، والفجوة بينهما صغيرة.", "teal")
with c3:
    card("الإفراط — Overfitting",
         "طالب حفظ حلول التمارين حرفيًا بلا فهم. كامل في التمارين، فاشل في الامتحان.<br><br>"
         "<b>الأعراض:</b> خطأ شبه معدوم في التدريب، لكنه عالٍ في الاختبار.", "pink")

st.table({
    "**الحالة**": ["**Underfitting**", "**Good fit**", "**Overfitting**"],
    "**خطأ التدريب**": [":red-badge[عالٍ]", ":green-badge[منخفض]", ":green-badge[منخفض جدًا]"],
    "**خطأ الاختبار**": [":red-badge[عالٍ]", ":green-badge[منخفض]", ":red-badge[عالٍ]"],
    "**الفجوة بينهما**": ["صغيرة", "صغيرة", ":red-badge[كبيرة جدًا]"],
    "**السبب**": ["الموديل بسيط جدًا", "التعقيد مناسب", "الموديل معقّد جدًا / بيانات قليلة"],
}, border="horizontal")

st.warning(
    "**لاحظ الفخّ:** خطأ التدريب وحده لا يميّز بين «جيّد» و«مُفرِط»! "
    "كلاهما يعطي خطأ تدريب منخفضًا. لهذا **لا يمكنك أبدًا** الحكم على موديل من أداء التدريب — "
    "وهذا هو السبب الجوهري لوجود بيانات الاختبار.",
    icon=":material/visibility_off:",
)

# ---------------------------------------------------------------------------
st.header("2) التجربة الكلاسيكية: درجة كثيرة الحدود")

st.markdown(
    "البيانات أدناه وُلّدت من دالة منحنية + ضوضاء. سنحاول تقريبها بكثير حدود "
    f"({en('Polynomial')}) من الدرجة `d`. حرّك المؤشّر وراقب الفرق بين المنحنى والخطأين."
)

x, y = toy_curve(n=30, noise=0.35, seed=3)
rng = np.random.default_rng(77)
idx = rng.permutation(len(x))
tr_i, te_i = idx[:20], idx[20:]
xtr, ytr, xte, yte = x[tr_i], y[tr_i], x[te_i], y[te_i]

cA, cB = st.columns([1, 2.1], gap="medium")
with cA:
    d = st.slider("درجة كثير الحدود  d", 1, 18, 3)
    co = np.polyfit(xtr, ytr, d)
    etr = float(np.mean((ytr - np.polyval(co, xtr)) ** 2))
    ete = float(np.mean((yte - np.polyval(co, xte)) ** 2))
    st.metric("خطأ التدريب MSE", f"{etr:.4f}")
    st.metric("خطأ الاختبار MSE", f"{ete:.4f}", delta=f"فجوة {ete - etr:+.4f}",
              delta_color="inverse")
    st.metric("عدد المعاملات", f"{d + 1}")

    if d <= 2:
        st.warning("**تقصير:** المنحنى بسيط جدًا ولا يلتقط شكل البيانات.",
                   icon=":material/trending_flat:")
    elif d <= 6:
        st.success("**توازن جيّد:** المنحنى يتبع الاتجاه العام ويتجاهل الضوضاء.",
                   icon=":material/check_circle:")
    else:
        st.error("**إفراط:** المنحنى يلتوي ليمرّ بكل نقطة تدريب — بما فيها الضوضاء!",
                 icon=":material/error:")

with cB:
    xs = np.linspace(x.min() - .25, x.max() + .25, 300)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xtr, y=ytr, mode="markers", name="بيانات التدريب",
                             marker=dict(size=11, color=PALETTE["indigo"],
                                         line=dict(width=1.5, color="white"))))
    fig.add_trace(go.Scatter(x=xte, y=yte, mode="markers", name="بيانات الاختبار",
                             marker=dict(size=13, color=PALETTE["pink"], symbol="diamond",
                                         line=dict(width=1.5, color="white"))))
    fig.add_trace(go.Scatter(x=xs, y=np.sin(xs) + 0.15 * xs**2, mode="lines",
                             name="العلاقة الحقيقية f", line=dict(color=PALETTE["muted"],
                                                                  width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=xs, y=np.polyval(co, xs), mode="lines",
                             name=f"الموديل (d={d})",
                             line=dict(color=PALETTE["orange"], width=3.5)))
    style_fig(fig, height=430, xaxis_title="x", yaxis_title="y")
    fig.update_yaxes(range=[min(y) - 1.8, max(y) + 1.8])
    st.plotly_chart(fig, width="stretch")

st.subheader("أنيميشن: الدرجات من 1 إلى 18 تباعًا")
frames, degs = [], list(range(1, 19))
for dd in degs:
    cc = np.polyfit(xtr, ytr, dd)
    frames.append(go.Frame(
        data=[go.Scatter(x=xtr, y=ytr), go.Scatter(x=xs, y=np.polyval(cc, xs))],
        name=str(dd), layout=go.Layout(title=f"درجة كثير الحدود d = {dd}")))
fig = go.Figure()
fig.add_trace(go.Scatter(x=xtr, y=ytr, mode="markers", name="بيانات التدريب",
                         marker=dict(size=11, color=PALETTE["indigo"],
                                     line=dict(width=1.5, color="white"))))
fig.add_trace(go.Scatter(x=xs, y=np.polyval(np.polyfit(xtr, ytr, 1), xs), mode="lines",
                         name="الموديل", line=dict(color=PALETTE["orange"], width=3.5)))
fig.frames = frames
style_fig(fig, height=400, xaxis_title="x", yaxis_title="y", title="درجة كثير الحدود d = 1")
fig.update_yaxes(range=[min(y) - 2.5, max(y) + 2.5])
play_buttons(fig, frame_ms=650)
st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
st.header("3) منحنى التعقيد — حيث يلتقي كل شيء")

st.markdown(
    "هذا الرسم هو **الصورة الأشهر في تعلّم الآلة كلّه**. يرسم الخطأين مقابل تعقيد الموديل."
)

degrees = np.arange(1, 19)
e_tr, e_te = [], []
for dd in degrees:
    cc = np.polyfit(xtr, ytr, dd)
    e_tr.append(np.mean((ytr - np.polyval(cc, xtr)) ** 2))
    e_te.append(min(np.mean((yte - np.polyval(cc, xte)) ** 2), 12))
best = int(degrees[int(np.argmin(e_te))])

fig = go.Figure()
fig.add_trace(go.Scatter(x=degrees, y=e_tr, mode="lines+markers", name="خطأ التدريب",
                         line=dict(color=PALETTE["indigo"], width=3.5)))
fig.add_trace(go.Scatter(x=degrees, y=e_te, mode="lines+markers", name="خطأ الاختبار",
                         line=dict(color=PALETTE["pink"], width=3.5)))
fig.add_vrect(x0=0.5, x1=best - 0.5, fillcolor=PALETTE["orange"], opacity=0.07,
              line_width=0, annotation_text="منطقة التقصير", annotation_position="top left")
fig.add_vrect(x0=best + 0.5, x1=18.5, fillcolor=PALETTE["pink"], opacity=0.07,
              line_width=0, annotation_text="منطقة الإفراط", annotation_position="top right")
fig.add_vline(x=best, line=dict(color=PALETTE["teal"], width=3, dash="dash"))
fig.add_annotation(x=best, y=max(e_te) * 0.85, text=f"★ التعقيد الأمثل (d = {best})",
                   showarrow=False, font=dict(color=PALETTE["teal"], size=15))
style_fig(fig, height=430, xaxis_title="تعقيد الموديل (درجة كثير الحدود) →",
          yaxis_title="الخطأ MSE")
st.plotly_chart(fig, width="stretch")

st.markdown(
    "**اقرأ الرسم بثلاث ملاحظات:**\n"
    "1. خطأ التدريب (الأزرق) **ينزل دائمًا** كلّما زاد التعقيد — ولا يتوقّف أبدًا. "
    "لهذا هو مقياس عديم الفائدة لوحده.\n"
    "2. خطأ الاختبار (الوردي) ينزل ثم **يرتدّ صاعدًا**. شكله حرف U.\n"
    "3. أدنى نقطة في المنحنى الوردي هي التعقيد الأمثل — وهي هدف كل مشروع."
)

# ---------------------------------------------------------------------------
st.header("4) أثر حجم البيانات")

st.markdown(
    "الإفراط ليس خاصية الموديل وحده، بل **علاقة بين تعقيد الموديل وكمّية البيانات**. "
    "نفس الموديل المعقّد قد يُفرط مع 20 نقطة وينجح تمامًا مع 2000."
)

n_pts = st.slider("عدد نقاط التدريب", 12, 300, 20, 4)
deg_fix = st.slider("درجة الموديل (ثابتة)", 3, 15, 12)

rng2 = np.random.default_rng(1)
xb = np.sort(rng2.uniform(-3, 3, n_pts))
yb = np.sin(xb) + 0.15 * xb**2 + rng2.normal(0, 0.35, n_pts)
xv = np.linspace(-3, 3, 200)
yv_true = np.sin(xv) + 0.15 * xv**2
cb_ = np.polyfit(xb, yb, min(deg_fix, n_pts - 1))
gap = float(np.mean((yv_true - np.polyval(cb_, xv)) ** 2))

h1, h2 = st.columns([2, 1], gap="medium")
with h1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xb, y=yb, mode="markers", name="التدريب",
                             marker=dict(size=8, color=PALETTE["indigo"], opacity=.8)))
    fig.add_trace(go.Scatter(x=xv, y=yv_true, mode="lines", name="الحقيقة",
                             line=dict(color=PALETTE["muted"], width=2.5, dash="dot")))
    fig.add_trace(go.Scatter(x=xv, y=np.polyval(cb_, xv), mode="lines",
                             name=f"الموديل (d={deg_fix})",
                             line=dict(color=PALETTE["orange"], width=3.5)))
    style_fig(fig, height=380, xaxis_title="x", yaxis_title="y")
    fig.update_yaxes(range=[-3, 4])
    st.plotly_chart(fig, width="stretch")
with h2:
    st.metric("انحراف الموديل عن الحقيقة", f"{gap:.4f}")
    st.markdown(
        "جرّب: اضبط الدرجة على 12 ثم حرّك عدد النقاط من 12 إلى 300.\n\n"
        "ستلاحظ أن نفس الموديل «المجنون» يتحوّل تدريجيًا إلى موديل ممتاز "
        "لمجرّد أن البيانات زادت.\n\n"
        ":green-badge[القاعدة] **البيانات الأكثر هي أقوى علاج للإفراط على الإطلاق.**"
    )

# ---------------------------------------------------------------------------
st.header("5) جدول العلاج — ماذا أفعل في كل حالة؟")

o1, o2 = st.columns(2, gap="medium")
with o1:
    st.error(
        "#### :material/trending_up: علاج الإفراط Overfitting\n"
        "1. **زد البيانات** — الحل الأقوى دائمًا (أو ولّدها بـ `Data augmentation`).\n"
        "2. **بسّط الموديل** — درجة أقل، عمق أقل، عقد أقل، متغيّرات أقل.\n"
        "3. **أضف تنظيمًا** — `Ridge (L2)`, `Lasso (L1)`, `Dropout`, `weight decay`.\n"
        "4. **التوقّف المبكّر** `Early stopping`.\n"
        "5. **التحقّق المتقاطع** لاختيار الهايبرباراميترز بدل التخمين.\n"
        "6. **اختيار المتغيّرات** — احذف الأعمدة عديمة الفائدة والمكرّرة.\n"
        "7. **تجميع الموديلات** `Bagging / Random Forest`."
    )
with o2:
    st.warning(
        "#### :material/trending_down: علاج التقصير Underfitting\n"
        "1. **عقّد الموديل** — درجة أعلى، عمق أكبر، طبقات أكثر.\n"
        "2. **أضف متغيّرات جديدة** أو متغيّرات تفاعل (`x₁·x₂`) أو تربيعية.\n"
        "3. **قلّل التنظيم** — خفّض `alpha` أو `lambda`.\n"
        "4. **درّب أطول** — حقب أكثر، أو معدّل تعلّم أنسب.\n"
        "5. **غيّر عائلة الخوارزمية** — من خطّي إلى شجري أو شبكة.\n"
        "6. **راجع جودة المتغيّرات** — ربما لا تحمل معلومة كافية أصلًا."
    )

eq(r"\text{Gap} = E_{\text{test}} - E_{\text{train}}",
   "فجوة كبيرة ⟵ إفراط · خطأ تدريب عالٍ أصلًا ⟵ تقصير · الاثنان منخفضان والفجوة صغيرة ⟵ جيّد")

st.info(
    "**ترتيب الخطوات عمليًا:** ابدأ بموديل بسيط جدًا وتأكّد أنه يُقصّر (خطأ عالٍ في الاثنين). "
    "ثم عقّده تدريجيًا حتى تبدأ فجوة الإفراط بالظهور. ثم ارجع خطوة وأضف تنظيمًا. "
    "هذا المسار أسرع وأأمن من البدء بموديل ضخم ومحاولة ترويضه.",
    icon=":material/stairs:",
)

st.divider()
st.markdown(
    "#### :material/arrow_circle_left: الخطوة التالية\n"
    "**مقايضة التحيّز والتباين**: التفسير الرياضي الدقيق لما رأيناه هنا."
)
footer()
