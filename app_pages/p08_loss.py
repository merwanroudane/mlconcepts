import numpy as np
import plotly.graph_objects as go
import streamlit as st

from lib.ui import PALETTE, SEQ, card, en, eq, footer, hero, setup, style_fig, toy_regression

setup("loss")

hero(
    "الصفحة 8 من 18",
    "دالة الخسارة ودالة التكلفة  Loss & Cost Function",
    "كيف نحوّل سؤالًا غامضًا مثل «هل هذا الموديل جيّد؟» إلى **رقم واحد** يمكن للحاسوب تصغيره؟ "
    "هذا الرقم هو بوصلة التعلّم كلّه، وإن اخترته خطأً تعلّم الموديل الشيء الخطأ بإتقان.",
)

# ---------------------------------------------------------------------------
st.header("1) الفرق بين Loss و Cost — سؤال يحيّر الكثيرين")

c1, c2 = st.columns(2, gap="medium")
with c1:
    with st.container(border=True):
        st.markdown("##### :material/looks_one: دالة الخسارة — Loss function")
        st.latex(r"L\big(y_i,\; \hat{y}_i\big)")
        st.markdown(
            "خطأ **عيّنة واحدة فقط**. تقيس: كم أخطأ الموديل في هذا المنزل تحديدًا؟\n\n"
            "مثال: `L = (240 − 215)² = 625`"
        )
with c2:
    with st.container(border=True):
        st.markdown("##### :material/functions: دالة التكلفة — Cost function")
        st.latex(r"J(\boldsymbol{\theta}) = \frac{1}{n}\sum_{i=1}^{n} L\big(y_i,\; \hat{y}_i\big)")
        st.markdown(
            "**متوسط** الخسارة على **كل** العيّنات. تقيس: ما مدى سوء الموديل إجمالًا؟\n\n"
            "هذا هو الرقم الذي نصغّره فعليًا في التدريب."
        )

st.markdown(
    f"باختصار: {en('Loss')} للفرد، {en('Cost')} للمجموع. "
    "كثير من الكتب تستعمل الاسمين بالتبادل ولا بأس، لكن معرفة الفرق تساعدك على قراءة الصيغ بدقّة. "
    f"وتُسمّى أيضًا {en('Objective function')} أو {en('Criterion')} في بعض المكتبات.",
)

card(
    "الفكرة المحورية في المجال كلّه",
    "<b>التدريب = مسألة تحسين (Optimization).</b><br>"
    "ابحث عن قيم المعاملات θ التي تجعل J(θ) أصغر ما يمكن:",
)
eq(r"\boldsymbol{\theta}^{*} \;=\; \arg\min_{\boldsymbol{\theta}} \; J(\boldsymbol{\theta})",
   "argmin تعني: «القيمة التي تحقّق الحدّ الأدنى» وليس الحدّ الأدنى نفسه")

# ---------------------------------------------------------------------------
st.header("2) دوال خسارة الانحدار — Regression losses")

st.subheader("المقارنة الرياضية")

l1, l2, l3 = st.columns(3, gap="medium")
with l1:
    with st.container(border=True):
        st.markdown("##### MSE — متوسط مربّع الخطأ")
        st.latex(r"\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}\big(y_i - \hat{y}_i\big)^2")
        st.markdown(
            ":green-badge[ميزة] قابلة للاشتقاق في كل نقطة، ومشتقّتها بسيطة.\n\n"
            ":red-badge[عيب] التربيع يضخّم القيم الشاذّة بشدّة.\n\n"
            "وحدة القياس **مربّعة** (دينار²) — غير قابلة للتفسير."
        )
with l2:
    with st.container(border=True):
        st.markdown("##### MAE — متوسط الخطأ المطلق")
        st.latex(r"\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}\big|y_i - \hat{y}_i\big|")
        st.markdown(
            ":green-badge[ميزة] قويّة أمام الشواذّ، ووحدتها نفس وحدة الهدف.\n\n"
            ":red-badge[عيب] غير قابلة للاشتقاق عند الصفر، والتدريب أبطأ قليلًا."
        )
with l3:
    with st.container(border=True):
        st.markdown("##### Huber — حلّ وسط")
        st.latex(r"""L_\delta = \begin{cases}
\tfrac{1}{2}e^2 & |e| \le \delta \\[4pt]
\delta\big(|e| - \tfrac{1}{2}\delta\big) & |e| > \delta
\end{cases}""")
        st.markdown(
            "تربيعية قرب الصفر (نعومة MSE) وخطّية بعيدًا (متانة MAE).\n\n"
            ":blue-badge[الأفضل] عند وجود شواذّ مع الحاجة لتدريب مستقرّ."
        )

st.subheader("شكل كل دالة — لماذا تتصرّف هكذا؟")

hd = st.slider("معامل δ في Huber", 0.5, 4.0, 1.5, 0.5)
e = np.linspace(-6, 6, 400)
mse = e**2
mae = np.abs(e)
hub = np.where(np.abs(e) <= hd, 0.5 * e**2, hd * (np.abs(e) - 0.5 * hd))

fig = go.Figure()
fig.add_trace(go.Scatter(x=e, y=mse, mode="lines", name="MSE  (e²)",
                         line=dict(color=PALETTE["indigo"], width=3.5)))
fig.add_trace(go.Scatter(x=e, y=mae, mode="lines", name="MAE  (|e|)",
                         line=dict(color=PALETTE["teal"], width=3.5)))
fig.add_trace(go.Scatter(x=e, y=hub, mode="lines", name=f"Huber (δ={hd})",
                         line=dict(color=PALETTE["orange"], width=3.5, dash="dash")))
fig.add_vline(x=-hd, line=dict(color="#C9D6F2", dash="dot"))
fig.add_vline(x=hd, line=dict(color="#C9D6F2", dash="dot"))
style_fig(fig, height=400, xaxis_title="الخطأ  e = y − ŷ", yaxis_title="قيمة الخسارة")
fig.update_yaxes(range=[0, 14])
st.plotly_chart(fig, width="stretch")

st.markdown(
    "اقرأ الرسم جيدًا: عند خطأ = 6، تعطي MAE خسارة 6 بينما تعطي MSE خسارة **36**. "
    "لهذا يصبح الموديل المدرَّب بـ MSE **مهووسًا** بالقيم الشاذّة ويضحّي بدقّة بقية النقاط "
    "من أجل تقليل خطأ نقطة واحدة متطرّفة."
)

st.subheader("تجربة: شاذّة واحدة تكفي لتحريف الموديل")

x, y = toy_regression(n=40, noise=2.0, seed=12)
add_out = st.toggle("أضف قيمة شاذّة واحدة", value=False)
xo, yo = x.copy(), y.copy()
if add_out:
    xo = np.append(xo, 9.5)
    yo = np.append(yo, 5.0)

from sklearn.linear_model import HuberRegressor, LinearRegression  # noqa: E402

lin = LinearRegression().fit(xo.reshape(-1, 1), yo)
hub_m = HuberRegressor(epsilon=1.35).fit(xo.reshape(-1, 1), yo)
xs = np.linspace(0, 10, 50).reshape(-1, 1)

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="البيانات",
                         marker=dict(size=9, color=PALETTE["indigo"],
                                     line=dict(width=1, color="white"))))
if add_out:
    fig.add_trace(go.Scatter(x=[9.5], y=[5.0], mode="markers", name="القيمة الشاذّة",
                             marker=dict(size=18, color=PALETTE["red"], symbol="x",
                                         line=dict(width=3))))
fig.add_trace(go.Scatter(x=xs.ravel(), y=lin.predict(xs), mode="lines",
                         name="مدرَّب بـ MSE", line=dict(color=PALETTE["orange"], width=3.5)))
fig.add_trace(go.Scatter(x=xs.ravel(), y=hub_m.predict(xs), mode="lines",
                         name="مدرَّب بـ Huber", line=dict(color=PALETTE["teal"],
                                                           width=3.5, dash="dash")))
style_fig(fig, height=390, xaxis_title="x", yaxis_title="y")
st.plotly_chart(fig, width="stretch")
if add_out:
    st.warning(
        "نقطة واحدة فقط أمالت خط MSE بوضوح، بينما بقي خط Huber ثابتًا تقريبًا. "
        "هذا هو الفرق العملي بين الخسائر، وليس مجرّد تفصيل نظري.",
        icon=":material/trending_down:",
    )

# ---------------------------------------------------------------------------
st.header("3) دوال خسارة التصنيف — Classification losses")

st.markdown(
    "في التصنيف لا يصلح MSE. السبب: الموديل يُخرج **احتمالًا**، ونريد أن نعاقبه بشدّة "
    "عندما يكون **واثقًا ومخطئًا**. الدالة المناسبة هي الإنتروبيا المتقاطعة."
)

st.subheader("Binary Cross-Entropy — الإنتروبيا المتقاطعة الثنائية (Log Loss)")
eq(r"L = -\Big[\, y\,\log(\hat{p}) \;+\; (1-y)\,\log(1-\hat{p}) \,\Big]",
   "y = 0 أو 1 (الحقيقة)،  p̂ = الاحتمال الذي تنبّأ به الموديل للفئة 1")

st.markdown("**لماذا تبدو معقّدة؟ لأنها مُختصَرة بحيلة أنيقة. فكّها:**")
a, b = st.columns(2)
with a:
    st.markdown("إذا كانت الحقيقة `y = 1` فالحدّ الثاني يُلغى:")
    st.latex(r"L = -\log(\hat{p})")
    st.caption("تنبّأت 0.99 → خسارة 0.01 (ممتاز) · تنبّأت 0.01 → خسارة 4.6 (كارثي)")
with b:
    st.markdown("إذا كانت الحقيقة `y = 0` فالحدّ الأول يُلغى:")
    st.latex(r"L = -\log(1-\hat{p})")
    st.caption("تنبّأت 0.01 → خسارة 0.01 (ممتاز) · تنبّأت 0.99 → خسارة 4.6 (كارثي)")

p = np.linspace(0.001, 0.999, 400)
fig = go.Figure()
fig.add_trace(go.Scatter(x=p, y=-np.log(p), mode="lines", name="الحقيقة y = 1",
                         line=dict(color=PALETTE["indigo"], width=3.5)))
fig.add_trace(go.Scatter(x=p, y=-np.log(1 - p), mode="lines", name="الحقيقة y = 0",
                         line=dict(color=PALETTE["pink"], width=3.5)))
style_fig(fig, height=370, xaxis_title="الاحتمال المتنبَّأ به  p̂  للفئة 1",
          yaxis_title="الخسارة")
fig.update_yaxes(range=[0, 6])
st.plotly_chart(fig, width="stretch")

st.success(
    "**الفكرة الجميلة:** الخسارة تنطلق إلى اللانهاية عندما يكون الموديل **واثقًا تمامًا ومخطئًا تمامًا**. "
    "هذا يعلّم الموديل أن يكون **متواضعًا** عندما لا يعرف — وهي خاصية ثمينة جدًا عمليًا.",
    icon=":material/psychology_alt:",
)

with st.expander("للتصنيف متعدّد الفئات: Categorical Cross-Entropy", icon=":material/category:"):
    st.latex(r"L = -\sum_{c=1}^{C} y_c \,\log(\hat{p}_c)")
    st.markdown(
        "- `C` = عدد الفئات، `y_c` = 1 للفئة الصحيحة و0 لغيرها (ترميز One-Hot).\n"
        "- عمليًا: الخسارة = ‎−log‎ (الاحتمال الذي أعطاه الموديل للفئة الصحيحة).\n"
        f"- يسبقها عادةً تحويل {en('Softmax')} الذي يحوّل أرقامًا حرّة إلى احتمالات مجموعها 1:"
    )
    st.latex(r"\text{softmax}(z)_c = \frac{e^{z_c}}{\sum_{j=1}^{C} e^{z_j}}")

with st.expander("Hinge Loss — خسارة المفصلة (المستعملة في SVM)", icon=":material/architecture:"):
    st.latex(r"L = \max\big(0,\; 1 - y\cdot\hat{y}\big), \qquad y \in \{-1, +1\}")
    st.markdown(
        "لا تكتفي بالتصنيف الصحيح، بل تطلب تصنيفًا صحيحًا **بهامش أمان** قدره 1. "
        "أي نقطة داخل الهامش تُعاقَب حتى لو صُنّفت صحيحًا — وهذا سرّ قوّة SVM."
    )

# ---------------------------------------------------------------------------
st.header("4) سطح التكلفة — Cost Surface")

card(
    "تصوّر المسألة",
    "دالة التكلفة J ليست رقمًا واحدًا، بل <b>منظر طبيعي</b> كامل: "
    "لكل تركيبة من قيم المعاملات (w, b) ارتفاع معيّن = قيمة الخطأ.<br>"
    "التدريب = <b>النزول إلى أخفض وادٍ في هذا المنظر</b>. "
    "وهذا تمامًا ما ستفعله خوارزمية الانحدار التدريجي في الصفحة التالية.",
    "violet",
)

xq, yq = toy_regression(n=50, noise=2.5, seed=21)


@st.cache_data
def cost_grid():
    ws = np.linspace(-1, 6, 70)
    bs = np.linspace(-8, 18, 70)
    W, B = np.meshgrid(ws, bs)
    J = np.array([[np.mean((yq - (w * xq + b)) ** 2) for w in ws] for b in bs])
    return ws, bs, W, B, J


ws, bs, W, B, J = cost_grid()
bw, bb = np.polyfit(xq, yq, 1)
bj = float(np.mean((yq - (bw * xq + bb)) ** 2))

v1, v2 = st.tabs([":material/view_in_ar: عرض ثلاثي الأبعاد", ":material/blur_circular: خطوط التساوي"])
with v1:
    fig = go.Figure(go.Surface(x=ws, y=bs, z=J, colorscale=SEQ[::-1], opacity=0.93,
                               showscale=False,
                               contours=dict(z=dict(show=True, usecolormap=True,
                                                    project_z=True))))
    fig.add_trace(go.Scatter3d(x=[bw], y=[bb], z=[bj], mode="markers+text",
                               text=["الحدّ الأدنى"], textposition="top center",
                               marker=dict(size=7, color=PALETTE["red"]), name="المنخفض"))
    fig.update_layout(height=520, template="plotly_white",
                      paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Cairo, sans-serif", size=13),
                      margin=dict(l=0, r=0, t=30, b=0),
                      scene=dict(xaxis_title="w (الميل)", yaxis_title="b (الثابت)",
                                 zaxis_title="J — التكلفة",
                                 camera=dict(eye=dict(x=1.6, y=1.5, z=1.1))))
    st.plotly_chart(fig, width="stretch")
    st.caption("دوّر الشكل بالفأرة. الهدف: الوصول إلى قاع الوادي الأحمر.")
with v2:
    fig = go.Figure(go.Contour(x=ws, y=bs, z=J, colorscale=SEQ,
                               contours=dict(showlabels=True,
                                             labelfont=dict(size=11, color="#4A5578")),
                               line=dict(width=1), colorbar=dict(title="J")))
    fig.add_trace(go.Scatter(x=[bw], y=[bb], mode="markers+text", text=["★ الأمثل"],
                             textposition="top center", name="الحدّ الأدنى",
                             marker=dict(size=16, color=PALETTE["red"], symbol="star",
                                         line=dict(width=1.5, color="white"))))
    style_fig(fig, height=480, xaxis_title="w (الميل)", yaxis_title="b (الثابت)")
    st.plotly_chart(fig, width="stretch")
    st.caption("كل حلقة = مجموعة النقاط التي تعطي نفس الخطأ، تمامًا كخطوط الارتفاع في خريطة جبل.")

st.info(
    "**لماذا الشكل بيضوي ناعم هنا؟** لأن MSE مع موديل خطي تعطي دالة **محدّبة** "
    f"({en('Convex')}): لها منخفض وحيد، وأي نزول سيصل إليه حتمًا. "
    "في الشبكات العصبية العميقة يكون السطح معقّدًا بمنخفضات محلية كثيرة — "
    "وهذا ما يجعل تدريبها أصعب بكثير.",
)

st.divider()
st.markdown(
    "#### :material/arrow_circle_left: الخطوة التالية\n"
    "**الانحدار التدريجي**: الخوارزمية التي تنزل في هذا الوادي خطوة بخطوة — مع أنيميشن كامل."
)
footer()
