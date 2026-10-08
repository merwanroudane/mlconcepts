import numpy as np
import plotly.graph_objects as go
import streamlit as st

from lib.ui import (PALETTE, SEQ, card, en, eq, footer, hero, play_buttons, setup,
                    style_fig, toy_regression)

setup("gradient")

hero(
    "الصفحة 9 من 18",
    "الانحدار التدريجي والتحسين  Gradient Descent & Optimization",
    "هذه هي الخوارزمية التي تُدرَّب بها تقريبًا كل موديلات تعلّم الآلة الحديثة، "
    "من الانحدار اللوجستي إلى أضخم الشبكات العصبية. الفكرة بسيطة بشكل مدهش: "
    "انزل دائمًا في اتجاه الهبوط الأشدّ.",
)

# ---------------------------------------------------------------------------
st.header("1) الحدس: رجل في الضباب على جبل")

card(
    "القصّة",
    "تخيّل أنك على جبل والضباب كثيف، لا ترى إلا مترًا واحدًا حولك، وتريد الوصول إلى الوادي.<br>"
    "ماذا تفعل؟ <b>تتحسّس ميل الأرض تحت قدميك، ثم تخطو خطوة في اتجاه الهبوط الأشدّ.</b> "
    "ثم تكرّر. هذه هي الخوارزمية بالكامل.<br><br>"
    "• الجبل = دالة التكلفة J<br>"
    "• موقعك = قيم المعاملات الحالية<br>"
    "• ميل الأرض = <b>المشتقّة / التدرّج</b> (Gradient)<br>"
    "• طول خطوتك = <b>معدّل التعلّم</b> (Learning rate)",
)

# ---------------------------------------------------------------------------
st.header("2) المشتقّة: ما معناها بالضبط؟")

st.markdown(
    "المشتقّة تجيب عن سؤال واحد: **إذا زدت المعامل قليلًا جدًا، هل ترتفع التكلفة أم تنخفض، وبأي سرعة؟**"
)
eq(r"\frac{dJ}{dw} \;=\; \lim_{h \to 0} \frac{J(w+h) - J(w)}{h}",
   "ببساطة: ميل المماس للمنحنى عند النقطة w")

st.markdown(
    "- مشتقّة **موجبة** ⟵ المنحنى صاعد ⟵ يجب أن **ننقص** `w` للنزول.\n"
    "- مشتقّة **سالبة** ⟵ المنحنى هابط ⟵ يجب أن **نزيد** `w` للنزول.\n"
    "- مشتقّة **صفر** ⟵ أرض مستوية ⟵ وصلنا إلى حدّ أدنى (أو قمة أو نقطة سرج).\n\n"
    "في الحالتين الأولى والثانية، القاعدة واحدة: **تحرّك عكس إشارة المشتقّة**. "
    "ومن هنا جاءت إشارة الناقص في القانون الأساسي:"
)

eq(r"w_{\text{new}} \;=\; w_{\text{old}} \;-\; \eta \,\frac{\partial J}{\partial w}",
   "η (إيتا) = معدّل التعلّم — الهايبرباراميتر الأهم في التدريب")

st.markdown(
    f"وعندما تكون المعاملات كثيرة نجمع كل المشتقّات الجزئية في متّجه واحد اسمه "
    f"**التدرّج** ({en('Gradient')}) ونرمز له بـ ∇J:"
)
eq(r"\nabla J = \Big(\frac{\partial J}{\partial w_1},\; \frac{\partial J}{\partial w_2},\; "
   r"\dots,\; \frac{\partial J}{\partial w_p}\Big)^{\!\top}"
   r"\qquad\Longrightarrow\qquad \boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta\,\nabla J")

with st.expander("اشتقاق القانون للانحدار الخطي خطوة بخطوة", icon=":material/calculate:"):
    st.markdown("دالة التكلفة (نضع ½ لتبسيط المشتقّة، ولا تغيّر موضع الحدّ الأدنى):")
    st.latex(r"J(w,b) = \frac{1}{2n}\sum_{i=1}^{n}\big(wx_i + b - y_i\big)^2")
    st.markdown("بقاعدة السلسلة، المشتقّة بالنسبة لـ `w`:")
    st.latex(r"\frac{\partial J}{\partial w} = \frac{1}{n}\sum_{i=1}^{n}\big(wx_i + b - y_i\big)\,x_i"
             r" = \frac{1}{n}\sum_{i=1}^{n} e_i\,x_i")
    st.markdown("وبالنسبة لـ `b`:")
    st.latex(r"\frac{\partial J}{\partial b} = \frac{1}{n}\sum_{i=1}^{n}\big(wx_i + b - y_i\big)"
             r" = \frac{1}{n}\sum_{i=1}^{n} e_i")
    st.markdown(
        "حيث `eᵢ = ŷᵢ − yᵢ` هو خطأ العيّنة. لاحظ المعنى: "
        "**التحديث يتناسب مع متوسط الأخطاء مرجّحًا بقيمة المتغيّر.** "
        "العيّنات التي أخطأنا فيها كثيرًا تدفع المعاملات أكثر."
    )

# ---------------------------------------------------------------------------
st.header("3) أنيميشن: النزول في دالة بسيطة")

st.markdown(
    "لنبدأ بأبسط حالة ممكنة: `J(w) = w²`، أي مشتقّتها `dJ/dw = 2w` والحدّ الأدنى عند `w = 0`. "
    "حرّك معدّل التعلّم ونقطة البداية، ثم اضغط **تشغيل**."
)

g1, g2 = st.columns([1, 2.2], gap="medium")
with g1:
    lr = st.select_slider("معدّل التعلّم η", [0.01, 0.05, 0.1, 0.3, 0.5, 0.9, 1.0, 1.05],
                          value=0.1)
    w0 = st.slider("نقطة البداية w₀", -9.0, 9.0, 8.0, 0.5)
    n_steps = st.slider("عدد الخطوات", 5, 60, 25)

    ws_path = [w0]
    for _ in range(n_steps):
        ws_path.append(ws_path[-1] - lr * 2 * ws_path[-1])
    ws_path = np.array(ws_path)
    final = ws_path[-1]

    if not np.isfinite(final) or abs(final) > 1e4:
        st.error("**تباعُد!** η كبير جدًا: القيم تنفجر نحو اللانهاية.",
                 icon=":material/dangerous:")
    elif abs(final) < 0.01:
        st.success(f"**تقارُب ممتاز.** w النهائي = {final:.4f}",
                   icon=":material/check_circle:")
    else:
        st.warning(f"**بطيء جدًا.** w النهائي = {final:.3f} ولم يصل بعد.",
                   icon=":material/hourglass_bottom:")

    st.caption(
        "جرّب: η = 0.01 (بطيء) · η = 0.3 (مثالي) · η = 0.9 (يتذبذب) · η = 1.05 (ينفجر)"
    )

with g2:
    wgrid = np.linspace(-10, 10, 200)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wgrid, y=wgrid**2, mode="lines", name="J(w) = w²",
                             line=dict(color=PALETTE["indigo"], width=3)))
    safe = np.clip(ws_path, -10, 10)
    fig.add_trace(go.Scatter(x=safe, y=safe**2, mode="lines+markers", name="المسار",
                             line=dict(color=PALETTE["orange"], width=2, dash="dot"),
                             marker=dict(size=7, color=PALETTE["orange"])))
    fig.add_trace(go.Scatter(x=[safe[0]], y=[safe[0]**2], mode="markers", name="الموقع",
                             marker=dict(size=20, color=PALETTE["red"],
                                         line=dict(width=2, color="white"))))
    frames = [go.Frame(data=[go.Scatter(x=wgrid, y=wgrid**2),
                             go.Scatter(x=safe[:k + 1], y=safe[:k + 1]**2),
                             go.Scatter(x=[safe[k]], y=[safe[k]**2])],
                       name=str(k))
              for k in range(len(safe))]
    fig.frames = frames
    style_fig(fig, height=420, xaxis_title="w — المعامل", yaxis_title="J(w) — التكلفة")
    fig.update_yaxes(range=[-5, 105])
    fig.update_xaxes(range=[-10.5, 10.5])
    play_buttons(fig, frame_ms=220)
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
st.header("4) معدّل التعلّم — الهايبرباراميتر الأهم")

st.markdown("هذه المقارنة الثلاثية هي أهم ما يجب أن تفهمه في هذه الصفحة:")

lr1, lr2, lr3 = st.columns(3, gap="medium")
cases = [
    ("η صغير جدًا", 0.02, PALETTE["teal"], lr1,
     "خطوات دقيقة لكن بطيئة جدًا. قد تحتاج آلاف التكرارات، وقد تعلق في منخفض محلّي ضحل."),
    ("η مناسب", 0.3, PALETTE["indigo"], lr2,
     "ينزل بسرعة ويستقرّ. هذا ما نريده — ويُوجد عادةً بالتجريب على مقياس لوغاريتمي."),
    ("η كبير جدًا", 1.02, PALETTE["red"], lr3,
     "يقفز فوق الحدّ الأدنى ويتباعد. التكلفة ترتفع بدل أن تنخفض، وقد تظهر NaN."),
]
for name, l, col, cont, desc in cases:
    path = [8.0]
    for _ in range(14):
        nxt = path[-1] - l * 2 * path[-1]
        path.append(np.clip(nxt, -14, 14))
    path = np.array(path)
    with cont:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wgrid, y=wgrid**2, mode="lines", showlegend=False,
                                 line=dict(color="#C9D4F5", width=2)))
        fig.add_trace(go.Scatter(x=path, y=path**2, mode="lines+markers", showlegend=False,
                                 line=dict(color=col, width=2),
                                 marker=dict(size=7, color=col)))
        style_fig(fig, height=250, legend_top=False, title=f"{name}  (η = {l})")
        fig.update_yaxes(range=[-5, 110], title="J")
        fig.update_xaxes(range=[-11, 11], title="w")
        st.plotly_chart(fig, width="stretch")
        st.caption(desc)

st.info(
    "**كيف أختار η عمليًا؟** جرّب على مقياس لوغاريتمي: "
    "`0.0001, 0.001, 0.01, 0.1, 1` وارسم منحنى التكلفة لكل قيمة. "
    "اختر أكبر قيمة تبقى عندها التكلفة **نازلة بثبات**. "
    f"وفي الممارسة الحديثة يُستعمل **جدول معدّل التعلّم** ({en('LR scheduler')}): "
    "نبدأ كبيرًا للسرعة ثم نصغّره تدريجيًا للدقّة.",
)

# ---------------------------------------------------------------------------
st.header("5) أنيميشن كامل: تدريب انحدار خطي حقيقي")

st.markdown(
    "الآن معاملان في آن واحد (`w` و `b`)، على بيانات حقيقية. "
    "ستشاهد ثلاثة أشياء تتحرّك معًا: الخط يتحسّن، والنقطة تنزل في خريطة التكلفة، والمنحنى يهبط."
)

x, y = toy_regression(n=50, noise=2.5, seed=21)


@st.cache_data
def run_gd(lr: float, steps: int, w0: float, b0: float):
    w, b = w0, b0
    n = len(x)
    W, B, C = [w], [b], [float(np.mean((y - (w * x + b)) ** 2))]
    for _ in range(steps):
        e = (w * x + b) - y
        gw = float(np.mean(e * x))
        gb = float(np.mean(e))
        w -= lr * 2 * gw
        b -= lr * 2 * gb
        W.append(w)
        B.append(b)
        C.append(float(np.mean((y - (w * x + b)) ** 2)))
    return np.array(W), np.array(B), np.array(C)


p1, p2 = st.columns([1, 3], gap="medium")
with p1:
    lr2v = st.select_slider("معدّل التعلّم η", [0.001, 0.005, 0.01, 0.02, 0.03],
                            value=0.01, key="lr2")
    steps = st.slider("عدد التكرارات", 10, 120, 60, key="st2")
    Wp, Bp, Cp = run_gd(lr2v, steps, 0.2, 0.0)
    st.metric("التكلفة الابتدائية", f"{Cp[0]:,.1f}")
    st.metric("التكلفة النهائية", f"{Cp[-1]:,.2f}",
              delta=f"{Cp[-1] - Cp[0]:,.1f}", delta_color="inverse")
    st.caption(f"w = {Wp[-1]:.3f} ,  b = {Bp[-1]:.3f}")

with p2:
    tabs = st.tabs([":material/show_chart: الخط يتعلّم",
                    ":material/map: المسار على خريطة التكلفة",
                    ":material/trending_down: منحنى التكلفة"])

    xs = np.linspace(x.min(), x.max(), 40)
    with tabs[0]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="البيانات",
                                 marker=dict(size=9, color=PALETTE["indigo"],
                                             line=dict(width=1, color="white"))))
        fig.add_trace(go.Scatter(x=xs, y=Wp[0] * xs + Bp[0], mode="lines", name="الموديل",
                                 line=dict(color=PALETTE["orange"], width=4)))
        fig.frames = [go.Frame(data=[go.Scatter(x=x, y=y),
                                     go.Scatter(x=xs, y=Wp[k] * xs + Bp[k])],
                               name=str(k)) for k in range(len(Wp))]
        style_fig(fig, height=420, xaxis_title="x", yaxis_title="y")
        play_buttons(fig, frame_ms=90)
        st.plotly_chart(fig, width="stretch")

    with tabs[1]:
        wg = np.linspace(min(0, Wp.min()) - .5, max(3.5, Wp.max()) + .5, 60)
        bg = np.linspace(min(-2, Bp.min()) - 1, max(8, Bp.max()) + 2, 60)
        Jg = np.array([[np.mean((y - (w * x + b)) ** 2) for w in wg] for b in bg])
        fig = go.Figure()
        fig.add_trace(go.Contour(x=wg, y=bg, z=Jg, colorscale=SEQ, showscale=False,
                                 contours=dict(showlabels=False), line=dict(width=1)))
        fig.add_trace(go.Scatter(x=Wp[:1], y=Bp[:1], mode="lines+markers", name="المسار",
                                 line=dict(color=PALETTE["orange"], width=3),
                                 marker=dict(size=6, color=PALETTE["orange"])))
        fig.add_trace(go.Scatter(x=[Wp[0]], y=[Bp[0]], mode="markers", name="الموقع",
                                 marker=dict(size=16, color=PALETTE["red"],
                                             line=dict(width=2, color="white"))))
        fig.frames = [go.Frame(data=[go.Contour(x=wg, y=bg, z=Jg),
                                     go.Scatter(x=Wp[:k + 1], y=Bp[:k + 1]),
                                     go.Scatter(x=[Wp[k]], y=[Bp[k]])],
                               name=str(k)) for k in range(len(Wp))]
        style_fig(fig, height=420, xaxis_title="w", yaxis_title="b")
        play_buttons(fig, frame_ms=90)
        st.plotly_chart(fig, width="stretch")
        st.caption("لاحظ أن المسار عمودي على خطوط التساوي — هذا هو معنى «اتجاه الهبوط الأشدّ».")

    with tabs[2]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(Cp))), y=Cp, mode="lines",
                                 name="التكلفة J", line=dict(color=PALETTE["teal"], width=3.5),
                                 fill="tozeroy", fillcolor="rgba(47,169,140,0.12)"))
        style_fig(fig, height=420, xaxis_title="رقم التكرار (Iteration)",
                  yaxis_title="J — التكلفة")
        st.plotly_chart(fig, width="stretch")
        st.markdown(
            "**هذا المنحنى هو أداة التشخيص الأولى عند أي باحث:**\n"
            "- ينزل بسلاسة ويستقرّ ← :green-badge[كل شيء سليم]\n"
            "- ينزل ببطء شديد ← :orange-badge[η صغير جدًا]\n"
            "- يتذبذب أو يصعد ← :red-badge[η كبير جدًا]\n"
            "- ينزل ثم يقفز فجأة ← :orange-badge[بيانات غير مقيَّسة أو تدرّج منفجر]"
        )

# ---------------------------------------------------------------------------
st.header("6) ثلاث نكهات من الانحدار التدريجي")

st.table({
    "**النوع**": ["**Batch GD**", "**Stochastic GD (SGD)**", "**Mini-batch GD**"],
    "**كم عيّنة لكل خطوة؟**": ["كل البيانات (n)", "عيّنة واحدة", "دفعة صغيرة (32, 64, 128…)"],
    "**الميزة**": ["مسار ناعم ومستقرّ", "سريع جدًا وقد يهرب من المنخفضات المحلية",
                   "توازن ممتاز + يستغلّ معالجة المصفوفات"],
    "**العيب**": ["بطيء جدًا مع البيانات الضخمة ويحتاج ذاكرة كبيرة",
                  "مسار شديد التذبذب وقد لا يستقرّ تمامًا", "يضيف هايبرباراميتر: حجم الدفعة"],
    "**الاستعمال**": [":gray-badge[نادر اليوم]", ":blue-badge[أحيانًا]",
                      ":green-badge[القياسي في كل مكان]"],
}, border="horizontal")

st.markdown(
    f"في الممارسة، {en('Mini-batch')} هو المعيار. وحجم الدفعة "
    f"({en('batch_size')}) يكون عادةً من قوى العدد 2 لأسباب تتعلّق بكفاءة المعالجات الرسومية."
)

st.subheader("كيف يبدو الفرق في المسار؟")
rng = np.random.default_rng(3)
fig = go.Figure()
for nm, noise, col in [("Batch GD", 0.0, PALETTE["indigo"]),
                       ("Mini-batch", 0.25, PALETTE["teal"]),
                       ("SGD", 0.8, PALETTE["orange"])]:
    pw, pb = 0.3, 0.0
    P = [(pw, pb)]
    for _ in range(45):
        e = (pw * x + pb) - y
        pw -= 0.012 * 2 * (np.mean(e * x) + rng.normal(0, noise * 3))
        pb -= 0.012 * 2 * (np.mean(e) + rng.normal(0, noise * 2))
        P.append((pw, pb))
    P = np.array(P)
    fig.add_trace(go.Scatter(x=P[:, 0], y=P[:, 1], mode="lines+markers", name=nm,
                             line=dict(color=col, width=2), marker=dict(size=5)))
style_fig(fig, height=400, xaxis_title="w", yaxis_title="b")
st.plotly_chart(fig, width="stretch")
st.caption("كلّما قلّ عدد العيّنات في الخطوة، زاد تذبذب المسار — لكن الاتجاه العام يبقى صحيحًا.")

# ---------------------------------------------------------------------------
st.header("7) المحسّنات الحديثة — Optimizers")

st.markdown("الانحدار التدريجي الكلاسيكي بطيء في بعض الأسطح. لذا طُوّرت تحسينات عليه:")

o1, o2 = st.columns(2, gap="medium")
with o1:
    with st.container(border=True):
        st.markdown("##### Momentum — الزخم")
        st.latex(r"v_t = \beta v_{t-1} + (1-\beta)\nabla J \;;\quad \theta \leftarrow \theta - \eta\, v_t")
        st.markdown(
            "يراكم «سرعة» من الخطوات السابقة، ككرة تتدحرج فتكتسب زخمًا. "
            "يخفّف التذبذب ويسرّع في الاتجاهات الثابتة. عادةً `β = 0.9`."
        )
    with st.container(border=True):
        st.markdown("##### RMSProp")
        st.markdown(
            "يعطي **معدّل تعلّم مختلفًا لكل معامل**، بقسمة التدرّج على جذر متوسط مربّعاته. "
            "المعاملات ذات التدرّجات الكبيرة تأخذ خطوات أصغر تلقائيًا."
        )
with o2:
    with st.container(border=True):
        st.markdown("##### Adam  :green-badge[الافتراضي]")
        st.latex(r"\theta \leftarrow \theta - \eta\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}")
        st.markdown(
            "يجمع Momentum + RMSProp معًا. هو المحسّن الافتراضي في التعلّم العميق منذ 2015. "
            "قيم شائعة: `η = 0.001`, `β₁ = 0.9`, `β₂ = 0.999`.\n\n"
            "إن لم تكن متأكدًا، ابدأ بـ Adam."
        )
    with st.container(border=True):
        st.markdown("##### AdamW")
        st.markdown(
            "نسخة من Adam تفصل تنظيم الأوزان (`weight decay`) عن التدرّج بشكل صحيح رياضيًا. "
            "هو المعيار الحالي في نماذج اللغة الكبيرة."
        )

with st.expander("ملاحظة مهمة: ليست كل الموديلات تُدرَّب بالانحدار التدريجي",
                 icon=":material/info:"):
    st.markdown(
        "- **الانحدار الخطي** له حلّ مغلق مباشر (المعادلات الطبيعية) لا يحتاج تكرارًا:"
    )
    st.latex(r"\hat{\boldsymbol{\beta}} = (\mathbf{X}^{\top}\mathbf{X})^{-1}\mathbf{X}^{\top}\mathbf{y}")
    st.markdown(
        "لكنه يتطلّب قلب مصفوفة بتعقيد ‎O(p³)‎، فيصبح مستحيلًا مع آلاف المتغيّرات — "
        "وهنا يعود الانحدار التدريجي ليكون أسرع.\n"
        "- **أشجار القرار** تُبنى بخوارزمية جشعة تختار أفضل تقسيم في كل عقدة، بلا تدرّجات أصلًا.\n"
        "- **KNN** لا يتدرّب إطلاقًا، بل يحفظ البيانات فقط."
    )

st.divider()
st.markdown(
    "#### :material/arrow_circle_left: الخطوة التالية\n"
    "**حلقة التدريب**: ما هو الـ Epoch والـ Batch والـ Iteration، وكيف تقرأ منحنيات التعلّم."
)
footer()
