import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from lib.ui import PALETTE, card, en, eq, footer, hero, setup, style_fig

setup("training")

hero(
    "الصفحة 10 من 18",
    "حلقة التدريب خطوة بخطوة  The Training Loop",
    "ماذا يحدث فعليًا داخل السطر الواحد `model.fit(X, y)`؟ "
    "هنا نفكّك الحلقة إلى خطواتها الخمس، ونفهم Epoch و Batch و Iteration، "
    "ونتعلّم قراءة منحنيات التعلّم — وهي أهم مهارة تشخيصية في المجال.",
)

# ---------------------------------------------------------------------------
st.header("1) الحلقة في خمس خطوات")

st.markdown(
    """
```mermaid
graph TD
    S["ابدأ: معاملات عشوائية<br/>Initialize θ"] --> A["1. Forward pass<br/>احسب التنبؤات ŷ"]
    A --> B["2. Compute loss<br/>احسب التكلفة J"]
    B --> C["3. Backward pass<br/>احسب التدرّجات ∇J"]
    C --> D["4. Update<br/>θ ← θ − η∇J"]
    D --> E{"5. هل انتهينا؟"}
    E -- "لا" --> A
    E -- "نعم" --> F["الموديل المدرَّب"]
    style S fill:#EEF2FC,stroke:#5B6FD6
    style B fill:#FDF1E3,stroke:#EF9036,stroke-width:2px
    style D fill:#E7F6F2,stroke:#2FA98C,stroke-width:2px
    style F fill:#F6EFFC,stroke:#8A6FD1,stroke-width:2px
```
"""
)

cols = st.columns(5, gap="small")
STEPS = [
    (":material/arrow_forward:", "تمرير أمامي", "Forward pass",
     "ندخل البيانات في الموديل بمعاملاته الحالية ونحصل على التنبؤات ŷ."),
    (":material/calculate:", "حساب الخسارة", "Compute loss",
     "نقارن ŷ بـ y الحقيقية عبر دالة التكلفة فنحصل على رقم واحد J."),
    (":material/arrow_back:", "تمرير خلفي", "Backpropagation",
     "نحسب مشتقّة J بالنسبة لكل معامل: من يتحمّل أي قدر من المسؤولية عن الخطأ؟"),
    (":material/update:", "تحديث المعاملات", "Update",
     "نحرّك كل معامل عكس تدرّجه بمقدار يحدّده معدّل التعلّم."),
    (":material/replay:", "كرّر", "Repeat",
     "نعيد الكرّة حتى يتوقّف التحسّن أو ننهي العدد المحدّد من الحقب."),
]
for c, (icon, ar, en_, desc) in zip(cols, STEPS):
    with c:
        with st.container(border=True):
            st.markdown(f"##### {icon} {ar}")
            st.caption(f"`{en_}`")
            st.markdown(desc)

st.markdown("**ترجمة الحلقة إلى كود صريح (تعليمي):**")
st.code(
    """# تهيئة
w, b = 0.0, 0.0
eta = 0.01

for epoch in range(n_epochs):              # 5. كرّر
    y_hat = w * X + b                      # 1. تمرير أمامي
    loss  = np.mean((y_hat - y) ** 2)      # 2. حساب الخسارة

    grad_w = 2 * np.mean((y_hat - y) * X)  # 3. التدرّجات
    grad_b = 2 * np.mean(y_hat - y)

    w -= eta * grad_w                      # 4. التحديث
    b -= eta * grad_b

    if epoch % 10 == 0:
        print(f"epoch {epoch}: loss = {loss:.4f}")""",
    language="python",
)

st.info(
    "كل المكتبات الحديثة (`scikit-learn`, `PyTorch`, `TensorFlow`) تفعل هذا بالضبط، "
    "لكنها تخفيه خلف `.fit()` وتضيف تحسينات في الأداء وحساب المشتقّات آليًا "
    f"({en('Automatic differentiation')}).",
    icon=":material/code_blocks:",
)

# ---------------------------------------------------------------------------
st.header("2) Epoch و Batch و Iteration — فرّق بينها جيدًا")

st.markdown("هذه الثلاثة تُخلط دائمًا، ومع ذلك العلاقة بينها بسيطة جدًا:")

b1, b2, b3 = st.columns(3, gap="medium")
with b1:
    card("Epoch — حقبة",
         "مرور <b>كامل</b> على كل بيانات التدريب مرّة واحدة. "
         "إذا كان عندك 1000 عيّنة، فالحقبة الواحدة = رأى الموديل الألف كلها مرّة.", "teal")
with b2:
    card("Batch — دفعة",
         "مجموعة جزئية من العيّنات تُعالَج معًا في خطوة تحديث واحدة. "
         "الأحجام الشائعة: 32, 64, 128, 256.", "orange")
with b3:
    card("Iteration — تكرار",
         "خطوة تحديث واحدة للمعاملات = معالجة دفعة واحدة. "
         "هي الوحدة الحقيقية للعمل.", "pink")

eq(r"\text{عدد التكرارات في الحقبة} \;=\; \left\lceil \frac{n}{\text{batch size}} \right\rceil",
   "n = عدد عيّنات التدريب")

cc1, cc2 = st.columns([1, 1.4], gap="medium")
with cc1:
    n_tr = st.number_input("عدد عيّنات التدريب n", 100, 100000, 1000, step=100)
    bsz = st.select_slider("حجم الدفعة batch_size", [1, 8, 16, 32, 64, 128, 256, 512], 32)
    n_ep = st.slider("عدد الحقب epochs", 1, 100, 10)
    it_per = int(np.ceil(n_tr / bsz))
    st.metric("تكرارات في كل حقبة", f"{it_per:,}")
    st.metric("إجمالي تحديثات المعاملات", f"{it_per * n_ep:,}")
with cc2:
    st.markdown(
        f"""
**ترجمة الأرقام إلى كلام:**

عندك **{n_tr:,}** عيّنة. في كل حقبة يقسّمها الموديل إلى **{it_per:,}** دفعة
حجم كل منها **{bsz}**. بعد كل دفعة يحدّث المعاملات مرّة.

وبما أنك طلبت **{n_ep}** حقبة، فالموديل سيحدّث معاملاته
**{it_per * n_ep:,}** مرّة إجمالًا، وسيرى كل عيّنة **{n_ep}** مرّات.

:blue-badge[قاعدة] تصغير الدفعة ⟵ تحديثات أكثر وأسرع لكن أكثر ضجيجًا.
تكبيرها ⟵ تحديثات أقلّ وأنعم لكن أبطأ في التقارب.
"""
    )

# ---------------------------------------------------------------------------
st.header("3) شاهد التدريب حيًا")

st.markdown(
    "اضغط الزر لتشاهد التدريب يحدث أمامك خطوة بخطوة، مع تحديث منحنى التكلفة في كل حقبة."
)

rng = np.random.default_rng(5)
Xd = np.sort(rng.uniform(0, 10, 120))
yd = 2.5 * Xd + 3 + rng.normal(0, 2.5, 120)
split = 90
Xtr, ytr, Xva, yva = Xd[:split], yd[:split], Xd[split:], yd[split:]

run_c1, run_c2 = st.columns([1, 3])
with run_c1:
    eta_live = st.select_slider("معدّل التعلّم", [0.002, 0.005, 0.01, 0.02], 0.01, key="live_lr")
    ep_live = st.slider("عدد الحقب", 10, 80, 40, key="live_ep")
    go_btn = st.button("ابدأ التدريب", icon=":material/play_arrow:", type="primary")

if go_btn:
    ph_metrics = st.empty()
    ph_chart = st.columns(2)
    ph_fit = ph_chart[0].empty()
    ph_loss = ph_chart[1].empty()
    prog = st.progress(0, text="جارٍ التدريب…")

    w, b = 0.0, 0.0
    hist_tr, hist_va = [], []
    xs = np.linspace(0, 10, 40)
    for ep in range(ep_live):
        e = (w * Xtr + b) - ytr
        w -= eta_live * 2 * float(np.mean(e * Xtr))
        b -= eta_live * 2 * float(np.mean(e))
        hist_tr.append(float(np.mean(((w * Xtr + b) - ytr) ** 2)))
        hist_va.append(float(np.mean(((w * Xva + b) - yva) ** 2)))

        with ph_metrics.container():
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("الحقبة", f"{ep + 1} / {ep_live}")
            k2.metric("خسارة التدريب", f"{hist_tr[-1]:.3f}")
            k3.metric("خسارة التحقّق", f"{hist_va[-1]:.3f}")
            k4.metric("المعاملات", f"w={w:.2f} · b={b:.2f}")

        f1 = go.Figure()
        f1.add_trace(go.Scatter(x=Xtr, y=ytr, mode="markers", name="تدريب",
                                marker=dict(size=7, color=PALETTE["indigo"], opacity=.75)))
        f1.add_trace(go.Scatter(x=Xva, y=yva, mode="markers", name="تحقّق",
                                marker=dict(size=9, color=PALETTE["pink"])))
        f1.add_trace(go.Scatter(x=xs, y=w * xs + b, mode="lines", name="الموديل",
                                line=dict(color=PALETTE["orange"], width=4)))
        style_fig(f1, height=300, xaxis_title="x", yaxis_title="y")
        ph_fit.plotly_chart(f1, width="stretch")

        f2 = go.Figure()
        f2.add_trace(go.Scatter(y=hist_tr, mode="lines", name="خسارة التدريب",
                                line=dict(color=PALETTE["indigo"], width=3)))
        f2.add_trace(go.Scatter(y=hist_va, mode="lines", name="خسارة التحقّق",
                                line=dict(color=PALETTE["pink"], width=3, dash="dash")))
        style_fig(f2, height=300, xaxis_title="الحقبة", yaxis_title="MSE")
        ph_loss.plotly_chart(f2, width="stretch")

        prog.progress((ep + 1) / ep_live, text=f"الحقبة {ep + 1} من {ep_live}")
        time.sleep(0.06)
    prog.empty()
    st.success(
        f"انتهى التدريب. المعاملات النهائية: w = {w:.3f} ، b = {b:.3f} "
        f"(القيم الحقيقية التي وُلّدت منها البيانات: w = 2.5 ، b = 3).",
        icon=":material/task_alt:",
    )

# ---------------------------------------------------------------------------
st.header("4) قراءة منحنيات التعلّم — Learning Curves")

card(
    "لماذا هذه المهارة هي الأهم؟",
    "منحنى التعلّم يرسم الخسارة على محور الحقب، لمنحنيين: التدريب والتحقّق. "
    "شكل المنحنيين وحده يخبرك ما المشكلة وما العلاج، قبل أن تجرّب أي شيء عشوائيًا. "
    "تعلّم قراءة الأشكال الأربعة التالية وستوفّر على نفسك ساعات.",
    "violet",
)

ep = np.arange(1, 61)
SHAPES = {
    "مثالي — Good fit": (
        2.6 * np.exp(-ep / 12) + 0.32, 2.8 * np.exp(-ep / 12) + 0.40,
        "المنحنيان ينزلان معًا ويستقرّان، والفجوة بينهما صغيرة.",
        "لا شيء — هذا ما نريد. يمكنك محاولة تحسين طفيف أو التوقّف هنا.", "success"),
    "تقصير — Underfitting": (
        1.9 * np.exp(-ep / 40) + 1.55, 2.0 * np.exp(-ep / 40) + 1.62,
        "المنحنيان مرتفعان ومتقاربان، ويستقرّان عند قيمة سيّئة.",
        "الموديل بسيط جدًا: زد تعقيده، أضف متغيّرات، درّب أطول، أو قلّل التنظيم.", "warning"),
    "إفراط — Overfitting": (
        2.6 * np.exp(-ep / 9) + 0.06, 2.6 * np.exp(-ep / 9) + 0.30 + 0.022 * ep,
        "خسارة التدريب تواصل النزول، لكن خسارة التحقّق تنزل ثم **ترتفع**.",
        "أوقف التدريب عند أدنى نقطة في منحنى التحقّق (Early stopping)، "
        "أو أضف تنظيمًا، أو زد البيانات، أو بسّط الموديل.", "error"),
    "تذبذب — Unstable": (
        2.4 * np.exp(-ep / 14) + 0.35 + 0.22 * np.sin(ep * 1.6),
        2.5 * np.exp(-ep / 14) + 0.45 + 0.28 * np.sin(ep * 1.6 + 0.6),
        "المنحنيان يقفزان صعودًا وهبوطًا بلا استقرار.",
        "معدّل التعلّم كبير جدًا، أو حجم الدفعة صغير جدًا، أو البيانات غير مقيَّسة.", "warning"),
}
pick = st.segmented_control("اختر الحالة لتشخيصها", list(SHAPES), default="مثالي — Good fit")
tr_c, va_c, read, fix, kind = SHAPES[pick]

d1, d2 = st.columns([1.7, 1], gap="medium")
with d1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ep, y=tr_c, mode="lines", name="خسارة التدريب",
                             line=dict(color=PALETTE["indigo"], width=3.5)))
    fig.add_trace(go.Scatter(x=ep, y=va_c, mode="lines", name="خسارة التحقّق",
                             line=dict(color=PALETTE["pink"], width=3.5, dash="dash")))
    if "إفراط" in pick:
        k = int(np.argmin(va_c))
        fig.add_vline(x=ep[k], line=dict(color=PALETTE["teal"], width=2, dash="dot"))
        fig.add_annotation(x=ep[k], y=float(va_c.max()), text="أوقف هنا ★",
                           showarrow=False, font=dict(color=PALETTE["teal"], size=14))
    style_fig(fig, height=380, xaxis_title="الحقبة Epoch", yaxis_title="الخسارة Loss")
    st.plotly_chart(fig, width="stretch")
with d2:
    st.markdown("##### :material/visibility: ما تراه")
    st.markdown(read)
    st.markdown("##### :material/build: العلاج")
    getattr(st, kind)(fix, icon=":material/healing:")

# ---------------------------------------------------------------------------
st.header("5) متى نتوقّف؟ — Stopping criteria")

s1, s2, s3 = st.columns(3, gap="medium")
with s1:
    with st.container(border=True):
        st.markdown("##### عدد حقب ثابت")
        st.markdown("أبسط طريقة: `epochs=100`. العيب: قد نتوقّف مبكرًا جدًا أو متأخّرًا جدًا.")
with s2:
    with st.container(border=True):
        st.markdown("##### عتبة تحسّن")
        st.markdown("نتوقّف إذا لم تنخفض الخسارة بأكثر من `tol` (مثلًا 1e-4) لعدّة حقب.")
with s3:
    with st.container(border=True):
        st.markdown("##### :material/star: Early stopping")
        st.markdown(
            "نراقب خسارة **التحقّق**: إذا لم تتحسّن لمدة `patience` حقبة، نتوقّف "
            "ونُرجع أفضل معاملات. **الطريقة المفضّلة.**")

st.code(
    """from sklearn.neural_network import MLPRegressor

model = MLPRegressor(
    max_iter=500,
    early_stopping=True,       # فعّل التوقّف المبكّر
    validation_fraction=0.15,  # يقتطع 15% من التدريب للمراقبة
    n_iter_no_change=10,       # patience: اصبر 10 حقب
    tol=1e-4,
)""",
    language="python",
)

st.divider()
st.markdown(
    "#### :material/arrow_circle_left: الخطوة التالية\n"
    "ندخل قسم **جودة التعلّم**: الإفراط والتقصير — أهم مشكلتين في تعلّم الآلة."
)
footer()
