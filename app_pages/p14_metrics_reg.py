import numpy as np
import plotly.graph_objects as go
import streamlit as st

from lib.ui import PALETTE, card, en, eq, footer, hero, setup, style_fig

setup("metrics_reg")

hero(
    "الصفحة 14 من 18",
    "معايير تقييم الانحدار  Regression Metrics",
    "درّبنا الموديل… فكيف نحكم عليه؟ وما الفرق بين MAE و RMSE؟ ولماذا قد يكون R² مرتفعًا "
    "وموديلك سيّئًا؟ هذه الصفحة تجيب بالأرقام وبالرسوم.",
)

st.info(
    "**تمييز مهم:** دالة الخسارة هي ما **يُحسّنه** الموديل أثناء التدريب. "
    "ومعيار التقييم هو ما **تقيس به أنت** الجودة وتبلّغه للناس. "
    "قد يتطابقان وقد يختلفان: يمكن أن تدرّب بـ MSE وتبلّغ بـ MAE لأنه أسهل تفسيرًا.",
    icon=":material/compare_arrows:",
)

# ---------------------------------------------------------------------------
st.header("1) المعايير الأساسية الخمسة")

m1, m2 = st.columns(2, gap="medium")
with m1:
    with st.container(border=True):
        st.markdown("##### MAE — متوسط الخطأ المطلق")
        st.latex(r"\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}\big|y_i - \hat{y}_i\big|")
        st.markdown(
            "**التفسير المباشر:** «في المتوسط، يُخطئ الموديل بمقدار MAE وحدة».\n\n"
            ":green-badge[ميزة] نفس وحدة الهدف — سهل جدًا شرحه لغير المختصّ.\n\n"
            ":blue-badge[خاصية] يعامل كل الأخطاء بالتساوي، فهو متين أمام الشواذّ.")
    with st.container(border=True):
        st.markdown("##### MSE — متوسط مربّع الخطأ")
        st.latex(r"\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}\big(y_i - \hat{y}_i\big)^2")
        st.markdown(
            ":red-badge[عيب] وحدته **مربّعة** (دينار²) فلا معنى له عمليًا.\n\n"
            ":green-badge[ميزة] قابل للاشتقاق ⟵ ممتاز كدالة خسارة للتدريب.")
    with st.container(border=True):
        st.markdown("##### RMSE — جذر متوسط مربّع الخطأ")
        st.latex(r"\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\big(y_i - \hat{y}_i\big)^2}")
        st.markdown(
            "جذر MSE، فيعود إلى وحدة الهدف. **المعيار الأكثر استعمالًا في التقارير.**\n\n"
            ":orange-badge[انتبه] يعاقب الأخطاء الكبيرة أكثر بكثير من MAE.")
with m2:
    with st.container(border=True):
        st.markdown("##### R² — معامل التحديد")
        st.latex(r"R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}"
                 r" = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}")
        st.markdown(
            "**نسبة التباين في الهدف التي يفسّرها الموديل.**\n\n"
            "- `R² = 1` ⟵ تنبؤ مثالي.\n"
            "- `R² = 0` ⟵ الموديل ليس أفضل من التنبؤ بالمتوسط دائمًا.\n"
            "- `R² < 0` ⟵ :red-badge[أسوأ من المتوسط] — ممكن جدًا على بيانات الاختبار!\n\n"
            "ميزته: **بلا وحدة**، فيُقارَن بين مسائل مختلفة.")
    with st.container(border=True):
        st.markdown("##### MAPE — متوسط الخطأ النسبي المئوي")
        st.latex(r"\text{MAPE} = \frac{100}{n}\sum_{i=1}^{n}"
                 r"\left|\frac{y_i - \hat{y}_i}{y_i}\right|")
        st.markdown(
            "خطأ كنسبة مئوية — مفهوم جدًا للإدارة: «نُخطئ بنسبة 8% في المتوسط».\n\n"
            ":red-badge[خطر] ينفجر إذا كانت هناك قيم `y` قريبة من الصفر، "
            "ويعاقب التقديرات الزائدة أكثر من الناقصة.")

st.markdown("**وهناك R² المعدّل، وهو مهم في البحث العلمي:**")
eq(r"R^2_{\text{adj}} = 1 - \big(1 - R^2\big)\,\frac{n-1}{n-p-1}",
   "n = عدد العيّنات،  p = عدد المتغيّرات")
st.markdown(
    "لماذا؟ لأن `R²` العادي **يرتفع دائمًا** كلّما أضفت متغيّرًا جديدًا، حتى لو كان عشوائيًا تمامًا. "
    "النسخة المعدّلة تعاقب على كثرة المتغيّرات، فترتفع فقط إذا كان المتغيّر الجديد يستحقّ."
)

# ---------------------------------------------------------------------------
st.header("2) آلة حاسبة تفاعلية — جرّب بنفسك")

st.markdown("حرّك المؤشّرات لترى كيف يستجيب كل معيار لنوع مختلف من الأخطاء.")

c1, c2 = st.columns([1, 2], gap="medium")
with c1:
    n = st.slider("عدد العيّنات", 20, 200, 60)
    noise = st.slider("مستوى الضوضاء العام", 0.0, 30.0, 8.0, 1.0)
    bias_sh = st.slider("إزاحة منهجية (تحيّز ثابت)", -20.0, 20.0, 0.0, 1.0)
    n_out = st.slider("عدد القيم الشاذّة", 0, 10, 0)

rng = np.random.default_rng(42)
y_true = rng.uniform(50, 300, n)
y_pred = y_true + rng.normal(bias_sh, noise, n)
if n_out:
    oi = rng.choice(n, n_out, replace=False)
    y_pred[oi] += rng.choice([-1, 1], n_out) * rng.uniform(90, 180, n_out)

err = y_true - y_pred
mae = float(np.mean(np.abs(err)))
mse = float(np.mean(err**2))
rmse = float(np.sqrt(mse))
r2 = 1 - np.sum(err**2) / np.sum((y_true - y_true.mean()) ** 2)
mape = float(np.mean(np.abs(err / y_true)) * 100)

with c2:
    k = st.columns(5)
    k[0].metric("MAE", f"{mae:,.2f}")
    k[1].metric("MSE", f"{mse:,.1f}")
    k[2].metric("RMSE", f"{rmse:,.2f}")
    k[3].metric("R²", f"{r2:.3f}")
    k[4].metric("MAPE", f"{mape:.1f}%")

    fig = go.Figure()
    lim = [min(y_true.min(), y_pred.min()) - 10, max(y_true.max(), y_pred.max()) + 10]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="التنبؤ المثالي (y = ŷ)",
                             line=dict(color=PALETTE["muted"], width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", name="التنبؤات",
                             marker=dict(size=9, color=PALETTE["indigo"], opacity=.75,
                                         line=dict(width=1, color="white"))))
    style_fig(fig, height=360, xaxis_title="القيمة الحقيقية  y",
              yaxis_title="القيمة المتنبَّأ بها  ŷ")
    st.plotly_chart(fig, width="stretch")

st.markdown(
    f"**جرّب هذه التجارب الثلاث لتفهم الفروق:**\n"
    f"1. اجعل عدد الشواذّ = 0 ثم 5: لاحظ أن **RMSE يقفز أكثر بكثير من MAE**. "
    f"الفرق بينهما مؤشّر مباشر على وجود شواذّ.\n"
    f"2. حرّك الإزاحة المنهجية: كل المعايير تسوء، والنقاط تنزاح كلّها فوق أو تحت الخط — "
    f"هذه علامة **تحيّز منهجي** يجب إصلاحه في الموديل لا في البيانات.\n"
    f"3. زد الضوضاء حتى يصبح `R²` سالبًا: عندها موديلك أسوأ من مجرّد قول «المتوسط»."
)

st.subheader("رسم البواقي — Residual plot")
st.markdown("أهم رسم تشخيصي في الانحدار. يجب أن تبدو البواقي **عشوائية بلا أي نمط**.")

fig = go.Figure()
fig.add_trace(go.Scatter(x=y_pred, y=err, mode="markers", name="البواقي",
                         marker=dict(size=9, color=PALETTE["teal"], opacity=.75,
                                     line=dict(width=1, color="white"))))
fig.add_hline(y=0, line=dict(color=PALETTE["red"], width=2.5))
style_fig(fig, height=330, xaxis_title="القيمة المتنبَّأ بها ŷ",
          yaxis_title="الباقي  e = y − ŷ")
st.plotly_chart(fig, width="stretch")

st.markdown(
    "**كيف تقرأ هذا الرسم؟**\n"
    "- سحابة عشوائية حول الصفر ← :green-badge[ممتاز]\n"
    "- شكل منحنٍ (قوس) ← :orange-badge[علاقة غير خطية لم يلتقطها الموديل] — أضف حدودًا تربيعية\n"
    "- شكل مخروطي يتّسع ← :orange-badge[تباين غير ثابت Heteroscedasticity] — جرّب تحويل log على y\n"
    "- البواقي كلّها فوق أو تحت الصفر في منطقة ← :red-badge[تحيّز منهجي]"
)

# ---------------------------------------------------------------------------
st.header("3) MAE أم RMSE؟ — كيف تختار")

st.table({
    "**السؤال**": ["هل الأخطاء الكبيرة أخطر بكثير من الصغيرة؟",
                   "هل في بياناتك قيم شاذّة حقيقية؟",
                   "هل ستشرح النتيجة لغير مختصّ؟",
                   "هل تقارن بين موديلات على نفس البيانات؟",
                   "هل الهدف يتراوح على مقاييس مختلفة جدًا؟"],
    "**إذا نعم**": [":blue-badge[RMSE]", ":green-badge[MAE]", ":green-badge[MAE]",
                    ":blue-badge[أيّهما بثبات]", ":violet-badge[MAPE أو R²]"],
    "**لماذا**": ["التربيع يضخّم الخطأ الكبير فيدفع الموديل لتجنّبه",
                  "MAE لا يتأثّر بها كثيرًا فيعطي صورة أصدق",
                  "«نُخطئ بـ 12 ألف دينار» أوضح من أي رقم آخر",
                  "المهم الاتّساق، لا نوع المعيار",
                  "المعايير النسبية تسمح بالمقارنة عبر المقاييس"],
}, border="horizontal")

eq(r"\text{MAE} \;\le\; \text{RMSE} \;\le\; \sqrt{n}\cdot\text{MAE}",
   "متباينة دائمة الصحّة: كلّما ابتعد RMSE عن MAE زاد تفاوت أحجام الأخطاء")

# ---------------------------------------------------------------------------
st.header("4) خط الأساس — بدونه كل الأرقام بلا معنى")

card(
    "لماذا هذا ضروري؟",
    "لنفترض أن موديلك أعطى RMSE = 25. هل هذا جيّد؟ <b>لا تستطيع الجواب</b> دون مرجع. "
    "خط الأساس يعطيك المرجع: ما هو خطأ أبسط حلّ ممكن؟",
    "teal",
)

base_pred = np.full_like(y_true, y_true.mean())
base_rmse = float(np.sqrt(np.mean((y_true - base_pred) ** 2)))
imp = (base_rmse - rmse) / base_rmse * 100

b1, b2, b3 = st.columns(3)
b1.metric("RMSE لخط الأساس (المتوسط)", f"{base_rmse:,.2f}")
b2.metric("RMSE لموديلك", f"{rmse:,.2f}")
b3.metric("نسبة التحسّن", f"{imp:,.1f}%",
          delta="أفضل من الأساس" if imp > 0 else "أسوأ من الأساس!")

st.code(
    """from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 1) خط الأساس: تنبّأ دائمًا بالمتوسط
baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)

# 2) قارن
for name, m in [("Baseline", baseline), ("My model", model)]:
    pred = m.predict(X_test)
    print(f"{name:10s} "
          f"MAE={mean_absolute_error(y_test, pred):8.3f}  "
          f"RMSE={np.sqrt(mean_squared_error(y_test, pred)):8.3f}  "
          f"R2={r2_score(y_test, pred):6.3f}")""",
    language="python",
)

st.warning(
    "**لا تبلّغ أبدًا عن رقم واحد فقط.** في أي تقرير أو بحث، اذكر: "
    "المعيار + قيمة خط الأساس + عدد عيّنات الاختبار + "
    f"الانحراف المعياري عبر طيّات التحقّق المتقاطع ({en('± std')}). "
    "الرقم بلا سياق وبلا قياس لعدم اليقين لا يُقنع مُحكِّمًا.",
    icon=":material/fact_check:",
)

st.divider()
st.markdown(
    "#### :material/arrow_circle_left: الخطوة التالية\n"
    "**معايير تقييم التصنيف**: مصفوفة الالتباس، الدقّة والاستدعاء، ومنحنى ROC."
)
footer()
