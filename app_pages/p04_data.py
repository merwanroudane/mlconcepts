import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.ui import PALETTE, card, en, eq, footer, hero, setup, style_fig

setup("data")

hero(
    "الصفحة 4 من 18",
    "فهم البيانات وتنظيفها  Understanding & Cleaning Data",
    "«بيانات سيّئة ← موديل سيّئ» مهما كانت الخوارزمية عبقرية. "
    "في هذه الصفحة: أنواع المتغيّرات، القيم المفقودة، القيم الشاذّة، وأخطر فخّ في المجال كلّه: تسريب البيانات.",
)


@st.cache_data
def make_messy(seed: int = 42, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    area = rng.normal(140, 45, n).clip(40, 400)
    rooms = np.round(area / 40 + rng.normal(0, 0.6, n)).clip(1, 8)
    age = rng.integers(0, 45, n).astype(float)
    city = rng.choice(["الجزائر", "وهران", "قسنطينة", "عنابة"], n, p=[.4, .25, .2, .15])
    quality = rng.choice(["ممتاز", "جيد", "متوسط"], n, p=[.3, .45, .25])
    qmap = {"ممتاز": 1.25, "جيد": 1.0, "متوسط": 0.82}
    cmap = {"الجزائر": 1.4, "وهران": 1.1, "قسنطينة": 0.95, "عنابة": 0.9}
    price = (area * 1.8 + rooms * 12 - age * 1.6) * \
            np.array([qmap[q] for q in quality]) * \
            np.array([cmap[c] for c in city]) + rng.normal(0, 18, n)
    df = pd.DataFrame({"المساحة": area.round(1), "الغرف": rooms, "عمر_البناء": age,
                       "المدينة": city, "الحالة": quality, "السعر": price.round(1)})
    # نحقن فوضى واقعية
    df.loc[rng.choice(n, 28, replace=False), "عمر_البناء"] = np.nan
    df.loc[rng.choice(n, 14, replace=False), "الحالة"] = None
    df.loc[rng.choice(n, 4, replace=False), "المساحة"] = [1200, 980, 1500, 1100]
    df = pd.concat([df, df.iloc[:6]], ignore_index=True)  # صفوف مكرّرة
    return df


df = make_messy()

# ---------------------------------------------------------------------------
st.header("1) أول نظرة — اعرف ما بين يديك")

m1, m2, m3, m4 = st.columns(4)
m1.metric("عدد الصفوف (عيّنات)", f"{df.shape[0]:,}")
m2.metric("عدد الأعمدة", f"{df.shape[1]}")
m3.metric("قيم مفقودة", f"{int(df.isna().sum().sum())}")
m4.metric("صفوف مكرّرة", f"{int(df.duplicated().sum())}")

st.dataframe(df.head(8), width="stretch")

st.code(
    """import pandas as pd
df = pd.read_csv("data.csv")

df.shape            # (عدد الصفوف، عدد الأعمدة)
df.head()           # أول 5 صفوف
df.info()           # أنواع الأعمدة وعدد القيم غير الفارغة
df.describe()       # إحصاءات وصفية للأعمدة الرقمية
df.isna().sum()     # عدد القيم المفقودة في كل عمود
df.duplicated().sum()   # عدد الصفوف المكرّرة
df["المدينة"].value_counts()   # تكرار كل فئة""",
    language="python",
)

# ---------------------------------------------------------------------------
st.header("2) أنواع المتغيّرات — Variable types")

st.markdown("التمييز بينها ضروري لأن كل نوع يُعالَج بطريقة مختلفة تمامًا.")

v1, v2 = st.columns(2, gap="medium")
with v1:
    card("كمّية متّصلة — Continuous",
         "أي قيمة ضمن مجال: المساحة، السعر، درجة الحرارة.<br>"
         "<b>المعالجة:</b> تقييس (Scaling)، معالجة الالتواء.", "teal")
    card("كمّية منفصلة — Discrete / Count",
         "أعداد صحيحة: عدد الغرف، عدد الأطفال.<br>"
         "<b>المعالجة:</b> كالكمّية غالبًا.", "teal")
with v2:
    card("اسمية — Nominal categorical",
         "فئات بلا ترتيب: المدينة، اللون، نوع الوقود.<br>"
         f"<b>المعالجة:</b> {en('One-Hot Encoding')} — عمود لكل فئة.", "orange")
    card("ترتيبية — Ordinal categorical",
         "فئات لها ترتيب طبيعي: متوسط &lt; جيد &lt; ممتاز.<br>"
         f"<b>المعالجة:</b> {en('Ordinal Encoding')} — نعطيها أرقامًا تحترم الترتيب (0,1,2).",
         "orange")

st.error(
    "**خطأ قاتل شائع:** ترميز المدن بأرقام (الجزائر=1، وهران=2، قسنطينة=3). "
    "الموديل سيفهم أن «قسنطينة أكبر من وهران» وأن «وهران = متوسط الجزائر وقسنطينة»، "
    "وهذا هراء. المتغيّر الاسمي يحتاج One-Hot وليس ترقيمًا.",
    icon=":material/dangerous:",
)

# ---------------------------------------------------------------------------
st.header("3) القيم المفقودة — Missing values")

st.markdown("أولًا، افهم **لماذا** هي مفقودة، فذلك يحدّد العلاج:")

t1, t2, t3 = st.tabs(["MCAR — مفقود عشوائيًا تمامًا",
                      "MAR — مفقود عشوائيًا مشروطًا",
                      "MNAR — مفقود لسبب غير عشوائي"])
with t1:
    st.markdown(
        "**Missing Completely At Random:** الفقدان لا علاقة له بأي شيء "
        "(جهاز قياس تعطّل يومًا عشوائيًا).\n\n"
        ":green-badge[آمن] الحذف أو التعويض البسيط لا يُحدث تحيّزًا."
    )
with t2:
    st.markdown(
        "**Missing At Random:** الفقدان يعتمد على متغيّرات **أخرى مرصودة** "
        "(كبار السن يتركون خانة الدخل فارغة أكثر).\n\n"
        ":orange-badge[انتبه] التعويض الذكي (بالانحدار أو KNN) أفضل من المتوسط."
    )
with t3:
    st.markdown(
        "**Missing Not At Random:** الفقدان يعتمد على **القيمة المفقودة نفسها** "
        "(أصحاب الدخول العالية جدًا يرفضون الإفصاح).\n\n"
        ":red-badge[خطر] أي تعويض سيُحدث تحيّزًا. الحل: أضف عمودًا ثنائيًا "
        "`was_missing` يخبر الموديل أن القيمة كانت غائبة — فالغياب نفسه معلومة!"
    )

st.subheader("طرق المعالجة ومقارنتها")

st.table(pd.DataFrame({
    "الطريقة": ["حذف الصفوف", "حذف العمود", "تعويض بالمتوسط Mean",
                "تعويض بالوسيط Median", "تعويض بالمنوال Mode",
                "تعويض بـ KNN / Iterative", "قيمة خاصة + عمود مؤشّر"],
    "متى تُستعمل": ["المفقود < 5% والبيانات كثيرة", "العمود مفقود > 50%",
                    "عمود رقمي متماثل بلا قيم شاذّة", "عمود رقمي ملتوٍ أو فيه شواذّ",
                    "عمود فئوي", "علاقات قوية بين الأعمدة", "الفقدان نفسه له معنى"],
    "الخطر": ["فقدان بيانات وتحيّز", "فقدان معلومة قد تكون مهمة",
              "يقلّل التباين الحقيقي", "آمن نسبيًا", "يضخّم الفئة الأكثر شيوعًا",
              "بطيء وقد يُفرط في التعلّم", "لا شيء تقريبًا — مستحسن"],
}), border="horizontal")

st.code(
    """from sklearn.impute import SimpleImputer, KNNImputer

num_imp = SimpleImputer(strategy="median")    # للأعمدة الرقمية
cat_imp = SimpleImputer(strategy="most_frequent")   # للأعمدة الفئوية

# مهم جدًا: نتعلّم القيمة من التدريب فقط
num_imp.fit(X_train[num_cols])
X_train[num_cols] = num_imp.transform(X_train[num_cols])
X_test[num_cols]  = num_imp.transform(X_test[num_cols])   # transform فقط، بلا fit!""",
    language="python",
)

st.subheader("تجربة: أثر طريقة التعويض على التوزيع")

col = "عمر_البناء"
method = st.segmented_control(
    "اختر طريقة التعويض", ["بلا تعويض (حذف)", "المتوسط Mean", "الوسيط Median", "صفر"],
    default="بلا تعويض (حذف)",
)
s = df[col].copy()
if method == "المتوسط Mean":
    s = s.fillna(s.mean())
elif method == "الوسيط Median":
    s = s.fillna(s.median())
elif method == "صفر":
    s = s.fillna(0)
else:
    s = s.dropna()

fig = go.Figure()
fig.add_trace(go.Histogram(x=df[col].dropna(), name="الأصلي (بلا مفقود)", nbinsx=24,
                           marker_color=PALETTE["indigo"], opacity=0.55))
fig.add_trace(go.Histogram(x=s, name=f"بعد: {method}", nbinsx=24,
                           marker_color=PALETTE["orange"], opacity=0.55))
fig.update_layout(barmode="overlay")
style_fig(fig, height=330, xaxis_title="عمر البناء", yaxis_title="التكرار")
st.plotly_chart(fig, width="stretch")
st.caption(
    "لاحظ كيف يصنع التعويض بالمتوسط «عمودًا صناعيًا» عند قيمة واحدة — "
    "هذا يقلّل التباين الحقيقي ويضلّل الموديل إن كانت النسبة كبيرة."
)

# ---------------------------------------------------------------------------
st.header("4) القيم الشاذّة — Outliers")

st.markdown(
    "قيمة بعيدة جدًا عن بقية البيانات. قد تكون **خطأ إدخال** (مساحة = 1200 م² في حيّ شقق) "
    "أو **حالة حقيقية نادرة** (قصر فعلًا). الفرق حاسم: الأولى تُصحَّح، والثانية قد تكون أهم ما في البيانات."
)

st.markdown("**الطريقة الأشهر للكشف: قاعدة المدى الربيعي IQR**")
eq(r"\text{IQR} = Q_3 - Q_1 \qquad\Longrightarrow\qquad "
   r"\text{شاذّة إذا } \; x < Q_1 - 1.5\,\text{IQR} \;\;\text{أو}\;\; x > Q_3 + 1.5\,\text{IQR}",
   "Q₁ = الربيع الأول (25%)، Q₃ = الربيع الثالث (75%)")

st.markdown("**وطريقة ثانية: الدرجة المعيارية Z-score** (تصلح للتوزيع الطبيعي فقط)")
eq(r"z_i = \frac{x_i - \mu}{\sigma} \qquad\Longrightarrow\qquad \text{شاذّة إذا } |z_i| > 3",
   "μ = المتوسط،  σ = الانحراف المعياري")

o1, o2 = st.columns([1, 2], gap="medium")
with o1:
    thr = st.slider("معامل الـ IQR", 1.0, 3.0, 1.5, 0.1)
    q1, q3 = df["المساحة"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - thr * iqr, q3 + thr * iqr
    mask = (df["المساحة"] < lo) | (df["المساحة"] > hi)
    st.metric("عدد القيم الشاذّة", int(mask.sum()))
    st.caption(f"الحدّ الأدنى = {lo:,.1f} ، الحدّ الأعلى = {hi:,.1f}")
    st.markdown(
        "**ماذا تفعل بها؟**\n"
        "- تحقّق من المصدر أولًا.\n"
        "- خطأ واضح ← صحّح أو احذف.\n"
        "- حالة حقيقية ← أبقِها، واستعمل موديلًا قويًا أمام الشواذّ "
        "(الأشجار) أو خسارة Huber.\n"
        "- أو قصّ القيم `Winsorizing` عند حدّ معيّن."
    )
with o2:
    fig = go.Figure()
    fig.add_trace(go.Box(x=df["المساحة"], name="المساحة", boxpoints="all",
                         jitter=0.4, pointpos=-1.6, marker_color=PALETTE["indigo"],
                         marker=dict(size=5, opacity=0.6), line_color=PALETTE["indigo"]))
    fig.add_vline(x=lo, line=dict(color=PALETTE["red"], dash="dash"))
    fig.add_vline(x=hi, line=dict(color=PALETTE["red"], dash="dash"))
    style_fig(fig, height=330, legend_top=False, xaxis_title="المساحة (م²)")
    st.plotly_chart(fig, width="stretch")
    st.caption("صندوق الشِّوار (Box plot): الصندوق يغطّي 50% الوسطى، والنقاط خارج الخطين شاذّة.")

# ---------------------------------------------------------------------------
st.header("5) العلاقات بين المتغيّرات — Correlation")

num = df.select_dtypes("number").dropna()
corr = num.corr().round(2)
fig = go.Figure(go.Heatmap(
    z=corr.values, x=corr.columns, y=corr.columns, zmin=-1, zmax=1,
    colorscale=[[0, "#E05C6E"], [0.5, "#FFFFFF"], [1, "#5B6FD6"]],
    text=corr.values, texttemplate="%{text}",
    textfont=dict(size=14), colorbar=dict(title="r"),
))
style_fig(fig, height=400, legend_top=False)
st.plotly_chart(fig, width="stretch")

st.markdown(
    "معامل الارتباط الخطي **بيرسون** يتراوح بين ‎−1‎ و‎+1‎: "
    "قريب من ‎+1‎ = علاقة طردية قوية، قريب من ‎−1‎ = عكسية قوية، قريب من 0 = لا علاقة **خطية**."
)
eq(r"r_{xy} = \frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}"
   r"{\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2}\;\sqrt{\sum_{i=1}^{n}(y_i-\bar{y})^2}}")

w1, w2 = st.columns(2, gap="medium")
with w1:
    st.warning(
        "**الارتباط ≠ السببية.** ارتفاع مبيعات المثلّجات يرتبط بارتفاع حوادث الغرق، "
        "والسبب الحقيقي ثالث: الصيف. الموديل يستغلّ الارتباط للتنبؤ، لكن لا يثبت السببية.",
        icon=":material/link_off:",
    )
with w2:
    st.warning(
        f"**الارتباط العالي بين متغيّرين مستقلّين** ({en('Multicollinearity')}) "
        "يجعل معاملات الانحدار الخطي غير مستقرّة وصعبة التفسير. "
        "الحل: احذف أحدهما، أو ادمجهما، أو استعمل Ridge.",
        icon=":material/join_inner:",
    )

# ---------------------------------------------------------------------------
st.header("6) تسريب البيانات — Data Leakage :red-badge[الأخطر]")

card(
    "ما هو؟",
    "أن تتسرّب إلى بيانات التدريب معلومة <b>لن تكون متاحة وقت التنبؤ الحقيقي</b>، "
    "أو معلومة مأخوذة من بيانات الاختبار. النتيجة: نتائج رائعة في الدفتر، وفشل ذريع في الواقع.",
    "pink",
)

st.markdown("**الأشكال الثلاثة الأشهر:**")

l1, l2, l3 = st.columns(3, gap="medium")
with l1:
    with st.container(border=True):
        st.markdown("##### :material/schedule: تسريب زمني")
        st.markdown(
            "استعمال معلومة من **المستقبل**.\n\n"
            "مثال: للتنبؤ بترك العميل للخدمة، نُدخل عمود «تاريخ إلغاء الاشتراك»! "
            "هذا العمود موجود فقط **بعد** حدوث ما نريد التنبؤ به."
        )
with l2:
    with st.container(border=True):
        st.markdown("##### :material/calculate: تسريب التجهيز")
        st.markdown(
            "حساب المتوسط/الانحراف من **كل** البيانات ثم التقسيم.\n\n"
            "بيانات الاختبار أثّرت في التحويل، فتسرّبت معلومة عنها إلى التدريب."
        )
        st.code("# خطأ\nX = scaler.fit_transform(X)\nsplit(X)\n\n"
                "# صواب\nsplit(X)\nscaler.fit(X_train)", language="python")
with l3:
    with st.container(border=True):
        st.markdown("##### :material/content_copy: تسريب التكرار")
        st.markdown(
            "صفوف مكرّرة أو شديدة التشابه موزّعة بين التدريب والاختبار.\n\n"
            "الموديل «يحفظ» الصف ثم يراه في الاختبار فيبدو عبقريًا."
        )

st.success(
    "**العلاج المنهجي الوحيد:** ضع كل خطوات التجهيز داخل `Pipeline` من `scikit-learn`، "
    "واجعل التقسيم أول شيء يحدث. حينها يستحيل تقريبًا أن تسرّب شيئًا دون قصد. "
    "سنبني هذا الـ Pipeline كاملًا في صفحة المشروع.",
    icon=":material/verified_user:",
)

with st.expander("علامة إنذار: متى أشكّ في وجود تسريب؟", icon=":material/notification_important:"):
    st.markdown(
        "- دقّة مرتفعة بشكل **غير معقول** (99.8% في مسألة يعرف الخبراء أنها صعبة).\n"
        "- متغيّر واحد يفسّر كل شيء تقريبًا في أهمية المتغيّرات.\n"
        "- الأداء على الاختبار **أفضل** من التدريب.\n"
        "- الأداء ينهار تمامًا عند النشر على بيانات جديدة.\n\n"
        "في كل هذه الحالات: ارجع وافحص كل عمود واسأل «هل هذه المعلومة متاحة فعلًا لحظة التنبؤ؟»"
    )

st.divider()
st.markdown(
    "#### :material/arrow_circle_left: الخطوة التالية\n"
    "**هندسة المتغيّرات والتقييس**: كيف نحوّل الأعمدة إلى أرقام، ولماذا التقييس ضروري لبعض الخوارزميات دون غيرها."
)
footer()
