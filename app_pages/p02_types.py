import numpy as np
import plotly.graph_objects as go
import streamlit as st

from lib.ui import (PALETTE, card, en, footer, hero, setup, style_fig,
                    toy_classification, toy_regression)

setup("types")

hero(
    "الصفحة 2 من 18",
    "أنواع التعلّم  Types of Machine Learning",
    "قبل أن تكتب سطرًا واحدًا من الكود، عليك أن تعرف: ما نوع المسألة التي بين يديك؟ "
    "هذا السؤال وحده يحدّد الخوارزميات المسموح بها، ومعايير التقييم، وطريقة تجهيز البيانات.",
)

st.header("الخريطة الكبرى")

st.markdown(
    """
```mermaid
graph TD
    ML["Machine Learning"] --> SL["Supervised Learning<br/>تعلّم بإشراف"]
    ML --> UL["Unsupervised Learning<br/>تعلّم بدون إشراف"]
    ML --> RL["Reinforcement Learning<br/>تعلّم معزّز"]
    SL --> REG["Regression<br/>الهدف رقم"]
    SL --> CLF["Classification<br/>الهدف فئة"]
    UL --> CLU["Clustering<br/>تجميع"]
    UL --> DIM["Dimensionality Reduction<br/>تقليل الأبعاد"]
    UL --> ANO["Anomaly Detection<br/>كشف الشذوذ"]
    style ML fill:#EEF2FC,stroke:#5B6FD6,stroke-width:2px
    style SL fill:#E7F6F2,stroke:#2FA98C
    style UL fill:#FDF1E3,stroke:#EF9036
    style RL fill:#F6EFFC,stroke:#8A6FD1
```
"""
)

st.markdown(
    "السؤال المفتاحي الذي يفرّق بين الفروع الثلاثة هو: "
    f"**هل عندي عمود الأجوبة ({en('Label')}) أم لا؟**",
)

# ---------------------------------------------------------------------------
st.header("1) التعلّم بإشراف — Supervised Learning")

card(
    "التعريف",
    "عندنا بيانات فيها <b>المدخلات والأجوبة الصحيحة معًا</b>. الموديل يتعلّم العلاقة بينهما، "
    "تمامًا كطالب يدرس مسائل محلولة ثم يُمتحن في مسائل جديدة. "
    "هذا هو النوع الأكثر استعمالًا (أكثر من 90% من المشاريع العملية).",
    "teal",
)

st.markdown(
    "ينقسم إلى نوعين حسب **طبيعة عمود الهدف**، وهذا التقسيم ليس شكليًا: "
    "يغيّر دالة الخسارة ومعايير التقييم تمامًا."
)

t1, t2 = st.tabs([":material/show_chart: الانحدار — Regression",
                  ":material/scatter_plot: التصنيف — Classification"])

with t1:
    a, b = st.columns([1, 1.4], gap="medium")
    with a:
        st.markdown(
            "**الهدف رقم متّصل** يمكن أن يأخذ أي قيمة.\n\n"
            "**أمثلة:**\n"
            "- سعر منزل بالدينار\n"
            "- درجة حرارة الغد\n"
            "- عدد المبيعات الشهر القادم\n"
            "- عمر شخص من صورته\n\n"
            "**خوارزميات شائعة:**\n"
            "`LinearRegression`, `Ridge`, `RandomForestRegressor`, `XGBRegressor`\n\n"
            "**معايير التقييم:** MAE, RMSE, R²"
        )
    with b:
        x, y = toy_regression(n=45, noise=2.5)
        c = np.polyfit(x, y, 1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="بيانات",
                                 marker=dict(size=9, color=PALETTE["indigo"],
                                             line=dict(width=1, color="white"))))
        xs = np.linspace(x.min(), x.max(), 40)
        fig.add_trace(go.Scatter(x=xs, y=np.polyval(c, xs), mode="lines",
                                 name="الموديل", line=dict(color=PALETTE["orange"], width=3)))
        style_fig(fig, height=330, xaxis_title="المتغيّر", yaxis_title="الهدف (رقم متّصل)")
        st.plotly_chart(fig, width="stretch")
        st.caption("المخرَج قيمة على محور متّصل — لا يوجد «صواب/خطأ» بل «قريب/بعيد».")

with t2:
    a, b = st.columns([1, 1.4], gap="medium")
    with a:
        st.markdown(
            "**الهدف فئة** من مجموعة محدودة.\n\n"
            "**أمثلة:**\n"
            "- بريد: مزعج / عادي (ثنائي — Binary)\n"
            "- صورة: قط / كلب / حصان (متعدّد — Multiclass)\n"
            "- مريض: سيُشفى / لن يُشفى\n"
            "- عميل: سيترك الخدمة / سيبقى\n\n"
            "**خوارزميات شائعة:**\n"
            "`LogisticRegression`, `SVC`, `RandomForestClassifier`\n\n"
            "**معايير التقييم:** Accuracy, Precision, Recall, F1, ROC-AUC"
        )
    with b:
        X, yc = toy_classification(n=160, sep=1.5)
        fig = go.Figure()
        for k, (nm, col) in enumerate([("الفئة 0", PALETTE["indigo"]),
                                       ("الفئة 1", PALETTE["pink"])]):
            m = yc == k
            fig.add_trace(go.Scatter(x=X[m, 0], y=X[m, 1], mode="markers", name=nm,
                                     marker=dict(size=9, color=col,
                                                 line=dict(width=1, color="white"))))
        xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 10)
        fig.add_trace(go.Scatter(x=xs, y=-1.1 * xs, mode="lines", name="حدّ القرار",
                                 line=dict(color=PALETTE["teal"], width=3, dash="dash")))
        style_fig(fig, height=330, xaxis_title="المتغيّر 1", yaxis_title="المتغيّر 2")
        st.plotly_chart(fig, width="stretch")
        st.caption("الموديل يرسم **حدّ قرار** (Decision boundary) يفصل الفئات.")

st.warning(
    "**تنبيه مهم:** `LogisticRegression` اسمها «انحدار» لكنها خوارزمية **تصنيف**! "
    "التسمية تاريخية لأنها تستعمل انحدارًا داخليًا ثم تمرّره على دالة تحويل. "
    "لا تنخدع بالاسم.",
    icon=":material/warning:",
)

# ---------------------------------------------------------------------------
st.header("2) التعلّم بدون إشراف — Unsupervised Learning")

card(
    "التعريف",
    "عندنا المدخلات <b>فقط</b>، بلا أجوبة. المطلوب أن يكتشف الموديل بنية خفيّة في البيانات بنفسه. "
    "لا يوجد «جواب صحيح» نقارن به، ولهذا التقييم هنا أصعب بكثير وغالبًا ما يعتمد على حكم الخبير.",
    "orange",
)

u1, u2, u3 = st.columns(3, gap="medium")
with u1:
    with st.container(border=True):
        st.markdown("##### :material/bubble_chart: التجميع — Clustering")
        st.markdown(
            "تقسيم العيّنات إلى مجموعات متشابهة.\n\n"
            "**مثال:** تقسيم الزبائن إلى شرائح تسويقية.\n\n"
            "`KMeans`, `DBSCAN`, `AgglomerativeClustering`"
        )
with u2:
    with st.container(border=True):
        st.markdown("##### :material/compress: تقليل الأبعاد")
        st.markdown(
            "ضغط 100 متغيّر إلى 2 أو 3 مع الحفاظ على أهم المعلومات.\n\n"
            "**مثال:** تصوّر بيانات معقّدة في رسم ثنائي.\n\n"
            "`PCA`, `t-SNE`, `UMAP`"
        )
with u3:
    with st.container(border=True):
        st.markdown("##### :material/report: كشف الشذوذ")
        st.markdown(
            "إيجاد العيّنات الغريبة جدًا عن البقية.\n\n"
            "**مثال:** عملية بنكية احتيالية.\n\n"
            "`IsolationForest`, `OneClassSVM`"
        )

st.subheader("تجربة: نفس النقاط، بإشراف وبدون إشراف")

cc1, cc2 = st.columns([1, 2], gap="medium")
with cc1:
    show_labels = st.toggle("أظهر الأجوبة الحقيقية (Labels)", value=False)
    k = st.slider("عدد المجموعات k في KMeans", 2, 5, 2)
    st.caption(
        "بدون إشراف: الخوارزمية لا ترى الألوان الحقيقية إطلاقًا، "
        "بل تبحث عن تجمّعات بناءً على المسافات فقط."
    )
with cc2:
    from sklearn.cluster import KMeans

    Xc, yc2 = toy_classification(n=220, sep=1.8, seed=5)
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xc)
    fig = go.Figure()
    if show_labels:
        for kk, (nm, col) in enumerate([("الفئة الحقيقية 0", PALETTE["indigo"]),
                                        ("الفئة الحقيقية 1", PALETTE["pink"])]):
            m = yc2 == kk
            fig.add_trace(go.Scatter(x=Xc[m, 0], y=Xc[m, 1], mode="markers", name=nm,
                                     marker=dict(size=10, color=col,
                                                 line=dict(width=1, color="white"))))
    else:
        cols = [PALETTE["indigo"], PALETTE["teal"], PALETTE["orange"],
                PALETTE["pink"], PALETTE["violet"]]
        for kk in range(k):
            m = km.labels_ == kk
            fig.add_trace(go.Scatter(x=Xc[m, 0], y=Xc[m, 1], mode="markers",
                                     name=f"المجموعة {kk + 1}",
                                     marker=dict(size=10, color=cols[kk],
                                                 line=dict(width=1, color="white"))))
        fig.add_trace(go.Scatter(x=km.cluster_centers_[:, 0], y=km.cluster_centers_[:, 1],
                                 mode="markers", name="المراكز",
                                 marker=dict(size=20, color="white", symbol="x",
                                             line=dict(width=3, color=PALETTE["ink"]))))
    style_fig(fig, height=380, xaxis_title="المتغيّر 1", yaxis_title="المتغيّر 2")
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
st.header("3) التعلّم المعزّز — Reinforcement Learning")

card(
    "التعريف",
    "لا توجد بيانات جاهزة أصلًا. يوجد <b>وكيل</b> (Agent) يتصرّف داخل <b>بيئة</b> (Environment)، "
    "ويتلقّى <b>مكافأة</b> (Reward) أو عقوبة بعد كل فعل، فيتعلّم بالتجربة والخطأ السياسة "
    "(Policy) التي تعظّم مجموع المكافآت على المدى الطويل.<br>"
    "<b>أمثلة:</b> برامج تلعب الشطرنج، السيارات ذاتية القيادة، إدارة الطاقة في مركز بيانات.",
    "violet",
)

st.info(
    "التعلّم المعزّز حقل واسع وله رياضيات مختلفة (معادلة بلمان، عمليات ماركوف القرارية). "
    "هذه المنصة تركّز على **التعلّم بإشراف** لأنه مدخل كل باحث، "
    "لكن من المهم أن تعرف أن هذا الفرع موجود.",
    icon=":material/info:",
)

# ---------------------------------------------------------------------------
st.header("كيف أحدّد نوع مسألتي؟ — شجرة قرار عملية")

st.markdown(
    """
```mermaid
graph TD
    Q1{"هل عندي عمود<br/>أجوبة صحيحة؟"}
    Q1 -- "لا" --> Q3{"ماذا أريد؟"}
    Q1 -- "نعم" --> Q2{"نوع العمود؟"}
    Q2 -- "رقم متّصل" --> R["Regression<br/>انحدار"]
    Q2 -- "فئات" --> C["Classification<br/>تصنيف"]
    Q3 -- "مجموعات" --> CL["Clustering"]
    Q3 -- "تبسيط" --> PCA["Dimensionality Reduction"]
    Q3 -- "حالات غريبة" --> AD["Anomaly Detection"]
    style R fill:#E7F6F2,stroke:#2FA98C,stroke-width:2px
    style C fill:#FCE9F1,stroke:#D7568C,stroke-width:2px
    style CL fill:#FDF1E3,stroke:#EF9036
    style PCA fill:#FDF1E3,stroke:#EF9036
    style AD fill:#FDF1E3,stroke:#EF9036
```
"""
)

with st.expander("حالة خاصة شائعة: الهدف رقم لكن المسألة تصنيف", icon=":material/quiz:"):
    st.markdown(
        "لو كان الهدف «عدد النجوم من 1 إلى 5» فهو رقم، لكن الفروق بينه غير متساوية "
        "(الفرق بين 1 و2 ليس كالفرق بين 4 و5 من ناحية المعنى). "
        "هذه تُسمّى مسألة **ترتيبية** (`Ordinal`)، ويمكن معالجتها كانحدار أو كتصنيف، "
        "والاختيار يعتمد على ما يهمّك: الترتيب أم التطابق التام."
    )

st.divider()
st.markdown(
    "#### :material/arrow_circle_left: الخطوة التالية\n"
    "في الصفحة القادمة سنرى **خريطة المشروع الكاملة**: الخطوات الثماني "
    "التي يمرّ بها أي مشروع تعلّم آلة من الفكرة إلى النشر."
)
footer()
