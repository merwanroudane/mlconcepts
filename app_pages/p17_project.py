import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.ui import PALETTE, card, en, footer, hero, setup, steps_bar, style_fig

setup("project")

hero(
    "الصفحة 17 من 18",
    "مشروع كامل خطوة بخطوة  End-to-End ML Project",
    "الآن نجمع كل شيء. مسألة تصنيف حقيقية — التنبؤ بتسرّب العملاء — "
    "تُنفَّذ أمامك فعليًا بالكود ونتائجه، من البيانات الخام إلى موديل جاهز للنشر.",
)

STEPS = ["المشكلة", "البيانات", "الاستكشاف", "التقسيم", "الأنبوب",
         "خط الأساس", "المقارنة", "الضبط", "التقييم", "التفسير", "الحفظ"]
steps_bar(STEPS)

# ===========================================================================
# البيانات
# ===========================================================================
@st.cache_data
def build_dataset(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.gamma(2.2, 9, n).clip(1, 72).round()
    monthly = rng.normal(68, 26, n).clip(18, 140)
    support = rng.poisson(1.3, n)
    contract = rng.choice(["شهري", "سنة", "سنتان"], n, p=[.55, .27, .18])
    payment = rng.choice(["بطاقة بنكية", "تحويل تلقائي", "شيك ورقي", "محفظة إلكترونية"],
                         n, p=[.32, .28, .22, .18])
    internet = rng.choice(["ألياف بصرية", "DSL", "بلا إنترنت"], n, p=[.45, .38, .17])
    senior = rng.choice([0, 1], n, p=[.84, .16])

    logit = (
        -1.15
        - 0.052 * tenure
        + 0.021 * (monthly - 68)
        + 0.33 * support
        + np.where(contract == "شهري", 1.25, np.where(contract == "سنة", 0.15, -0.65))
        + np.where(payment == "شيك ورقي", 0.52, 0.0)
        + np.where(internet == "ألياف بصرية", 0.46, 0.0)
        + 0.33 * senior
        + rng.normal(0, 0.65, n)
    )
    churn = (1 / (1 + np.exp(-logit)) > rng.random(n)).astype(int)

    df = pd.DataFrame({
        "مدة_الاشتراك_شهر": tenure,
        "الفاتورة_الشهرية": monthly.round(2),
        "مكالمات_الدعم": support,
        "كبير_السن": senior,
        "نوع_العقد": contract,
        "طريقة_الدفع": payment,
        "نوع_الإنترنت": internet,
        "تسرّب": churn,
    })
    # فوضى واقعية
    df.loc[rng.choice(n, 120, replace=False), "الفاتورة_الشهرية"] = np.nan
    df.loc[rng.choice(n, 70, replace=False), "نوع_الإنترنت"] = None
    return df


df = build_dataset()
TARGET = "تسرّب"

# ===========================================================================
tabs = st.tabs([
    ":material/flag: 1–3 الإطار والبيانات",
    ":material/call_split: 4–5 التقسيم والأنبوب",
    ":material/model_training: 6–8 التدريب والضبط",
    ":material/analytics: 9–10 التقييم والتفسير",
    ":material/save: 11 النشر والقائمة النهائية",
])

# ---------------------------------------------------------------------------
with tabs[0]:
    st.header("الخطوة 1 — تحديد المشكلة")
    card(
        "وثيقة المشروع (اكتبها قبل أي كود)",
        "<b>السؤال:</b> بالنظر إلى بيانات عميل، ما احتمال أن يُلغي اشتراكه خلال الفترة القادمة؟<br>"
        "<b>نوع المسألة:</b> تصنيف ثنائي (Binary classification).<br>"
        "<b>الهدف y:</b> `تسرّب` — 1 = ألغى، 0 = بقي.<br>"
        "<b>الفائدة العملية:</b> استهداف العملاء المعرّضين للتسرّب بعرض احتفاظ قبل أن يرحلوا.<br>"
        "<b>معيار النجاح:</b> نركّز على `Recall` لأن تفويت عميل سيرحل أغلى من عرض تخفيض "
        "لعميل كان سيبقى — ثم نوازن بـ `F1` و `ROC-AUC`.<br>"
        "<b>خط الأساس:</b> التنبؤ بالفئة الأكثر شيوعًا (الجميع يبقى).",
    )
    st.warning(
        "**تكلفة الخطأين ليست متساوية هنا:**\n"
        "- إنذار كاذب (FP): نعرض تخفيضًا على عميل كان سيبقى ← تكلفة محدودة.\n"
        "- تفويت (FN): يرحل العميل دون أن نحاول ← خسارة إيراداته كاملة.\n\n"
        "هذا القرار التجاري هو ما سيحدّد لاحقًا **عتبة القرار** ومعيار التقييم.",
        icon=":material/payments:",
    )

    st.header("الخطوة 2 — البيانات")
    st.code(
        """import pandas as pd
df = pd.read_csv("customer_churn.csv")
df.shape, df.dtypes, df.head()""", language="python")

    k = st.columns(4)
    k[0].metric("عدد العملاء", f"{len(df):,}")
    k[1].metric("عدد المتغيّرات", f"{df.shape[1] - 1}")
    k[2].metric("نسبة التسرّب", f"{df[TARGET].mean():.1%}")
    k[3].metric("قيم مفقودة", f"{int(df.isna().sum().sum())}")
    st.dataframe(df.head(10), width="stretch")

    st.header("الخطوة 3 — الاستكشاف EDA")
    e1, e2 = st.columns(2, gap="medium")
    with e1:
        vc = df[TARGET].value_counts().sort_index()
        fig = go.Figure(go.Bar(x=["بقي (0)", "تسرّب (1)"], y=vc.values,
                               marker_color=[PALETTE["indigo"], PALETTE["pink"]],
                               text=[f"{v:,}<br>({v / len(df):.1%})" for v in vc.values],
                               textposition="outside"))
        style_fig(fig, height=330, legend_top=False, title="توازن الفئات",
                  yaxis_title="عدد العملاء")
        fig.update_yaxes(range=[0, vc.max() * 1.22])
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"نسبة {df[TARGET].mean():.1%} تعني اختلالًا معتدلًا: "
            "الدقّة الكلية ستكون مضلّلة، وسنستعمل `class_weight='balanced'`.")
    with e2:
        num_col = st.selectbox("متغيّر عددي للمقارنة",
                               ["مدة_الاشتراك_شهر", "الفاتورة_الشهرية", "مكالمات_الدعم"])
        fig = go.Figure()
        for v, nm, col in [(0, "بقي", PALETTE["indigo"]), (1, "تسرّب", PALETTE["pink"])]:
            fig.add_trace(go.Violin(y=df.loc[df[TARGET] == v, num_col].dropna(), name=nm,
                                    box_visible=True, meanline_visible=True,
                                    line_color=col, opacity=.7))
        style_fig(fig, height=330, title=f"{num_col} حسب التسرّب", yaxis_title=num_col)
        st.plotly_chart(fig, width="stretch")

    cat = st.selectbox("متغيّر فئوي", ["نوع_العقد", "طريقة_الدفع", "نوع_الإنترنت"])
    g = df.groupby(cat, observed=True)[TARGET].agg(["mean", "count"]).reset_index()
    fig = go.Figure(go.Bar(x=g[cat], y=g["mean"], marker_color=PALETTE["orange"],
                           text=[f"{v:.1%}" for v in g["mean"]], textposition="outside"))
    fig.add_hline(y=df[TARGET].mean(), line=dict(color=PALETTE["muted"], dash="dash"))
    style_fig(fig, height=330, legend_top=False,
              title=f"معدّل التسرّب حسب {cat} (الخط = المتوسط العام)",
              yaxis_title="معدّل التسرّب")
    fig.update_yaxes(tickformat=".0%", range=[0, g["mean"].max() * 1.3])
    st.plotly_chart(fig, width="stretch")
    st.success(
        "**ملاحظات الاستكشاف:** العقد الشهري ومدّة الاشتراك القصيرة ومكالمات الدعم الكثيرة "
        "كلها مرتبطة بتسرّب أعلى — وهذا منطقي تجاريًا، وهي علامة جيّدة على أن البيانات سليمة.",
        icon=":material/visibility:",
    )

# ---------------------------------------------------------------------------
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_curve,  # noqa: E402
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,  # noqa: E402
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

NUM = ["مدة_الاشتراك_شهر", "الفاتورة_الشهرية", "مكالمات_الدعم", "كبير_السن"]
CAT = ["نوع_العقد", "طريقة_الدفع", "نوع_الإنترنت"]

X = df.drop(columns=[TARGET])
y = df[TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def make_pre() -> ColumnTransformer:
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, NUM), ("cat", cat_pipe, CAT)])


with tabs[1]:
    st.header("الخطوة 4 — التقسيم")
    st.code(
        """from sklearn.model_selection import train_test_split

X = df.drop(columns=["تسرّب"])
y = df["تسرّب"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,          # يحافظ على نسبة التسرّب في الجزأين
    random_state=42,     # لإعادة إنتاج النتيجة
)""", language="python")
    k = st.columns(4)
    k[0].metric("عيّنات التدريب", f"{len(X_tr):,}")
    k[1].metric("عيّنات الاختبار", f"{len(X_te):,}")
    k[2].metric("نسبة التسرّب — تدريب", f"{y_tr.mean():.2%}")
    k[3].metric("نسبة التسرّب — اختبار", f"{y_te.mean():.2%}")
    st.caption("النسبتان متطابقتان تقريبًا بفضل `stratify=y`.")

    st.header("الخطوة 5 — الأنبوب Pipeline")
    card(
        "لماذا الأنبوب ليس رفاهية؟",
        "كل تحويل «يتعلّم» شيئًا من البيانات (الوسيط للتعويض، المتوسط والانحراف للتقييس، "
        "فئات الترميز). الأنبوب يضمن أن هذا التعلّم يحدث <b>داخل بيانات التدريب فقط</b>، "
        "وأنه يُعاد بشكل صحيح في <b>كل طيّة</b> من التحقّق المتقاطع.<br>"
        "بدونه، التسريب شبه مؤكّد — وهو أخطر خطأ في المجال.",
        "teal",
    )
    st.code(
        """from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUM = ["مدة_الاشتراك_شهر", "الفاتورة_الشهرية", "مكالمات_الدعم", "كبير_السن"]
CAT = ["نوع_العقد", "طريقة_الدفع", "نوع_الإنترنت"]

num_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),   # تعويض بالوسيط
    ("sc",  StandardScaler()),                   # تقييس معياري
])

cat_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("oh",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, NUM),
    ("cat", cat_pipe, CAT),
])

model = Pipeline([
    ("prep", preprocessor),
    ("clf",  LogisticRegression(class_weight="balanced", max_iter=1000)),
])""", language="python")

    st.markdown(
        """
```mermaid
graph LR
    R["البيانات الخام"] --> N["أعمدة عددية"]
    R --> C["أعمدة فئوية"]
    N --> N1["SimpleImputer<br/>median"] --> N2["StandardScaler"]
    C --> C1["SimpleImputer<br/>most_frequent"] --> C2["OneHotEncoder"]
    N2 --> M["مصفوفة موحّدة"]
    C2 --> M
    M --> F["Classifier"]
    style M fill:#EEF2FC,stroke:#5B6FD6,stroke-width:2px
    style F fill:#E7F6F2,stroke:#2FA98C,stroke-width:2px
```
"""
    )
    pre_demo = make_pre().fit(X_tr)
    st.metric("عدد الأعمدة بعد التحويل", pre_demo.transform(X_tr.head(5)).shape[1],
              delta=f"كانت {X_tr.shape[1]} قبل الترميز")

# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="جارٍ تدريب الموديلات ومقارنتها…")
def train_all():
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    models = {
        "خط الأساس (الفئة الأكثر)": DummyClassifier(strategy="most_frequent"),
        "الانحدار اللوجستي": LogisticRegression(class_weight="balanced", max_iter=2000),
        "الغابة العشوائية": RandomForestClassifier(n_estimators=250, min_samples_leaf=4,
                                                   class_weight="balanced", random_state=42),
        "التعزيز التدريجي": GradientBoostingClassifier(random_state=42),
    }
    rows, fitted = [], {}
    for nm, clf in models.items():
        pipe = Pipeline([("prep", make_pre()), ("clf", clf)])
        f1cv = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1", n_jobs=-1)
        auccv = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
        pipe.fit(X_tr, y_tr)
        fitted[nm] = pipe
        rows.append({"الموديل": nm,
                     "F1 (CV)": f"{f1cv.mean():.3f} ± {f1cv.std():.3f}",
                     "ROC-AUC (CV)": f"{auccv.mean():.3f} ± {auccv.std():.3f}",
                     "_f1": f1cv.mean()})
    return pd.DataFrame(rows), fitted


@st.cache_resource(show_spinner="جارٍ البحث عن أفضل هايبرباراميترز…")
def tune():
    pipe = Pipeline([("prep", make_pre()),
                     ("clf", RandomForestClassifier(class_weight="balanced",
                                                    random_state=42))])
    grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 8, 14],
        "clf__min_samples_leaf": [1, 4, 10],
    }
    gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                      scoring="f1", n_jobs=-1, refit=True)
    gs.fit(X_tr, y_tr)
    return gs


with tabs[2]:
    st.header("الخطوة 6 و 7 — خط الأساس ومقارنة الموديلات")
    st.code(
        """from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(5, shuffle=True, random_state=42)
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    f1  = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1",      n_jobs=-1)
    auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"{name}: F1={f1.mean():.3f}±{f1.std():.3f}  AUC={auc.mean():.3f}")""",
        language="python")

    res, fitted = train_all()
    st.dataframe(res.drop(columns="_f1"), width="stretch", hide_index=True)

    fig = go.Figure(go.Bar(x=res["_f1"], y=res["الموديل"], orientation="h",
                           marker_color=[PALETTE["muted"], PALETTE["indigo"],
                                         PALETTE["teal"], PALETTE["orange"]],
                           text=[f"{v:.3f}" for v in res["_f1"]], textposition="outside"))
    style_fig(fig, height=330, legend_top=False, xaxis_title="متوسط F1 في التحقّق المتقاطع")
    fig.update_xaxes(range=[0, max(res["_f1"]) * 1.2])
    st.plotly_chart(fig, width="stretch")
    st.info(
        "**قراءة النتائج:** خط الأساس يعطي F1 = 0 لأنه لا يتنبّأ بأي تسرّب إطلاقًا، "
        "رغم أن دقّته الكلية مرتفعة — وهذا تجسيد عملي لمفارقة الدقّة. "
        "كل الموديلات الحقيقية تتفوّق عليه بوضوح، والفروق بينها صغيرة نسبيًا "
        "مقارنةً بالانحراف المعياري.",
        icon=":material/leaderboard:",
    )

    st.header("الخطوة 8 — ضبط الهايبرباراميترز")
    st.code(
        """param_grid = {
    "clf__n_estimators":     [200, 400],
    "clf__max_depth":        [None, 8, 14],
    "clf__min_samples_leaf": [1, 4, 10],
}
search = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1", n_jobs=-1, refit=True)
search.fit(X_tr, y_tr)
print(search.best_params_, search.best_score_)""", language="python")

    gs = tune()
    a, b = st.columns([1, 1.4], gap="medium")
    with a:
        st.metric("أفضل F1 في التحقّق المتقاطع", f"{gs.best_score_:.4f}")
        st.markdown("**أفضل تركيبة:**")
        st.json({k.replace("clf__", ""): v for k, v in gs.best_params_.items()})
        st.caption(f"عدد التركيبات المجرَّبة: {len(gs.cv_results_['params'])} "
                   f"× 5 طيّات = {len(gs.cv_results_['params']) * 5} تدريبًا.")
    with b:
        cvr = pd.DataFrame(gs.cv_results_)
        top = cvr.nlargest(10, "mean_test_score")
        fig = go.Figure(go.Bar(
            x=top["mean_test_score"],
            y=[str(p).replace("clf__", "").replace("'", "") for p in top["params"]],
            orientation="h",
            error_x=dict(type="data", array=top["std_test_score"], color=PALETTE["muted"]),
            marker_color=PALETTE["indigo"]))
        style_fig(fig, height=400, legend_top=False, xaxis_title="متوسط F1 ± الانحراف",
                  title="أفضل 10 تركيبات")
        st.plotly_chart(fig, width="stretch")
        st.caption("لاحظ تداخل أعمدة الخطأ: الفروق بين التركيبات الأولى غير معنوية إحصائيًا.")

# ---------------------------------------------------------------------------
with tabs[3]:
    st.header("الخطوة 9 — التقييم النهائي على بيانات الاختبار")
    st.error(
        "**هذه أول وآخر مرة نلمس فيها `X_te`.** كل ما سبق تمّ داخل بيانات التدريب فقط.",
        icon=":material/lock_open:",
    )

    gs = tune()
    best = gs.best_estimator_
    proba = best.predict_proba(X_te)[:, 1]

    thr = st.slider("عتبة القرار", 0.10, 0.90, 0.50, 0.01,
                    help="خفّضها لرفع الاستدعاء Recall — المناسب لمسألتنا")
    pred = (proba >= thr).astype(int)

    TP = int(((pred == 1) & (y_te == 1)).sum())
    FP = int(((pred == 1) & (y_te == 0)).sum())
    FN = int(((pred == 0) & (y_te == 1)).sum())
    TN = int(((pred == 0) & (y_te == 0)).sum())

    k = st.columns(5)
    k[0].metric("Accuracy", f"{accuracy_score(y_te, pred):.1%}")
    k[1].metric("Precision", f"{precision_score(y_te, pred, zero_division=0):.1%}")
    k[2].metric("Recall", f"{recall_score(y_te, pred):.1%}")
    k[3].metric("F1", f"{f1_score(y_te, pred):.3f}")
    k[4].metric("ROC-AUC", f"{roc_auc_score(y_te, proba):.3f}")

    g1, g2 = st.columns(2, gap="medium")
    with g1:
        fig = go.Figure(go.Heatmap(
            z=[[TN, FP], [FN, TP]], x=["تنبؤ: بقي", "تنبؤ: تسرّب"],
            y=["حقيقة: بقي", "حقيقة: تسرّب"],
            colorscale=[[0, "#F6F9FE"], [1, "#5B6FD6"]], showscale=False,
            text=[[f"TN<br>{TN}", f"FP<br>{FP}"], [f"FN<br>{FN}", f"TP<br>{TP}"]],
            texttemplate="%{text}", textfont=dict(size=19)))
        style_fig(fig, height=330, legend_top=False, title="مصفوفة الالتباس")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"فوّتنا **{FN}** عميلًا سيتسرّب، وأنذرنا خطأً عن **{FP}** عميلًا سيبقى.")
    with g2:
        fpr, tpr, _ = roc_curve(y_te, proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="عشوائي",
                                 line=dict(color=PALETTE["muted"], dash="dash")))
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f"AUC = {roc_auc_score(y_te, proba):.3f}",
                                 line=dict(color=PALETTE["indigo"], width=4),
                                 fill="tozeroy", fillcolor="rgba(91,111,214,0.12)"))
        cf = FP / (FP + TN) if FP + TN else 0
        ct = TP / (TP + FN) if TP + FN else 0
        fig.add_trace(go.Scatter(x=[cf], y=[ct], mode="markers", name="عتبتك",
                                 marker=dict(size=16, color=PALETTE["red"], symbol="star")))
        style_fig(fig, height=330, xaxis_title="FPR", yaxis_title="TPR", title="منحنى ROC")
        st.plotly_chart(fig, width="stretch")

    prec_c, rec_c, th_c = precision_recall_curve(y_te, proba)
    f1_c = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-12)
    best_t = float(th_c[int(np.argmax(f1_c[:-1]))])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=th_c, y=prec_c[:-1], mode="lines", name="Precision",
                             line=dict(color=PALETTE["indigo"], width=3)))
    fig.add_trace(go.Scatter(x=th_c, y=rec_c[:-1], mode="lines", name="Recall",
                             line=dict(color=PALETTE["pink"], width=3)))
    fig.add_trace(go.Scatter(x=th_c, y=f1_c[:-1], mode="lines", name="F1",
                             line=dict(color=PALETTE["teal"], width=3.5)))
    fig.add_vline(x=thr, line=dict(color=PALETTE["red"], width=2.5))
    fig.add_vline(x=best_t, line=dict(color=PALETTE["orange"], width=2, dash="dash"))
    fig.add_annotation(x=best_t, y=1.02, text=f"أفضل F1 عند {best_t:.2f}", showarrow=False,
                       font=dict(size=13, color=PALETTE["orange"]))
    style_fig(fig, height=360, xaxis_title="العتبة", yaxis_title="القيمة",
              title="المعايير الثلاثة مقابل العتبة")
    st.plotly_chart(fig, width="stretch")
    st.success(
        f"**قرار العتبة:** العتبة التي تعظّم F1 هنا هي **{best_t:.2f}**. "
        "ولأن تفويت عميل أغلى من إنذار كاذب في مسألتنا، يمكن تبرير اختيار عتبة أقل من ذلك "
        "لرفع الاستدعاء. هذا قرار **تجاري** وليس إحصائيًا بحتًا — "
        "والمهم أن يُتَّخذ بوعي وأن يُوثَّق.",
        icon=":material/rule:",
    )

    st.header("الخطوة 10 — تفسير الموديل")
    st.code(
        """from sklearn.inspection import permutation_importance

r = permutation_importance(best_model, X_te, y_te, n_repeats=10,
                           random_state=42, scoring="f1")""", language="python")

    @st.cache_data(show_spinner="جارٍ حساب أهمية المتغيّرات…")
    def importance():
        r = permutation_importance(gs.best_estimator_, X_te, y_te, n_repeats=8,
                                   random_state=42, scoring="f1", n_jobs=-1)
        return (pd.DataFrame({"المتغيّر": X_te.columns,
                              "الأهمية": r.importances_mean,
                              "الانحراف": r.importances_std})
                .sort_values("الأهمية"))

    imp = importance()
    fig = go.Figure(go.Bar(x=imp["الأهمية"], y=imp["المتغيّر"], orientation="h",
                           error_x=dict(type="data", array=imp["الانحراف"],
                                        color=PALETTE["muted"]),
                           marker_color=PALETTE["teal"]))
    style_fig(fig, height=380, legend_top=False,
              xaxis_title="الانخفاض في F1 عند خلط هذا العمود عشوائيًا",
              title="أهمية المتغيّرات — Permutation Importance")
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        "**كيف تُقرأ؟** نخلط قيم عمود واحد عشوائيًا ونقيس كم انخفض الأداء. "
        "الانخفاض الكبير = العمود مهم. هذه الطريقة تعمل مع **أي** موديل، "
        "وهي أصدق من `feature_importances_` الخاصة بالأشجار "
        "لأنها تُقاس على بيانات الاختبار وليس على التدريب.\n\n"
        "النتيجة هنا منسجمة مع ما رأيناه في الاستكشاف: نوع العقد ومدّة الاشتراك هما الأقوى — "
        "وهذا **اتّساق منطقي** يزيد ثقتنا في الموديل."
    )
    st.info(
        f"للتفسير الأعمق على مستوى العميل الواحد، استعمل {en('SHAP')} أو {en('LIME')}: "
        "تخبرك لماذا تنبّأ الموديل بهذا الاحتمال **لهذا العميل تحديدًا** — "
        "وهو مطلوب قانونيًا في قطاعات مثل البنوك والتأمين.",
        icon=":material/query_stats:",
    )

# ---------------------------------------------------------------------------
with tabs[4]:
    st.header("الخطوة 11 — الحفظ والنشر")
    st.code(
        """import joblib

# الأنبوب كامل: التجهيز + الموديل في ملف واحد
joblib.dump(best_model, "churn_model.joblib")

# عند الاستعمال لاحقًا — لا حاجة لإعادة أي تجهيز يدوي
model = joblib.load("churn_model.joblib")
proba = model.predict_proba(new_customers_df)[:, 1]""", language="python")

    st.markdown("**خدمة ويب بسيطة بـ FastAPI:**")
    st.code(
        '''from fastapi import FastAPI
import pandas as pd, joblib

app = FastAPI()
model = joblib.load("churn_model.joblib")
THRESHOLD = 0.42        # العتبة التي اخترناها بوعي

@app.post("/predict")
def predict(customer: dict):
    df = pd.DataFrame([customer])
    p = float(model.predict_proba(df)[0, 1])
    return {"churn_probability": round(p, 4),
            "at_risk": bool(p >= THRESHOLD)}''', language="python")

    st.header("ماذا بعد النشر؟ — المراقبة")
    m1, m2 = st.columns(2, gap="medium")
    with m1:
        card("انزياح البيانات — Data drift",
             "توزيع المتغيّرات الداخلة يتغيّر مع الزمن (أسعار جديدة، شريحة عملاء جديدة). "
             "راقب إحصاءات الأعمدة شهريًا وقارنها ببيانات التدريب.", "orange")
    with m2:
        card("انزياح المفهوم — Concept drift",
             "العلاقة نفسها بين X و y تتغيّر (تغيّر سلوك السوق، دخول منافس). "
             "راقب الأداء الفعلي كلما توفّرت الأجوبة الحقيقية، وأعد التدريب دوريًا.", "pink")

    st.header("القائمة النهائية — اطبعها واستعملها في كل مشروع")
    st.markdown(
        """
- [ ] كتبت السؤال ومعيار النجاح **قبل** أي كود
- [ ] حدّدت خط أساس وقارنت به
- [ ] استكشفت البيانات ورسمتها قبل النمذجة
- [ ] عالجت القيم المفقودة والشاذّة بوعي (وليس آليًا)
- [ ] **قسّمت البيانات قبل أي تحويل**
- [ ] استعملت `stratify` في التصنيف
- [ ] وضعت كل التجهيز داخل `Pipeline`
- [ ] استعملت تحقّقًا متقاطعًا مناسبًا لطبيعة البيانات (`Stratified` / `TimeSeries` / `Group`)
- [ ] اخترت معيار تقييم يناسب تكلفة الأخطاء الحقيقية
- [ ] ضبطت الهايبرباراميترز على التحقّق **لا** على الاختبار
- [ ] قارنت خطأ التدريب بخطأ الاختبار لكشف الإفراط
- [ ] لمست بيانات الاختبار **مرة واحدة فقط** في النهاية
- [ ] اخترت عتبة القرار بوعي ووثّقتها
- [ ] فسّرت الموديل وتأكّدت أن النتائج منطقية في مجال الدراسة
- [ ] ثبّتت `random_state` في كل مكان
- [ ] بلّغت عن النتيجة مع عدم اليقين (`± std`) وليس رقمًا مجرّدًا
- [ ] حفظت الموديل والمكتبات وإصداراتها (`requirements.txt`)
- [ ] خطّطت لمراقبة الأداء وإعادة التدريب بعد النشر
"""
    )

    st.success(
        "**مبروك — أنت الآن تعرف كيف يُبنى مشروع تعلّم آلة كامل.** "
        "ما تبقّى هو التكرار على بياناتك الخاصة: كل مشروع جديد هو نفس هذه الخطوات بمحتوى مختلف.",
        icon=":material/workspace_premium:",
    )

st.divider()
st.markdown(
    "#### :material/arrow_circle_left: الخطوة التالية\n"
    "**قاموس المصطلحات والمراجع** — مرجع سريع تعود إليه كلما نسيت مصطلحًا."
)
footer()
