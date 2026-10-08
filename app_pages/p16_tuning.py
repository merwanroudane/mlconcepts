import numpy as np
import plotly.graph_objects as go
import streamlit as st

from lib.ui import PALETTE, SEQ, card, en, eq, footer, hero, setup, style_fig

setup("tuning")

hero(
    "الصفحة 16 من 18",
    "ضبط الهايبرباراميترز  Hyperparameter Tuning",
    "المعاملات يتعلّمها الموديل، أمّا الهايبرباراميترز فتختارها أنت. "
    "وبدل التخمين العشوائي، هناك طرق منهجية تبحث عنها — مع قاعدة ذهبية واحدة: "
    "لا تلمس بيانات الاختبار أبدًا أثناء البحث.",
)

# ---------------------------------------------------------------------------
st.header("1) تذكير بالفرق")

c1, c2 = st.columns(2, gap="medium")
with c1:
    card("Parameters — المعاملات",
         "تُتعلَّم <b>أثناء</b> التدريب من البيانات.<br>"
         "أمثلة: أوزان الانحدار `coef_`، عتبات الشجرة، أوزان الشبكة العصبية.<br>"
         "<b>أنت لا تتدخّل فيها إطلاقًا.</b>", "teal")
with c2:
    card("Hyperparameters — الهايبرباراميترز",
         "تُحدَّد <b>قبل</b> التدريب وتتحكّم في <b>كيفية</b> التعلّم.<br>"
         "أمثلة: `alpha`, `max_depth`, `n_estimators`, `learning_rate`, `C`, `k`.<br>"
         "<b>هذه مسؤوليتك أنت.</b>", "orange")

st.subheader("أهم الهايبرباراميترز في الخوارزميات الشائعة")
st.table({
    "**الخوارزمية**": ["Ridge / Lasso", "Logistic Regression", "KNN",
                       "Decision Tree", "Random Forest", "Gradient Boosting", "SVM"],
    "**الأهم ضبطًا**": ["`alpha`", "`C`, `penalty`", "`n_neighbors`, `weights`",
                        "`max_depth`, `min_samples_leaf`",
                        "`n_estimators`, `max_depth`, `max_features`",
                        "`learning_rate`, `n_estimators`, `max_depth`",
                        "`C`, `gamma`, `kernel`"],
    "**أثره**": ["↑ alpha = تنظيم أقوى = تحيّز أعلى",
                 "C = مقلوب التنظيم؛ ↓ C = تنظيم أقوى",
                 "↑ k = حدّ قرار أنعم = تباين أقل",
                 "↑ العمق = تعقيد أعلى = خطر إفراط",
                 "↑ الأشجار = أفضل دائمًا (لكن أبطأ)",
                 "↓ learning_rate مع ↑ n_estimators = أدقّ وأبطأ",
                 "↑ C و ↑ gamma = حدّ أكثر التواءً"],
}, border="horizontal")

# ---------------------------------------------------------------------------
st.header("2) الطرق الثلاث للبحث")

st.markdown("لنفترض أن عندنا هايبرباراميترين نريد ضبطهما. كيف نستكشف الفضاء؟")

rng = np.random.default_rng(8)
n_pts = st.slider("ميزانية التجارب (عدد التركيبات المسموح بها)", 9, 64, 25)


def score_surface(a, b):
    """سطح أداء وهمي: القمة الحقيقية تعتمد على المحور الأول أكثر بكثير."""
    return np.exp(-((a - 0.62) ** 2) / 0.05) * (1 + 0.10 * np.exp(-((b - 0.4) ** 2) / 0.4))


gx = np.linspace(0, 1, 120)
GX, GY = np.meshgrid(gx, gx)
GZ = score_surface(GX, GY)

side = int(np.sqrt(n_pts))
grid_a, grid_b = np.meshgrid(np.linspace(0.08, 0.92, side), np.linspace(0.08, 0.92, side))
grid_pts = np.c_[grid_a.ravel(), grid_b.ravel()]
rand_pts = rng.uniform(0.02, 0.98, (n_pts, 2))

tabs = st.tabs([":material/grid_on: Grid Search", ":material/casino: Random Search",
                ":material/auto_graph: Bayesian Search"])

for tab, nm, pts, desc in [
    (tabs[0], "Grid Search — البحث الشبكي", grid_pts,
     """
**كيف يعمل؟** تحدّد قائمة قيم لكل هايبرباراميتر، ويجرّب **كل التركيبات** الممكنة.

:green-badge[ميزة] شامل، منهجي، قابل للتوازي بالكامل، ونتائجه قابلة لإعادة الإنتاج.

:red-badge[عيب] **اللعنة الأسّية**: 3 باراميترات × 5 قيم = 125 تركيبة، × 5 طيّات CV = **625 تدريبًا**.
وأسوأ من ذلك: يهدر معظم التجارب على قيم للباراميتر غير المهم.

```python
GridSearchCV(model, {"alpha": [0.01, 0.1, 1, 10]}, cv=5)
```
"""),
    (tabs[1], "Random Search — البحث العشوائي", rand_pts,
     """
**كيف يعمل؟** تحدّد **توزيعًا** لكل باراميتر، ويسحب منه `n_iter` تركيبة عشوائية.

:green-badge[ميزة] تتحكّم في الميزانية مباشرةً، و**يجرّب قيمًا مختلفة للباراميتر المهم في كل محاولة**
بدل تكرار نفس القيم. أثبت Bergstra & Bengio (2012) أنه أكفأ من الشبكي عمليًا.

:blue-badge[القاعدة] إذا كان عندك أكثر من 3 هايبرباراميترات، فالبحث العشوائي شبه دائمًا الخيار الأفضل.

```python
from scipy.stats import loguniform
RandomizedSearchCV(model, {"alpha": loguniform(1e-4, 1e2)}, n_iter=40, cv=5)
```
"""),
    (tabs[2], "Bayesian Optimization — التحسين البايزي", None,
     """
**كيف يعمل؟** يبني نموذجًا احتماليًا للعلاقة بين الهايبرباراميترز والأداء،
ثم يختار النقطة التالية **بذكاء**: حيث يتوقّع أداءً جيدًا أو حيث عدم اليقين كبير
(موازنة الاستكشاف والاستغلال).

:green-badge[ميزة] يحتاج تجارب أقل بكثير للوصول إلى أداء أفضل — مهم جدًا
عندما يستغرق التدريب الواحد ساعات.

:orange-badge[عيب] أعقد، غير قابل للتوازي بالكامل، ويحتاج مكتبة خارجية.

```python
# pip install optuna
import optuna

def objective(trial):
    alpha = trial.suggest_float("alpha", 1e-4, 1e2, log=True)
    depth = trial.suggest_int("max_depth", 2, 20)
    return cross_val_score(make_model(alpha, depth), X, y, cv=5).mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print(study.best_params)
```
"""),
]:
    with tab:
        a, b = st.columns([1.5, 1], gap="medium")
        with a:
            fig = go.Figure()
            fig.add_trace(go.Contour(x=gx, y=gx, z=GZ, colorscale=SEQ, showscale=False,
                                     contours=dict(showlabels=False), line=dict(width=0.6)))
            if pts is not None:
                sc = score_surface(pts[:, 0], pts[:, 1])
                fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="markers",
                                         name="التجارب",
                                         marker=dict(size=9, color=PALETTE["orange"],
                                                     line=dict(width=1.3, color="white"))))
                bi = int(np.argmax(sc))
                fig.add_trace(go.Scatter(x=[pts[bi, 0]], y=[pts[bi, 1]], mode="markers",
                                         name="الأفضل الموجود",
                                         marker=dict(size=19, color=PALETTE["red"],
                                                     symbol="star",
                                                     line=dict(width=1.5, color="white"))))
            else:
                bpts = np.array([[0.2, 0.8], [0.8, 0.2], [0.5, 0.5], [0.6, 0.35],
                                 [0.63, 0.42], [0.61, 0.38], [0.62, 0.40]])
                fig.add_trace(go.Scatter(x=bpts[:, 0], y=bpts[:, 1], mode="lines+markers+text",
                                         text=[str(i + 1) for i in range(len(bpts))],
                                         textposition="top center", name="ترتيب التجارب",
                                         line=dict(color=PALETTE["orange"], width=2, dash="dot"),
                                         marker=dict(size=12, color=PALETTE["orange"],
                                                     line=dict(width=1.3, color="white"))))
                fig.add_trace(go.Scatter(x=[0.62], y=[0.40], mode="markers",
                                         name="الأفضل الموجود",
                                         marker=dict(size=19, color=PALETTE["red"],
                                                     symbol="star")))
            style_fig(fig, height=400, xaxis_title="الهايبرباراميتر المهم →",
                      yaxis_title="الهايبرباراميتر غير المهم →", title=nm)
            st.plotly_chart(fig, width="stretch")
            if pts is not None:
                st.caption(f"أفضل أداء وُجد: {score_surface(pts[:, 0], pts[:, 1]).max():.4f} "
                           f"(الأقصى الممكن ≈ 1.10)")
        with b:
            st.markdown(desc)

st.success(
    "**لاحظ الدرس الأهم في الرسوم الثلاثة:** في البحث الشبكي، كل الأعمدة تجرّب نفس قيم "
    "الباراميتر المهم — فمهما زدت النقاط، عدد القيم المختلفة المجرّبة للباراميتر المهم يبقى "
    "`√n` فقط. أمّا البحث العشوائي فيجرّب `n` قيمة مختلفة. "
    "هذا هو السبب الرياضي لتفوّقه.",
    icon=":material/emoji_objects:",
)

# ---------------------------------------------------------------------------
st.header("3) البحث الصحيح: التداخل مع التحقّق المتقاطع")

st.markdown(
    "البحث لا يُقيَّم على بيانات التدريب (سيختار دائمًا الأعقد)، ولا على بيانات الاختبار "
    "(سيلوّثها). الصواب: **التحقّق المتقاطع داخل بيانات التدريب**."
)

st.markdown(
    """
```mermaid
graph TD
    D["كل البيانات"] --> T["Train + Validation (80%)"]
    D --> TE["Test (20%) — مقفلة"]
    T --> CV["لكل تركيبة هايبرباراميترات:<br/>K-Fold Cross-Validation"]
    CV --> B["اختر التركيبة ذات أفضل متوسط"]
    B --> R["أعد التدريب على كل الـ 80% بهذه التركيبة"]
    R --> F["قِس مرة واحدة فقط على الـ Test"]
    style TE fill:#FCE9F1,stroke:#D7568C,stroke-width:2px
    style F fill:#E7F6F2,stroke:#2FA98C,stroke-width:2px
```
"""
)

st.code(
    """from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          stratify=y, random_state=42)

# الأنبوب يضمن أن التقييس يُحسب داخل كل طيّة — لا تسريب
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(random_state=42)),
])

param_grid = {
    "clf__n_estimators": [100, 300, 500],
    "clf__max_depth": [None, 5, 10, 20],
    "clf__min_samples_leaf": [1, 2, 5],
}

search = GridSearchCV(
    pipe, param_grid,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="f1",         # اختر المعيار المناسب لمسألتك
    n_jobs=-1,            # استعمل كل أنوية المعالج
    refit=True,           # يعيد التدريب تلقائيًا على كل بيانات التدريب
    verbose=1,
)
search.fit(X_tr, y_tr)

print("أفضل تركيبة:", search.best_params_)
print("أفضل نتيجة CV:", search.best_score_)

# مرة واحدة فقط، وفي النهاية:
print("نتيجة الاختبار:", search.score(X_te, y_te))""",
    language="python",
)

st.error(
    "**الخطأ الأخطر في هذه المرحلة:** تجريب 200 تركيبة، اختيار الأفضل **على بيانات الاختبار**، "
    "ثم التبليغ عن تلك النتيجة. هذا يسمّى "
    f"{en('overfitting the test set')} وهو شكل خفيّ من التسريب. "
    "النتيجة المبلَّغ عنها ستكون متفائلة بشكل منهجي، وستفشل عند النشر.",
    icon=":material/block:",
)

# ---------------------------------------------------------------------------
st.header("4) التحقّق المتقاطع المتداخل — Nested CV")

st.markdown(
    "إذا كنت تريد تقديرًا **غير متحيّز** لأداء «منهجيتك كاملة» (بما فيها البحث عن الهايبرباراميترز)، "
    "فالطريقة الصارمة هي حلقتان متداخلتان:"
)

k1, k2 = st.columns(2, gap="medium")
with k1:
    with st.container(border=True):
        st.markdown("##### الحلقة الداخلية — Inner loop")
        st.markdown("تبحث عن أفضل هايبرباراميترز داخل كل طيّة تدريب (مثلًا 5 طيّات).")
with k2:
    with st.container(border=True):
        st.markdown("##### الحلقة الخارجية — Outer loop")
        st.markdown("تقيّم أداء التركيبة المختارة على الطيّة المحجوزة (مثلًا 5 طيّات).")

st.code(
    """from sklearn.model_selection import cross_val_score, GridSearchCV, KFold

inner = KFold(5, shuffle=True, random_state=1)
outer = KFold(5, shuffle=True, random_state=2)

search = GridSearchCV(pipe, param_grid, cv=inner, scoring="f1", n_jobs=-1)
nested = cross_val_score(search, X, y, cv=outer, scoring="f1")

print(f"الأداء غير المتحيّز: {nested.mean():.3f} ± {nested.std():.3f}")""",
    language="python",
)
st.caption(
    "التكلفة: 5 × 5 × عدد التركيبات تدريبًا — مكلف جدًا. "
    "يُستعمل عادةً في الأبحاث المنشورة وعند البيانات القليلة، لا في كل مشروع."
)

# ---------------------------------------------------------------------------
st.header("5) نصائح عملية توفّر عليك ساعات")

t1, t2 = st.columns(2, gap="medium")
with t1:
    st.markdown(
        "- **ابدأ خشنًا ثم دقّق**: جرّب `[0.001, 0.01, 0.1, 1, 10]` أولًا، "
        "ثم ضيّق حول الأفضل.\n"
        "- **استعمل المقياس اللوغاريتمي** للباراميترات مثل `alpha`, `C`, `learning_rate`. "
        "الفرق بين 0.001 و0.01 أهم بكثير من الفرق بين 10 و11.\n"
        "- **اضبط الأهم أولًا**: في Gradient Boosting ابدأ بـ `learning_rate` و`n_estimators` "
        "قبل الباقي.\n"
        "- **`n_jobs=-1`** لاستعمال كل أنوية المعالج."
    )
with t2:
    st.markdown(
        "- **إن كانت القيمة المختارة على حافّة المجال**، وسّع المجال وأعد البحث.\n"
        "- **تحقّق من الاستقرار**: انظر إلى `cv_results_` وليس إلى `best_score_` وحده. "
        "فرق 0.002 بين تركيبتين لا يعني شيئًا إذا كان الانحراف المعياري 0.03.\n"
        "- **فضّل البساطة عند التعادل**: بين موديلين متقاربين، اختر الأبسط "
        "(موس أوكام) — فهو أسرع وأكثر استقرارًا.\n"
        "- **ثبّت `random_state`** في كل مكان."
    )

eq(r"\text{عدد التدريبات} = \underbrace{\prod_{j} |V_j|}_{\text{التركيبات}} "
   r"\;\times\; \underbrace{K}_{\text{الطيّات}} \;\;(+\;1\;\text{لإعادة التدريب النهائي})",
   "احسب هذا الرقم قبل تشغيل البحث — قد يكون الفرق بين 3 دقائق و3 أيام")

st.divider()
st.markdown(
    "#### :material/arrow_circle_left: الخطوة التالية\n"
    "**المشروع الكامل**: نطبّق كل ما سبق على بيانات حقيقية، خطوة بخطوة، مع التنفيذ الحيّ."
)
footer()
