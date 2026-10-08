"""منصة تعلّم الآلة التفاعلية — من الصفر إلى مشروع كامل.

إعداد: د. مروان رودان — Dr Merwan Roudane
تشغيل:  streamlit run streamlit_app.py
"""

import streamlit as st

from lib.ui import AUTHOR

st.set_page_config(
    page_title="منصة تعلّم الآلة — Machine Learning من الصفر",
    page_icon=":material/model_training:",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.session_state["_page_cfg"] = True

P = "app_pages"

pages = {
    "الأساسيات — Foundations": [
        st.Page(f"{P}/p01_intro.py", title="ما هو تعلّم الآلة؟", icon=":material/lightbulb:", default=True),
        st.Page(f"{P}/p02_types.py", title="أنواع التعلّم", icon=":material/category:"),
        st.Page(f"{P}/p03_workflow.py", title="خريطة المشروع الكاملة", icon=":material/route:"),
    ],
    "البيانات — Data": [
        st.Page(f"{P}/p04_data.py", title="فهم البيانات وتنظيفها", icon=":material/dataset:"),
        st.Page(f"{P}/p05_features.py", title="هندسة المتغيّرات والتقييس", icon=":material/tune:"),
        st.Page(f"{P}/p06_split.py", title="تقسيم البيانات والتحقّق المتقاطع", icon=":material/call_split:"),
    ],
    "قلب التعلّم — Learning core": [
        st.Page(f"{P}/p07_model.py", title="الموديل والخوارزمية والمعاملات", icon=":material/function:"),
        st.Page(f"{P}/p08_loss.py", title="دالة الخسارة ودالة التكلفة", icon=":material/show_chart:"),
        st.Page(f"{P}/p09_gradient.py", title="الانحدار التدريجي والتحسين", icon=":material/trending_down:"),
        st.Page(f"{P}/p10_training.py", title="حلقة التدريب خطوة بخطوة", icon=":material/autorenew:"),
    ],
    "جودة التعلّم — Generalization": [
        st.Page(f"{P}/p11_fitting.py", title="الإفراط والتقصير في التعلّم", icon=":material/balance:"),
        st.Page(f"{P}/p12_bias_variance.py", title="مقايضة التحيّز والتباين", icon=":material/scatter_plot:"),
        st.Page(f"{P}/p13_regularization.py", title="التنظيم Regularization", icon=":material/compress:"),
    ],
    "التقييم والضبط — Evaluation": [
        st.Page(f"{P}/p14_metrics_reg.py", title="معايير تقييم الانحدار", icon=":material/straighten:"),
        st.Page(f"{P}/p15_metrics_clf.py", title="معايير تقييم التصنيف", icon=":material/fact_check:"),
        st.Page(f"{P}/p16_tuning.py", title="ضبط الهايبرباراميترز", icon=":material/settings_input_component:"),
    ],
    "التطبيق — Putting it together": [
        st.Page(f"{P}/p17_project.py", title="مشروع كامل خطوة بخطوة", icon=":material/rocket_launch:"),
        st.Page(f"{P}/p18_glossary.py", title="قاموس المصطلحات والمراجع", icon=":material/menu_book:"),
    ],
}

page = st.navigation(pages, position="sidebar")

with st.sidebar:
    st.caption(f":material/person: {AUTHOR}")

page.run()
