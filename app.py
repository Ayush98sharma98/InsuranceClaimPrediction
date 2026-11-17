
import streamlit as st
import pandas as pd
import plotly.express as px
import json
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ClaimIQ | Insurance Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "outputs" / "cleaned_claims"
METRICS_FILE = BASE_DIR / "outputs" / "evaluation_results.json"

# --------------------------------------------------
# DASHBOARD THEMES
# --------------------------------------------------

THEMES = {
    "Midnight Blue": {
        "bg": "#0B1020",
        "sidebar": "#101729",
        "panel": "#131B2E",
        "text": "#EDF2FF",
        "muted": "#8995AD",
        "border": "#263550",
        "accent": "#659BFF"
    },
    "Emerald": {
        "bg": "#0B1715",
        "sidebar": "#10251F",
        "panel": "#142B25",
        "text": "#ECFFF5",
        "muted": "#A1C6B5",
        "border": "#29483D",
        "accent": "#39C99A"
    },
    "Pearl Light": {
        "bg": "#F4F6FB",
        "sidebar": "#E9EDF5",
        "panel": "#FFFFFF",
        "text": "#17243A",
        "muted": "#64748B",
        "border": "#D9E0EB",
        "accent": "#3769D5"
    },
    "Amethyst": {
        "bg": "#130F1C",
        "sidebar": "#21172C",
        "panel": "#291E36",
        "text": "#FFF2FF",
        "muted": "#C1A9D5",
        "border": "#49345D",
        "accent": "#A78BFA"
    }
}

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():

    files = list(DATA_DIR.glob("part-*.csv"))

    if not files:
        raise FileNotFoundError(
            "No Spark CSV files found in outputs/cleaned_claims."
        )

    data = pd.concat(
        [pd.read_csv(file) for file in files],
        ignore_index=True
    )

    data["billed_amount"] = pd.to_numeric(
        data["billed_amount"],
        errors="coerce"
    )

    if "claim_billing_date" in data.columns:
        data["claim_billing_date"] = pd.to_datetime(
            data["claim_billing_date"],
            errors="coerce"
        )

    return data


try:
    data = load_data()

except Exception as error:
    st.error(f"Unable to load the cleaned dataset: {error}")
    st.stop()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 📊 ClaimIQ")
    st.caption("Insurance Analytics Platform")

    st.divider()

    theme_name = st.selectbox(
        "🎨 Choose Dashboard Theme",
        list(THEMES.keys())
    )

    THEME = THEMES[theme_name]

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Claim Analytics",
            "Provider Insights",
            "Model Performance",
            "Claim Prediction"
        ]
    )

    st.divider()

    st.markdown("**PROJECT INFORMATION**")
    st.caption("Insurance Claim Prediction")
    st.caption("Powered by Apache Spark & Logistic Regression")

    st.divider()

    st.caption("Academic Project | 2026")


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(f"""
<style>

    .stApp {{
        background-color: {THEME["bg"]};
        color: {THEME["text"]};
    }}

    [data-testid="stSidebar"] {{
        background-color: {THEME["sidebar"]};
        border-right: 1px solid {THEME["border"]};
    }}

    .main-title {{
        font-size: 34px;
        font-weight: 800;
        color: {THEME["text"]};
        letter-spacing: -1px;
        margin-bottom: 2px;
    }}

    .subtitle {{
        color: {THEME["muted"]};
        font-size: 14px;
        margin-bottom: 25px;
    }}

    .section-title {{
        color: {THEME["text"]};
        font-size: 21px;
        font-weight: 700;
        margin-top: 22px;
        margin-bottom: 14px;
    }}

    [data-testid="stMetric"] {{
        background: {THEME["panel"]};
        border: 1px solid {THEME["border"]};
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 5px 18px rgba(0,0,0,0.08);
    }}

    [data-testid="stMetricLabel"] {{
        color: {THEME["muted"]};
    }}

    [data-testid="stMetricValue"] {{
        color: {THEME["text"]};
        font-weight: 750;
    }}

    div[data-testid="stPlotlyChart"] {{
        background: {THEME["panel"]};
        border: 1px solid {THEME["border"]};
        border-radius: 15px;
        padding: 12px;
    }}

    [data-testid="stDataFrame"] {{
        border: 1px solid {THEME["border"]};
        border-radius: 12px;
    }}

    .stButton button {{
        background: {THEME["accent"]};
        color: white;
        border: none;
        border-radius: 9px;
    }}

    .stButton button:hover {{
        filter: brightness(1.12);
    }}

    .footer {{
        color: {THEME["muted"]};
        font-size: 12px;
        text-align: center;
        margin-top: 40px;
    }}

    h1, h2, h3 {{
        color: {THEME["text"]};
    }}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">Insurance Analytics Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Monitor claim activity, explore provider '
    'performance and evaluate machine learning results.</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# GLOBAL FILTERS
# --------------------------------------------------

with st.expander("🔎 Dashboard Filters", expanded=False):

    col1, col2 = st.columns(2)

    with col1:

        providers = sorted(
            data["insurance_provider"].dropna().unique().tolist()
        )

        selected_providers = st.multiselect(
            "Insurance Provider",
            providers,
            default=providers
        )

    with col2:

        statuses = sorted(
            data["claim_status"].dropna().unique().tolist()
        )

        selected_statuses = st.multiselect(
            "Claim Status",
            statuses,
            default=statuses
        )


filtered_data = data[
    data["insurance_provider"].isin(selected_providers)
    & data["claim_status"].isin(selected_statuses)
].copy()


# --------------------------------------------------
# CHART THEME
# --------------------------------------------------

chart_layout = dict(
    paper_bgcolor=THEME["panel"],
    plot_bgcolor=THEME["panel"],
    font=dict(color=THEME["text"]),
    margin=dict(l=25, r=25, t=45, b=25),
    legend=dict(font=dict(color=THEME["muted"])),
    title_font=dict(color=THEME["text"], size=16),
    xaxis=dict(
        gridcolor=THEME["border"],
        zerolinecolor=THEME["border"]
    ),
    yaxis=dict(
        gridcolor=THEME["border"],
        zerolinecolor=THEME["border"]
    )
)


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.markdown(
        '<div class="section-title">Overview</div>',
        unsafe_allow_html=True
    )

    total_claims = len(filtered_data)

    paid_claims = (
        filtered_data["claim_status"] == "Paid"
    ).sum()

    denied_claims = (
        filtered_data["claim_status"] == "Denied"
    ).sum()

    average_billed = filtered_data["billed_amount"].mean()

    paid_rate = (
        paid_claims / total_claims * 100
        if total_claims else 0
    )

    denied_rate = (
        denied_claims / total_claims * 100
        if total_claims else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Claims",
        f"{total_claims:,}"
    )

    col2.metric(
        "Paid Claims",
        f"{paid_claims:,}"
    )

    col3.metric(
        "Denied Claims",
        f"{denied_claims:,}"
    )

    col4.metric(
        "Average Billed Amount",
        f"{average_billed:,.2f}"
        if pd.notna(average_billed) else "N/A"
    )

    st.markdown(
        '<div class="section-title">Claim Summary</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Paid Claim Rate",
            f"{paid_rate:.2f}%"
        )

        status_data = (
            filtered_data["claim_status"]
            .value_counts()
            .rename_axis("Status")
            .reset_index(name="Claims")
        )

        fig = px.pie(
            status_data,
            names="Status",
            values="Claims",
            hole=0.65,
            title="Claim Status Distribution",
            color="Status",
            color_discrete_map={
                "Paid": "#22c55e",
                "Denied": "#ef4444"
            }
        )

        fig.update_layout(**chart_layout)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.metric(
            "Denied Claim Rate",
            f"{denied_rate:.2f}%"
        )

        provider_data = (
            filtered_data.groupby("insurance_provider")["claim_status"]
            .count()
            .reset_index(name="Claims")
            .sort_values("Claims", ascending=False)
        )

        fig = px.bar(
            provider_data,
            x="insurance_provider",
            y="Claims",
            title="Claims by Insurance Provider",
            color="Claims",
            color_continuous_scale="Blues"
        )

        fig.update_layout(**chart_layout)
        fig.update_xaxes(title="Insurance Provider")
        fig.update_yaxes(title="Number of Claims")

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# --------------------------------------------------
# CLAIM ANALYTICS
# --------------------------------------------------

elif page == "Claim Analytics":

    st.markdown(
        '<div class="section-title">Claim Analytics</div>',
        unsafe_allow_html=True
    )

    if "claim_billing_date" in filtered_data.columns:

        monthly_data = filtered_data.dropna(
            subset=["claim_billing_date"]
        ).copy()

        monthly_data["Month"] = (
            monthly_data["claim_billing_date"]
            .dt.to_period("M")
            .astype(str)
        )

        monthly_summary = (
            monthly_data.groupby(["Month", "claim_status"])
            .size()
            .reset_index(name="Claims")
        )

        fig = px.line(
            monthly_summary,
            x="Month",
            y="Claims",
            color="claim_status",
            markers=True,
            title="Monthly Claim Trends",
            color_discrete_map={
                "Paid": "#22c55e",
                "Denied": "#ef4444"
            }
        )

        fig.update_layout(**chart_layout)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            filtered_data,
            x="billed_amount",
            color="claim_status",
            nbins=35,
            title="Billed Amount Distribution",
            color_discrete_map={
                "Paid": "#22c55e",
                "Denied": "#ef4444"
            }
        )

        fig.update_layout(**chart_layout)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        status_amount = (
            filtered_data.groupby("claim_status")["billed_amount"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            status_amount,
            x="claim_status",
            y="billed_amount",
            title="Average Billed Amount by Claim Status",
            color="claim_status",
            color_discrete_map={
                "Paid": "#22c55e",
                "Denied": "#ef4444"
            }
        )

        fig.update_layout(**chart_layout)

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# --------------------------------------------------
# PROVIDER INSIGHTS
# --------------------------------------------------

elif page == "Provider Insights":

    st.markdown(
        '<div class="section-title">Provider Insights</div>',
        unsafe_allow_html=True
    )

    provider_summary = (
        filtered_data.groupby("insurance_provider")
        .agg(
            Total_Claims=("claim_status", "count"),
            Paid_Claims=(
                "claim_status",
                lambda x: (x == "Paid").sum()
            ),
            Denied_Claims=(
                "claim_status",
                lambda x: (x == "Denied").sum()
            ),
            Average_Billed=("billed_amount", "mean")
        )
        .reset_index()
    )

    provider_summary["Denial_Rate"] = (
        provider_summary["Denied_Claims"]
        / provider_summary["Total_Claims"].replace(0, pd.NA)
        * 100
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            provider_summary,
            x="insurance_provider",
            y=["Paid_Claims", "Denied_Claims"],
            title="Paid vs Denied Claims by Provider",
            barmode="group",
            color_discrete_map={
                "Paid_Claims": "#22c55e",
                "Denied_Claims": "#ef4444"
            }
        )

        fig.update_layout(**chart_layout)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.bar(
            provider_summary.sort_values(
                "Denial_Rate",
                ascending=False
            ),
            x="insurance_provider",
            y="Denial_Rate",
            title="Denial Rate by Provider",
            color="Denial_Rate",
            color_continuous_scale="Reds"
        )

        fig.update_layout(**chart_layout)
        fig.update_yaxes(title="Denial Rate (%)")

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("### Provider Summary Table")

    st.dataframe(
        provider_summary.round(2),
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.markdown(
        '<div class="section-title">Model Performance</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Classification model: PySpark Logistic Regression"
    )

    if METRICS_FILE.exists():

        with open(METRICS_FILE, "r") as file:
            metrics = json.load(file)

        metric_columns = st.columns(4)

        metric_names = [
            ("Accuracy", "accuracy"),
            ("Weighted Precision", "weightedPrecision"),
            ("Weighted Recall", "weightedRecall"),
            ("F1 Score", "f1")
        ]

        for column, (label, key) in zip(
            metric_columns,
            metric_names
        ):

            value = metrics.get(key)

            if value is not None:
                column.metric(
                    label,
                    f"{float(value) * 100:.2f}%"
                )
            else:
                column.metric(label, "N/A")

        st.markdown("### Saved Evaluation Results")

        st.json(metrics)

    else:

        st.warning(
            "Evaluation results file was not found."
        )


# --------------------------------------------------
# CLAIM PREDICTION
# --------------------------------------------------

elif page == "Claim Prediction":

    st.markdown(
        '<div class="section-title">Claim Prediction</div>',
        unsafe_allow_html=True
    )

    st.info(
        "The Logistic Regression model has been trained and saved. "
        "We will connect the prediction interface to the trained model "
        "in the next development step."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    '<div class="footer">ClaimIQ | Built with Streamlit, Plotly '
    'and Apache Spark</div>',
    unsafe_allow_html=True
)
