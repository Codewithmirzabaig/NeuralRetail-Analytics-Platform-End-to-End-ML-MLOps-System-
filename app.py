import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Retail Analytics Dashboard", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_transaction_data() -> pd.DataFrame:
    path = DATA_DIR / "final_cleaned_data.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
    return df


@st.cache_data
def load_customer_data() -> pd.DataFrame:
    path = DATA_DIR / "customer_model_data.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if "CustomerID" in df.columns:
        df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
    return df


@st.cache_data
def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (reference_date - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalPrice": "sum"
    }).reset_index()

    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
    rfm = rfm[(rfm["Frequency"] > 0) & (rfm["Monetary"] > 0)].copy()

    rfm["R_score"] = pd.qcut(rfm["Recency"], 4, labels=[4, 3, 2, 1])
    rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4])
    rfm["M_score"] = pd.qcut(rfm["Monetary"], 4, labels=[1, 2, 3, 4])

    rfm["RFM_Score"] = (
        rfm["R_score"].astype(str)
        + rfm["F_score"].astype(str)
        + rfm["M_score"].astype(str)
    )

    def segment_customer(row):
        if row["RFM_Score"] == "444":
            return "Best Customers"
        elif int(row["R_score"]) >= 3 and int(row["F_score"]) >= 3:
            return "Loyal Customers"
        elif int(row["R_score"]) == 4:
            return "New Customers"
        elif int(row["R_score"]) <= 2 and int(row["F_score"]) >= 3:
            return "At Risk Loyal"
        elif int(row["R_score"]) <= 2:
            return "At Risk"
        else:
            return "Regular"

    rfm["Segment"] = rfm.apply(segment_customer, axis=1)
    return rfm


# -----------------------------
# Read Files
# -----------------------------
try:
    tx = load_transaction_data()
    customers = load_customer_data()
    rfm = build_rfm(tx)
except FileNotFoundError:
    st.error(
        "Data file not found. Make sure these files exist inside data/processed/: "
        "final_cleaned_data.csv and customer_model_data.csv"
    )
    st.stop()

# -----------------------------
# Page Title
# -----------------------------
st.title("Retail Analytics & Churn Prediction Dashboard")
st.markdown("""
This dashboard provides insights into sales performance, customer behavior,
customer segmentation, and churn risk using machine learning and RFM analysis.
""")

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")

min_date = tx["InvoiceDate"].min().date()
max_date = tx["InvoiceDate"].max().date()

selected_dates = st.sidebar.date_input(
    "Invoice Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

country_options = sorted(tx["Country"].dropna().unique().tolist())
selected_countries = st.sidebar.multiselect(
    "Country",
    options=country_options,
    default=country_options
)

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
else:
    start_date, end_date = min_date, max_date

# -----------------------------
# Apply Filters
# -----------------------------
filtered_tx = tx[
    (tx["InvoiceDate"].dt.date >= start_date)
    & (tx["InvoiceDate"].dt.date <= end_date)
    & (tx["Country"].isin(selected_countries))
].copy()

filtered_customer_ids = filtered_tx["CustomerID"].dropna().unique()

filtered_customers = customers[customers["CustomerID"].isin(filtered_customer_ids)].copy()
filtered_rfm = rfm[rfm["CustomerID"].isin(filtered_customer_ids)].copy()

# -----------------------------
# KPI Cards
# -----------------------------
revenue = filtered_tx["TotalPrice"].sum()
orders = filtered_tx["InvoiceNo"].nunique()
active_customers = filtered_tx["CustomerID"].nunique()
churn_rate = filtered_customers["Churn"].mean() * 100 if not filtered_customers.empty else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Revenue", f"${revenue:,.2f}")
k2.metric("Orders", f"{orders:,}")
k3.metric("Customers", f"{active_customers:,}")
k4.metric("Customer Churn", f"{churn_rate:.1f}%")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Overview",
    "Demand Intelligence",
    "Customer Hub",
    "Churn Insights"
])

# -----------------------------
# Executive Overview
# -----------------------------
with tab1:
    col1, col2 = st.columns(2)

    daily_sales = (
        filtered_tx.groupby(filtered_tx["InvoiceDate"].dt.date)["TotalPrice"]
        .sum()
        .reset_index(name="Sales")
    )
    fig_sales = px.line(
        daily_sales,
        x="InvoiceDate",
        y="Sales",
        title="Daily Sales Trend"
    )
    col1.plotly_chart(fig_sales, use_container_width=True)

    country_sales = (
        filtered_tx.groupby("Country")["TotalPrice"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    fig_country = px.bar(
        country_sales,
        x="Country",
        y="TotalPrice",
        title="Top 10 Countries by Revenue",
        color_discrete_sequence=["#FF4B4B"]
    )
    col2.plotly_chart(fig_country, use_container_width=True)

    st.info("Most revenue is driven by a small number of countries, with the United Kingdom contributing the highest share.")

    st.subheader("Top 10 Products by Revenue")
    top_products = (
        filtered_tx.groupby("Description")["TotalPrice"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    st.dataframe(top_products, use_container_width=True)

# -----------------------------
# Demand Intelligence
# -----------------------------
with tab2:
    monthly_sales = filtered_tx.copy()
    monthly_sales["YearMonth"] = monthly_sales["InvoiceDate"].dt.to_period("M").astype(str)
    monthly_sales = monthly_sales.groupby("YearMonth")["TotalPrice"].sum().reset_index()

    fig_month = px.bar(
        monthly_sales,
        x="YearMonth",
        y="TotalPrice",
        title="Monthly Revenue Trend",
        color_discrete_sequence=["#FF4B4B"]
    )
    st.plotly_chart(fig_month, use_container_width=True)

    st.info("Monthly sales patterns help identify seasonal spikes and business demand fluctuations.")

    top_skus = (
        filtered_tx.groupby("StockCode")
        .agg(Revenue=("TotalPrice", "sum"), Quantity=("Quantity", "sum"))
        .sort_values("Revenue", ascending=False)
        .head(15)
        .reset_index()
    )

    st.subheader("Top SKUs by Revenue")
    st.dataframe(top_skus, use_container_width=True)

# -----------------------------
# Customer Hub
# -----------------------------
with tab3:
    col1, col2 = st.columns(2)

    if not filtered_rfm.empty:
        segment_counts = filtered_rfm["Segment"].value_counts().reset_index()
        segment_counts.columns = ["Segment", "Count"]

        fig_segment = px.bar(
            segment_counts,
            x="Segment",
            y="Count",
            title="Customer Segments",
            color_discrete_sequence=["#FF4B4B"]
        )
        col1.plotly_chart(fig_segment, use_container_width=True)

    top_customers = (
        filtered_tx.groupby("CustomerID")["TotalPrice"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig_customers = px.bar(
        top_customers,
        x="CustomerID",
        y="TotalPrice",
        title="Top 10 Customers by Spend",
        color_discrete_sequence=["#FF4B4B"]
    )
    col2.plotly_chart(fig_customers, use_container_width=True)

    st.info("A small number of customers contribute a significant share of revenue, highlighting the importance of retention and loyalty strategies.")

    st.subheader("RFM Segment Preview")
    st.dataframe(filtered_rfm.head(20), use_container_width=True)

# -----------------------------
# Churn Insights
# -----------------------------
with tab4:
    if not filtered_customers.empty:
        churn_counts = filtered_customers["Churn"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Count"]
        churn_counts["ChurnLabel"] = churn_counts["Churn"].map({0: "Active", 1: "Churned"})

        fig_churn = px.pie(
            churn_counts,
            names="ChurnLabel",
            values="Count",
            title="Customer Churn Distribution",
            color_discrete_sequence=["#FF4B4B", "#1F77B4"]
        )
        st.plotly_chart(fig_churn, use_container_width=True)

        st.info("Customers with long inactivity periods are classified as churned, helping the business identify retention opportunities early.")

        feature_cols = [
            "CustomerID",
            "TotalOrders",
            "TotalQuantity",
            "TotalSpend",
            "Recency",
            "CustomerLifetime",
            "AvgOrderValue",
            "Churn"
        ]

        st.subheader("Customer Model Data Preview")
        st.dataframe(filtered_customers[feature_cols].head(25), use_container_width=True)

    else:
        st.warning("No customer data available for the selected filters.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built with Streamlit, Plotly, Python, and machine learning outputs from the NeuralRetail project.")