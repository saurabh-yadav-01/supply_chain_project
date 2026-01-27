import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time

from data_loader import load_data
from analytics import calculate_kpis
from model import train_late_delivery_model, get_feature_importance

from visualizations import (
    delivery_delay_by_shipping_mode,
    order_trend_over_time,
    discount_vs_late_risk,
    market_share_by_orders,
    top_products_late_risk
)

# ---------------- PAGE CONFIG ----------------

st.set_page_config(page_title="Supply Chain Dashboard", layout="wide")

st.markdown("""
<style>
h1, h2 {color:red;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Supply Chain Analytics & Late Delivery Prediction</h1>", unsafe_allow_html=True)

with st.spinner("Loading dashboard..."):
    time.sleep(1)

# ---------------- LOAD DATA ----------------

df = load_data()

# ---------------- SIDEBAR FILTERS ----------------

st.sidebar.header("🔍 Filters")

region_filter = st.sidebar.multiselect(
    "Region",
    df['Order_Region'].unique(),
    default=df['Order_Region'].unique()
)

shipping_filter = st.sidebar.multiselect(
    "Shipping Mode",
    df['Shipping_Mode'].unique(),
    default=df['Shipping_Mode'].unique()
)

category_filter = st.sidebar.multiselect(
    "Category",
    df['Category_Name'].unique(),
    default=df['Category_Name'].unique()
)

df = df[
    (df['Order_Region'].isin(region_filter)) &
    (df['Shipping_Mode'].isin(shipping_filter)) &
    (df['Category_Name'].isin(category_filter))
]

# ---------------- KPIs ----------------

total_orders, total_sales, total_profit, late_rate, avg_hours, same_day_eff = calculate_kpis(df)

k1, k2, k3, k4, k5,  k6 = st.columns(6)
k1.metric("Total Orders", total_orders)
k2.metric("Total Sales", f"${total_sales:,.0f}")
k3.metric("Total Profit", f"${total_profit:,.0f}")
k4.metric("Late Delivery %", f"{late_rate:.2f}%")
k5.metric("Avg Delivery Time (hrs)", f"{avg_hours:.1f}")
k6.metric("Same Day Efficiency", f"{same_day_eff:.2f}%")

st.markdown("---")

# ---------------- VISUALS ----------------

st.subheader("Analytics")

col1, col2 = st.columns(2)

with col1:
    st.pyplot(delivery_delay_by_shipping_mode(df))
    st.pyplot(order_trend_over_time(df))
    st.pyplot(market_share_by_orders(df))

with col2:
    st.pyplot(discount_vs_late_risk(df))
    st.pyplot(top_products_late_risk(df))

st.markdown("---")

# ---------------- ML TRAINING ----------------

st.subheader("Gradient Boosting Late Delivery Prediction")

if st.button("Train Prediction Model"):

    model, accuracy, ship_enc, region_enc = train_late_delivery_model(df)

    st.session_state.model = model
    st.session_state.ship_enc = ship_enc
    st.session_state.region_enc = region_enc

    st.success("Model trained successfully ✅")
    st.info(f"Accuracy: {accuracy*100:.2f}%")

# ---------------- PREDICTION INPUT ----------------

st.subheader("Predict New Order")

shipping_mode = st.selectbox("Shipping Mode", df['Shipping_Mode'].unique())
scheduled_days = st.number_input("Scheduled Shipping Days", 1, 15, 3)
real_days = st.number_input("Actual Shipping Days", 1, 15, 4)
discount_rate = st.slider("Discount Rate", 0.0, 1.0, 0.1)
quantity = st.number_input("Order Quantity", 1, 10, 2)
sales = st.number_input("Sales Amount", 10.0, 10000.0, 200.0)
order_region = st.selectbox("Order Region", df['Order_Region'].unique())

threshold = st.slider("Risk Threshold (%)", 10, 90, 50)

if st.button("Predict Late Delivery"):

    if 'model' not in st.session_state:
        st.warning("Train model first")
    else:

        input_df = pd.DataFrame({
            'Shipping_Mode': [shipping_mode],
            'Days_for_shipment_(scheduled)': [scheduled_days],
            'Order_Item_Discount_Rate': [discount_rate],
            'Order_Item_Quantity': [quantity],
            'Sales': [sales],
            'Order_Region': [order_region]
        })


        input_df['Shipping_Mode'] = st.session_state.ship_enc.transform(input_df['Shipping_Mode'])
        input_df['Order_Region'] = st.session_state.region_enc.transform(input_df['Order_Region'])

        proba = st.session_state.model.predict_proba(input_df)[0]

        late_prob = proba[1] * 100
        ontime_prob = proba[0] * 100

        if late_prob >= threshold:
            st.error("⚠️ HIGH RISK of Late Delivery")
        else:
            st.success("✅ LOW RISK of Late Delivery")

        st.progress(int(late_prob))

        st.info(f"""
Late Delivery Probability: {late_prob:.2f}%  
On-Time Probability: {ontime_prob:.2f}%  
Threshold Applied: {threshold}%
""")

# ---------------- FEATURE IMPORTANCE ----------------

st.markdown("---")
st.subheader("🧠 Feature Importance")

if 'model' in st.session_state:

    feature_names = [
        'Shipping_Mode',
        'Scheduled_Days',
        'Discount_Rate',
        'Quantity',
        'Sales',
        'Region'
    ]

    imp_df = get_feature_importance(st.session_state.model, feature_names)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Importance Table")
        st.dataframe(imp_df)

    with col2:
        st.subheader("Importance Chart")

        fig, ax = plt.subplots()
        ax.barh(imp_df['Feature'], imp_df['Importance'])
        ax.set_title("Random Forest Feature Importance")

        st.pyplot(fig)
