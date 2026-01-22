import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time

# Import project modules
from data_loader import load_data
from analytics import calculate_kpis, region_wise_late_risk
from visualizations import sales_by_category, shipping_mode_vs_delay, delivery_delay_by_shipping_mode, discount_vs_late_risk
from model import train_late_delivery_model, get_feature_importance


with st.spinner("Loading dashboard..."):
    time.sleep(1)


# Page configuration
st.set_page_config(
    page_title="Supply Chain Analytics Dashboard",
    layout="wide"
)

# App title
st.markdown(
    "<h1 style='color:red;'>📦 Supply Chain Analytics Dashboard</h1>",
    unsafe_allow_html=True
)

# Load data
df = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filters")

region_filter = st.sidebar.multiselect(
    "Select Region",
    options=df['Order_Region'].unique(),
    default=df['Order_Region'].unique()
)

st.sidebar.subheader("📅 Date Filter")

min_date = df['order_date_(DateOrders)'].min()
max_date = df['order_date_(DateOrders)'].max()

start_date, end_date = st.sidebar.date_input(
    "Select Order Date Range",
    [min_date, max_date]
)

df = df[
    (df['order_date_(DateOrders)'] >= pd.to_datetime(start_date)) &
    (df['order_date_(DateOrders)'] <= pd.to_datetime(end_date))
]

shipping_filter = st.sidebar.multiselect(
    "Shipping Mode",
    df['Shipping_Mode'].unique(),
    default=df['Shipping_Mode'].unique()
)

df = df[df['Shipping_Mode'].isin(shipping_filter)]

category_filter = st.sidebar.multiselect(
    "Product Category",
    df['Category_Name'].unique(),
    default=df['Category_Name'].unique()
)

df = df[df['Category_Name'].isin(category_filter)]


# Apply filters
df = df[df['Order_Region'].isin(region_filter)]

# Calculate KPIs
total_orders, total_sales, total_profit, late_rate, avg_delay, aov = calculate_kpis(df)

# KPI display
col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric("📦 Total Orders", total_orders)
col2.metric("💰 Total Sales", f"${total_sales:,.0f}")
col3.metric("📈 Total Profit", f"${total_profit:,.0f}")
col4.metric("⚠️ Late Delivery %", f"{late_rate:.2f}%")
col5.metric("Avg Delivery Delay (hrs)", f"{avg_delay:.2f}")
col6.metric("Avg Order Value", f"${aov:,.0f}")

st.markdown("---")

# Charts section
st.subheader("📊 Visual Analytics")

col1, col2 = st.columns(2)

with col1:
    st.pyplot(sales_by_category(df))

with col2:
    st.pyplot(shipping_mode_vs_delay(df))

col3, col4 =st.columns(2)

with col3:
    st.pyplot(delivery_delay_by_shipping_mode(df))

with col4:
    st.pyplot(discount_vs_late_risk(df))

# Risk table
st.subheader("📋 Order Level Data")

st.dataframe(
    df[['Order_Id', 'Shipping_Mode', 'Sales', 'Late_delivery_risk']],
    use_container_width=True
)

st.markdown("---")
st.subheader("🤖 Late Delivery Prediction (ML)")

if st.button("Train Prediction Model"):
    model, accuracy, ship_enc, region_enc = train_late_delivery_model(df)

    # Store everything in session_state
    st.session_state.model = model
    st.session_state.ship_enc = ship_enc
    st.session_state.region_enc = region_enc

    st.success("Model trained successfully ✅")
    st.info(f"Accuracy: {accuracy*100:.2f}%")


st.markdown("---")
st.subheader("📦 Predict Late Delivery for a New Order")
# ---- User Inputs ----
shipping_mode = st.selectbox(
    "Shipping Mode",
    df['Shipping_Mode'].unique()
)

scheduled_days = st.number_input(
    "Scheduled Shipping Days",
    min_value=1,
    max_value=10,
    value=3
)

discount_rate = st.slider(
    "Discount Rate",
    min_value=0.0,
    max_value=1.0,
    value=0.1
)

order_region = st.selectbox(
    "Order Region",
    df['Order_Region'].unique()
)

if st.button("Predict Late Delivery"):

    # Safety check
    if 'model' not in st.session_state:
        st.warning("⚠️ Please train the model first")
    else:
        input_df = pd.DataFrame({
            'Shipping_Mode': [shipping_mode],
            'Days_for_shipment_(scheduled)': [scheduled_days],
            'Order_Item_Discount_Rate': [discount_rate],
            'Order_Region': [order_region]
        })

        # Use SAME encoders used during training
        input_df['Shipping_Mode'] = st.session_state.ship_enc.transform(
            input_df['Shipping_Mode']
        )
        input_df['Order_Region'] = st.session_state.region_enc.transform(
            input_df['Order_Region']
        )

        proba = st.session_state.model.predict_proba(input_df)[0]

        late_prob = proba[1] * 100   # Probability of late delivery
        on_time_prob = proba[0] * 100

        st.write("### 🔍 Prediction Result")

        if late_prob > 50:
            st.error(f"⚠️ High Risk of Late Delivery")
        else:
            st.success(f"✅ Likely On-Time Delivery")

        st.progress(int(late_prob))

        st.info(
            f"""
            📊 **Prediction Confidence**
            - Late Delivery Probability: **{late_prob:.2f}%**
            - On-Time Delivery Probability: **{on_time_prob:.2f}%**
            """
        )

st.markdown("---")
st.subheader("🧠 Why the Model Predicts Late Delivery")

if 'model' in st.session_state:

    feature_names = [
        'Shipping_Mode',
        'Scheduled_Shipping_Days',
        'Discount_Rate',
        'Order_Region'
    ]

    importance_df = get_feature_importance(
        st.session_state.model,
        feature_names
    )

    st.dataframe(importance_df)

    fig, ax = plt.subplots()

    ax.barh(
        importance_df['Feature'],
        importance_df['Absolute_Impact']
    )

    ax.set_xlabel("Impact Strength")
    ax.set_title("Feature Importance for Late Delivery Prediction")

    st.pyplot(fig)

