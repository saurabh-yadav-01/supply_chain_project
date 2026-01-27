import pandas as pd
import matplotlib.pyplot as plt

# 1. Delivery delay by shipping mode
def delivery_delay_by_shipping_mode(df):

    df = df.copy()
    df['Delivery_Delay'] = (
        df['Days_for_shipping_(real)'] -
        df['Days_for_shipment_(scheduled)']
    )

    delay_summary = df.groupby('Shipping_Mode')['Delivery_Delay'].mean()

    fig, ax = plt.subplots()
    delay_summary.plot(kind='bar', ax=ax)

    ax.set_title("Average Delivery Delay by Shipping Mode")
    ax.set_ylabel("Delay (Days)")
    ax.set_xlabel("Shipping Mode")

    return fig


# 2. Order trend over time
def order_trend_over_time(df):

    order_trend = df.groupby(
        df['order_date_(DateOrders)'].dt.to_period("M")
    )['Order_Id'].nunique()

    fig, ax = plt.subplots()
    order_trend.plot(ax=ax)

    ax.set_title("Order Volume Trend Over Time")
    ax.set_ylabel("Number of Orders")
    ax.set_xlabel("Month")

    return fig


# 3. Discount vs late risk
def discount_vs_late_risk(df):

    discount_risk = df.groupby(
        pd.cut(df['Order_Item_Discount_Rate'], bins=5)
    )['Late_delivery_risk'].mean()

    fig, ax = plt.subplots()
    discount_risk.plot(kind='bar', ax=ax)

    ax.set_title("Late Delivery Risk by Discount Level")
    ax.set_ylabel("Late Delivery Probability")
    ax.set_xlabel("Discount Range")

    return fig


# 4. Market share pie
def market_share_by_orders(df):

    market_orders = df['Market'].value_counts()

    fig, ax = plt.subplots()
    ax.pie(market_orders, labels=market_orders.index, autopct='%1.1f%%')

    ax.set_title("Market Share by Order Volume")

    return fig


# 5. Top products late risk
def top_products_late_risk(df):

    product_risk = (
        df.groupby('Product_Name')['Late_delivery_risk']
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots()
    product_risk.plot(kind='bar', ax=ax)

    ax.set_title("Top 10 Products by Late Delivery Risk")
    ax.set_ylabel("Late Delivery Probability")
    ax.set_xlabel("Product")

    return fig
