import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def sales_by_category(df):
    """
    Bar chart: Sales by product category
    """

    category_sales = df.groupby('Category_Name')['Sales'].sum().sort_values(ascending=False)

    fig, ax = plt.subplots()
    category_sales.head(10).plot(kind='bar', ax=ax)

    ax.set_title("Top 10 Categories by Sales")
    ax.set_ylabel("Sales Amount")
    ax.set_xlabel("Category")

    return fig


def shipping_mode_vs_delay(df):
    """
    Bar chart: Shipping mode vs late delivery risk
    """

    shipping_delay = df.groupby('Shipping_Mode')['Late_delivery_risk'].mean().sort_values(ascending=False)

    fig, ax = plt.subplots()
    shipping_delay.plot(kind='bar', ax=ax)

    ax.set_title("Late Delivery Risk by Shipping Mode")
    ax.set_ylabel("Late Delivery Probability")

    return fig

def delivery_delay_by_shipping_mode(df):
    """
    Bar chart showing average delivery delay by shipping mode
    """

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

def discount_vs_late_risk(df):
    """
    Bar chart showing late delivery risk by discount bucket
    """

    discount_risk = df.groupby(
        pd.cut(df['Order_Item_Discount_Rate'], bins=5)
    )['Late_delivery_risk'].mean()

    fig, ax = plt.subplots()
    discount_risk.plot(kind='bar', ax=ax)

    ax.set_title("Late Delivery Risk by Discount Level")
    ax.set_ylabel("Late Delivery Probability")
    ax.set_xlabel("Discount Range")

    return fig
