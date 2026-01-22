def calculate_kpis(df):
    """
    Calculates important KPIs for dashboard
    """

    # Total number of orders
    total_orders = df['Order_Id'].nunique()

    # Total sales amount
    total_sales = df['Sales'].sum()

    # Total profit
    total_profit = df['Order_Profit_Per_Order'].sum()

    # Late delivery percentage
    late_delivery_rate = df['Late_delivery_risk'].mean() * 100

    # Average Delivery Days
    avg_delay = ((df['Days_for_shipping_(real)'] - df['Days_for_shipment_(scheduled)']).mean())*24

    # Average Order Values
    aov = df['Sales'].sum() / df['Order_Id'].nunique()


    return total_orders, total_sales, total_profit, late_delivery_rate,avg_delay, aov


def region_wise_late_risk(df):
    """
    Calculates late delivery risk per region
    """

    return df.groupby('Order_Region')['Late_delivery_risk'].mean().sort_values(ascending=False)
