def calculate_kpis(df):

    total_orders = df['Order_Id'].nunique()
    total_sales = df['Sales'].sum()
    total_profit = df['Order_Profit_Per_Order'].sum()
    late_delivery_rate = df['Late_delivery_risk'].mean() * 100

    # KPI 1: Average delivery time in HOURS
    avg_delivery_hours = (df['Days_for_shipping_(real)'] * 24).mean()

    # KPI 2: Same day delivery efficiency
    same_day_df = df[df['Shipping_Mode'] == "Same Day"]

    if len(same_day_df) > 0:
        same_day_efficiency = (
            (same_day_df['Late_delivery_risk'] == 0).mean() * 100
        )
    else:
        same_day_efficiency = 0

    return (
        total_orders,
        total_sales,
        total_profit,
        late_delivery_rate,
        avg_delivery_hours,
        same_day_efficiency
    )
