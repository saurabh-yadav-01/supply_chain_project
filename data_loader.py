import pandas as pd

def load_data():
    """
    This function loads the supply chain dataset
    and performs basic preprocessing.
    """

    # Read CSV file into pandas DataFrame
    df = pd.read_csv("DataCoSupplyChainDataset1.csv")

    # Convert order date column to datetime
    df['order_date_(DateOrders)'] = pd.to_datetime(
        df['order_date_(DateOrders)'], errors='coerce'
    )

    # Fill missing values (simple & safe method)
    df.fillna(0, inplace=True)

    # Return cleaned dataframe
    return df
