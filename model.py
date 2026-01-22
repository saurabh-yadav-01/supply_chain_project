import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_late_delivery_model(df):
    """
    Trains ML model and returns model + encoders
    """

    # Feature selection
    X = df[[
        'Shipping_Mode',
        'Days_for_shipment_(scheduled)',
        'Order_Item_Discount_Rate',
        'Order_Region'
    ]].copy()

    y = df['Late_delivery_risk']

    # Create separate encoders
    shipping_encoder = LabelEncoder()
    region_encoder = LabelEncoder()

    # Encode categorical columns
    X['Shipping_Mode'] = shipping_encoder.fit_transform(X['Shipping_Mode'])
    X['Order_Region'] = region_encoder.fit_transform(X['Order_Region'])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Accuracy
    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, accuracy, shipping_encoder, region_encoder

def get_feature_importance(model, feature_names):
    """
    Extracts feature importance from Logistic Regression model
    """

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0]
    })

    # Absolute value for ranking importance
    importance_df['Absolute_Impact'] = importance_df['Coefficient'].abs()

    # Sort by importance
    importance_df = importance_df.sort_values(
        by='Absolute_Impact', ascending=False
    )

    return importance_df
