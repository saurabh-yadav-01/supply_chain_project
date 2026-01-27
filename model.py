import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

def train_late_delivery_model(df):

    # Feature selection (NO leakage)
    X = df[[
        'Shipping_Mode',
        'Days_for_shipment_(scheduled)',
        'Order_Item_Discount_Rate',
        'Order_Item_Quantity',
        'Sales',
        'Order_Region'
    ]].copy()

    y = df['Late_delivery_risk']

    ship_enc = LabelEncoder()
    region_enc = LabelEncoder()

    X['Shipping_Mode'] = ship_enc.fit_transform(X['Shipping_Mode'])
    X['Order_Region'] = region_enc.fit_transform(X['Order_Region'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Gradient Boosting
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)

    return model, accuracy, ship_enc, region_enc


def get_feature_importance(model, feature_names):

    return pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
