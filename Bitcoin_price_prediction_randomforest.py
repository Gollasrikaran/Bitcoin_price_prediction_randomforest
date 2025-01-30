import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
# Streamlit UI
st.title('Bitcoin Price Prediction App (Random Forest)')
# Upload file
uploaded_file = st.file_uploader("Upload Bitcoin Data CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])  # Convert Date column to datetime
    df = df.sort_values(by='Date')  # Ensure data is in order
    # Clean numerical columns (remove commas, convert percentages)
    numeric_columns = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']
    for col in numeric_columns:
        df[col] = df[col].astype(str).str.replace(',', '')  # Remove commas
        if col == 'Change %':
            df[col] = df[col].str.replace('%', '').astype(float)  # Remove %
        elif col == 'Vol.':
            df[col] = df[col].str.replace('K', '').astype(float) * 1000  # Convert K to number
        else:
            df[col] = df[col].astype(float)  # Convert to float
    # Create lag features (using past 60 days to predict next day)
    lookback_days = 60
    for i in range(1, lookback_days + 1):
        df[f'PriceLag{i}'] = df['Price'].shift(i)
    df.dropna(inplace=True)  # Drop rows with NaN values after shifting
    # Define features and target
    features = [col for col in df.columns if 'Lag' in col]
    target = 'Price'  # Predict next day's price
    # Split into training and testing sets
    train_size = int(0.8 * len(df))  # 80% train, 20% test
    X_train, X_test = df[features][:train_size], df[features][train_size:]
    y_train, y_test = df[target][:train_size], df[target][train_size:]
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    # Predict next day's price
    y_pred = model.predict(X_test)
    # Evaluate model
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f'Mean Absolute Error: {mae:.2f}')
    # Check if investing today is profitable
    last_features = df[features].iloc[-1].values.reshape(1, -1)
    next_day_price = model.predict(last_features)[0]
    current_price = df['Price'].iloc[-1]
    st.write(f'Current Price: {current_price:.2f}')
    st.write(f'Predicted Next Day Price: {next_day_price:.2f}')
    if next_day_price > current_price:
        st.success("Recommendation: BUY (Price expected to increase)")
    else:
        st.error("Recommendation: DON'T BUY (Price expected to decrease or stay same)")
    # Plot actual vs predicted prices
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['Date'][train_size:], y_test, label='Actual Price', color='blue')
    ax.plot(df['Date'][train_size:], y_pred, label='Predicted Price', color='red', linestyle='dashed')
    ax.set_xlabel('Date')
    ax.set_ylabel('Bitcoin Price')
    ax.legend()
    ax.set_title('Bitcoin Price Prediction (Random Forest)')
    st.pyplot(fig)
"Instead of random forest use super vector regression"