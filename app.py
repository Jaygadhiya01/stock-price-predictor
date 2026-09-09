import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from math import sqrt

st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Stock Price Prediction")
st.markdown("Linear Regression + Time Series | NSE | ")

# -----------------------
# Sidebar
# -----------------------
st.sidebar.header("Stock Settings")

# User enters stock name
stock_input = st.sidebar.text_input(
    "Enter NSE Stock Name (e.g., RELIANCE, TCS, INFY)",
    value="RELIANCE"
)

# Auto append .NS safely (અહીં સુધારો કર્યો છે જેથી બે વાર .NS ના ઉમેરાય)
stock = stock_input.upper().strip()
if stock and not stock.endswith(".NS"):
    stock += ".NS"

# Years of historical data
years = st.sidebar.slider(
    "Select Historical Data (Years)",
    min_value=1,
    max_value=10,
    value=5
)

# Lag window
window = st.sidebar.slider(
    "Lag Window (Days)",
    min_value=3,
    max_value=10,
    value=5
)

# -----------------------
# Fetch Data
# -----------------------
period = f"{years}y"

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data(symbol, period_val):
    try:
        # multi_level_column=False ઉમેરવાથી MultiIndex ઈશ્યૂ સોલ્વ થઈ જશે
        df = yf.download(symbol, period=period_val, interval="1d", multi_level_column=False)
        if df.empty:
            return None
        return df
    except Exception as e:
        return None

df = load_data(stock, period)

if df is None or df.empty:
    st.error("❌ Invalid stock or no data found. Try RELIANCE, TCS, INFY, etc.")
    st.stop()

st.subheader("📊 Latest Stock Data")
st.dataframe(df.tail())

# -----------------------
# Feature Engineering
# -----------------------
data = pd.DataFrame()
data['Close'] = df['Close']
data['Target'] = data['Close'].shift(-1)

# Lag features
for i in range(1, window + 1):
    data[f'lag_{i}'] = data['Close'].shift(i)

# Moving averages
data['MA_5'] = data['Close'].rolling(5).mean()
data['MA_10'] = data['Close'].rolling(10).mean()
data.dropna(inplace=True)

features = [f'lag_{i}' for i in range(1, window + 1)] + ['MA_5', 'MA_10']

X = data[features]
y = data['Target']

# Train/Test split (80/20)
split = int(len(data) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -----------------------
# Train Model
# -----------------------
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# -----------------------
# Metrics
# -----------------------
mae = mean_absolute_error(y_test, predictions)
rmse = sqrt(mean_squared_error(y_test, predictions))

st.subheader("📈 Model Performance")
col1, col2 = st.columns(2)
col1.metric("MAE", f"{mae:.2f}")
col2.metric("RMSE", f"{rmse:.2f}")

# -----------------------
# Next Day Prediction
# -----------------------
latest_features = data[features].iloc[-1].values.reshape(1, -1)
next_day_price = float(model.predict(latest_features)[0])

st.subheader("🔮 Next Trading Day Prediction")
st.success(f"Predicted Price: ₹ {next_day_price:.2f}")

# -----------------------
# Auto Refresh Current Price
# -----------------------
current_price = float(df['Close'].iloc[-1])
st.subheader("💹 Current Price")
st.info(f"₹ {current_price:.2f}")

# -----------------------
# Plot Actual vs Predicted
# -----------------------
st.subheader("📉 Stock Price Prediction Chart")

fig, ax = plt.subplots(figsize=(14,6))

# Plot actual prices
ax.plot(y_test.index, y_test.values, label="Actual Price", color="blue", linewidth=2)

# Plot predicted prices
ax.plot(y_test.index, predictions, label="Predicted Price", color="red", linestyle="--", linewidth=2)

# Next day predicted price point
latest_index = y_test.index[-1]
ax.scatter(latest_index, next_day_price, color='orange', s=100, zorder=5)  # orange dot
ax.annotate(f"Next Predicted: ₹{next_day_price:.2f}",
            xy=(latest_index, next_day_price),
            xytext=(latest_index, next_day_price+15),
            fontsize=12, color='orange',
            arrowprops=dict(facecolor='orange', shrink=0.05))

# Current price point
ax.scatter(latest_index, current_price, color='green', s=100, zorder=5)  # green dot
ax.annotate(f"Current: ₹{current_price:.2f}",
            xy=(latest_index, current_price),
            xytext=(latest_index, current_price-15),
            fontsize=12, color='green',
            arrowprops=dict(facecolor='green', shrink=0.05))

# Formatting
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Price (₹)", fontsize=12)
ax.set_title(f"{stock} | Actual vs Predicted Prices", fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=45)

st.pyplot(fig)
