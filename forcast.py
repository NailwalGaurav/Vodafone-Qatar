import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ================================
# UI Setup
# ================================
st.set_page_config(layout="wide")
st.markdown("<style>.main {padding-top: 0px;}</style>", unsafe_allow_html=True)

st.sidebar.image("https://www.truesky.com/wp-content/uploads/2016/08/revenue_forecasting.jpg", use_container_width=True)
st.markdown(
    "<h1 style='text-align: center; margin-top: -20px;'>📊 Vodafone Qatar Revenue Forecasting</h1>",
    unsafe_allow_html=True,
)

# ================================
# Sidebar Inputs
# ================================
st.sidebar.header("Forecast Settings")

n_future = st.sidebar.number_input(
    "Forecast Quarters Ahead", min_value=4, max_value=12, value=8, step=1
)

future_subscriber = st.sidebar.number_input(
    "Expected Subscribers per Quarter",
    min_value=80_000,
    max_value=40000000,
    value=300_000,
    step=10_000
)
@st.cache_data
if st.sidebar.button("Run Forecast"):
    # ================================
    # Load and Preprocess CSV
    # ================================
    df = pd.read_csv("voda.csv")

    quarter_to_month = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}
    df["Month"] = df["Quarter"].map(quarter_to_month)
    df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"].astype(str))
    df.set_index("Date", inplace=True)
    df = df.drop(columns=["Month"])

    ts_df = df.groupby(level=0).agg(
        {
            "Revenue_QR_Mn": "sum",
            "Subscribers": "sum",
            "ARPU_QR": "mean",
            "Data_Usage_GB": "sum",
        }
    ).round(2)

    ts_df = ts_df.asfreq("QS-JAN").sort_index()

    # ================================
    # Train/Test Split
    # ================================
    n_test = 4
    train_end = len(ts_df) - n_test

    train_revenue = ts_df["Revenue_QR_Mn"][:train_end]
    test_revenue = ts_df["Revenue_QR_Mn"][train_end:]

    exog = ts_df[["Subscribers"]]
    scaler = StandardScaler()
    exog_scaled = pd.DataFrame(
        scaler.fit_transform(exog), columns=exog.columns, index=exog.index
    )
    train_exog = exog_scaled[:train_end]
    test_exog = exog_scaled[train_end:]

    # ================================
    # SARIMAX (with exogenous)
    # ================================
    sarimax_model = SARIMAX(
        train_revenue,
        exog=train_exog,
        order=(1, 1, 1),
        seasonal_order=(0, 0, 0, 4),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    sarimax_forecast = sarimax_model.get_forecast(steps=n_test, exog=test_exog)
    sarimax_pred = sarimax_forecast.predicted_mean

    # ================================
    # ARIMA (no exogenous)
    # ================================
    arima_model = ARIMA(train_revenue, order=(1, 1, 1)).fit()
    arima_forecast = arima_model.get_forecast(steps=n_test)
    arima_pred = arima_forecast.predicted_mean

    # ================================
    # Forecast Future Quarters
    # ================================
    last_date = ts_df.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.offsets.QuarterBegin(), periods=n_future, freq="QS-JAN"
    )

    future_exog_scaled = pd.DataFrame(
        scaler.transform(pd.DataFrame({"Subscribers": [future_subscriber] * n_future})),
        columns=["Subscribers"],
        index=future_dates
    )

    future_forecast = sarimax_model.get_forecast(steps=n_future, exog=future_exog_scaled)
    future_pred = future_forecast.predicted_mean
    future_ci = future_forecast.conf_int()


     # ================================
    #  Revenue Metrics (in Billion QAR)
    # ================================
    def custom_kpi(label, value):
          st.markdown(
        f"""
        <div style="
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;">
            <h3 style="margin-bottom: 5px; color: #333;">{label}</h3>
            <h2 style="margin-top: 0; color: #2c7be5;">{value}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )


    st.divider()
    st.markdown("### 💼 Revenue Metrics")

# Calculate values
    total_revenue_billion = ts_df["Revenue_QR_Mn"].sum() / 1000
    forecast_revenue_billion = future_pred.iloc[-1] / 1000

    col1, col2 = st.columns(2)

    with col1:
         custom_kpi("💰 Total Historical Revenue", f"{total_revenue_billion:,.3f} bn QAR")

    with col2:
         custom_kpi("🔮 Forecasted Revenue", f"{forecast_revenue_billion:,.3f} bn QAR")
   
   
    st.divider()
   
     # ================================
    # Forecast Table (Bn QAR)
    # ================================
    forecast_df = pd.DataFrame({
    "Quarter": future_pred.index.to_period("Q").astype(str),
    "Forecast_Revenue_bn_QAR": future_pred.values / 1000,
})

    st.subheader("📋 Forecasted Quarterly Revenue")

# Convert the DataFrame to HTML with inline styles
    table_html = "<table style='width:100%; border-collapse: collapse; font-family: Arial, sans-serif;'>"

# Header
    table_html += "<thead><tr style='background-color: #4CAF50; color: white;'>"
    for col in forecast_df.columns:
         table_html += f"<th style='padding: 12px; text-align:center;'>{col}</th>"
    table_html += "</tr></thead>"

# Body with alternating row colors
    row_colors = ["#ffffff", "#f2f2f2"]
    table_html += "<tbody>"
    for i, (_, row) in enumerate(forecast_df.iterrows()):
         bg = row_colors[i % 2]
         table_html += f"<tr style='background-color: {bg}; text-align:center;'>"
         for val in row:
               if isinstance(val, float):
                     table_html += f"<td style='padding: 10px; color: black;'>{val:.3f}</td>"
               else:
                     table_html += f"<td style='padding: 10px; color: black;'>{val}</td>"
         table_html += "</tr>"
    table_html += "</tbody></table>"

# Wrap in a card-like container
    st.markdown(
    f"""
    <div style="
        background-color: #E6F2FF;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        overflow-x:auto;">
        {table_html}
    </div>
    """,
    unsafe_allow_html=True
)


   
    # ================================
    # Plot
    # ================================
    st.divider()
    st.markdown(f"### 📈 Historical and Forecasted Quarterly Revenue)")
    fig, ax = plt.subplots(figsize=(14, 5))
    
# Plot history
    ax.plot(ts_df.index, ts_df["Revenue_QR_Mn"] / 1000, label="History", color="blue")
    for i, (x, y) in enumerate(zip(ts_df.index, ts_df["Revenue_QR_Mn"] / 1000)):
         if i % 2 == 0:  # label every 2nd point
              ax.text(x, y + 0.2, f"{y:.3f}bn", ha="center", va="bottom", fontsize=8, color="blue")

# Train/Test split marker
    ax.axvline(x=ts_df.index[train_end], color="gray", linestyle="--", label="Train/Test Split")

# SARIMAX predictions
    ax.plot(test_revenue.index, sarimax_pred / 1000, label="SARIMAX Predictions", color="green")
    for i, (x, y) in enumerate(zip(test_revenue.index, sarimax_pred / 1000)):
         if i % 2 == 0:
              ax.text(x, y + 0.2, f"{y:.3f}bn", ha="center", va="bottom", fontsize=8, color="green")

# Future forecast
    ax.plot(future_pred.index, future_pred / 1000, label="Future Forecast", color="red")
    for i, (x, y) in enumerate(zip(future_pred.index, future_pred / 1000)):
         if i % 2 == 0:
               ax.text(x, y + 0.4, f"{y:.3f}bn", ha="center", va="bottom", fontsize=8, color="red")  # shifted up more

# Confidence interval shading
    ax.fill_between(
    future_ci.index, future_ci.iloc[:, 0] / 1000, future_ci.iloc[:, 1] / 1000,
    color="lightcoral", alpha=0.3
)

# Labels and title
    ax.set_title("Revenue Forecasting")
    ax.set_ylabel("Revenue (Bn QAR)")
    ax.set_xlabel("Date")
    ax.legend()
    st.pyplot(fig)


