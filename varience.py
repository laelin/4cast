import pandas as pd
from prophet import Prophet
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Streamlit app title
st.title("Prediction Error Over Time with Prophet")

# Stopwatch Code
stopwatch_placeholder = st.empty()
start_time = time.time()

# Update the stopwatch
while True:
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(int(elapsed_time), 60)
    hours, minutes = divmod(minutes, 60)
    stopwatch_placeholder.markdown(
        f"<h1 style='text-align: center; color: #4CAF50; font-size: 48px;'>🕒 {hours:02d}:{minutes:02d}:{seconds:02d}</h1>",
        unsafe_allow_html=True
    )
    time.sleep(1)
    break

# Download historical data for Zillow (Z)
data = yf.download('Z', start='2015-01-01', end='2024-10-01')
data.columns = data.columns.get_level_values(0)  # Flatten MultiIndex columns if present

# Check if data is empty
if data.empty:
    raise ValueError("No data downloaded. Please check the ticker symbol and date range.")

# Display completion message
stopwatch_placeholder.markdown(
    f"<h1 style='text-align: center; color: #E91E63; font-size: 48px;'>✔️ Analysis Complete in {hours:02d}:{minutes:02d}:{seconds:02d}</h1>",
    unsafe_allow_html=True
)

# Download historical data for Zillow (Z)
data = yf.download('Z', start='2015-01-01', end='2024-10-01')
data.columns = data.columns.get_level_values(0)  # Flatten MultiIndex columns if present

# Check if data is empty
if data.empty:
    raise ValueError("No data downloaded. Please check the ticker symbol and date range.")

# Print the length of data
print("Length of data:", len(data))

# Prepare the data for Prophet
data = data.reset_index()
data['Date'] = data['Date'].dt.tz_localize(None)  # Remove timezone information
data = data.rename(columns={'Date': 'ds', 'Close': 'y'})
data = data[['ds', 'y']]
data = data.dropna(subset=['y'])
data['y'] = pd.to_numeric(data['y'], errors='coerce')

# Number of prediction waves and increments for training data
num_waves = 100
# Adjust num_waves if data length is less
num_waves = min(num_waves, len(data))

increment = int(len(data) / num_waves)

# Ensure increment is at least 1
increment = max(1, increment)

# Check if increment is zero
if increment == 0:
    raise ValueError("Increment calculated to be zero. Adjust num_waves or ensure data has sufficient length.")

# Initialize lists to store errors and corresponding dates
dates_of_wave = []
errors = []
yhat_list = []
y_actual_list = []

# Generate forecast data for each wave and calculate errors
for i in range(increment, len(data), increment):
    subset_data = data.iloc[:i]

    # Initialize and fit the model
    model = Prophet()
    model.fit(subset_data)

    # Make future predictions
    future = model.make_future_dataframe(periods=180)
    forecast = model.predict(future)

    # Get the last date in the subset (current wave date)
    t_i = subset_data['ds'].iloc[-1]

    # Compute the date 30 days ahead
    t_i_plus_30 = t_i + pd.Timedelta(days=30)

    # Check if t_i_plus_30 is within the available data range
    if t_i_plus_30 > data['ds'].iloc[-1]:
        break

    # Get the forecasted value at t_i_plus_30
    forecast_future = forecast[forecast['ds'] >= t_i_plus_30]
    if forecast_future.empty:
        continue  # Skip if no forecast is available for t_i_plus_30
    forecast_date_index = forecast_future.index[0]
    yhat_i = forecast.loc[forecast_date_index, 'yhat']

    # Get the actual value at t_i_plus_30
    data_future = data[data['ds'] >= t_i_plus_30]
    if data_future.empty:
        continue  # Skip if no actual data is available for t_i_plus_30
    data_date_index = data_future.index[0]
    y_actual_i = data.loc[data_date_index, 'y']

    # Compute the error
    error_i = yhat_i - y_actual_i

    # Append the results to the lists
    dates_of_wave.append(t_i)
    errors.append(error_i)
    yhat_list.append(yhat_i)
    y_actual_list.append(y_actual_i)

# Create a DataFrame with the collected error data
error_df = pd.DataFrame({
    'Date': dates_of_wave,
    'Error': errors,
    'Predicted': yhat_list,
    'Actual': y_actual_list
})

# Calculate Percentage Error
error_df['Percentage Error'] = (error_df['Error'] / error_df['Actual']) * 100

# Add a column to track if error exceeds $5 threshold
error_df['OutOfBounds'] = error_df['Error'].apply(lambda x: abs(x) > 5)

# Add rolling mean and rolling standard deviation columns
error_df['Rolling_Mean_Error'] = error_df['Error'].rolling(window=10).mean()
error_df['Rolling_Std_Error'] = error_df['Error'].rolling(window=10).std()

from prophet.diagnostics import cross_validation, performance_metrics

# Perform cross-validation with an initial training period and horizon
df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='30 days')
df_p = performance_metrics(df_cv)
st.write("Cross-Validation Performance Metrics")
st.dataframe(df_p)

# Show Metrics
st.write("Mean Absolute Error (MAE):", error_df['Error'].abs().mean())
st.write("Root Mean Squared Error (RMSE):", (error_df['Error'] ** 2).mean() ** 0.5)
st.write("Mean Absolute Percentage Error (MAPE):", error_df['Percentage Error'].abs().mean())

# Plot Error Histogram
fig_hist, ax_hist = plt.subplots()
sns.histplot(error_df['Error'], bins=20, kde=True, ax=ax_hist)
ax_hist.set_title("Error Distribution Histogram")
ax_hist.set_xlabel("Prediction Error")
st.pyplot(fig_hist)

# Box Plot of Errors
fig_box, ax_box = plt.subplots()
sns.boxplot(y=error_df['Error'], ax=ax_box)
ax_box.set_title("Box Plot of Prediction Errors")
st.pyplot(fig_box)

# Plot Rolling Mean and Rolling Standard Deviation
fig_rolling, ax_rolling = plt.subplots()
ax_rolling.plot(error_df['Date'], error_df['Rolling_Mean_Error'], label='Rolling Mean Error')
ax_rolling.plot(error_df['Date'], error_df['Rolling_Std_Error'], label='Rolling Std Error')
ax_rolling.set_title("Rolling Mean and Std of Prediction Errors")
ax_rolling.legend()
st.pyplot(fig_rolling)

# Create a line chart of the prediction error over time
fig_error = px.line(
    error_df,
    x='Date',
    y='Error',
    title='Prediction Error Over Time (30 Days Ahead)',
    labels={'Error': 'Prediction Error', 'Date': 'Date of Wave'}
)
fig_error.update_layout(width=900, height=600)

# Create a line chart comparing Predicted and Actual values at t+30 days
fig_compare = go.Figure()
fig_compare.add_trace(go.Scatter(
    x=error_df['Date'] + pd.Timedelta(days=30),
    y=error_df['Actual'],
    mode='lines+markers',
    name='Actual'
))
fig_compare.add_trace(go.Scatter(
    x=error_df['Date'] + pd.Timedelta(days=30),
    y=error_df['Predicted'],
    mode='lines+markers',
    name='Predicted'
))
fig_compare.update_layout(
    title='Predicted vs Actual Values at t+30 Days',
    xaxis_title='Date (t + 30 days)',
    yaxis_title='Stock Price',
    width=900,
    height=600
)

# Display the error plot in Streamlit
st.plotly_chart(fig_error)

# Display the comparison plot in Streamlit
st.plotly_chart(fig_compare)
