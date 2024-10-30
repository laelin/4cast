import pandas as pd
from prophet import Prophet
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
import plotly.express as px

# Streamlit app title
st.title("Prediction Error Over Time with Prophet")

# Download historical data for Zillow (Z)
data = yf.download('Z', start='2015-01-01', end='2024-10-01')
data.columns = data.columns.get_level_values(0)  # Flatten MultiIndex columns if present

# Prepare the data for Prophet
data = data.reset_index()
data['Date'] = data['Date'].dt.tz_localize(None)  # Remove timezone information
data = data.rename(columns={'Date': 'ds', 'Close': 'y'})
data = data[['ds', 'y']]
data = data.dropna(subset=['y'])
data['y'] = pd.to_numeric(data['y'], errors='coerce')

# Number of prediction waves and increments for training data
num_waves = 100
increment = int(len(data) / num_waves)

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
    forecast_date_index = forecast[forecast['ds'] >= t_i_plus_30].index[0]
    yhat_i = forecast.loc[forecast_date_index, 'yhat']

    # Get the actual value at t_i_plus_30
    data_date_index = data[data['ds'] >= t_i_plus_30].index[0]
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