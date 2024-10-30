import pandas as pd
from prophet import Prophet
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

# Streamlit app title
st.title("Evolution of Forecast Over Time with Prophet")

# Function to load data and cache it
@st.cache_data
def load_data():
    # Download historical data for Apple (AAPL)
    data = yf.download('AAPL', start='2015-01-01', end='2024-10-01')
    data.columns = data.columns.get_level_values(0)  # Flatten MultiIndex columns if present

    # Prepare the data for Prophet
    data = data.reset_index()
    data['Date'] = data['Date'].dt.tz_localize(None)  # Remove timezone information
    data = data.rename(columns={'Date': 'ds', 'Close': 'y'})
    data = data[['ds', 'y']]
    data = data.dropna(subset=['y'])
    data['y'] = pd.to_numeric(data['y'], errors='coerce')
    return data

# Function to generate forecast frames and cache them
@st.cache_data
def generate_forecast_frames(data, num_waves):
    increment = int(len(data) / num_waves)
    frames = []
    for i in range(increment, len(data), increment):
        # Subset data up to the current point in the wave
        subset_data = data.iloc[:i]

        # Initialize and fit the model
        model = Prophet()
        model.fit(subset_data)

        # Make future predictions (up to mid-2025)
        future = model.make_future_dataframe(periods=90)
        forecast = model.predict(future)

        # Append the current frame data
        frames.append({
            'ds': forecast['ds'],
            'yhat': forecast['yhat'],
            'yhat_lower': forecast['yhat_lower'],
            'yhat_upper': forecast['yhat_upper'],
            'actual_ds': subset_data['ds'],
            'actual_y': subset_data['y']
        })
    return frames

# Function to generate monitor data and cache it
@st.cache_data
def generate_monitor_data(frames):
    monitor_data = {
        'dates': [],
        'waves': [],
        'yhat': [],
        'yhat_upper': [],
        'yhat_lower': []
    }

    for wave_idx, frame in enumerate(frames):
        for date, yhat, yhat_upper, yhat_lower in zip(frame['ds'], frame['yhat'], frame['yhat_upper'], frame['yhat_lower']):
            monitor_data['dates'].append(date)
            monitor_data['waves'].append(wave_idx + 1)
            monitor_data['yhat'].append(yhat)
            monitor_data['yhat_upper'].append(yhat_upper)
            monitor_data['yhat_lower'].append(yhat_lower)
    return pd.DataFrame(monitor_data)

# Load data
data = load_data()

# Number of prediction waves
num_waves = 1000

# Generate forecast data for each wave
frames = generate_forecast_frames(data, num_waves)

# Create an interactive plot with a slider in Plotly
fig = go.Figure()

# Add initial forecast line and actual data points
forecast_wave = frames[0]
fig.add_trace(go.Scatter(x=forecast_wave['actual_ds'], y=forecast_wave['actual_y'],
                         mode='lines', name='Actual', line=dict(color='gray')))
fig.add_trace(go.Scatter(x=forecast_wave['ds'], y=forecast_wave['yhat'],
                         mode='lines', name='Forecast', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=forecast_wave['ds'], y=forecast_wave['yhat_upper'],
                         mode='lines', line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=forecast_wave['ds'], y=forecast_wave['yhat_lower'],
                         mode='lines', fill='tonexty', line=dict(width=0), showlegend=False))

# Define frames for each wave to update the plot dynamically
plotly_frames = []
for i, frame in enumerate(frames):
    plotly_frames.append(go.Frame(data=[
        go.Scatter(x=frame['actual_ds'], y=frame['actual_y']),
        go.Scatter(x=frame['ds'], y=frame['yhat']),
        go.Scatter(x=frame['ds'], y=frame['yhat_upper']),
        go.Scatter(x=frame['ds'], y=frame['yhat_lower'])
    ], name=str(i)))

# Add frames to the figure
fig.frames = plotly_frames

# Add slider and animation settings
fig.update_layout(
    updatemenus=[dict(type="buttons", showactive=False,
                      buttons=[dict(label="Play",
                                    method="animate",
                                    args=[None, {"frame": {"duration": 100, "redraw": True},
                                                 "fromcurrent": True}]),
                               dict(label="Pause",
                                    method="animate",
                                    args=[[None], {"frame": {"duration": 0, "redraw": False},
                                                   "mode": "immediate",
                                                   "transition": {"duration": 0}}])])],
    sliders=[dict(steps=[dict(method="animate",
                              args=[[str(k)], {"frame": {"duration": 100, "redraw": True},
                                               "mode": "immediate",
                                               "transition": {"duration": 0}}],
                              label=f"Wave {k+1}") for k in range(len(plotly_frames))],
                 transition={"duration": 0},
                 x=0.1, xanchor="left", y=0, yanchor="top",
                 active=0)]
)

fig.update_layout(title="Evolution of Forecast Over Time with Prophet",
                  xaxis_title="Date", yaxis_title="Zillow (Z) Stock Price",
                  width=900, height=600)

# Display the Plotly figure in Streamlit
st.plotly_chart(fig)

# Add second interactive plot for monitoring bounded region over waves
st.title("Monitor Bounded Region for Selected Date Over Waves")

# Generate monitor data
monitor_df = generate_monitor_data(frames)

# User selects a date to monitor
selected_date = st.date_input("Select a date to monitor:", value=pd.to_datetime('2016-01-01'))

# Convert available dates to a list
available_dates = monitor_df['dates'].dt.date.unique()
selected_date = st.slider("Select a date to monitor:", min_value=min(available_dates), max_value=max(available_dates), value=min(available_dates))


# Ensure selected_date is within the forecast dates
if selected_date not in monitor_df['dates'].dt.date.values:
    st.write("Selected date is not within the forecast dates.")
else:
    selected_date_df = monitor_df[monitor_df['dates'] == pd.to_datetime(selected_date)]

    # Create a new figure for monitoring the bounded region
    fig2 = go.Figure()

    # Add traces for yhat, yhat_upper, and yhat_lower over waves for the selected date
    fig2.add_trace(go.Scatter(x=selected_date_df['waves'], y=selected_date_df['yhat'], mode='lines+markers', name='Forecast (yhat)', line=dict(color='blue')))
    fig2.add_trace(go.Scatter(x=selected_date_df['waves'], y=selected_date_df['yhat_upper'], mode='lines+markers', name='Upper Bound (yhat_upper)', line=dict(color='red')))
    fig2.add_trace(go.Scatter(x=selected_date_df['waves'], y=selected_date_df['yhat_lower'], mode='lines+markers', name='Lower Bound (yhat_lower)', line=dict(color='green')))

    # Update layout for the second figure
    fig2.update_layout(title="Monitoring Bounded Region Over Waves for Selected Date",
                       xaxis_title="Wave", yaxis_title="Zillow (Z) Stock Price",
                       width=900, height=600)

    # Display the second Plotly figure in Streamlit
    st.plotly_chart(fig2)
