import pandas as pd
from prophet import Prophet
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

# Streamlit app title
st.title("Evolution of Forecast Over Time with Prophet")

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

# Generate forecast data for each wave
frames = []
for i in range(increment, len(data), increment):
    # Subset data up to the current point in the wave
    subset_data = data.iloc[:i]

    # Initialize and fit the model
    model = Prophet()
    model.fit(subset_data)

    # Make future predictions (up to mid-2025)
    future = model.make_future_dataframe(periods=365)
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