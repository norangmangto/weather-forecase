import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.predict import WeatherPredictor
from src.monitor import WeatherMonitor
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Weather Prediction Dashboard",
    page_icon="🌤️",
    layout="wide"
)

# Title and Description
st.title("🌤️ Weather Prediction Dashboard")
st.markdown("""
This dashboard provides a **7-day weather forecast** for Düsseldorf (Station 01078) using
advanced Machine Learning models (XGBoost and Random Forest).
""")

# Initialize Predictor
@st.cache_resource
def get_predictor():
    predictor = WeatherPredictor()
    predictor.load_latest_models()
    return predictor

predictor = get_predictor()

# Sidebar for Controls
st.sidebar.header("Forecast Settings")
target_date = st.sidebar.date_input(
    "Prediction Start Date",
    datetime.now().date(),
    help="Select the date for which you want to start the 7-day forecast."
)

st.sidebar.info(f"Models loaded: {len(predictor.models)}")

# Fetch Forecast
def fetch_forecast(date_str):
    try:
        xgb_df = predictor.predict_7_days('xgboost', start_date=date_str)
        rf_df = predictor.predict_7_days('random_forest', start_date=date_str)
        return xgb_df, rf_df
    except Exception as e:
        st.error(f"Error generating forecast: {e}")
        return None, None

date_str = target_date.strftime('%Y-%m-%d')
with st.spinner("Generating 7-day forecast..."):
    xgb_df, rf_df = fetch_forecast(date_str)

if xgb_df is not None and not xgb_df.empty:
    # ---------------------------------------------------------
    # UI Layout: Top Metrics
    # ---------------------------------------------------------
    st.subheader(f"Next 7 Days (Starting {xgb_df['date'].iloc[0]})")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Max Temp", f"{xgb_df['temp_max'].mean():.1f}°C")
    with col2:
        st.metric("Avg Min Temp", f"{xgb_df['temp_min'].mean():.1f}°C")
    with col3:
        st.metric("Total Precip.", f"{xgb_df['precipitation_mm'].sum():.1f} mm")
    with col4:
        st.metric("Avg Humidity", f"{xgb_df['humidity_mean'].mean():.1f}%")

    # ---------------------------------------------------------
    # Charts
    # ---------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(["🌡️ Temperature", "🌧️ Precipitation", "💨 Wind & Other", "📊 MLOps: Performance"])

    with tab1:
        st.markdown("### Temperature Forecast")
        # Combine XGB and RF for comparison
        comp_temp = pd.DataFrame({
            'Date': pd.to_datetime(xgb_df['date']),
            'XGB Max': xgb_df['temp_max'],
            'RF Max': rf_df['temp_max'],
            'XGB Min': xgb_df['temp_min'],
            'RF Min': rf_df['temp_min']
        })

        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(x=comp_temp['Date'], y=comp_temp['XGB Max'], name='XGB Max Temp', line=dict(color='firebrick', width=4)))
        fig_temp.add_trace(go.Scatter(x=comp_temp['Date'], y=comp_temp['RF Max'], name='RF Max Temp', line=dict(color='firebrick', dash='dash')))
        fig_temp.add_trace(go.Scatter(x=comp_temp['Date'], y=comp_temp['XGB Min'], name='XGB Min Temp', line=dict(color='royalblue', width=4)))
        fig_temp.add_trace(go.Scatter(x=comp_temp['Date'], y=comp_temp['RF Min'], name='RF Min Temp', line=dict(color='royalblue', dash='dash')))

        fig_temp.update_layout(title='Max/Min Temperature Comparison', xaxis_title='Date', yaxis_title='Temperature (°C)')
        st.plotly_chart(fig_temp, use_container_width=True)

    with tab2:
        st.markdown("### Precipitation & Rain Probability")

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_precip = px.bar(xgb_df, x='date', y='precipitation_mm', title="Precipitation (mm) - XGBoost")
            st.plotly_chart(fig_precip, use_container_width=True)
        with col_c2:
            fig_prob = px.line(xgb_df, x='date', y='rain_probability', title="Rain Probability (%) - XGBoost")
            fig_prob.update_yaxes(range=[0, 1])
            st.plotly_chart(fig_prob, use_container_width=True)

    with tab3:
        st.markdown("### Wind, Pressure & Visibility")

        col_c3, col_c4 = st.columns(2)
        with col_c3:
            fig_wind = go.Figure()
            fig_wind.add_trace(go.Scatter(x=xgb_df['date'], y=xgb_df['wind_speed_mean'], name='Mean Wind'))
            fig_wind.add_trace(go.Scatter(x=xgb_df['date'], y=xgb_df['wind_gust_max'], name='Max Gust'))
            fig_wind.update_layout(title="Wind Speed & Gusts (m/s)")
            st.plotly_chart(fig_wind, use_container_width=True)
        with col_c4:
            fig_pressure = px.line(xgb_df, x='date', y='pressure_hpa', title="Air Pressure (hPa)")
            st.plotly_chart(fig_pressure, use_container_width=True)

    with tab4:
        st.markdown("### Model Performance Analysis (MLOps)")
        st.markdown("""
        This section compares historical predictions with actual observed weather data
        to track accuracy and detect model drift.
        """)

        monitor = WeatherMonitor()
        try:
            metrics_df = monitor.get_accuracy_metrics()
        except Exception as e:
            st.error(f"Error fetching accuracy metrics: {e}")
            metrics_df = pd.DataFrame()

        if not metrics_df.empty:
            # Aggregate metrics for summary
            col_m1, col_m2 = st.columns(2)

            with col_m1:
                st.markdown("#### Mean Absolute Error (MAE) by Target")
                fig_mae = px.bar(
                    metrics_df,
                    x='target_name',
                    y='mae',
                    color='model_type',
                    barmode='group',
                    title="Model Accuracy (Lower is better)"
                )
                st.plotly_chart(fig_mae, use_container_width=True)

            with col_m2:
                st.markdown("#### RMSE by Target")
                fig_rmse = px.bar(
                    metrics_df,
                    x='target_name',
                    y='rmse',
                    color='model_type',
                    barmode='group',
                    title="Model Error Variance"
                )
                st.plotly_chart(fig_rmse, use_container_width=True)

            st.markdown("#### Performance Summary Table")
            st.dataframe(metrics_df.sort_values(['target_name', 'mae']), use_container_width=True)
        else:
            st.info("No historical accuracy data available yet. Metrics will appear once predictions are compared with new actual data.")

    # ---------------------------------------------------------
    # Raw Data
    # ---------------------------------------------------------
    with st.expander("Show Raw Forecast Data"):
        st.write("XGBoost Predictions")
        st.dataframe(xgb_df)
        st.write("Random Forest Predictions")
        st.dataframe(rf_df)

else:
    st.warning("No models or data available. Run the pipeline first to train models.")

# Footer
st.divider()
st.caption("Powered by XGBoost, Random Forest, DuckDB and dbt.")
