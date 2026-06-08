import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from predict import PredictionEngine  # Your existing module
from data_loader import get_data
import sys
sys.path.append('.')  # Ensure modules are found

st.set_page_config(page_title="CVX Political Trading Bot", layout="wide")
st.title("🚀 CVX Political Trading Bot Dashboard")
st.markdown("**Real-time predictions with GDELT, Political Events (Anti-Weaponization Fund), Technical Patterns & Ensemble ML**")

# Sidebar
st.sidebar.header("Controls")
ticker = st.sidebar.text_input("Ticker", "CVX")
run_button = st.sidebar.button("🔄 Generate Latest Prediction", type="primary")

if run_button:
    with st.spinner("Running full prediction engine (data + GDELT + models)..."):
        try:
            engine = PredictionEngine()
            results = engine.predict_next_day()
            
            # Current Status
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Price", f"${results['current_price']:.2f}")
            col2.metric("Ensemble Prediction", f"${results['ensemble_prediction']:.2f}")
            col3.metric("Signal", results['signal']['signal'], delta=results['signal']['price_change_pct'])
            col4.metric("Confidence", f"{results['signal']['confidence']:.1%}")
            
            # Charts
            st.subheader("Price Chart + Patterns")
            df = get_data()
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="CVX"))
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed Results
            st.subheader("🧠 Model Predictions")
            model_col1, model_col2 = st.columns(2)
            model_col1.metric("LSTM", f"${results['lstm_prediction']:.2f}")
            model_col2.metric("XGBoost", f"${results['xgb_prediction']:.2f}")
            
            # Political & GDELT Insights
            st.subheader("🌍 Geopolitical & Political Analysis")
            pol = results['political_impact']
            st.info(f"**Risk Level**: {pol['risk_level']} | Anti-Weaponization Fund / Trump-related impact active")
            st.write(f"Political Risk Score: {results.get('political_impact', {}).get('score', 'N/A')}")
            
            # Full Results Table
            st.subheader("📊 Full Recommendation")
            signal = results['signal']
            data = {
                "Metric": ["Signal", "Target Price", "Stop Loss", "Take Profit", "Position Size", "Political Adjustment"],
                "Value": [signal['signal'], f"${results['ensemble_prediction']:.2f}", 
                         f"${signal['stop_loss']:.2f}", f"${signal['take_profit']:.2f}", 
                         f"{signal['position_size']:.1%}", pol['risk_level']]
            }
            st.table(pd.DataFrame(data))
            
        except Exception as e:
            st.error(f"Error running prediction: {e}")

# Footer
st.caption("Educational tool only • Not financial advice • Past performance ≠ future results")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
