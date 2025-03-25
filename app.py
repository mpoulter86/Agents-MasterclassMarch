import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Streamlit Page Config
st.set_page_config(page_title="📈 Revenue Forecasting Agent", page_icon="📊", layout="wide")
st.title("🤖 AI Revenue Forecasting App using Prophet")

# API Key check
if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# File Upload
uploaded_file = st.file_uploader("📤 Upload your Excel file with 'Date' and 'Revenue' columns", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)

        # Data Validation
        if not {'Date', 'Revenue'}.issubset(df.columns):
            st.error("❌ File must contain 'Date' and 'Revenue' columns.")
            st.stop()

        df = df[['Date', 'Revenue']].dropna()
        df.columns = ['ds', 'y']  # Prophet requires these column names
        df['ds'] = pd.to_datetime(df['ds'])

        st.success("✅ File uploaded and processed successfully!")
        st.write("📊 Preview of the uploaded data:")
        st.dataframe(df.head())

        # Prophet Forecasting
        st.subheader("🔮 Forecasting Revenue with Prophet")
        periods = st.slider("Select forecast horizon (days):", 30, 365, 90)

        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        # Plot Forecast
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

        # Display Forecast Table
        st.subheader("📅 Forecast Table")
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods))

        # AI Commentary
        st.subheader("🧠 AI Analysis of Forecast")
        client = Groq(api_key=GROQ_API_KEY)

        json_data = forecast[['ds', 'yhat']].tail(periods).to_json(orient="records", date_format="iso")

        prompt = f"""
        You are a financial planning and analysis (FP&A) expert. You have been given a time series forecast of revenue.
        Please analyze it and provide:
        - Key trends and inflection points.
        - Potential business implications.
        - A short, clear summary a CFO would care about.
        Forecast data (in JSON): {json_data}
        """

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a financial analyst skilled in forecasting interpretation."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
        )

        commentary = response.choices[0].message.content
        st.markdown("#### 💬 AI Commentary:")
        st.write(commentary)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
