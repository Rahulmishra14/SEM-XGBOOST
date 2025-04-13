import json
import os
import joblib
import nbformat
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import google.generativeai as genai

# Load API key and model features
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("models/gemini-2.0-flash")
chat = model.start_chat(history=[])

with open('./Models/model_features.json') as f:
    trained_features = json.load(f)

def get_gemini_response(question):
    try:
        response = chat.send_message(question, stream=True)
        return response
    except Exception as e:
        return [type('ResponseChunk', (object,), {'text': f"Error: {str(e)}"})()]

def get_energy_management_advice(electricity_usage, solar_generation, other_energy_sources):
    question = (
        f"How can I manage my energy consumption with the following data? "
        f"Electricity Usage: {electricity_usage} kWh, "
        f"Solar Generation: {solar_generation} kWh, "
        f"Other Energy Sources: {other_energy_sources} kWh"
    )
    return get_gemini_response(question)

st.set_page_config(page_title="Home Energy Manager", page_icon="💡", layout="wide", initial_sidebar_state="expanded")
st.header("💡 Home Energy Manager")

tab1, tab2, tab3 = st.tabs(["Chat Assistant", "Energy Management", "AI/ML Data Study"])

with tab1:
    st.subheader("🤖 Ask a Question")
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    input = st.text_input("Type your question here:", key="input")
    submit = st.button("Ask")

    if submit and input:
        response = get_gemini_response(input)
        st.session_state['chat_history'].append(("You", input))
        st.subheader("Response:")
        for chunk in response:
            st.write(chunk.text)
            st.session_state['chat_history'].append(("Bot", chunk.text))

    st.subheader("Chat History:")
    for role, text in st.session_state['chat_history']:
        st.write(f"{role}: {text}")

with tab2:
    st.subheader("🌞 Manage Your Home Energy Consumption")
    electricity_usage = st.number_input("Electricity Usage (kWh):", min_value=0.0, step=0.1)
    solar_generation = st.number_input("Solar Generation (kWh):", min_value=0.0, step=0.1)
    other_energy_sources = st.number_input("Other Energy Sources (kWh):", min_value=0.0, step=0.1)
    submit_energy_data = st.button("Get Advice")

    if submit_energy_data:
        st.success("Energy data submitted successfully!")
        advice_response = get_energy_management_advice(electricity_usage, solar_generation, other_energy_sources)
        st.subheader("🔍 Energy Management Advice")
        for chunk in advice_response:
            st.write(chunk.text)

with tab3:
    st.subheader("📊 AI/ML Data Study")
    st.title("🔌 Appliance Energy Prediction")

    try:
        model = joblib.load("./Models/tuned_appliance_energy_model.pkl")
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'tuned_appliance_energy_model.pkl' exists.")
        st.stop()

    st.header("Enter Required Sensor Readings")
    col1, col2, col3 = st.columns(3)

    with col1:
        lights = st.slider("Lights Energy Usage (Wh)", 0.0, 100.0, 10.0)
        RH_1 = st.number_input("Kitchen Humidity (RH_1)", value=40.0)
        T2 = st.number_input("Living Room Temp (T2)", value=21.0)
        T4 = st.number_input("Office Temp (T4)", value=21.0)
        RH_4 = st.number_input("Office Humidity (RH_4)", value=40.0)
        T1 = st.number_input("Kitchen Temp (T1)", value=20.0)
        T3 = st.number_input("Bathroom Temp (T3)", value=21.0)
        RH_2 = st.number_input("Living Room Humidity (RH_2)", value=40.0)

    with col2:
        RH_5 = st.number_input("Bathroom Humidity (RH_5)", value=45.0)
        T7 = st.number_input("Ironing Room Temp (T7)", value=20.0)
        T8 = st.number_input("Teen Room Temp (T8)", value=21.0)
        RH_8 = st.number_input("Teen Room Humidity (RH_8)", value=50.0)
        RH_9 = st.number_input("Parents Room Humidity (RH_9)", value=42.0)
        T5 = st.number_input("Toilet Temp (T5)", value=21.0)
        T6 = st.number_input("Laundry Room Temp (T6)", value=20.0)
        RH_3 = st.number_input("Bathroom Humidity (RH_3)", value=45.0)

    with col3:
        T_out = st.number_input("Outside Temp (T_out)", value=10.0)
        RH_out = st.number_input("Outside Humidity (RH_out)", value=50.0)
        Windspeed = st.number_input("Wind Speed", value=4.0)
        Visibility = st.number_input("Visibility", value=40.0)
        Tdewpoint = st.number_input("Dew Point Temp", value=5.0)
        T9 = st.number_input("Parents Room Temp (T9)", value=20.0)
        RH_6 = st.number_input("Laundry Room Humidity (RH_6)", value=40.0)
        RH_7 = st.number_input("Hall Humidity (RH_7)", value=39.0)
        Press_mm_hg = st.number_input("Pressure (mm Hg)", value=755.0)
        rv1 = st.number_input("Random Variable 1 (rv1)", value=0.0)
        rv2 = st.number_input("Random Variable 2 (rv2)", value=0.0)

    input_df = pd.DataFrame([{
        'lights': lights, 'RH_1': RH_1, 'T2': T2, 'T4': T4, 'RH_4': RH_4,
        'RH_5': RH_5, 'T7': T7, 'T8': T8, 'RH_8': RH_8, 'RH_9': RH_9,
        'T_out': T_out, 'RH_out': RH_out, 'Windspeed': Windspeed,
        'Visibility': Visibility, 'Tdewpoint': Tdewpoint,
        'T1': T1, 'T3': T3, 'T5': T5, 'T6': T6, 'T9': T9,
        'RH_2': RH_2, 'RH_3': RH_3, 'RH_6': RH_6, 'RH_7': RH_7,
        'Press_mm_hg': Press_mm_hg, 'rv1': rv1, 'rv2': rv2
    }])

    if st.button("Predict Energy Usage"):
        try:
            input_df = input_df[trained_features]
            prediction = model.predict(input_df)[0]
            st.success(f"⚡ Predicted Appliance Energy Consumption: {prediction:.2f} Wh")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    if st.button("Compare with Real Data"):
        try:
            raw_data = pd.read_csv("./Data Set/energydata_complete.csv")
            raw_data.dropna(subset=trained_features + ['Appliances'], inplace=True)
            scaler = StandardScaler()
            scaled_raw = scaler.fit_transform(raw_data[trained_features])
            scaled_input = scaler.transform(input_df[trained_features])
            raw_data["distance"] = euclidean_distances(scaled_raw, scaled_input).flatten()
            top_similar = raw_data.sort_values(by="distance").head(5)
            st.write("### 🔍 Top 5 Most Similar Data Points")
            st.dataframe(top_similar[trained_features + ["Appliances"]])

            X_similar = top_similar[trained_features]
            y_true = top_similar["Appliances"]
            y_pred = model.predict(X_similar)
            residuals = y_true.values - y_pred

            result_df = pd.DataFrame({
                "Actual (Wh)": y_true.values,
                "Predicted (Wh)": y_pred,
                "Residual (Wh)": residuals
            })
            st.write("### 📈 Prediction vs Actual on Similar Inputs")
            st.dataframe(result_df)

            st.subheader("📈 Actual vs Predicted Plot")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(data=result_df[['Actual (Wh)', 'Predicted (Wh)']])
            ax.set_title("Actual vs Predicted Energy Usage")
            st.pyplot(fig)

            st.markdown("### 📌 Prediction Summary")
            st.markdown(f"""
                - **Mean Actual**: `{y_true.mean():.2f}` Wh  
                - **Mean Predicted**: `{y_pred.mean():.2f}` Wh  
                - **Mean Residual**: `{residuals.mean():.2f}` Wh  
                - **MAE**: `{abs(residuals).mean():.2f}` Wh
            """)

            st.subheader("🔍 Feature Importance (XGBoost Model)")
            importances = model.named_steps['model'].feature_importances_
            importance_df = pd.DataFrame({
                'Feature': trained_features,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=importance_df.head(15), y='Feature', x='Importance', palette='viridis')
            ax.set_title('Top 15 Most Important Features')
            st.pyplot(fig)

            with st.expander("📋 See All Feature Importances"):
                st.dataframe(importance_df)

        except Exception as e:
            st.error(f"Error comparing with real data: {e}")
