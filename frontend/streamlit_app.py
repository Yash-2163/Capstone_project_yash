# This is the code used when we spin up instances locally

import streamlit as st
import pandas as pd
import requests
import io

# --- Streamlit Page Configuration ---
# Sets up the page title, icon, and layout for the app
st.set_page_config(
    page_title="Lead Conversion Predictor",
    page_icon="🚀",
    layout="centered"
)

# --- App Title and Description ---
# Displays the main title and some context for the user
st.title("🚀 Lead Conversion Predictor (POC)")
st.write(
    "Upload a CSV file with lead data to predict which leads are most likely to convert. "
    "This is a proof-of-concept using a trained machine learning model."
)

# --- File Uploader ---
# Allows user to upload a CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # --- Data Preview ---
    try:
        # Read the uploaded CSV file and show a preview
        df_preview = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(df_preview.head())

        # Reset the file pointer after reading for preview
        uploaded_file.seek(0)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()

    # --- Prediction Button ---
    if st.button("Get Predictions"):
        with st.spinner("Sending data to the model... Please wait."):
            # Define backend API endpoint
            # 'localhost' is used here for local testing.
            # In a Docker network, replace with service name like 'backend'.
            backend_url = "http://localhost:5001/predict"
            
            try:
                # Construct POST request with uploaded file contents
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                
                # Send file to the Flask backend server
                response = requests.post(backend_url, files=files, timeout=30)

                # --- Handle the Response ---
                if response.status_code == 200:
                    st.success("✅ Predictions received successfully!")
                    
                    # Convert JSON response to DataFrame and display
                    results_data = response.json()
                    results_df = pd.DataFrame(results_data["predictions"])
                    
                    st.write("Prediction Results:")
                    st.dataframe(results_df)

                else:
                    # Display status code and optionally the JSON/error message
                    st.error(f"Error from backend: {response.status_code}")
                    try:
                        st.json(response.json())
                    except requests.exceptions.JSONDecodeError:
                        st.text(response.text)

            except requests.exceptions.ConnectionError:
                st.error(
                    "Connection Error: Could not connect to the backend service. "
                    "Please ensure the backend container is running and accessible."
                )
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
else:
    # Prompt user to upload a file if none has been uploaded yet
    st.info("Please upload a CSV file to begin.")
