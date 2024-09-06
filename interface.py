import streamlit as st
import pandas as pd
import joblib
def detect_fraudulent_transactions(new_data: pd.DataFrame, model) -> pd.DataFrame:
    # Predict anomalies in the new dataset
    predictions = model.predict(new_data)
    
    # Assuming that the model returns 0 for normal transactions and 1 for anomalies
    fraudulent_transactions = new_data[predictions == 1]
    
    return fraudulent_transactions

def main():
    st.title("Credit Card Fraud Detection")

    uploaded_file = st.file_uploader("Upload a CSV file with credit card transactions", type="csv")
    model_file = st.file_uploader("Upload the trained anomaly detection model", type="pkl")

    if uploaded_file is not None and model_file is not None:
        new_data = pd.read_csv(uploaded_file)
        model = joblib.load(model_file)
        
        st.write("New Transactions Data:")
        st.write(new_data)

        fraudulent_transactions = detect_fraudulent_transactions(new_data, model)
        
        st.write("Fraudulent Transactions:")
        st.write(fraudulent_transactions)

if __name__ == "__main__":
    main()
