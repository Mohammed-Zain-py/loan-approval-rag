import streamlit as st
import pandas as pd
import joblib
import shap
from streamlit_shap import st_shap
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# 1. Setup & Loading
# @st.cache_resource ensures we only load the model once to keep the app fast
@st.cache_resource
def load_model_and_data():
    # Load the trained XGBoost model
    model = joblib.load('xgb_model.pkl')
    
    # Dynamically get the exact column names the model expects
    expected_columns = model.get_booster().feature_names
        
    return model, expected_columns

@st.cache_resource
def initialize_vector_db():
    loader = TextLoader("bank_policy.txt")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

def main():
    st.set_page_config(page_title="Loan Approval Predictor", layout="wide")
    
    # Sidebar for API Key
    st.sidebar.header("Settings")
    api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Enter your Gemini API Key to enable AI explanations.")
    if api_key and HAS_GENAI:
        genai.configure(api_key=api_key)
        
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Model Diagnostics")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric(label="Accuracy", value="93.6%") 
    with col2:
        st.metric(label="ROC-AUC", value="0.941")   

    st.sidebar.caption("Deployed Model: XGBoost 1.7.3")
        
    st.title("Loan Approval Predictor")
    st.write("Enter the applicant's details below to predict their loan approval status and see the AI's explanation.")
    
    # Load model and expected column structure
    model, _ = load_model_and_data()

    # 2. Clean UI: Inputs for primary features
    st.subheader("Applicant Information")
    
    col1, col2 = st.columns(2)
    with col1:
        person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
        person_income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
        person_emp_length = st.number_input("Employment Length (Years)", min_value=0.0, max_value=60.0, value=5.0)
    with col2:
        loan_amnt = st.number_input("Loan Amount ($)", min_value=100, value=10000, step=500)
        loan_int_rate = st.slider("Loan Interest Rate (%)", min_value=0.0, max_value=30.0, value=10.0, step=0.1)
        loan_percent_income = st.slider("Loan Percent of Income", min_value=0.0, max_value=1.0, value=0.2, step=0.01)

    if st.button("Predict Loan Status", type="primary"):
        # 1. Get exact expected columns
        expected_columns = model.get_booster().feature_names
        
        # 2. Create the dictionary with scalar values (no lists inside the dict)
        input_dict = {col: 0.0 for col in expected_columns}
        
        # 3. Update the 6 specific values
        input_dict['person_age'] = person_age
        input_dict['person_income'] = person_income
        input_dict['person_emp_length'] = person_emp_length
        input_dict['loan_amnt'] = loan_amnt
        input_dict['loan_int_rate'] = loan_int_rate
        input_dict['loan_percent_income'] = loan_percent_income
        
        # 4. Create the DataFrame by wrapping the dict in a list AND enforcing the columns argument
        # And slicing it strictly just in case pandas still messes up the order
        input_data = pd.DataFrame([input_dict], columns=expected_columns)[expected_columns]
        
        # 5. Prediction
        # 0 = Approved (Non-Default), 1 = Rejected (Default)
        pred = model.predict(input_data)[0]
        pred_proba = model.predict_proba(input_data)[0][1] # Probability of default (class 1)
        
        st.divider()
        st.subheader("Prediction Result")
        
        # Display colored success/error message
        if pred == 0:
            status_text = "APPROVED"
            st.success(f"✅ **Loan APPROVED!** (Probability of Default: {pred_proba:.1%})")
        else:
            status_text = "REJECTED"
            st.error(f"❌ **Loan REJECTED!** (Probability of Default: {pred_proba:.1%})")
            
        # 6. GenAI Explanation
        if HAS_GENAI and api_key:
            with st.spinner("🤖 Generating policy-backed explanation..."):
                try:
                    # 1. Search the Vector DB
                    db = initialize_vector_db()
                    search_query = f"Rules for age {person_age}, income {person_income}, loan amount {loan_amnt}, employment length {person_emp_length}"
                    docs = db.similarity_search(search_query, k=2)
                    retrieved_context = "\n".join([doc.page_content for doc in docs])
                    
                    # 2. Call Gemini
                    llm = genai.GenerativeModel("gemini-2.5-flash")
                    prompt = f"Act as an expert bank underwriter. Explain why a loan application for a {person_age}-year-old earning ${person_income}/year, requesting ${loan_amnt} at {loan_int_rate}% with {person_emp_length} years of employment, resulted in a status of: {status_text}. YOU MUST CITE the specific Section Number from the provided Context. Keep it to strictly 3 short sentences. CONTEXT: {retrieved_context}"
                    
                    response = llm.generate_content(prompt)
                    st.markdown("**⚡ Policy Explanation:**")
                    st.info(response.text)
                except Exception as e:
                    st.error(f"Error generating explanation: {e}")
        elif not api_key and HAS_GENAI:
            st.info("💡 Enter your Gemini API Key in the sidebar to generate a plain-English explanation.")
        elif not HAS_GENAI:
            st.warning("google-generativeai module is not installed. Run `pip install google-generativeai` to enable GenAI explanations.")
        
        # 7. SHAP Integration
        with st.expander("📊 View Technical Underwriting Math (For Auditors)"):
            st.write("This waterfall chart explains why the model made its decision. Features pushing the prediction higher (towards Reject) are in pink, and features pushing it lower (towards Approve) are in blue.")
            
            with st.spinner("Calculating SHAP values..."):
                # Initialize TreeExplainer and calculate SHAP values on the 1-row DataFrame
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(input_data)
                
                # Render the SHAP waterfall plot directly in Streamlit
                st_shap(shap.plots.waterfall(shap_values[0]))

if __name__ == "__main__":
    main()
