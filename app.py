import streamlit as st
import pandas as pd
import joblib
import shap
import json
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

@st.cache_data
def load_config():
    with open('policy_limits.json', 'r') as f:
        return json.load(f)

# 1. Setup & Loading
@st.cache_resource
def load_model_and_data():
    model = joblib.load('xgb_model.pkl')
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
    st.set_page_config(page_title="AI Underwriting Copilot", layout="wide")
    
    # Enforce Layer 1 Configuration Loading
    try:
        config = load_config()
    except FileNotFoundError:
        st.error("Configuration file 'policy_limits.json' not found. Please create it in the root directory.")
        st.stop()
        
    # Sidebar for API Key & Diagnostics
    st.sidebar.header("Settings")
    api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Enter API Key for RAG Compliance Engine")
    if api_key and HAS_GENAI:
        genai.configure(api_key=api_key)
        
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Model Diagnostics")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric(label="Accuracy", value="90.78%") 
    with col2:
        st.metric(label="ROC-AUC", value="0.9459")   
    st.sidebar.caption("Deployed Model: XGBoost 1.7.3")
        
    st.title("Human-in-the-Loop Underwriting Copilot")
    st.write("Enterprise architecture featuring Deterministic Guardrails, ML Risk Assessment, and RAG-driven Compliance.")
    
    model, expected_columns = load_model_and_data()

    # SECTION 1: UI Inputs (Strict Widgets to Prevent Prompt Injection)
    st.subheader("1. Applicant Information")
    col1, col2 = st.columns(2)
    with col1:
        person_age = st.number_input("Age", min_value=18, max_value=120, value=26)
        person_income = st.number_input("Annual Income ($)", min_value=1000, value=35000, step=1000)
        person_emp_length = st.number_input("Employment Length (Years)", min_value=0.0, max_value=70.0, value=3.0, step=0.5)
    with col2:
        loan_amnt = st.number_input("Loan Amount Requested ($)", min_value=100, value=25000, step=500)
        loan_intent = st.selectbox("Loan Purpose", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
        home_ownership = st.selectbox("Home Ownership Status", ["RENT", "OWN", "MORTGAGE", "OTHER"])

    # SECTION 2: Credit Bureau (CIBIL)
    st.subheader("2. Credit Bureau (CIBIL) & Assessment")
    col3, col4 = st.columns(2)
    with col3:
        loan_grade = st.selectbox("Assigned Loan Risk Grade", ["A", "B", "C", "D", "E", "F", "G"])
        loan_int_rate = st.slider("Approved Interest Rate (%)", min_value=0.0, max_value=30.0, value=13.5, step=0.1)
    with col4:
        cb_cred_hist = st.number_input("Credit Bureau History Length (Years)", min_value=0.0, max_value=50.0, value=3.0, step=1.0)
        prior_default = st.selectbox("Prior Default on Record?", ["N", "Y"])

    if st.button("Evaluate Credit Application", type="primary"):
        
        # --- LAYER 1: DETERMINISTIC RULES ENGINE (PYTHON + JSON) ---
        layer_1_violation = None
        dti_ratio = loan_amnt / person_income
        
        if dti_ratio >= config["max_dti"]:
            layer_1_violation = f"Debt-to-Income ratio of {dti_ratio:.1%} exceeds the strict {config['max_dti']*100}% maximum."
        elif person_age < config["min_age"] or person_age > config["max_age"]:
            layer_1_violation = f"Applicant age {person_age} is outside the valid range ({config['min_age']}-{config['max_age']})."
        elif person_emp_length > config["max_emp_length"]:
            layer_1_violation = f"Employment length of {person_emp_length} years exceeds maximum verifiable limit."
        elif prior_default == 'Y' and loan_grade in config["premium_grades"]:
            layer_1_violation = f"Assigned premium grade {loan_grade} contradicts prior default history."
            
        # Hard Stop: Bypass ML to save compute and enforce absolute legal boundaries
        if layer_1_violation:
            final_status = "REJECTED"
            pred_proba = 1.0 # Forced probability flag for rejection path
            st.error(f"🛑 **HARD STOP (Layer 1):** {layer_1_violation}")
            
        else:
            # --- LAYER 2: STATISTICAL RISK ENGINE (XGBOOST) ---
            input_dict = {col: 0.0 for col in expected_columns}
            input_dict['person_age'] = person_age
            input_dict['person_income'] = person_income
            input_dict['person_emp_length'] = person_emp_length
            input_dict['loan_amnt'] = loan_amnt
            input_dict['loan_int_rate'] = loan_int_rate
            if 'cb_person_cred_hist_length' in expected_columns:
                input_dict['cb_person_cred_hist_length'] = cb_cred_hist
            
            input_dict['loan_percent_income'] = dti_ratio
            
            if f'person_home_ownership_{home_ownership}' in expected_columns:
                input_dict[f'person_home_ownership_{home_ownership}'] = 1.0
            if f'loan_intent_{loan_intent}' in expected_columns:
                input_dict[f'loan_intent_{loan_intent}'] = 1.0
            if f'loan_grade_{loan_grade}' in expected_columns:
                input_dict[f'loan_grade_{loan_grade}'] = 1.0
            if f'cb_person_default_on_file_{prior_default}' in expected_columns:
                input_dict[f'cb_person_default_on_file_{prior_default}'] = 1.0

            input_data = pd.DataFrame([input_dict], columns=expected_columns)
            
            pred = model.predict(input_data)[0]
            pred_proba = model.predict_proba(input_data)[0][1] 
            
            if pred == 0:
                final_status = "APPROVED"
                st.success(f"✅ **ML ASSESSMENT (Layer 2): APPROVED** (Probability of Default: {pred_proba:.1%})")
            else:
                final_status = "REJECTED"
                st.warning(f"⚠️ **ML ASSESSMENT (Layer 2): REJECTED** (Probability of Default: {pred_proba:.1%})")
                
            # --- LAYER 3: EXPLAINABLE AI AUDIT (SHAP) ---
            with st.expander("📊 View Technical Underwriting Math (SHAP Audit)"):
                st.write("Waterfall chart mapping the specific feature weights that influenced the XGBoost probability score.")
                with st.spinner("Calculating SHAP weights..."):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer(input_data)
                    st_shap(shap.plots.waterfall(shap_values[0]))

        st.divider()
        st.subheader("Regulatory Communication")

        # --- LAYER 4: COMPLIANCE COMMUNICATOR (RAG / GEMINI) ---
        if HAS_GENAI and api_key:
            with st.spinner("🤖 Drafting RBI-Compliant explanation letter..."):
                try:
                    db = initialize_vector_db()
                    
                    # 1. DYNAMIC QUERY CONSTRUCTION
                    if final_status == "REJECTED" and layer_1_violation:
                        # Search ChromaDB specifically for the rule that was broken
                        search_query = f"Mandatory policy constraints, thresholds, and rules regarding: {layer_1_violation}"
                        decision_detail = f"REJECTED by automated policy check. Reason: {layer_1_violation}"
                        
                    elif final_status == "REJECTED" and not layer_1_violation:
                        # Search ChromaDB for overarching risk and default policies
                        search_query = "General risk-based underwriting criteria, default probability, and ML credit risk assessment"
                        decision_detail = f"REJECTED due to elevated probability of default ({pred_proba:.1%}) under composite risk assessment."
                        
                    else:
                        # Search ChromaDB for standard approval and eligibility criteria
                        search_query = "Standard approval criteria, eligible applicant profiles, and acceptable risk thresholds"
                        decision_detail = f"APPROVED. The application meets credit criteria with a low default probability of {pred_proba:.1%}."

                    # 2. SURGICAL VECTOR RETRIEVAL
                    docs = db.similarity_search(search_query, k=2) # Reduced to k=2 for tighter context
                    retrieved_context = "\n".join([doc.page_content for doc in docs])
                    
                    llm = genai.GenerativeModel("gemini-2.5-flash")
                    
                    prompt = (
                        "You are an automated regulatory communications agent for Global Merchants Bank.\n\n"
                        f"MANDATORY DECISION OUTCOME: {final_status}\n"
                        f"DETERMINED REASON: {decision_detail}\n\n"
                        f"BANK POLICY CONTEXT:\n{retrieved_context}\n\n"
                        "INSTRUCTIONS:\n"
                        "1. Write an official, polite notice to the applicant communicating ONLY the Mandatory Decision Outcome above.\n"
                        "2. Ground your explanation strictly on the provided Bank Policy Context.\n"
                        "3. You MUST cite the specific Section Number (e.g., 'Section 3.3') from the Bank Policy Context that governs this decision.\n"
                        "4. Do NOT dispute, recalculate, or alter the determined decision or metrics.\n"
                        "5. Keep the response to exactly 3 or 4 professional sentences."
                    )
                    
                    response = llm.generate_content(prompt)
                    st.info(f"**Regulatory Decision Notice:**\n\n{response.text}")
                    
                except Exception as e:
                    st.error(f"Error generating communication: {e}")
        else:
            st.info("💡 Enter your Gemini API Key in the sidebar to automatically draft the regulatory compliance explanation.")

if __name__ == "__main__":
    main()