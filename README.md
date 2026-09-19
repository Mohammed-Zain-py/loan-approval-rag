# Loan Approval RAG & Explainable AI

A hybrid loan approval MVP that combines traditional **Machine Learning (XGBoost)** with **Explainable AI (SHAP)** and **Agentic RAG (Gemini + ChromaDB)** to make credit decisions explainable and policy-compliant.

## The Architecture & Why I Built It

Most beginner AI loan projects either rely entirely on basic ML predictions (which are black boxes and cannot cite bank policy) or rely purely on LLM prompts (which hallucinate financial math and are too slow/expensive for strict rules).

I built this project to demonstrate a realistic **Human-in-the-Loop (HITL)** system that uses the right tool for the right job. It processes applications through a 4-step workflow:

1. **Deterministic Rules Engine (Python + JSON):** The system first checks strict parameters (like a 50% Debt-to-Income cap) using a JSON config file. This ensures zero math hallucinations and saves compute by instantly rejecting policy violations.
2. **Statistical Risk Engine (XGBoost):** If the application passes the hard rules, an XGBoost model evaluates the historical probability of default based on 32,000+ past records.
3. **Explainable AI Audit (SHAP):** Generates a waterfall visualization to show the human underwriter exactly which features (like income, loan grade, or homeownership) influenced the ML prediction, exposing any historical data biases.
4. **Compliance Communicator (RAG + Gemini):** Instead of letting the LLM make the decision, the deterministic outcome is passed to Gemini. Gemini searches the local ChromaDB vector store (`bank_policy.txt`) and drafts a professional, RBI-compliant explanation letter citing specific policy sections.

## Tech Stack

- **Python & Streamlit** (Frontend Dashboard)
- **XGBoost, Pandas, Scikit-Learn** (Machine Learning Pipeline)
- **SHAP** (Model Explainability)
- **LangChain & ChromaDB** (Vector Database & RAG)
- **HuggingFace MiniLM** (Embeddings)
- **Google Gemini 2.5 Flash API** (LLM Communication Engine)

## Project Structure

```text
loan-approval-rag/
├── app.py                   # Main Streamlit application and 4-layer logic
├── policy_limits.json       # Configurable rules engine (DTI limits, age, etc.)
├── bank_policy.txt          # Mock institutional underwriting policy document
├── xgb_model.pkl            # Trained XGBoost classifier
├── clean_data.py            # Data preprocessing script
├── train_xgboost.py         # Model training and evaluation script
├── credit_risk_dataset.csv  # Raw dataset
├── loan_processed.csv       # Cleaned feature matrix
├── requirements.txt         # Project dependencies
├── .gitignore               # Git ignore rules
└── README.md                # Documentation
```

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/Mohammed-Zain-py/loan-approval-rag.git
cd loan-approval-rag
```

### 2. Create and activate a virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your default web browser.

## Security & Privacy Features

### Prompt Injection Prevention

The Streamlit UI strictly uses dropdowns, number inputs, and sliders. By eliminating free-text inputs, the system is physically protected from prompt injection attacks.

### Data Minimization in RAG

Raw applicant figures (like exact salary and requested amounts) are evaluated locally in Python. Only the final decision status and specific policy violation are sent to the Gemini API, minimizing the exposure of financial data.

## Note on Gemini API Key & Model Versions

The Gemini API key is **not stored in the source code**. You will need to enter your API key in the Streamlit sidebar at runtime to activate the RAG compliance letter generation. 

**Model Versioning:** The `app.py` script currently points to a specific model version (e.g., `gemini-2.5-flash`). Because Google regularly updates its models and deprecates older endpoints, you might receive a "Model discontinued" error with newer API keys. If this happens, simply open `app.py` and update the `genai.GenerativeModel("gemini-2.5-flash")` string to the latest supported version (such as `gemini-3.5-flash` or `gemini-3.8-flash`).

If no API key is provided, the ML and SHAP layers will still function normally.

## Disclaimer

This is an educational portfolio project demonstrating hybrid AI architecture. The included bank policies and rules are fictional and are not intended to represent a real financial institution.