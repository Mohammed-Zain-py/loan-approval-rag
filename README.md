# Loan Approval RAG & Explainable AI

A loan approval system that combines **Machine Learning, RAG, and SHAP** to make loan decisions more explainable.

## What this project does

Loan approval is a sensitive process where simply predicting whether an application should be approved or rejected is not always enough. When a loan is rejected, there should also be a clear reason behind the decision.

This project combines three components to address this:

- **XGBoost** — predicts whether a loan should be approved or rejected.
- **SHAP** — shows which features influenced the model's decision, mainly for bank employees and auditors.
- **RAG** — retrieves relevant sections from the bank's underwriting policy and uses them to generate a policy-based explanation for the customer.

### Decision flow

1. Applicant enters their loan information.
2. XGBoost predicts the loan status and probability of default.
3. SHAP explains the factors that influenced the prediction.
4. The RAG pipeline retrieves relevant sections from `bank_policy.txt`.
5. Gemini generates a short explanation based on the retrieved policy.

## Tech Stack

- Python
- Streamlit
- XGBoost
- SHAP
- Pandas
- LangChain
- ChromaDB
- Hugging Face Embeddings
- Google Gemini API
- Joblib

## Project Structure

```text
loan-approval-rag/
├── app.py
├── clean_data.py
├── train_xgboost.py
├── bank_policy.txt
├── credit_risk_dataset.csv
├── loan_processed.csv
├── xgb_model.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/Mohammed-Zain-py/loan-approval-rag.git
cd loan-approval-rag
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## Gemini API Key

The Gemini API key is **not stored in the source code**.

The application accepts the API key through the Streamlit sidebar at runtime.

## Model

The project uses an XGBoost classifier trained on the processed credit-risk dataset.

The trained model is stored in:

```text
xgb_model.pkl
```

## Explainability

### Employee / Auditor View

SHAP provides a waterfall visualization showing how individual features influenced the model's prediction toward approval or rejection.

### Customer View

The RAG pipeline retrieves relevant sections from `bank_policy.txt` and provides them as context to Gemini for generating the explanation.

## Important Note

This is an educational/portfolio project demonstrating explainable loan decisioning and policy-grounded RAG.

The included bank and underwriting policy are fictional and are not intended to represent a real financial institution or production banking system.
