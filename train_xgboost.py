import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

def main():
    print("Starting XGBoost training process...")
    
    # 1. Load Data
    try:
        # Load the cleaned dataset
        df = pd.read_csv('loan_processed.csv')
        print(f"Successfully loaded processed data with shape: {df.shape}")
    except FileNotFoundError:
        print("Error: 'loan_processed.csv' not found. Please make sure to run clean_data.py first.")
        return

    # 2. Separate features (X) and target (y)
    # The target variable we want to predict is 'loan_status'
    if 'loan_status' not in df.columns:
        print("Error: Target column 'loan_status' not found in the dataset.")
        return
        
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    print(f"Features (X) and Target (y) separated. Predicting: 'loan_status'.")
    
    # 3. Split the data into training (80%) and testing (20%) sets
    # Using random_state=42 ensures reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split completed: {X_train.shape[0]} training samples and {X_test.shape[0]} testing samples.")

    # 4. Initialize and Train the XGBoost Classifier
    # We set standard parameters to prevent overfitting
    print("\nTraining XGBoost Classifier...")
    model = xgb.XGBClassifier(
        max_depth=5, 
        learning_rate=0.1, 
        n_estimators=100, 
        random_state=42,
        eval_metric='logloss' # To prevent a deprecation warning regarding the default evaluation metric
    )
    
    # Fit the model to the training data
    model.fit(X_train, y_train)
    print("Model training completed.")

    # 5. Evaluate the model on the test set
    print("\nEvaluating model on the test set...")
    # Predict discrete classes for Accuracy
    y_pred = model.predict(X_test)
    
    # Predict probabilities for ROC-AUC
    # predict_proba returns [prob_class_0, prob_class_1], we want prob_class_1
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Print the evaluation metrics
    print(f"--- Model Evaluation ---")
    print(f"Accuracy:      {accuracy:.4f} ({(accuracy*100):.2f}%)")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"------------------------")

    # 6. Save the trained model to a file
    # joblib is efficient for models with large internal arrays like XGBoost/Random Forest
    model_filename = 'xgb_model.pkl'
    joblib.dump(model, model_filename)
    print(f"\nModel successfully saved to '{model_filename}'.")

if __name__ == "__main__":
    main()
