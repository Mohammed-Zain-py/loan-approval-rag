import pandas as pd

def main():
    print("Starting data cleaning process...")
    
    # 1. Load Data
    try:
        # Load the raw dataset
        df = pd.read_csv('credit_risk_dataset.csv')
        original_shape = df.shape
        print(f"Original Dataframe Shape: {original_shape} (Rows: {original_shape[0]}, Columns: {original_shape[1]})")
    except FileNotFoundError:
        print("Error: 'credit_risk_dataset.csv' not found in the current directory.")
        return

    # 2. Impute Missing Values
    # Fill NaN values in 'person_emp_length' and 'loan_int_rate' with their respective medians
    # Median is robust to outliers which is why it's preferred here
    emp_length_median = df['person_emp_length'].median()
    int_rate_median = df['loan_int_rate'].median()
    
    df['person_emp_length'] = df['person_emp_length'].fillna(emp_length_median)
    df['loan_int_rate'] = df['loan_int_rate'].fillna(int_rate_median)
    print("Imputed missing values for 'person_emp_length' and 'loan_int_rate' with medians.")

    # 3. Remove Outliers
    # Drop rows where person_age > 100 or person_emp_length > 60 as these are likely data entry errors
    # We keep rows where age <= 100 AND emp_length <= 60
    df = df[(df['person_age'] <= 100) & (df['person_emp_length'] <= 60)]
    print("Removed outliers (age > 100 or employment length > 60).")

    # 4. Encode Categorical Variables
    # One-hot encode the specified text columns to convert them into numerical format for modeling
    # drop_first=True is used to avoid the dummy variable trap (multicollinearity)
    categorical_cols = [
        'person_home_ownership', 
        'loan_intent', 
        'loan_grade', 
        'cb_person_default_on_file'
    ]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Convert resulting boolean columns from get_dummies to integers (1/0)
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)
    print("One-hot encoded categorical variables and converted booleans to integers (1/0).")

    # Logging: Show final shape
    cleaned_shape = df.shape
    rows_removed = original_shape[0] - cleaned_shape[0]
    print(f"\nCleaned Dataframe Shape: {cleaned_shape} (Rows: {cleaned_shape[0]}, Columns: {cleaned_shape[1]})")
    print(f"Total rows modified/removed during cleaning: {rows_removed}")

    # 5. Export
    # Save the cleaned dataset to a new CSV file, excluding the DataFrame index
    output_filename = 'loan_processed.csv'
    df.to_csv(output_filename, index=False)
    print(f"\nSuccess! Cleaned dataset saved as '{output_filename}'.")

if __name__ == "__main__":
    main()
