import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def generate_synthetic_data(num_samples=2000, fraud_ratio=0.2):
    """Generate synthetic data for fraud detection.
    
    This function creates realistic fraud detection data with:
    1. Strong signals (clear indicators of fraud)
    2. Weak signals (subtle indicators that are hard to detect)
    3. Noisy features (irrelevant to fraud detection)
    4. Sparse features (rarely occurring but highly indicative)
    """
    np.random.seed(42)
    n_fraud = int(num_samples * fraud_ratio)
    n_nonfraud = num_samples - n_fraud
    
    # Create fraud and non-fraud indices
    fraud_indices = list(range(n_fraud))
    non_fraud_indices = list(range(n_fraud, num_samples))
    all_indices = fraud_indices + non_fraud_indices

    # Empty data dictionary
    data = {}
    
    # 1. Strong signals with clear differences between fraud/non-fraud
    # Income tends to be lower for fraudulent cases
    data['income'] = np.zeros(num_samples)
    data['income'][fraud_indices] = np.random.uniform(0.1, 0.4, n_fraud)
    data['income'][non_fraud_indices] = np.random.uniform(0.3, 0.9, n_nonfraud)
    
    # Name-email similarity tends to be lower for fraudulent cases
    data['name_email_similarity'] = np.zeros(num_samples)
    data['name_email_similarity'][fraud_indices] = np.random.uniform(0, 0.4, n_fraud)
    data['name_email_similarity'][non_fraud_indices] = np.random.uniform(0.3, 1.0, n_nonfraud)
    
    # 2. Weak signals (subtle differences)
    # Previous address months - fraudsters tend to have slightly shorter history
    data['prev_address_months_count'] = np.zeros(num_samples)
    data['prev_address_months_count'][fraud_indices] = np.random.choice([-1] + list(range(0, 30)), n_fraud)
    data['prev_address_months_count'][non_fraud_indices] = np.random.choice([-1] + list(range(0, 381)), n_nonfraud)
    
    # Current address months - similar pattern
    data['current_address_months_count'] = np.zeros(num_samples)
    data['current_address_months_count'][fraud_indices] = np.random.choice([-1] + list(range(0, 40)), n_fraud)
    data['current_address_months_count'][non_fraud_indices] = np.random.choice([-1] + list(range(0, 430)), n_nonfraud)
    
    # 3. Regular features with some signal
    data['customer_age'] = np.zeros(num_samples)
    data['customer_age'][fraud_indices] = np.random.randint(18, 35, n_fraud)  # Fraudsters tend to be younger
    data['customer_age'][non_fraud_indices] = np.random.randint(10, 91, n_nonfraud)
    
    data['days_since_request'] = np.random.randint(0, 80, num_samples)
    data['intended_balcon_amount'] = np.random.uniform(-16, 114, num_samples)
    
    # Payment type has some correlation with fraud
    payment_types = ['A', 'B', 'C', 'D', 'E']
    fraud_payment_probs = [0.4, 0.3, 0.15, 0.1, 0.05]  # Type A and B more common in fraud
    non_fraud_payment_probs = [0.1, 0.15, 0.25, 0.25, 0.25]  # More evenly distributed for non-fraud
    
    data['payment_type'] = np.zeros(num_samples, dtype=object)
    data['payment_type'][fraud_indices] = np.random.choice(payment_types, n_fraud, p=fraud_payment_probs)
    data['payment_type'][non_fraud_indices] = np.random.choice(payment_types, n_nonfraud, p=non_fraud_payment_probs)
    
    # 4. Velocity features (fraudsters have higher activity rates)
    data['velocity_6h'] = np.zeros(num_samples)
    data['velocity_6h'][fraud_indices] = np.random.uniform(100, 16819, n_fraud)
    data['velocity_6h'][non_fraud_indices] = np.random.uniform(-175, 100, n_nonfraud)
    
    data['velocity_24h'] = np.zeros(num_samples)
    data['velocity_24h'][fraud_indices] = np.random.randint(5000, 9587, n_fraud)
    data['velocity_24h'][non_fraud_indices] = np.random.randint(1297, 5000, n_nonfraud)
    
    data['velocity_4w'] = np.random.randint(2825, 7021, num_samples)
    
    # 5. Sparse features that are highly indicative but rare
    # Few bank branches indicate higher risk
    data['bank_branch_count_8w'] = np.zeros(num_samples)
    data['bank_branch_count_8w'][fraud_indices] = np.random.choice([0, 1, 2], n_fraud, p=[0.6, 0.3, 0.1])
    data['bank_branch_count_8w'][non_fraud_indices] = np.random.randint(1, 2405, n_nonfraud)
    
    # Multiple emails with same date of birth is suspicious
    data['date_of_birth_distinct_emails_4w'] = np.zeros(num_samples)
    data['date_of_birth_distinct_emails_4w'][fraud_indices] = np.random.choice([0, 1, 2, 3, 20, 30, 40], n_fraud, p=[0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1])
    data['date_of_birth_distinct_emails_4w'][non_fraud_indices] = np.random.choice([0, 1, 2], n_nonfraud, p=[0.7, 0.2, 0.1])
    
    # 6. Categorical features with some relationship to fraud
    # Employment status
    emp_statuses = ['E1', 'E2', 'E3', 'E4', 'E5']
    data['employment_status'] = np.zeros(num_samples, dtype=object)
    data['employment_status'][fraud_indices] = np.random.choice(emp_statuses, n_fraud, p=[0.5, 0.3, 0.1, 0.05, 0.05])
    data['employment_status'][non_fraud_indices] = np.random.choice(emp_statuses, n_nonfraud)
    
    # Credit risk score - fraudsters tend to have lower scores
    data['credit_risk_score'] = np.zeros(num_samples)
    data['credit_risk_score'][fraud_indices] = np.random.uniform(-191, 100, n_fraud)
    data['credit_risk_score'][non_fraud_indices] = np.random.uniform(-100, 390, n_nonfraud)
    
    # 7. Binary features with fraud relationships
    # Free email providers more common in fraud
    data['email_is_free'] = np.zeros(num_samples)
    data['email_is_free'][fraud_indices] = np.random.choice([0, 1], n_fraud, p=[0.2, 0.8])
    data['email_is_free'][non_fraud_indices] = np.random.choice([0, 1], n_nonfraud, p=[0.6, 0.4])
    
    # Housing status
    data['housing_status'] = np.random.choice(['H1', 'H2', 'H3'], num_samples)
    
    # Invalid phone numbers more common in fraud
    data['phone_home_valid'] = np.zeros(num_samples)
    data['phone_home_valid'][fraud_indices] = np.random.choice([0, 1], n_fraud, p=[0.4, 0.6])
    data['phone_home_valid'][non_fraud_indices] = np.random.choice([0, 1], n_nonfraud, p=[0.1, 0.9])
    
    data['phone_mobile_valid'] = np.zeros(num_samples)
    data['phone_mobile_valid'][fraud_indices] = np.random.choice([0, 1], n_fraud, p=[0.3, 0.7])
    data['phone_mobile_valid'][non_fraud_indices] = np.random.choice([0, 1], n_nonfraud, p=[0.05, 0.95])
    
    # 8. Other potentially relevant features
    data['bank_months_count'] = np.random.choice([-1] + list(range(0, 33)), num_samples)
    data['has_other_cards'] = np.random.choice([0, 1], num_samples)
    data['proposed_credit_limit'] = np.random.randint(200, 2001, num_samples)
    
    # Foreign requests more common in fraud
    data['foreign_request'] = np.zeros(num_samples)
    data['foreign_request'][fraud_indices] = np.random.choice([0, 1], n_fraud, p=[0.5, 0.5])
    data['foreign_request'][non_fraud_indices] = np.random.choice([0, 1], n_nonfraud, p=[0.9, 0.1])
    
    # Source type
    data['source'] = np.random.choice(['INTERNET', 'TELEAPP'], num_samples)
    
    # Session characteristics
    data['session_length_in_minutes'] = np.random.choice([-1] + list(range(0, 108)), num_samples)
    data['device_os'] = np.random.choice(['Windows', 'macOS', 'Linux'], num_samples)
    data['keep_alive_session'] = np.random.choice([0, 1], num_samples)
    
    # Device metrics
    data['device_distinct_emails'] = np.random.choice([-1, 0, 1], num_samples)
    
    # Known fraud device
    data['device_fraud_count'] = np.zeros(num_samples)
    data['device_fraud_count'][fraud_indices] = np.random.choice([0, 1], n_fraud, p=[0.7, 0.3])
    data['device_fraud_count'][non_fraud_indices] = np.random.choice([0, 1], n_nonfraud, p=[0.99, 0.01])
    
    # Time-based features
    data['month'] = np.random.randint(0, 12, num_samples)
    
    # Target variable
    data['fraud_Cases'] = np.zeros(num_samples)
    data['fraud_Cases'][fraud_indices] = 1

    df = pd.DataFrame(data)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

def preprocess_data(df):
    """Preprocess the data including scaling and encoding."""
    categorical_features = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
    numerical_features = [col for col in df.columns if col not in categorical_features + ['fraud_Cases']]

    # Scale numerical features
    scaler = MinMaxScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # Encode categorical features
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_cats = encoder.fit_transform(df[categorical_features])
    encoded_df = pd.DataFrame(
        encoded_cats,
        columns=encoder.get_feature_names_out(categorical_features)
    )

    # Combine numerical and encoded categorical features
    df = df.drop(columns=categorical_features).reset_index(drop=True)
    df = pd.concat([df, encoded_df], axis=1)
    
    return df.select_dtypes(include=[np.number])

def downsample_data(df, target='fraud_Cases'):
    """Balance the dataset through downsampling."""
    fraud = df[df[target] == 1]
    non_fraud = df[df[target] == 0]
    # Use a more balanced ratio (e.g., 1:2 instead of 1:4) to make patterns more challenging
    non_fraud_downsampled = non_fraud.sample(n=len(fraud) * 2, random_state=42)
    return pd.concat([fraud, non_fraud_downsampled]).sample(frac=1, random_state=42).reset_index(drop=True)

def apply_pca(df, n_components=7):
    """Apply PCA transformation to the features."""
    features = df.drop(columns=['fraud_Cases'])
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(features)
    
    pca_df = pd.DataFrame(
        pca_result,
        columns=[f'PCA_{i+1}' for i in range(n_components)]
    )
    pca_df['fraud_Cases'] = df['fraud_Cases']
    
    return pca_df

def discretize_data(data, pca_columns, bins=3):
    """Discretize continuous PCA components.
    
    This function converts continuous PCA values into discrete bins, which is necessary
    for discrete Bayesian Networks. Using bins=3 creates 'low', 'med', and 'high' categories.
    """
    discretized = data.copy()
    for col in pca_columns:
        discretized[col] = pd.qcut(
            pd.to_numeric(discretized[col], errors='coerce').fillna(discretized[col].median()),
            q=bins,
            labels=['low', 'med', 'high'],
            duplicates='drop'
        ).astype(str)
    return discretized

def prepare_data_pipeline(num_samples=2000, fraud_ratio=0.2, n_components=7):
    """Complete data preparation pipeline."""
    # Generate synthetic data
    df = generate_synthetic_data(num_samples, fraud_ratio)
    
    # Preprocess data
    df_preprocessed = preprocess_data(df)
    
    # Downsample
    df_balanced = downsample_data(df_preprocessed)
    
    # Apply PCA
    df_pca = apply_pca(df_balanced, n_components)
    
    # Discretize PCA components
    pca_columns = [f'PCA_{i+1}' for i in range(n_components)]
    df_discretized = discretize_data(df_pca, pca_columns)
    
    # Split data
    X = df_discretized.drop(columns=['fraud_Cases'])
    y = df_discretized['fraud_Cases']
    
    return train_test_split(X, y, test_size=0.2, random_state=42) 