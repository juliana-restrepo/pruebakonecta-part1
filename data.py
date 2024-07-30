import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the training dataset
df = pd.read_csv('train.csv')

# Initial exploration
def initial_exploration(df):
    """Perform initial exploration of the DataFrame."""
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\nDataFrame info:")
    print(df.info())
    print("\nMissing values per column:")
    print(df.isnull().sum())

# Initial exploration of the training data
initial_exploration(df)

# Step 2: Data Cleaning
# Drop irrelevant columns
df = df.drop(columns=['id', 'CustomerId', 'Surname'])

# Handle missing values and encode categorical features
categorical_features = ['Geography', 'Gender']
numeric_features = df.drop(columns=categorical_features + ['Churn']).columns

# Define the column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Apply the transformations
df_processed = preprocessor.fit_transform(df)

# Convert the result back to a DataFrame
df = pd.DataFrame(df_processed, columns=numeric_features.tolist() + preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist())

# Add the Churn column back
df['Churn'] = pd.read_csv('train.csv')['Churn'].values

# Verify the cleaning
initial_exploration(df)

# Step 3: Exploratory Data Analysis (EDA)
# Plot distributions of key variables
def plot_distributions(df):
    """Plot distributions of key variables."""
    # Plot target variable distribution
    plt.figure(figsize=(8, 6))
    df['Churn'].value_counts().plot(kind='bar')
    plt.title('Distribution of Churn')
    plt.xlabel('Churn')
    plt.ylabel('Count')
    plt.show()
    
    # Plot binary variables distributions
    binary_columns = ['HasCrCard', 'IsActiveMember', 'NumOfProducts']
    plt.figure(figsize=(16, 4))
    for i, col in enumerate(binary_columns, 1):
        plt.subplot(1, len(binary_columns), i)
        df[col].value_counts().plot(kind='bar')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    # Plot correlation matrix
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = df[numeric_columns].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
    
    # Print the correlation line for Churn
    print("\nCorrelation of Churn with other variables:")
    print(corr_matrix['Churn'])

# Plot distributions
plot_distributions(df)

