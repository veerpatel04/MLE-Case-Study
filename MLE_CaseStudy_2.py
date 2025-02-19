# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    FunctionTransformer
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    roc_curve,
    ConfusionMatrixDisplay
)
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')
import pyarrow.parquet as pq

# Read the Parquet file's schema
schema = pq.read_schema('case_study_dataset.parquet')
print(schema)

# Load data
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd

# Read the Parquet file as a PyArrow Table
table = pq.read_table('case_study_dataset.parquet')

# Check if the table has metadata and if it includes the 'pandas' key.
if table.schema.metadata is not None and b'pandas' in table.schema.metadata:
    # Remove the 'pandas' metadata to avoid custom dtypes like 'dbdate'
    metadata = table.schema.metadata
    new_metadata = {k: v for k, v in metadata.items() if k != b'pandas'}
    new_schema = table.schema.with_metadata(new_metadata)
    # Rebuild the table with the new schema
    table = pa.Table.from_arrays(table.columns, schema=new_schema)

# Convert to a Pandas DataFrame
df = table.to_pandas()
print(df.head())

# EDA
def perform_eda(df):
    print("===== EDA =====")
    print(f"Shape: {df.shape}")
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nTarget distribution (bookingLabel):")
    print(df['bookingLabel'].value_counts(normalize=True))
    
    # Plot target distribution
    sns.countplot(x='bookingLabel', data=df)
    plt.title('Booking Label Distribution')
    plt.savefig('target_distribution.png')
    plt.close()
    
    # Plot numerical features
    num_cols = ['numRooms', 'starLevel', 'customerReviewScore', 'minPrice']
    df[num_cols].hist(bins=20, figsize=(10, 8))
    plt.tight_layout()
    plt.savefig('numerical_distributions.png')
    plt.close()

perform_eda(df)

# Preprocessing and Feature Engineering
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Convert dates
        date_cols = ['checkInDate', 'checkOutDate', 'searchDate']
        for col in date_cols:
            X[col] = pd.to_datetime(X[col])
        
        # Days to check-in and length of stay
        X['days_to_checkin'] = (X['checkInDate'] - X['searchDate']).dt.days
        X['length_of_stay'] = (X['checkOutDate'] - X['checkInDate']).dt.days
        
        # Discount percentage
        X['valid_discount'] = X['minStrikePrice'] > X['minPrice']
        X['discount_pct'] = np.where(
            X['valid_discount'],
            (X['minStrikePrice'] - X['minPrice'])/X['minStrikePrice']*100,
            0
)
        
        # Drop unnecessary columns
        drop_cols = [
            'searchId', 'userId', 'hotelId', 'checkInDate', 'checkOutDate',
            'searchDate', 'clickLabel', 'rank', 'minStrikePrice', 'minPrice'
        ]
        return X.drop(columns=drop_cols, errors='ignore')

# Define features and target
X = df.drop(columns=['bookingLabel'])
y = df['bookingLabel']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing pipeline
categorical_features = [
    'destinationName', 'deviceCode', 'vipTier', 'signedInFlag',
    'brandId', 'freeBreakfastFlag', 'freeInternetFlag'
]
numerical_features = [
    'numRooms', 'starLevel', 'customerReviewScore', 'reviewCount',
    'days_to_checkin', 'length_of_stay', 'discount_pct'
]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ]
)

# Define models
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced'),
    'Random Forest': RandomForestClassifier(class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Train and evaluate
results = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('feature_engineer', FeatureEngineer()),
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=3)
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1
    )
    
    # Fit on full training data
    pipeline.fit(X_train, y_train)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        'CV AUC Mean': np.mean(cv_scores),
        'Test AUC': auc
    }
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.savefig('roc_curves.png')
plt.close()

# Print results
print("\n===== Model Results =====")
for model, metrics in results.items():
    print(f"{model}: CV AUC = {metrics['CV AUC Mean']:.3f}, Test AUC = {metrics['Test AUC']:.3f}")

# Feature importance for Random Forest
rf_pipeline = Pipeline(steps=[
    ('feature_engineer', FeatureEngineer()),
    ('preprocessor', preprocessor),
    ('classifier', models['Random Forest'])
])
rf_pipeline.fit(X_train, y_train)
importances = rf_pipeline.named_steps['classifier'].feature_importances_
cat_encoder = rf_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
cat_features = cat_encoder.get_feature_names_out(categorical_features)
all_features = numerical_features + list(cat_features)
feat_imp = pd.Series(importances, index=all_features).sort_values(ascending=False)
feat_imp.head(10).plot(kind='barh')
plt.title('Top 10 Feature Importances (Random Forest)')
plt.savefig('feature_importances.png')
plt.close()
