import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# --- Step 1: Missing Value Handling ---
def handle_missing_values_v2(df):
    """
    Handles missing values based on feature type and missing proportion.
    
    Strategy:
    - Numerical columns with <5% missing: fill with median (robust to outliers).
    - Numerical columns with 5-50% missing: KNN Imputer used to consider patterns.
    - Numerical columns with >50% missing: fill with 0 and add missing flag.
    - Boolean columns: fill with False.
    - Categorical columns:
        - <50% missing: fill with mode.
        - >50%: fill with 'Unknown' (preserves missing info).
    """
    df = df.copy()
    miss = df.isnull().mean()
    num_cols = df.select_dtypes(include=['int64','float64']).columns
    bool_cols = df.select_dtypes(include=['bool','boolean']).columns
    cat_cols = df.select_dtypes(include=['object','category']).columns

    for col in num_cols:
        m = miss[col]
        if m == 0:
            continue
        if m < 0.05:
            df[col] = df[col].fillna(df[col].median())  # Robust to outliers
        elif m < 0.5:
            continue  # Will use KNN imputer later
        else:
            df[f'{col}_missing_flag'] = df[col].isnull().astype(int)
            df[col] = df[col].fillna(0)  # Assume 0 as safe default, flag preserves info

    knn_cols = [c for c in num_cols if 0.05 <= miss[c] < 0.5]
    if knn_cols:
        imputer = KNNImputer(n_neighbors=3)
        df[knn_cols] = imputer.fit_transform(df[knn_cols])  # KNN uses patterns from other rows

    for col in bool_cols:
        if miss[col] > 0:
            df[col] = df[col].fillna(False)  # False is often a conservative default

    for col in cat_cols:
        m = miss[col]
        if m == 0:
            continue
        if m < 0.5:
            df[col] = df[col].fillna(df[col].mode()[0])  # Most common category
        else:
            if pd.api.types.is_categorical_dtype(df[col]):
                df[col] = df[col].cat.add_categories('Unknown')
            df[col] = df[col].fillna('Unknown')  # Preserves missing info in label

    return df

# --- Step 2: Skew and Outlier Handling ---
def handle_skew_and_outliers(df):
    """
    Applies log transformation to skewed features and caps outliers using IQR.
    
    Why:
    - Log transformation reduces skew, which improves model performance for algorithms sensitive to distribution.
    - IQR-based clipping reduces impact of extreme values without removing them.
    """
    df = df.copy()

    # Apply log1p to skewed variables (log1p handles 0 values safely)
    for col in ['TotalVisits', 'Page Views Per Visit']:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    # IQR-based clipping for outliers
    numeric_cols = [
        'TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit',
        'Asymmetrique Activity Score', 'Asymmetrique Profile Score'
    ]
    for col in numeric_cols:
        if col in df.columns:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lo, upper=hi)
    return df

# --- Step 3: Optimize Dtypes ---
def optimize_dtypes(df):
    """
    Converts certain features to 'category' or 'bool' to reduce memory and 
    help transformers/encoders work more efficiently.
    """
    df = df.copy()

    # Explicitly convert ordinal columns to categorical
    ord_cols = ['Lead Quality', 'Asymmetrique Activity Index', 'Asymmetrique Profile Index']
    for col in ord_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Convert binary integer columns to boolean
    bin_int = [
        'Do Not Email', 'Do Not Call', 'Search', 'Newspaper Article',
        'X Education Forums', 'Newspaper', 'Digital Advertisement',
        'Through Recommendations', 'A free copy of Mastering The Interview'
    ]
    for col in bin_int:
        if col in df.columns:
            vals = set(df[col].dropna().unique())
            if vals.issubset({0,1}):  # Only 0 and 1 values
                df[col] = df[col].astype('bool')
    return df

# --- Step 4: Categorical Transformation ---
class CategoricalPreprocessor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to handle:
    - Binary column mapping ('Yes'/'No' to 1/0)
    - Ordinal encoding using predefined maps
    - Dropping low-variance or irrelevant columns
    """

    def __init__(self):
        self.bin_cols = [
            'Do Not Email','Do Not Call','Search','Newspaper Article',
            'X Education Forums','Newspaper','Digital Advertisement',
            'Through Recommendations','A free copy of Mastering The Interview'
        ]
        self.ord_map = {
            'Lead Quality':{'Worst':0,'Low in Relevance':1,'Not Sure':2,'Might be':3,'High in Relevance':4},
            'Asymmetrique Activity Index':{'03.Low':0,'02.Medium':1,'01.High':2},
            'Asymmetrique Profile Index':{'03.Low':0,'02.Medium':1,'01.High':2}
        }
        self.drop_cols = [
            'Magazine', 'Receive More Updates About Our Courses',
            'Update me on Supply Chain Content', 'Get updates on DM Content',
            'I agree to pay the amount through cheque'
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        df = df.copy()

        # Drop irrelevant columns (either always null, redundant, or uninformative)
        for col in self.drop_cols:
            if col in df.columns:
                df.drop(columns=col, inplace=True)

        # Binary mapping: convert 'Yes'/'No' to 1/0
        for col in self.bin_cols:
            if col in df.columns:
                df[col] = df[col].map({'Yes': 1, 'No': 0})

        # Apply manual ordinal encoding
        for col, m in self.ord_map.items():
            if col in df.columns:
                df[col] = df[col].map(m)

        return df

# --- Helper functions for column selection ---
def select_numeric_columns(df):
    return df.select_dtypes(include=['int64', 'float64']).columns.tolist()

def select_categorical_columns(df):
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

# --- Step 5: Build Preprocessing Pipeline ---
def build_full_preprocessing_pipeline():
    """
    Constructs full preprocessing pipeline with the following stages:
    1. Handle missing values.
    2. Correct skewness and outliers.
    3. Optimize data types.
    4. Apply categorical transformations (ordinal + binary).
    5. Scale numeric and one-hot encode categorical features.
    """
    missing_tf    = FunctionTransformer(handle_missing_values_v2, validate=False)
    skew_out_tf   = FunctionTransformer(handle_skew_and_outliers, validate=False)
    dtype_opt_tf  = FunctionTransformer(optimize_dtypes, validate=False)
    cat_proc_tf   = CategoricalPreprocessor()

    # Scalers and Encoders
    numeric_scaler = MinMaxScaler()  # Brings all features to 0–1 range, useful for models sensitive to scale
    categorical_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # ColumnTransformer applies different transformers to different column types
    scale_encode = ColumnTransformer(transformers=[
        ('num', numeric_scaler, select_numeric_columns),
        ('cat', categorical_encoder, select_categorical_columns)
    ])

    # Final preprocessing pipeline
    pipeline = Pipeline([
        ('missing', missing_tf),
        ('skew_outliers', skew_out_tf),
        ('dtypes', dtype_opt_tf),
        ('cat_transform', cat_proc_tf),
        ('scale_encode', scale_encode)
    ])

    return pipeline
