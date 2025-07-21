import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

# These imports are necessary for joblib to unpickle the model pipeline,
# even if they are not called directly in this script.
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

from preprocessing import handle_missing_values_v2, handle_skew_and_outliers, optimize_dtypes, CategoricalPreprocessor, select_numeric_columns, select_categorical_columns

# --- Define custom transformers ---
# These classes and functions MUST be defined at the top level
# so that joblib.load() can find them when unpickling the model.
class CategoricalPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.bin_cols = [
            'do_not_email','do_not_call','search','newspaper_article',
            'x_education_forums','newspaper','digital_advertisement',
            'through_recommendations','a_free_copy_of_mastering_the_interview'
        ]
        self.ord_map = {
            'lead_quality':{'Worst':0,'Low in Relevance':1,'Not Sure':2,'Might be':3,'High in Relevance':4},
            'asymmetrique_activity_index':{'03.Low':0,'02.Medium':1,'01.High':2},
            'asymmetrique_profile_index':{'03.Low':0,'02.Medium':1,'01.High':2}
        }
        self.drop_cols = [
            'magazine', 'receive_more_updates_about_our_courses',
            'update_me_on_supply_chain_content', 'get_updates_on_dm_content',
            'i_agree_to_pay_the_amount_through_cheque'
        ]
    def fit(self, X, y=None):
        return self
    def transform(self, df):
        df = df.copy()
        for col in self.drop_cols:
            if col in df.columns:
                df.drop(columns=col, inplace=True, errors='ignore')
        for col in self.bin_cols:
            if col in df.columns:
                df[col] = df[col].map({'Yes': 1, 'No': 0})
        for col, m in self.ord_map.items():
            if col in df.columns:
                df[col] = df[col].map(m)
        return df

def handle_missing_values_v2(df):
    df = df.copy()
    miss = df.isnull().mean()
    num_cols = df.select_dtypes(include=['int64','float64']).columns
    bool_cols = df.select_dtypes(include=['bool','boolean']).columns
    cat_cols = df.select_dtypes(include=['object','category']).columns

    for col in num_cols:
        m = miss.get(col, 0)
        if m == 0: continue
        if m < 0.05:
            df[col] = df[col].fillna(df[col].median())
        elif m >= 0.5:
            df[f'{col}_missing_flag'] = df[col].isnull().astype(int)
            df[col] = df[col].fillna(0)
    
    knn_cols = [c for c in num_cols if 0.05 <= miss.get(c, 0) < 0.5]
    if knn_cols:
        imputer = KNNImputer(n_neighbors=3)
        df[knn_cols] = imputer.fit_transform(df[knn_cols])

    for col in bool_cols:
        if miss.get(col, 0) > 0: df[col] = df[col].fillna(False)

    for col in cat_cols:
        m = miss.get(col, 0)
        if m == 0: continue
        if m < 0.5:
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            if pd.api.types.is_categorical_dtype(df[col]):
                if 'Unknown' not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories('Unknown')
            df[col] = df[col].fillna('Unknown')
    return df

def handle_skew_and_outliers(df):
    df = df.copy()
    # Corrected column names to match training data (e.g., 'totalvisits')
    for col in ['totalvisits', 'page_views_per_visit']:
        if col in df.columns: df[col] = np.log1p(df[col])
    numeric_cols = ['totalvisits', 'total_time_spent_on_website', 'page_views_per_visit', 'asymmetrique_activity_score', 'asymmetrique_profile_score']
    for col in numeric_cols:
        if col in df.columns:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lo, upper=hi)
    return df

def optimize_dtypes(df):
    df = df.copy()
    ord_cols = ['lead_quality', 'asymmetrique_activity_index', 'asymmetrique_profile_index']
    for col in ord_cols:
        if col in df.columns: df[col] = df[col].astype('category')
    bin_int = ['do_not_email', 'do_not_call', 'search', 'newspaper_article', 'x_education_forums', 'newspaper', 'digital_advertisement', 'through_recommendations', 'a_free_copy_of_mastering_the_interview']
    for col in bin_int:
        if col in df.columns:
            if set(df[col].dropna().unique()).issubset({0,1}):
                df[col] = df[col].astype('bool')
    return df

def select_numeric_columns(df): return df.select_dtypes(include=['int64', 'float64']).columns.tolist()
def select_categorical_columns(df): return df.select_dtypes(include=['object', 'category']).columns.tolist()

# --- Load the model pipeline ---
@st.cache_resource
def load_model():
    """Loads the model from the model.pkl file."""
    try:
        return joblib.load("final_model.pkl")
    except FileNotFoundError:
        st.error("Error: 'model.pkl' not found. Please ensure the model file is in the same directory as this script.")
        return None

model = load_model()

# --- Define the expected columns for the model ---
# This list exactly matches the feature columns from your df.info() output.
EXPECTED_COLUMNS = [
    'prospect_id', 'lead_number', 'lead_origin', 'lead_source', 'do_not_email', 'do_not_call',
    'totalvisits', 'total_time_spent_on_website', 'page_views_per_visit', 'last_activity',
    'country', 'specialization', 'how_did_you_hear_about_x_education', 
    'what_is_your_current_occupation', 'what_matters_most_to_you_in_choosing_a_course',
    'search', 'magazine', 'newspaper_article', 'x_education_forums', 'newspaper',
    'digital_advertisement', 'through_recommendations', 'receive_more_updates_about_our_courses',
    'tags', 'lead_quality', 'update_me_on_supply_chain_content', 'get_updates_on_dm_content',
    'lead_profile', 'city', 'asymmetrique_activity_index', 'asymmetrique_profile_index',
    'asymmetrique_activity_score', 'asymmetrique_profile_score', 
    'i_agree_to_pay_the_amount_through_cheque', 'a_free_copy_of_mastering_the_interview',
    'last_notable_activity', 'x education forums', 'newspaper article'
]

# --- Helper function to align columns ---
def align_columns(input_df, expected_cols):
    """Aligns the uploaded dataframe to the format expected by the model."""
    df_to_process = input_df.copy()
    
    # Standardize uploaded column names to match the training data format (lowercase, snake_case)
    df_to_process.columns = [col.lower().replace(' ', '_') for col in df_to_process.columns]
    
    # Ensure all expected columns are present, adding missing ones with NaN
    for col in expected_cols:
        if col not in df_to_process.columns:
            # Handle the special case of columns with spaces in the training data
            original_name_with_space = col.replace('_', ' ')
            if original_name_with_space in df_to_process.columns:
                 df_to_process.rename(columns={original_name_with_space: col}, inplace=True)
            else:
                df_to_process[col] = np.nan
            
    # Reorder and select only the expected columns
    return df_to_process[expected_cols]


# --- Streamlit UI ---
st.set_page_config(page_title="Lead Conversion Predictor", layout="wide")
st.title("🚀 Lead Conversion Predictor")
st.markdown("Upload a CSV of leads and get predictions of which are likely to convert.")

if model is not None:
    uploaded_file = st.file_uploader("📄 Upload CSV", type="csv")

    if uploaded_file:
        try:
            df_original = pd.read_csv(uploaded_file)
            st.subheader("🔍 Data Preview")
            st.dataframe(df_original.head())

            if st.button("Get Predictions"):
                with st.spinner("Aligning data and running predictions..."):
                    
                    # 1. Align the uploaded data to the model's expected format
                    aligned_df = align_columns(df_original, EXPECTED_COLUMNS)
                    
                    # 2. Run predictions on the aligned data
                    predictions = model.predict(aligned_df)
                    
                    # 3. Add results to the original dataframe for display
                    df_original["Prediction"] = ["✅ Converted" if p == 1 else "❌ Not Converted" for p in predictions]
                
                st.success("Predictions complete!")
                st.subheader("🎯 Prediction Results")
                st.dataframe(df_original)

                # Allow downloading the results
                csv = df_original.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="lead_predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.info("⬆️ Please upload a CSV file to begin.")
