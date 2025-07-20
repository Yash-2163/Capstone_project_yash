from data_loader import load_data                    # Imports the custom function to load the dataset
from preprocessing import build_full_preprocessing_pipeline  # Imports the function that builds the preprocessing pipeline
import joblib                                         # For saving the pipeline as a binary file

# Load the raw dataset (features and target)
X, y = load_data()

# Build the full preprocessing pipeline
# This pipeline performs:
# 1. Missing value imputation (smart strategies including KNN)
# 2. Skew correction and outlier clipping (log + IQR method)
# 3. Type optimization (bools, categoricals)
# 4. Categorical encoding (ordinal + one-hot)
# 5. Normalization (MinMaxScaler for numeric columns)
pipeline = build_full_preprocessing_pipeline()

# Fit the pipeline to the data
# This step is needed to compute any statistics required (e.g., median, mode, encoder fitting, etc.)
pipeline.fit(X)

# Save the entire preprocessing pipeline for reuse
# Storing the pipeline ensures consistency during both training and inference
joblib.dump(pipeline, '../final_model/preprocessing_pipeline.pkl')

# Confirmation message
print("✅ Preprocessing pipeline saved.")
