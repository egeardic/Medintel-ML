import time
import joblib
import pandas as pd
import time

# Timer for sklearn.linear_model import
start_import = time.time()
from sklearn.linear_model import LogisticRegression
print(f"Imported LogisticRegression in {time.time() - start_import:.4f} seconds")

# Timer for sklearn.model_selection import
start_import = time.time()
from sklearn.model_selection import train_test_split
print(f"Imported train_test_split in {time.time() - start_import:.4f} seconds")

# Timer for sklearn.metrics import
start_import = time.time()
from sklearn.metrics import classification_report
print(f"Imported classification_report in {time.time() - start_import:.4f} seconds")

# Timer for sklearn.pipeline import
start_import = time.time()
from sklearn.pipeline import Pipeline
print(f"Imported Pipeline in {time.time() - start_import:.4f} seconds")

# Timer for sklearn.preprocessing import
start_import = time.time()
from sklearn.preprocessing import StandardScaler
print(f"Imported StandardScaler in {time.time() - start_import:.4f} seconds")

# Timer for scipy.sparse import
start_import = time.time()
from scipy.sparse import csr_matrix
print(f"Imported csr_matrix in {time.time() - start_import:.4f} seconds")

# ----------- CONFIG ----------------
max_iterations = 1        # Tune this
tolerance = 1e-1             # Larger tol = faster but less precise
penalty_type = 'l1'          # 'l1' for sparse feature selection
model_filename = f"disease_predictor_saga_{max_iterations}.pkl"
# ----------------------------------

# Start timer
start_time = time.time()

# 1. Read dataset
df = pd.read_csv("Final_Augmented_dataset_Diseases_and_Symptoms.csv")
# 2. Separate features and target
X = df.drop('diseases', axis=1)
y = df['diseases']

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Convert to sparse matrix (helpful if data has many 0s)
X_train_sparse = csr_matrix(X_train.values)
X_test_sparse = csr_matrix(X_test.values)

# 5. Create pipeline
model = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),  # with_mean=False is required for sparse
    ('clf', LogisticRegression(
        max_iter=max_iterations,
        solver='saga',
        penalty=penalty_type,
        tol=tolerance,
        n_jobs=-1  # Use all CPU cores
    ))
])

# 6. Train
model.fit(X_train_sparse, y_train)

# 7. Predict and evaluate
y_pred = model.predict(X_test_sparse)
print(classification_report(y_test, y_pred))

# 8. Save model
joblib.dump(model, model_filename)

# 9. Print timing
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds")
