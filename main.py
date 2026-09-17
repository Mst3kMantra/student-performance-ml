import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV

# Step B1: Preprocess dataset
# Load dataset
df = pd.read_csv("StudentPerformanceFactors.csv")

# Inspect dataset
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

# Handle missing values
categorical_columns = [
    "Teacher_Quality",
    "Parental_Education_Level",
    "Distance_from_Home"
]

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

# Separate features and target
x_features = df.drop("Exam_Score", axis=1)
y_target = df["Exam_Score"]

# Convert categorical variables to numerical variables
x_features = pd.get_dummies(x_features, drop_first=True)

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x_features,
    y_target,
    test_size=0.20,
    random_state=42
)

processed_df = x_features.copy()
processed_df["Exam_Score"] = y_target

processed_df.to_csv("StudentPerformanceFactors_preprocessed.csv", index=False)

print("\nTraining rows:", len(x_train))
print("Testing rows:", len(x_test))
print("Number of features:", x_train.shape[1])

# Step B2: Build the AI/ML algorithm
# Build the Random Forest Regression model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Step B3: Train the model using the AI/ML algorithm
# Train the model
model.fit(x_train, y_train)

# Step B4: Evaluate model accuracy using metrics like accuracy, precision, recall, and F1 score.
# Make predictions
y_pred = model.predict(x_test)

# Evaluate model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# B5: Apply cross-validation techniques
kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model,
    x_train,
    y_train,
    cv=kfold,
    scoring="r2"
)

cv_rmse = -cross_val_score(
    model,
    x_train,
    y_train,
    cv=kfold,
    scoring="neg_root_mean_squared_error"
)

# B6: Use hyperparameter tuning to optimize the model
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

grid_search.fit(x_train, y_train)

print("Best parameters:")
print(grid_search.best_params_)

print("Best CV RMSE:")
print(-grid_search.best_score_)

best_model = grid_search.best_estimator_

y_pred_optimized = best_model.predict(x_test)

rmse_optimized = np.sqrt(mean_squared_error(y_test, y_pred_optimized))
r2_optimized = r2_score(y_test, y_pred_optimized)

cv_scores_optimized = cross_val_score(
    best_model,
    x_train,
    y_train,
    cv=kfold,
    scoring="r2"
)

cv_rmse_optimized = -cross_val_score(
    best_model,
    x_train,
    y_train,
    cv=kfold,
    scoring="neg_root_mean_squared_error"
)

print("\nUnoptimized Metrics:")

print("RMSE:", rmse)
print("R²:", r2)

print("Average cross-validation RMSE:", cv_rmse.mean())
print("Cross-validation R² scores:", cv_scores)
print("Average cross-validation R²:", cv_scores.mean(), "\n")

print("Optimized Metrics:")
print("Optimized RMSE:", rmse_optimized)
print("Optimized R²:", r2_optimized)

print("Optimized Average cross-validation RMSE:", cv_rmse_optimized.mean())
print("Optimized Cross-validation R² scores:", cv_scores_optimized)
print("Optimized Average cross-validation R²:", cv_scores_optimized.mean())