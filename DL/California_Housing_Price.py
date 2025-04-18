import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import warnings
warnings.filterwarnings('ignore')

# Load the California Housing dataset
print("Loading the California Housing dataset...")
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = pd.DataFrame(california.target, columns=["Price"])
print(f"Dataset loaded with {X.shape[0]} samples and {X.shape[1]} features.")

# 1. Preparing the data
print("\n1. Preparing the data...")

# Understanding the data
print("Summary statistics of features:")
print(X.describe().T[['mean', 'std', 'min', 'max']])

# Feature names and their descriptions
print("\nFeature names and descriptions:")
feature_descriptions = {
    'MedInc': 'Median income in block group',
    'HouseAge': 'Median house age in block group',
    'AveRooms': 'Average number of rooms per household',
    'AveBedrms': 'Average number of bedrooms per household',
    'Population': 'Block group population',
    'AveOccup': 'Average number of household members',
    'Latitude': 'Block group latitude',
    'Longitude': 'Block group longitude'
}
for feature in california.feature_names:
    print(f"{feature}: {feature_descriptions[feature]}")

# Checking for missing values
print("\nChecking for missing values:")
print(X.isnull().sum())

# Visualization of target variable distribution
plt.figure(figsize=(10, 6))
sns.histplot(y, kde=True)
plt.title('Distribution of Housing Prices')
plt.xlabel('Median House Value (in $100,000s)')
plt.tight_layout()
plt.savefig('price_distribution.png')
plt.close()

# Correlation analysis
plt.figure(figsize=(12, 10))
correlation_matrix = pd.concat([X, y], axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

print("Top 3 features correlated with housing prices:")
correlations = correlation_matrix['Price'].sort_values(ascending=False)
print(correlations.iloc[1:4])  # Excluding Price's correlation with itself

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"\nData split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")

# 2. Building the Neural Network
print("\n2. Building the neural network...")

def create_model(input_dim=8):  # California Housing has 8 features
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model

# Create and train the model
model = create_model(X_train.shape[1])
print(model.summary())

# Training with early stopping
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
    ],
    verbose=0
)

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Training and Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.close()

# 3. Validating using K-fold validation
print("\n3. Validating using K-fold validation...")

def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Function to perform k-fold validation
def kfold_validation(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_scores = []
    r2_scores = []
    
    fold_idx = 1
    for train_idx, val_idx in kf.split(X):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        model = create_model(X.shape[1])
        model.fit(
            X_train_fold, y_train_fold,
            epochs=50,
            batch_size=32,
            verbose=0
        )
        
        y_pred = model.predict(X_val_fold)
        
        rmse = rmse_score(y_val_fold, y_pred)
        r2 = r2_score(y_val_fold, y_pred)
        
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        
        print(f"Fold {fold_idx}: RMSE = {rmse:.4f}, R² = {r2:.4f}")
        fold_idx += 1
    
    print(f"\nAverage RMSE: {np.mean(rmse_scores):.4f} (±{np.std(rmse_scores):.4f})")
    print(f"Average R²: {np.mean(r2_scores):.4f} (±{np.std(r2_scores):.4f})")
    
    return rmse_scores, r2_scores

# Perform k-fold validation
rmse_scores, r2_scores = kfold_validation(X_scaled, y)

# Plot k-fold validation results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(range(1, 6), rmse_scores)
plt.axhline(y=np.mean(rmse_scores), color='r', linestyle='-', label=f'Mean: {np.mean(rmse_scores):.2f}')
plt.title('RMSE Scores Across 5 Folds')
plt.xlabel('Fold')
plt.ylabel('RMSE')
plt.xticks(range(1, 6))
plt.legend()

plt.subplot(1, 2, 2)
plt.bar(range(1, 6), r2_scores)
plt.axhline(y=np.mean(r2_scores), color='r', linestyle='-', label=f'Mean: {np.mean(r2_scores):.2f}')
plt.title('R² Scores Across 5 Folds')
plt.xlabel('Fold')
plt.ylabel('R²')
plt.xticks(range(1, 6))
plt.legend()

plt.tight_layout()
plt.savefig('kfold_results.png')
plt.close()

# 4. Final evaluation on test set
print("\n4. Final evaluation on test set...")

# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate metrics
test_rmse = rmse_score(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test R²: {test_r2:.4f}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Actual vs Predicted Housing Prices')
plt.xlabel('Actual Price ($100,000s)')
plt.ylabel('Predicted Price ($100,000s)')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.close()

# 5. Feature importance analysis
print("\n5. Feature importance analysis...")

# Create a simple model to assess feature importance
feature_importance_model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_dim=X_train.shape[1], use_bias=False)
])

feature_importance_model.compile(optimizer='adam', loss='mse')
feature_importance_model.fit(X_train, y_train, epochs=100, verbose=0)

# Extract weights
weights = feature_importance_model.get_weights()[0].flatten()
feature_names = X.columns

# Create a DataFrame with feature names and their importance
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': np.abs(weights)
})
importance_df = importance_df.sort_values('Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print("\nTop 5 most important features:")
print(importance_df.head())

# 6. Wrapping up
print("\n6. Wrapping up...")
print("Summary of the California Housing Price Regression Model:")
print(f"- Number of samples: {X.shape[0]}")
print(f"- Number of features: {X.shape[1]}")
print(f"- Model architecture: 3-layer neural network with dropout")
print(f"- Cross-validation RMSE: {np.mean(rmse_scores):.4f} (±{np.std(rmse_scores):.4f})")
print(f"- Cross-validation R²: {np.mean(r2_scores):.4f} (±{np.std(r2_scores):.4f})")
print(f"- Test set RMSE: {test_rmse:.4f}")
print(f"- Test set R²: {test_r2:.4f}")
print(f"- Most important feature: {importance_df.iloc[0]['Feature']}")

print("\nThe model successfully predicts California housing prices with reasonable accuracy.")