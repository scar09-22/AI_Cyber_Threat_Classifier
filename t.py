
import numpy as np
import pandas as pd
import json
import joblib

print("Creating test file from your model's test set...")

# Load everything
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')
scaler = joblib.load('data/scaler.pkl')

with open('data/feature_names.json', 'r') as f:
    features = json.load(f)

# Get 5 attack samples and 5 benign samples
attack_indices = np.where(y_test == 1)[0][:5]
benign_indices = np.where(y_test == 0)[0][:5]
all_indices = np.concatenate([attack_indices, benign_indices])

# Get the scaled data
X_samples = X_test[all_indices]

# INVERSE TRANSFORM to get original values
X_original = scaler.inverse_transform(X_samples)

# Create DataFrame
df = pd.DataFrame(X_original, columns=features)

# Add labels for reference (but don't include in upload)
labels = ['Attack'] * 5 + ['Benign'] * 5
df['Expected_Result'] = labels

# Save
df.to_csv('dashboard_test.csv', index=False)

print(f"\n✅ Created dashboard_test.csv with 10 samples:")
print(f"   - 5 Attacks (rows 1-5)")
print(f"   - 5 Benign (rows 6-10)")
print(f"\nTo test:")
print("1. Remove 'Expected_Result' column before uploading")
print("2. Upload in Batch Analysis")
print("3. Model should detect 5 attacks correctly!")
