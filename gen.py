import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import json
from pathlib import Path

# Setup
output_dir = Path('outputs')
output_dir.mkdir(exist_ok=True)

print("🎨 Generating Visualization")
print("="*60)

# Load model and data
print("📂 Loading model and test data...")
model = joblib.load('models/threat_classifier.pkl')
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

with open('models/model_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"✅ Loaded {len(X_test)} test samples")
print(f"   Model: {metadata['model_type']}")

# Make predictions
print("\n🔮 Making predictions on test set...")
y_pred = model.predict(X_test)

# Calculate metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred)
}

print("\n📊 Performance Metrics:")
for metric, value in metrics.items():
    print(f"   {metric.capitalize()}: {value:.4f}")

# Generate confusion matrix
print("\n🎨 Creating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=['Benign', 'Attack'],
           yticklabels=['Benign', 'Attack'],
           cbar_kws={'label': 'Count'})
plt.title(f'Confusion Matrix - {metadata["model_type"]}', fontsize=14, pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)

# Add accuracy text
accuracy_text = f"Accuracy: {metrics['accuracy']:.2%}\nF1-Score: {metrics['f1']:.2%}"
plt.text(1, -0.3, accuracy_text, ha='center', fontsize=10, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(output_dir / f'confusion_matrix_{metadata["model_type"]}.png', 
           dpi=300, bbox_inches='tight')
plt.close()

print(f"   💾 Saved: confusion_matrix_{metadata['model_type']}.png")

# Detailed classification report
print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred, 
                           target_names=['Benign', 'Attack'],
                           digits=4))

# Create performance summary plot
print("\n🎨 Creating performance metrics chart...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Metrics bar chart
metrics_df = pd.DataFrame([metrics])
metrics_df.T.plot(kind='bar', ax=ax1, legend=False, color='steelblue')
ax1.set_title('Model Performance Metrics', fontsize=14, pad=15)
ax1.set_ylabel('Score', fontsize=12)
ax1.set_xlabel('Metric', fontsize=12)
ax1.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'], rotation=45)
ax1.set_ylim([0, 1.05])
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(metrics.values()):
    ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

# Confusion matrix as percentage
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='RdYlGn', 
           xticklabels=['Benign', 'Attack'],
           yticklabels=['Benign', 'Attack'],
           ax=ax2, cbar_kws={'label': 'Percentage'})
ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, pad=15)
ax2.set_ylabel('True Label', fontsize=12)
ax2.set_xlabel('Predicted Label', fontsize=12)

plt.tight_layout()
plt.savefig(output_dir / 'model_performance_detailed.png', dpi=300, bbox_inches='tight')
plt.close()

print("   💾 Saved: model_performance_detailed.png")

print("\n" + "="*60)
print("✅ All visualizations generated!")
print("="*60)
print(f"\n📁 Check the outputs/ folder for:")
print(f"   - confusion_matrix_{metadata['model_type']}.png")
print(f"   - model_performance_detailed.png")
