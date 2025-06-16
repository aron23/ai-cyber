#!/usr/bin/env python3
"""
Day 3: Neural Network Development - CPU Optimized
Target: 93-96% F1-Score building on exceptional 92.6% baseline
Uses scikit-learn MLPClassifier for CPU-optimized training
"""
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import ML libraries
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

print('=== DAY 3: NEURAL NETWORK DEVELOPMENT (CPU OPTIMIZED) ===')
print('Deadline: 18/06/2025 18:00 (HIGH PRIORITY)')
print('Target: 93-96% F1-Score with CPU-optimized neural networks')
print('Foundation: Building on exceptional 92.6% SVM baseline')
print('Platform: scikit-learn MLPClassifier (no CUDA required)')
print()

# Set random seeds for reproducibility
np.random.seed(42)

print('STEP 1: Loading Baseline Infrastructure')
try:
    # Load TF-IDF vectorizer and label encoder
    tfidf_vectorizer = joblib.load('models/day2_baselines_corrected/tfidf_vectorizer.joblib')
    label_encoder = joblib.load('models/day2_baselines_corrected/label_encoder.joblib')
    
    # Load clean data
    train_df = pd.read_csv('data/clean/train_clean.csv')
    val_df = pd.read_csv('data/clean/val_clean.csv')
    
    print(f'✓ Training samples: {len(train_df)}')
    print(f'✓ Validation samples: {len(val_df)}')
    print(f'✓ TF-IDF vectorizer loaded')
    print(f'✓ Label encoder loaded')
    
except Exception as e:
    print(f'❌ Error loading infrastructure: {e}')
    exit(1)

print('\nSTEP 2: Feature Engineering & Data Preparation')

# Text preprocessing (same as baseline for consistency)
import re
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'http[s]?://\S+', ' URL ', text)
    text = re.sub(r'\b\d{10,}\b', ' PHONE ', text)   
    text = re.sub(r'[£$]\d+', ' MONEY ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

# Preprocess and transform text
train_texts = train_df['text'].apply(preprocess_text).values
val_texts = val_df['text'].apply(preprocess_text).values

# Transform to TF-IDF features
X_train = tfidf_vectorizer.transform(train_texts)
X_val = tfidf_vectorizer.transform(val_texts)

# Encode labels
y_train = label_encoder.transform(train_df['label'].values)
y_val = label_encoder.transform(val_df['label'].values)

print(f'✓ Training features: {X_train.shape}')
print(f'✓ Validation features: {X_val.shape}')
print(f'✓ Feature sparsity: {1 - X_train.nnz / np.prod(X_train.shape):.3f}')
print(f'✓ Class distribution - Train: Ham={np.sum(y_train==0)}, Spam={np.sum(y_train==1)}')
print(f'✓ Class distribution - Val: Ham={np.sum(y_val==0)}, Spam={np.sum(y_val==1)}')

print('\nSTEP 3: CPU-Optimized Neural Network Architectures')
print('Target Performance:')
print('- Deep MLP Networks: 93-94% F1-Score')
print('- Wide MLP Networks: 93-94% F1-Score')
print('- Regularized Networks: 93-95% F1-Score')
print('- Optimized Ensemble: 94-96% F1-Score')
print()

def create_mlp_configs():
    """Define various MLP configurations optimized for CPU training"""
    configs = {
        'deep_network': {
            'hidden_layer_sizes': (512, 256, 128, 64),
            'alpha': 0.001,
            'learning_rate_init': 0.001,
            'max_iter': 500,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 20,
            'random_state': 42,
            'description': 'Deep 4-layer network'
        },
        'wide_network': {
            'hidden_layer_sizes': (1024, 512),
            'alpha': 0.001,
            'learning_rate_init': 0.001,
            'max_iter': 500,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 20,
            'random_state': 42,
            'description': 'Wide 2-layer network'
        },
        'regularized_network': {
            'hidden_layer_sizes': (256, 128, 64),
            'alpha': 0.01,
            'learning_rate_init': 0.0005,
            'max_iter': 600,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 25,
            'random_state': 42,
            'description': 'Regularized 3-layer network'
        },
        'balanced_network': {
            'hidden_layer_sizes': (400, 200, 100),
            'alpha': 0.005,
            'learning_rate_init': 0.002,
            'max_iter': 400,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 15,
            'random_state': 42,
            'description': 'Balanced architecture'
        }
    }
    return configs

print('STEP 4: Model Training & Evaluation')

mlp_configs = create_mlp_configs()
results = {}
models = {}

print('\n=== TRAINING CPU-OPTIMIZED NEURAL NETWORKS ===')

for config_name, config in mlp_configs.items():
    print(f'\nTraining {config_name.replace("_", " ").title()}...')
    print(f'Description: {config["description"]}')
    
    # Remove description from config for MLPClassifier
    mlp_params = {k: v for k, v in config.items() if k != 'description'}
    
    # Create and train model
    model = MLPClassifier(**mlp_params)
    
    print(f'✓ Architecture: {config["hidden_layer_sizes"]}')
    print(f'✓ Learning rate: {config["learning_rate_init"]}')
    
    # Train model
    print('  Training...', end='', flush=True)
    model.fit(X_train, y_train)
    print(' ✓ Complete')
    
    # Evaluate on validation set
    val_pred = model.predict(X_val)
    
    f1 = f1_score(y_val, val_pred)
    precision = precision_score(y_val, val_pred)
    recall = recall_score(y_val, val_pred)
    
    results[config_name] = {
        'val_f1': f1,
        'val_precision': precision,
        'val_recall': recall,
        'training_iterations': model.n_iter_,
        'hidden_layers': config['hidden_layer_sizes'],
        'alpha': config['alpha'],
        'description': config['description']
    }
    
    models[config_name] = model
    
    print(f'✓ Validation F1: {f1:.3f}')
    print(f'✓ Validation Precision: {precision:.3f}')
    print(f'✓ Validation Recall: {recall:.3f}')
    print(f'✓ Training iterations: {model.n_iter_}')

print('\nSTEP 5: Hyperparameter Optimization')

# Grid search for optimal hyperparameters
param_grid = {
    'hidden_layer_sizes': [
        (400, 200, 100),
        (500, 250),
        (600, 300, 150),
        (512, 256, 128)
    ],
    'alpha': [0.001, 0.005, 0.01],
    'learning_rate_init': [0.001, 0.002, 0.005]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    MLPClassifier(
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42
    ),
    param_grid,
    cv=cv,
    scoring='f1',
    n_jobs=-1,
    verbose=0
)

print('Hyperparameter optimization...', end='', flush=True)
grid_search.fit(X_train, y_train)
print(' ✓ Complete')

best_params = grid_search.best_params_
best_cv_score = grid_search.best_score_

print(f'\nBest hyperparameters:')
print(f'  Hidden layers: {best_params["hidden_layer_sizes"]}')
print(f'  Alpha: {best_params["alpha"]}')
print(f'  Learning rate: {best_params["learning_rate_init"]}')
print(f'  CV F1-Score: {best_cv_score:.3f}')

# Train final optimized model
optimized_model = MLPClassifier(
    hidden_layer_sizes=best_params['hidden_layer_sizes'],
    alpha=best_params['alpha'],
    learning_rate_init=best_params['learning_rate_init'],
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42
)

print('\nTraining optimized model...', end='', flush=True)
optimized_model.fit(X_train, y_train)
print(' ✓ Complete')

opt_val_pred = optimized_model.predict(X_val)
opt_f1 = f1_score(y_val, opt_val_pred)
opt_precision = precision_score(y_val, opt_val_pred)
opt_recall = recall_score(y_val, opt_val_pred)

results['optimized_final'] = {
    'val_f1': opt_f1,
    'val_precision': opt_precision,
    'val_recall': opt_recall,
    'training_iterations': optimized_model.n_iter_,
    'hidden_layers': best_params['hidden_layer_sizes'],
    'alpha': best_params['alpha'],
    'description': 'Hyperparameter-optimized final model'
}

models['optimized_final'] = optimized_model

print(f'✓ Optimized F1: {opt_f1:.3f}')

print('\nSTEP 6: Results Summary')
print('=' * 75)
print(f'{"Model":<20} {"Val F1":<10} {"Precision":<10} {"Recall":<10} {"Target":<15}')
print('=' * 75)

targets = {
    'deep_network': '93-94%',
    'wide_network': '93-94%',
    'regularized_network': '93-95%',
    'balanced_network': '93-94%',
    'optimized_final': '94-96%'
}

for model_name, metrics in results.items():
    target = targets.get(model_name, 'N/A')
    print(f'{model_name:<20} {metrics["val_f1"]:.3f}      {metrics["val_precision"]:.3f}      {metrics["val_recall"]:.3f}      {target}')

print('=' * 75)

best_model_name = max(results.keys(), key=lambda x: results[x]['val_f1'])
best_f1 = results[best_model_name]['val_f1']

print(f'\n🎯 Best Model: {best_model_name.upper().replace("_", " ")} (F1: {best_f1:.3f})')

print('\nSTEP 7: Detailed Analysis')
best_metrics = results[best_model_name]
print(f'Model: {best_model_name}')
print(f'F1-Score: {best_metrics["val_f1"]:.3f}')
print(f'Precision: {best_metrics["val_precision"]:.3f}')
print(f'Recall: {best_metrics["val_recall"]:.3f}')
print(f'Architecture: {best_metrics["hidden_layers"]}')

if best_model_name in models:
    best_model = models[best_model_name]
    val_pred = best_model.predict(X_val)
    
    print('\nClassification Report:')
    print(classification_report(y_val, val_pred, target_names=['ham', 'spam']))
    
    print('\nConfusion Matrix:')
    cm = confusion_matrix(y_val, val_pred)
    print(f'         ham  spam')
    print(f'ham      {cm[0,0]:3d}  {cm[0,1]:3d}')
    print(f'spam     {cm[1,0]:3d}  {cm[1,1]:3d}')
    
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    print(f'\nBusiness Metrics:')
    print(f'False Positive Rate: {fpr:.3f} ({fpr*100:.1f}%)')
    print(f'False Negative Rate: {fnr:.3f} ({fnr*100:.1f}%)')

print('\nSTEP 8: Model Persistence')
os.makedirs('models/day3_neural_networks_cpu', exist_ok=True)

saved_models = []
for model_name, model in models.items():
    if results[model_name]['val_f1'] > 0.85:
        model_path = f'models/day3_neural_networks_cpu/{model_name}_model.joblib'
        joblib.dump(model, model_path)
        saved_models.append(model_name)
        print(f'✓ Saved {model_name} (F1: {results[model_name]["val_f1"]:.3f})')

# Save results
results_summary = {
    'timestamp': datetime.now().isoformat(),
    'day': 3,
    'phase': 'neural_networks_cpu',
    'best_model': best_model_name,
    'best_f1_score': best_f1,
    'baseline_comparison': {
        'baseline_svm_f1': 0.926,
        'best_neural_f1': best_f1,
        'improvement': best_f1 - 0.926
    },
    'optimization': best_params,
    'models_saved': saved_models
}

with open('models/day3_neural_network_cpu_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print('\nSTEP 9: Performance Comparison')
baseline_svm_f1 = 0.926
improvement = best_f1 - baseline_svm_f1
print(f'Baseline SVM: {baseline_svm_f1:.3f}')
print(f'Best Neural Network: {best_f1:.3f}')
print(f'Improvement: {improvement:+.3f} ({improvement*100:+.1f}%)')

if best_f1 >= 0.93:
    print('🎯 TARGET ACHIEVED: Exceeded 93% F1-Score!')
elif best_f1 >= baseline_svm_f1:
    print('✅ IMPROVEMENT over baseline!')
else:
    print('📊 Baseline remains competitive')

print('\n✅ DAY 3 CPU NEURAL NETWORKS COMPLETE!')
print(f'Best: {best_model_name.upper().replace("_", " ")} (F1: {best_f1:.3f})')
print('Next: Day 4 Ensemble Methods (Target: 94-97%)')

# Create note
timestamp = datetime.now().strftime("%d%m%Y-%H%M%S")
note_content = f"""# Day 3 CPU Neural Networks Complete

**Date**: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}
**Best Model**: {best_model_name} 
**Best F1**: {best_f1:.3f}
**Improvement**: {improvement:+.3f} vs baseline

## Models Trained
"""
for name, metrics in results.items():
    note_content += f"- {name}: {metrics['val_f1']:.3f} F1\n"

note_content += f"""
## Optimization Results
- Best architecture: {best_params['hidden_layer_sizes']}
- Best alpha: {best_params['alpha']}
- CV score: {best_cv_score:.3f}

## Next: Day 4 Ensemble Methods
"""

os.makedirs('notes', exist_ok=True)
with open(f'notes/{timestamp}-day3-cpu-neural-networks.md', 'w') as f:
    f.write(note_content)

print(f'📝 Note saved: {timestamp}-day3-cpu-neural-networks.md') 