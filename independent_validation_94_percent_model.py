#!/usr/bin/env python3
"""
Independent Validation - 94.12% F1-Score Neural Network Ensemble
Validating our successful ensemble on completely independent Dataset_5971.csv
Goal: Confirm 94%+ performance on real-world unseen data
"""

import warnings
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, 
    classification_report, confusion_matrix, roc_auc_score
)
import json

warnings.filterwarnings('ignore')

print("🧪 INDEPENDENT VALIDATION - 94.12% F1-Score MODEL")
print("=" * 70)
print(f"📅 Started: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print("🎯 Goal: Validate 94.12% ensemble on independent Dataset_5971.csv")
print("🔬 Model: Neural Network + Logistic Regression Ensemble")
print("=" * 70)
print()

def load_best_ensemble_and_vectorizer():
    """Load our best 94.12% F1-Score ensemble and vectorizer"""
    print("📦 LOADING BEST ENSEMBLE MODEL")
    print("-" * 40)
    
    # Load the best ensemble (94.12% F1-Score)
    ensemble_path = Path("ensemble_models/neural_ensemble_best_16062025_130235.joblib")
    if not ensemble_path.exists():
        print(f"❌ Best ensemble not found: {ensemble_path}")
        return None, None
    
    ensemble = joblib.load(ensemble_path)
    print(f"✅ Best ensemble loaded: {ensemble_path}")
    
    # Load the vectorizer
    vectorizer_path = Path("models/tfidf_vectorizer_v1.0.0.joblib")
    vectorizer = joblib.load(vectorizer_path)
    print(f"✅ TF-IDF Vectorizer loaded: {len(vectorizer.vocabulary_):,} features")
    
    print(f"📊 Ensemble details:")
    print(f"   Method: {type(ensemble).__name__}")
    print(f"   Base estimators: {len(ensemble.estimators_)}")
    print(f"   Meta-learner: {type(ensemble.final_estimator_).__name__}")
    print()
    
    return ensemble, vectorizer

def load_independent_dataset():
    """Load and prepare the independent validation dataset"""
    print("📊 LOADING INDEPENDENT DATASET")
    print("-" * 40)
    
    # Load Dataset_5971.csv
    dataset_path = Path("data/Dataset_5971.csv")
    if not dataset_path.exists():
        print(f"❌ Independent dataset not found: {dataset_path}")
        return None, None
    
    df = pd.read_csv(dataset_path)
    print(f"✅ Independent dataset loaded: {len(df):,} samples")
    
    # Check dataset structure
    print(f"📋 Columns: {list(df.columns)}")
    print(f"📊 Label distribution:")
    for label, count in df['LABEL'].value_counts().items():
        print(f"   {label}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Prepare data
    # Use TEXT column as message content
    X_independent = df['TEXT'].fillna('').astype(str)
    
    # Map labels to binary (ham=0, all types of spam=1)
    spam_labels = ['spam', 'Spam', 'Smishing']
    y_independent = df['LABEL'].apply(lambda x: 1 if x in spam_labels else 0)
    
    print(f"\n📈 Binary classification mapping:")
    print(f"   Ham (0): {sum(y_independent == 0):,} samples")
    print(f"   Spam (1): {sum(y_independent == 1):,} samples")
    print(f"   Spam rate: {sum(y_independent == 1)/len(y_independent)*100:.1f}%")
    print()
    
    return X_independent, y_independent

def validate_ensemble_performance(ensemble, vectorizer, X_independent, y_independent):
    """Comprehensive validation of ensemble performance"""
    print("🎯 VALIDATING ENSEMBLE PERFORMANCE")
    print("-" * 50)
    
    # Transform text data
    print("🔄 Transforming text data...")
    X_independent_vec = vectorizer.transform(X_independent)
    print(f"✅ Transformed: {X_independent_vec.shape} features")
    
    # Make predictions
    print("🧠 Making predictions...")
    start_time = datetime.now()
    y_pred = ensemble.predict(X_independent_vec)
    prediction_time = (datetime.now() - start_time).total_seconds()
    
    # Get prediction probabilities
    try:
        y_proba = ensemble.predict_proba(X_independent_vec)[:, 1]
        has_proba = True
    except:
        y_proba = None
        has_proba = False
    
    print(f"✅ Predictions completed in {prediction_time:.2f} seconds")
    print(f"⚡ Average prediction time: {prediction_time/len(y_independent)*1000:.2f} ms per sample")
    
    # Calculate comprehensive metrics
    print("\n📊 CALCULATING PERFORMANCE METRICS")
    print("-" * 40)
    
    # Basic metrics
    f1 = f1_score(y_independent, y_pred, average='binary', pos_label=1)
    precision = precision_score(y_independent, y_pred, average='binary', pos_label=1)
    recall = recall_score(y_independent, y_pred, average='binary', pos_label=1)
    accuracy = accuracy_score(y_independent, y_pred)
    
    # Additional metrics
    if has_proba:
        try:
            auc_roc = roc_auc_score(y_independent, y_proba)
        except:
            auc_roc = None
    else:
        auc_roc = None
    
    # Confusion matrix
    cm = confusion_matrix(y_independent, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Business metrics
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Results summary
    results = {
        "validation_metrics": {
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "auc_roc": auc_roc,
            "specificity": specificity
        },
        "business_metrics": {
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
            "spam_catch_rate": recall,
            "ham_accuracy": specificity
        },
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        },
        "performance_stats": {
            "total_samples": len(y_independent),
            "prediction_time_seconds": prediction_time,
            "predictions_per_second": len(y_independent) / prediction_time,
            "has_probabilities": has_proba
        }
    }
    
    return results, y_pred, y_proba

def display_comprehensive_results(results, target_f1=0.94):
    """Display comprehensive validation results"""
    print("🎉 INDEPENDENT VALIDATION RESULTS")
    print("=" * 60)
    
    metrics = results["validation_metrics"]
    business = results["business_metrics"]
    cm = results["confusion_matrix"]
    perf = results["performance_stats"]
    
    # Main performance metrics
    print("📊 CORE PERFORMANCE METRICS:")
    print(f"   🏆 F1-Score: {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print(f"   🎯 Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"   📈 Recall: {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"   ✅ Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    if metrics['auc_roc']:
        print(f"   📈 AUC-ROC: {metrics['auc_roc']:.4f}")
    
    # Target comparison
    print(f"\n🎯 TARGET COMPARISON:")
    print(f"   Target F1-Score: {target_f1*100:.2f}%")
    print(f"   Achieved F1-Score: {metrics['f1_score']*100:.2f}%")
    improvement = (metrics['f1_score'] - target_f1) * 100
    print(f"   {'✅ EXCEEDED' if improvement >= 0 else '❌ BELOW'} by {improvement:+.2f} percentage points")
    
    # Business impact metrics
    print(f"\n💼 BUSINESS IMPACT METRICS:")
    print(f"   📧 Spam catch rate: {business['spam_catch_rate']*100:.2f}% (how much spam detected)")
    print(f"   ✉️ Ham accuracy: {business['ham_accuracy']*100:.2f}% (how much ham preserved)")
    print(f"   🚫 False positive rate: {business['false_positive_rate']*100:.2f}% (ham marked as spam)")
    print(f"   ⚠️ False negative rate: {business['false_negative_rate']*100:.2f}% (spam marked as ham)")
    
    # User experience analysis
    print(f"\n👤 USER EXPERIENCE ANALYSIS:")
    user_satisfaction = "EXCELLENT" if business['false_positive_rate'] < 0.05 else "GOOD" if business['false_positive_rate'] < 0.10 else "FAIR"
    security_level = "HIGH" if business['spam_catch_rate'] > 0.85 else "MEDIUM" if business['spam_catch_rate'] > 0.75 else "LOW"
    print(f"   😊 User satisfaction: {user_satisfaction} (<{business['false_positive_rate']*100:.1f}% false positives)")
    print(f"   🛡️ Security level: {security_level} ({business['spam_catch_rate']*100:.1f}% spam caught)")
    
    # Confusion matrix
    print(f"\n📋 CONFUSION MATRIX:")
    print(f"              Predicted")
    print(f"           Ham    Spam   Total")
    print(f"    Ham   {cm['true_negative']:4d}   {cm['false_positive']:4d}   {cm['true_negative']+cm['false_positive']:4d}")
    print(f"   Spam   {cm['false_negative']:4d}   {cm['true_positive']:4d}   {cm['false_negative']+cm['true_positive']:4d}")
    print(f"  Total   {cm['true_negative']+cm['false_negative']:4d}   {cm['false_positive']+cm['true_positive']:4d}   {perf['total_samples']:4d}")
    
    # Performance statistics
    print(f"\n⚡ PERFORMANCE STATISTICS:")
    print(f"   📊 Total samples: {perf['total_samples']:,}")
    print(f"   ⏱️ Total time: {perf['prediction_time_seconds']:.2f} seconds")
    print(f"   🚀 Throughput: {perf['predictions_per_second']:.0f} predictions/second")
    print(f"   📈 Probabilities: {'✅ Available' if perf['has_probabilities'] else '❌ Not available'}")

def compare_with_validation_performance():
    """Compare independent results with validation performance"""
    print("\n🔍 COMPARISON WITH VALIDATION PERFORMANCE")
    print("-" * 50)
    
    validation_f1 = 0.9412  # Our validation F1-Score
    
    print(f"📊 Performance Comparison:")
    print(f"   Validation F1-Score: {validation_f1*100:.2f}%")
    print(f"   Independent F1-Score: Will be displayed above")
    print(f"   Expected range: {(validation_f1-0.02)*100:.2f}% - {(validation_f1+0.02)*100:.2f}%")
    print()
    print("🔬 Generalization Analysis:")
    print("   ✅ Consistent performance indicates good generalization")
    print("   ⚠️ Large difference may indicate overfitting")
    print("   📈 Better performance suggests robust model")

def save_validation_results(results, X_independent, y_independent, y_pred):
    """Save comprehensive validation results"""
    print("\n💾 SAVING VALIDATION RESULTS")
    print("-" * 40)
    
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    
    # Add metadata to results
    full_results = {
        "timestamp": timestamp,
        "validation_type": "independent_dataset",
        "dataset": "Dataset_5971.csv",
        "model": "neural_network_logistic_regression_ensemble",
        "validation_f1_baseline": 0.9412,
        **results,
        "sample_predictions": {
            "total_samples": len(y_independent),
            "correct_predictions": int(sum(y_pred == y_independent)),
            "accuracy_check": sum(y_pred == y_independent) / len(y_independent)
        }
    }
    
    # Save detailed results
    results_path = Path("ensemble_models") / f"independent_validation_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"✅ Validation results saved: {results_path}")
    
    # Save sample predictions for analysis
    predictions_df = pd.DataFrame({
        'message': X_independent.head(100),  # First 100 samples
        'true_label': y_independent.head(100),
        'predicted_label': y_pred[:100],
        'correct': (y_independent.head(100) == y_pred[:100])
    })
    
    predictions_path = Path("ensemble_models") / f"sample_predictions_{timestamp}.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"✅ Sample predictions saved: {predictions_path}")
    
    return full_results

def main():
    """Main execution for independent validation"""
    
    # Step 1: Load best ensemble and vectorizer
    ensemble, vectorizer = load_best_ensemble_and_vectorizer()
    if ensemble is None:
        print("❌ CRITICAL: Cannot load best ensemble model")
        return False
    
    # Step 2: Load independent dataset
    X_independent, y_independent = load_independent_dataset()
    if X_independent is None:
        print("❌ CRITICAL: Cannot load independent dataset")
        return False
    
    # Step 3: Validate performance
    results, y_pred, y_proba = validate_ensemble_performance(
        ensemble, vectorizer, X_independent, y_independent
    )
    
    # Step 4: Display comprehensive results
    display_comprehensive_results(results)
    
    # Step 5: Compare with validation performance
    compare_with_validation_performance()
    
    # Step 6: Save results
    full_results = save_validation_results(results, X_independent, y_independent, y_pred)
    
    # Final summary
    f1_score_achieved = results["validation_metrics"]["f1_score"]
    target_achieved = f1_score_achieved >= 0.94
    
    print("\n🎊 INDEPENDENT VALIDATION COMPLETED!")
    print("=" * 60)
    print(f"🏆 F1-Score Achieved: {f1_score_achieved*100:.2f}%")
    print(f"🎯 94% Target: {'✅ ACHIEVED' if target_achieved else '❌ NOT ACHIEVED'}")
    print(f"📊 Dataset: {len(y_independent):,} independent samples")
    print(f"🔬 Model: Neural Network + Logistic Regression Ensemble")
    
    return target_achieved

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 SUCCESS: Independent validation confirms 94%+ performance!")
    else:
        print("\n📊 ANALYSIS: Review independent validation results") 