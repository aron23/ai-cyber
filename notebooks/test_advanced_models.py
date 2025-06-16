#!/usr/bin/env python3
"""
DS-005 Advanced Models Testing Script
Quick test of XGBoost and LightGBM models
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import joblib
from datetime import datetime

def main():
    print(f"🚀 DS-005 Advanced Models Test Started: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    
    try:
        # Load data
        print("📁 Loading processed data...")
        train_df = pd.read_csv('../data/processed/train.csv')
        test_df = pd.read_csv('../data/processed/test.csv')
        
        print(f"📈 Training data shape: {train_df.shape}")
        print(f"📈 Test data shape: {test_df.shape}")
        
        # Check class distribution
        print(f"\n📊 Class Distribution: {train_df['label'].value_counts().to_dict()}")
        class_counts = train_df['label_encoded'].value_counts()
        class_weight_ratio = class_counts[0] / class_counts[1]
        print(f"⚖️ Class weight ratio (Ham:Spam): {class_weight_ratio:.2f}:1")
        
        # Load TF-IDF and prepare features
        print("🔄 Loading TF-IDF vectorizer...")
        tfidf_vectorizer = joblib.load('../models/tfidf_vectorizer_v1.0.0.joblib')
        
        # Use text_aggressive column for features
        X_train = tfidf_vectorizer.transform(train_df['text_aggressive'])
        X_test = tfidf_vectorizer.transform(test_df['text_aggressive'])
        y_train = train_df['label_encoded']
        y_test = test_df['label_encoded']
        
        print(f"✅ Features prepared: {X_train.shape[1]} TF-IDF features")
        print(f"✅ Training samples: {X_train.shape[0]}")
        print(f"✅ Test samples: {X_test.shape[0]}")
        
        baseline_f1 = 0.4358
        results = []
        
        # XGBoost Model
        print("\n🚀 Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            scale_pos_weight=class_weight_ratio,
            random_state=42,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1
        )
        
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)
        y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
        
        xgb_f1 = f1_score(y_test, y_pred_xgb)
        xgb_precision = precision_score(y_test, y_pred_xgb)
        xgb_recall = recall_score(y_test, y_pred_xgb)
        xgb_auc = roc_auc_score(y_test, y_pred_proba_xgb)
        
        print(f"📊 XGBoost Results:")
        print(f"  F1-Score: {xgb_f1:.4f} ({xgb_f1*100:.2f}%)")
        print(f"  Precision: {xgb_precision:.4f} ({xgb_precision*100:.2f}%)")
        print(f"  Recall: {xgb_recall:.4f} ({xgb_recall*100:.2f}%)")
        print(f"  AUC-ROC: {xgb_auc:.4f}")
        
        xgb_improvement = ((xgb_f1 - baseline_f1) / baseline_f1) * 100
        print(f"  Improvement over baseline: {xgb_improvement:+.2f}%")
        
        results.append({
            'Model': 'XGBoost',
            'F1': xgb_f1,
            'Precision': xgb_precision, 
            'Recall': xgb_recall,
            'AUC': xgb_auc,
            'Improvement_%': xgb_improvement
        })
        
        # LightGBM Model
        print("\n🚀 Training LightGBM...")
        n_pos = sum(y_train == 1)
        n_neg = sum(y_train == 0)
        lgb_class_weight = {0: 1.0, 1: n_neg/n_pos}
        
        lgb_model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            boosting_type='gbdt',
            class_weight=lgb_class_weight,
            random_state=42,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            n_jobs=-1,
            verbose=-1
        )
        
        lgb_model.fit(X_train, y_train)
        y_pred_lgb = lgb_model.predict(X_test)
        y_pred_proba_lgb = lgb_model.predict_proba(X_test)[:, 1]
        
        lgb_f1 = f1_score(y_test, y_pred_lgb)
        lgb_precision = precision_score(y_test, y_pred_lgb)
        lgb_recall = recall_score(y_test, y_pred_lgb)
        lgb_auc = roc_auc_score(y_test, y_pred_proba_lgb)
        
        print(f"📊 LightGBM Results:")
        print(f"  F1-Score: {lgb_f1:.4f} ({lgb_f1*100:.2f}%)")
        print(f"  Precision: {lgb_precision:.4f} ({lgb_precision*100:.2f}%)")
        print(f"  Recall: {lgb_recall:.4f} ({lgb_recall*100:.2f}%)")
        print(f"  AUC-ROC: {lgb_auc:.4f}")
        
        lgb_improvement = ((lgb_f1 - baseline_f1) / baseline_f1) * 100
        print(f"  Improvement over baseline: {lgb_improvement:+.2f}%")
        
        results.append({
            'Model': 'LightGBM',
            'F1': lgb_f1,
            'Precision': lgb_precision,
            'Recall': lgb_recall,
            'AUC': lgb_auc,
            'Improvement_%': lgb_improvement
        })
        
        # Summary
        print("\n" + "="*60)
        print("📊 COMPARISON SUMMARY")
        print("="*60)
        print(f"{'Model':<12} {'F1':>8} {'Precision':>10} {'Recall':>8} {'AUC':>8} {'Improve':>8}")
        print("-"*60)
        print(f"{'Baseline':<12} {baseline_f1:>8.4f} {'0.2797':>10} {'0.9865':>8} {'-':>8} {'-':>8}")
        
        for result in results:
            print(f"{result['Model']:<12} {result['F1']:>8.4f} {result['Precision']:>10.4f} {result['Recall']:>8.4f} {result['AUC']:>8.4f} {result['Improvement_%']:>7.1f}%")
        
        # Best model analysis
        best_result = max(results, key=lambda x: x['F1'])
        best_f1 = best_result['F1']
        
        print(f"\n🏆 BEST MODEL: {best_result['Model']} (F1={best_f1:.4f})")
        print(f"🎯 TARGET ANALYSIS:")
        print(f"  Minimum Target (F1≥70%): {'✅ ACHIEVED' if best_f1 >= 0.70 else '❌ NOT ACHIEVED'}")
        print(f"  Stretch Target (F1≥90%): {'✅ ACHIEVED' if best_f1 >= 0.90 else '❌ NOT ACHIEVED'}")
        print(f"  Gap to minimum: {max(0, (0.70 - best_f1)*100):.1f} percentage points")
        print(f"  Gap to stretch: {max(0, (0.90 - best_f1)*100):.1f} percentage points")
        
        # Save models if good performance
        timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
        if best_f1 > baseline_f1:
            if best_result['Model'] == 'XGBoost':
                model_path = f'../models/xgboost_advanced_v1.0.0_{timestamp}.joblib'
                joblib.dump(xgb_model, model_path)
                print(f"💾 Best model saved: {model_path}")
            else:
                model_path = f'../models/lightgbm_advanced_v1.0.0_{timestamp}.joblib'
                joblib.dump(lgb_model, model_path)
                print(f"💾 Best model saved: {model_path}")
        
        print(f"\n✅ DS-005 Phase 1 Complete: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        
        return best_f1 >= 0.70
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 