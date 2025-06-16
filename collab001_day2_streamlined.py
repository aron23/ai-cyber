#!/usr/bin/env python3
"""
COLLAB-001 Day 2: Streamlined Advanced Ensemble Optimization
SMS Spam Detection Project - Strategic Excellence Phase

Author: Data Scientist (AI-Enhanced)  
Date: 16/06/2025 08:00:00
Focus: Real model integration with XGBoost and LightGBM for 95%+ F1-Score
"""

import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
from scipy import sparse
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from scipy.optimize import minimize

print("="*80)
print("🚀 COLLAB-001 DAY 2: STREAMLINED ADVANCED ENSEMBLE OPTIMIZATION")
print("="*80)
print(f"📅 Day 2 Start: 16/06/2025 08:00:00")
print(f"🎯 Mission: Achieve confirmed 95%+ F1-Score with real models")
print(f"⚡ Focus: XGBoost + LightGBM ensemble optimization")
print()

class COLLAB001Day2Streamlined:
    """Streamlined ensemble optimization with real models"""
    
    def __init__(self, target_f1=0.95, baseline_f1=0.9467):
        self.target_f1 = target_f1
        self.baseline_f1 = baseline_f1
        self.start_time = time.time()
        
        self.base_models = {}
        
        print(f"🎯 COLLAB-001 Day 2 Streamlined Initialized")
        print(f"   Target F1-Score: {target_f1:.1%}")
        print(f"   Baseline to Beat: {baseline_f1:.2%}")
        
    def load_real_data(self):
        """Load actual processed data with proper type handling"""
        print("\n📊 LOADING REAL TRAINING DATA")
        print("-" * 50)
        
        try:
            # Load standardized features
            self.X_train = np.load('data/features/train_features_standard.npy')
            
            # Load sparse validation/test features
            val_sparse = sparse.load_npz('data/features/val_features.npz')
            test_sparse = sparse.load_npz('data/features/test_features.npz')
            
            self.X_val = val_sparse.toarray()
            self.X_test = test_sparse.toarray()
            
            # Load labels with proper encoding
            train_df = pd.read_csv('data/processed/train.csv')
            val_df = pd.read_csv('data/processed/validation.csv')
            test_df = pd.read_csv('data/processed/test.csv')
            
            # Convert labels to binary (0=ham, 1=spam)
            def encode_labels(labels):
                return (labels == 1).astype(int) if labels.dtype != 'object' else (labels == 'spam').astype(int)
            
            self.y_train = encode_labels(train_df['label'].values)
            self.y_val = encode_labels(val_df['label'].values)
            self.y_test = encode_labels(test_df['label'].values)
            
            print(f"✅ Real Data Loaded Successfully:")
            print(f"   Training: {self.X_train.shape[0]:,} samples, {self.X_train.shape[1]:,} features")
            print(f"   Validation: {self.X_val.shape[0]:,} samples")
            print(f"   Test: {self.X_test.shape[0]:,} samples")
            
            # Class distribution
            train_spam_rate = self.y_train.mean() * 100
            val_spam_rate = self.y_val.mean() * 100
            
            print(f"\n📊 Class Distribution:")
            print(f"   Training: {train_spam_rate:.1f}% spam")
            print(f"   Validation: {val_spam_rate:.1f}% spam")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def load_real_models(self):
        """Load actual trained XGBoost and LightGBM models"""
        print("\n🧠 LOADING REAL TRAINED MODELS")
        print("-" * 50)
        
        models_dir = Path('models')
        loaded_models = {}
        
        try:
            # Load XGBoost model
            print("⚡ Loading XGBoost model...")
            xgb_files = list(models_dir.glob('xgboost_advanced_*.joblib'))
            if xgb_files:
                latest_xgb = max(xgb_files, key=os.path.getctime)
                xgb_model = joblib.load(latest_xgb)
                loaded_models['xgboost'] = {
                    'model': xgb_model,
                    'file': latest_xgb.name,
                    'expected_f1': 0.8904
                }
                print(f"   ✅ Loaded: {latest_xgb.name}")
            
            # Load LightGBM model  
            print("🌟 Loading LightGBM model...")
            lgb_files = list(models_dir.glob('lightgbm_optimized_*.joblib'))
            if lgb_files:
                latest_lgb = max(lgb_files, key=os.path.getctime)
                lgb_model = joblib.load(latest_lgb)
                loaded_models['lightgbm'] = {
                    'model': lgb_model,
                    'file': latest_lgb.name,
                    'expected_f1': 0.8993
                }
                print(f"   ✅ Loaded: {latest_lgb.name}")
            
            print(f"\n✅ Real Models Loaded: {len(loaded_models)}")
            self.base_models = loaded_models
            return len(loaded_models) >= 2
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def evaluate_base_models(self):
        """Evaluate base models on validation set"""
        print("\n📊 EVALUATING BASE MODELS")
        print("-" * 50)
        
        base_results = {}
        
        for name, model_info in self.base_models.items():
            model = model_info['model']
            
            print(f"📈 Evaluating {name.title()}...")
            
            try:
                y_pred = model.predict(self.X_val)
                y_prob = model.predict_proba(self.X_val)[:, 1]
                
                f1 = f1_score(self.y_val, y_pred)
                precision = precision_score(self.y_val, y_pred)
                recall = recall_score(self.y_val, y_pred)
                auc = roc_auc_score(self.y_val, y_prob)
                
                base_results[name] = {
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'auc_roc': auc,
                    'predictions': y_pred,
                    'probabilities': y_prob
                }
                
                print(f"   F1: {f1:.4f} ({f1*100:.2f}%)")
                print(f"   Precision: {precision:.4f}, Recall: {recall:.4f}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        return base_results
    
    def optimized_weighted_voting(self, base_results):
        """Optimized weighted voting ensemble"""
        print("\n⚖️  OPTIMIZED WEIGHTED VOTING")
        print("-" * 50)
        
        if len(base_results) < 2:
            return None
        
        model_names = list(base_results.keys())
        prob_matrix = np.column_stack([base_results[name]['probabilities'] for name in model_names])
        
        def objective(weights):
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            ensemble_prob = np.dot(prob_matrix, weights)
            ensemble_pred = (ensemble_prob >= 0.5).astype(int)
            
            f1 = f1_score(self.y_val, ensemble_pred)
            return -f1
        
        print("🔍 Optimizing ensemble weights...")
        initial_weights = np.ones(len(model_names)) / len(model_names)
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
        bounds = [(0, 1) for _ in model_names]
        
        try:
            result = minimize(objective, initial_weights, method='SLSQP', 
                             bounds=bounds, constraints=constraints)
            optimal_weights = result.x / result.x.sum()
        except:
            print("   ⚠️ Optimization failed, using equal weights")
            optimal_weights = initial_weights
        
        # Apply optimal weights
        ensemble_prob = np.dot(prob_matrix, optimal_weights)
        ensemble_pred = (ensemble_prob >= 0.5).astype(int)
        
        f1 = f1_score(self.y_val, ensemble_pred)
        precision = precision_score(self.y_val, ensemble_pred)
        recall = recall_score(self.y_val, ensemble_pred)
        auc = roc_auc_score(self.y_val, ensemble_prob)
        
        improvement = ((f1 - self.baseline_f1) / self.baseline_f1) * 100
        
        results = {
            'method': 'Optimized Weighted Voting',
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'auc_roc': auc,
            'weights': dict(zip(model_names, optimal_weights)),
            'improvement_over_baseline': improvement
        }
        
        print(f"📊 Results: F1={f1:.4f} ({f1*100:.2f}%)")
        print(f"🏋️ Weights: {dict(zip(model_names, [f'{w:.3f}' for w in optimal_weights]))}")
        
        target_achieved = f1 >= self.target_f1
        print(f"🎯 Target (≥95.0%): {'✅ ACHIEVED' if target_achieved else '❌ NOT ACHIEVED'}")
        
        return results
    
    def stacking_ensemble(self, base_results):
        """Stacking ensemble with meta-learner"""
        print("\n🏗️  STACKING ENSEMBLE")
        print("-" * 50)
        
        if len(base_results) < 2:
            return None
        
        model_names = list(base_results.keys())
        
        # Meta-features: probabilities + predictions
        meta_features = []
        for name in model_names:
            meta_features.append(base_results[name]['probabilities'])
            meta_features.append(base_results[name]['predictions'].astype(float))
        
        X_meta = np.column_stack(meta_features)
        
        print(f"📊 Meta-features: {X_meta.shape}")
        
        # Cross-validation training
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        meta_predictions = np.zeros(len(self.y_val))
        meta_probabilities = np.zeros(len(self.y_val))
        
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_meta, self.y_val)):
            X_train_fold = X_meta[train_idx]
            X_val_fold = X_meta[val_idx]
            y_train_fold = self.y_val[train_idx]
            
            meta_learner = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
            meta_learner.fit(X_train_fold, y_train_fold)
            
            fold_pred = meta_learner.predict(X_val_fold)
            fold_prob = meta_learner.predict_proba(X_val_fold)[:, 1]
            
            meta_predictions[val_idx] = fold_pred
            meta_probabilities[val_idx] = fold_prob
            
            fold_f1 = f1_score(self.y_val[val_idx], fold_pred)
            fold_scores.append(fold_f1)
            
            print(f"   Fold {fold+1}: F1={fold_f1:.4f}")
        
        f1 = f1_score(self.y_val, meta_predictions)
        precision = precision_score(self.y_val, meta_predictions)
        recall = recall_score(self.y_val, meta_predictions)
        auc = roc_auc_score(self.y_val, meta_probabilities)
        
        improvement = ((f1 - self.baseline_f1) / self.baseline_f1) * 100
        
        results = {
            'method': 'Stacking',
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'auc_roc': auc,
            'cv_mean_f1': np.mean(fold_scores),
            'improvement_over_baseline': improvement
        }
        
        print(f"\n📊 Results: F1={f1:.4f} ({f1*100:.2f}%)")
        print(f"   CV Mean: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
        
        target_achieved = f1 >= self.target_f1
        print(f"🎯 Target (≥95.0%): {'✅ ACHIEVED' if target_achieved else '❌ NOT ACHIEVED'}")
        
        return results
    
    def run_day2_optimization(self):
        """Run complete Day 2 optimization"""
        
        print("\n" + "="*80)
        print("🚀 OPTIMIZATION PIPELINE")
        print("="*80)
        
        # Load data and models
        if not self.load_real_data():
            return False
            
        if not self.load_real_models():
            return False
        
        # Evaluate base models
        base_results = self.evaluate_base_models()
        if len(base_results) < 2:
            print("❌ Need at least 2 models for ensemble")
            return False
        
        # Ensemble methods
        ensemble_results = {}
        
        weighted_results = self.optimized_weighted_voting(base_results)
        if weighted_results:
            ensemble_results['weighted_voting'] = weighted_results
        
        stacking_results = self.stacking_ensemble(base_results)
        if stacking_results:
            ensemble_results['stacking'] = stacking_results
        
        # Analysis
        success = self.analyze_results(base_results, ensemble_results)
        return success
    
    def analyze_results(self, base_results, ensemble_results):
        """Analyze all results"""
        print("\n" + "="*80)
        print("📊 DAY 2 RESULTS ANALYSIS")
        print("="*80)
        
        all_results = []
        
        # Base models
        for name, results in base_results.items():
            all_results.append({
                'Method': f"{name.title()}",
                'F1-Score': results['f1_score'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'Type': 'Base'
            })
        
        # Ensemble methods
        for name, results in ensemble_results.items():
            all_results.append({
                'Method': results['method'],
                'F1-Score': results['f1_score'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'Type': 'Ensemble'
            })
        
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('F1-Score', ascending=False)
        
        print("📊 PERFORMANCE RANKING:")
        print(results_df.to_string(index=False, float_format='%.4f'))
        
        # Best method
        best_method = results_df.iloc[0]
        print(f"\n🏆 BEST METHOD: {best_method['Method']}")
        print(f"   F1-Score: {best_method['F1-Score']:.4f} ({best_method['F1-Score']*100:.2f}%)")
        
        # Target achievement
        target_achieved = best_method['F1-Score'] >= self.target_f1
        print(f"\n🎯 TARGET ACHIEVEMENT:")
        print(f"   F1 ≥ 95.0%: {'✅ ACHIEVED' if target_achieved else '❌ NOT ACHIEVED'}")
        
        if not target_achieved:
            gap = (self.target_f1 - best_method['F1-Score']) * 100
            print(f"   Gap to target: {gap:.1f} percentage points")
        
        # Performance progression
        print(f"\n📈 PROGRESSION:")
        print(f"   Baseline: {self.baseline_f1:.2%}")
        print(f"   Best Day 2: {best_method['F1-Score']:.2%}")
        print(f"   Improvement: {(best_method['F1-Score'] - self.baseline_f1)*100:+.2f} points")
        
        # Save results
        self.save_results(results_df, best_method, target_achieved)
        
        duration = (time.time() - self.start_time) / 60
        print(f"\n✅ DAY 2 COMPLETE: {duration:.1f} minutes")
        print(f"   Status: {'SUCCESS ✅' if target_achieved else 'PARTIAL SUCCESS ⚠️'}")
        
        return target_achieved
    
    def save_results(self, results_df, best_method, target_achieved):
        """Save Day 2 results"""
        timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
        
        results_summary = {
            'timestamp': timestamp,
            'phase': 'COLLAB-001 Day 2 Streamlined',
            'duration_minutes': (time.time() - self.start_time) / 60,
            'baseline_f1': self.baseline_f1,
            'target_f1': self.target_f1,
            'best_method': {
                'name': best_method['Method'],
                'f1_score': best_method['F1-Score'],
                'precision': best_method['Precision'],
                'recall': best_method['Recall']
            },
            'target_achieved': target_achieved
        }
        
        results_path = f'models/collab001_day2_streamlined_results_{timestamp}.json'
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        print(f"💾 Results saved: {results_path}")

def main():
    """Execute COLLAB-001 Day 2"""
    
    try:
        optimizer = COLLAB001Day2Streamlined()
        success = optimizer.run_day2_optimization()
        
        if success:
            print("\n🌟 DAY 2 SUCCESSFULLY COMPLETED!")
            print("📋 Ready for Day 3: Final Refinement")
        
        return success
        
    except Exception as e:
        print(f"\n❌ Error in Day 2: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
