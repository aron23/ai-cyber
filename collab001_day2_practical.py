#!/usr/bin/env python3
"""
COLLAB-001 Day 2: Practical Advanced Ensemble Optimization
SMS Spam Detection Project - Strategic Excellence Phase

Author: Data Scientist (AI-Enhanced)  
Date: 16/06/2025 08:10:00
Focus: Fresh model training + advanced ensemble techniques for 95%+ F1-Score
"""

import os
import time
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from scipy import sparse
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from scipy.optimize import minimize
import xgboost as xgb
import lightgbm as lgb

print("="*80)
print("🚀 COLLAB-001 DAY 2: PRACTICAL ADVANCED ENSEMBLE OPTIMIZATION")
print("="*80)
print(f"📅 Day 2 Start: 16/06/2025 08:10:00")
print(f"🎯 Mission: Achieve confirmed 95%+ F1-Score with advanced ensembles")
print(f"⚡ Strategy: Fresh model training + sophisticated ensemble methods")
print()

class COLLAB001Day2Practical:
    """Practical ensemble optimization with fresh model training"""
    
    def __init__(self, target_f1=0.95, baseline_f1=0.9467):
        self.target_f1 = target_f1
        self.baseline_f1 = baseline_f1
        self.start_time = time.time()
        
        self.base_models = {}
        
        print(f"🎯 COLLAB-001 Day 2 Practical Initialized")
        print(f"   Target F1-Score: {target_f1:.1%}")
        print(f"   Baseline to Beat: {baseline_f1:.2%}")
        
    def load_data(self):
        """Load data with proper encoding"""
        print("\n📊 LOADING TRAINING DATA")
        print("-" * 50)
        
        try:
            # Load features
            train_sparse = sparse.load_npz('data/features/train_features.npz')
            val_sparse = sparse.load_npz('data/features/val_features.npz')
            test_sparse = sparse.load_npz('data/features/test_features.npz')
            
            self.X_train = train_sparse.toarray()
            self.X_val = val_sparse.toarray()
            self.X_test = test_sparse.toarray()
            
            # Load labels
            train_df = pd.read_csv('data/processed/train.csv')
            val_df = pd.read_csv('data/processed/validation.csv')
            test_df = pd.read_csv('data/processed/test.csv')
            
            # Convert labels to binary (0=ham, 1=spam)
            def encode_labels(labels):
                return (labels == 1).astype(int) if labels.dtype != 'object' else (labels == 'spam').astype(int)
            
            self.y_train = encode_labels(train_df['label'].values)
            self.y_val = encode_labels(val_df['label'].values)
            self.y_test = encode_labels(test_df['label'].values)
            
            print(f"✅ Data Loaded Successfully:")
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
    
    def train_base_models(self):
        """Train diverse base models for ensemble"""
        print("\n🧠 TRAINING DIVERSE BASE MODELS")
        print("-" * 50)
        
        trained_models = {}
        
        # 1. XGBoost
        print("⚡ Training XGBoost...")
        try:
            # Calculate class weights
            n_pos = sum(self.y_train == 1)
            n_neg = sum(self.y_train == 0)
            scale_pos_weight = n_neg / n_pos
            
            xgb_model = xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1
            )
            
            xgb_model.fit(self.X_train, self.y_train)
            trained_models['xgboost'] = xgb_model
            print("   ✅ XGBoost trained successfully")
            
        except Exception as e:
            print(f"   ❌ XGBoost training failed: {e}")
        
        # 2. LightGBM
        print("🌟 Training LightGBM...")
        try:
            lgb_model = lgb.LGBMClassifier(
                objective='binary',
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            
            lgb_model.fit(self.X_train, self.y_train)
            trained_models['lightgbm'] = lgb_model
            print("   ✅ LightGBM trained successfully")
            
        except Exception as e:
            print(f"   ❌ LightGBM training failed: {e}")
        
        # 3. Random Forest
        print("🌳 Training Random Forest...")
        try:
            rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            rf_model.fit(self.X_train, self.y_train)
            trained_models['random_forest'] = rf_model
            print("   ✅ Random Forest trained successfully")
            
        except Exception as e:
            print(f"   ❌ Random Forest training failed: {e}")
        
        # 4. Neural Network
        print("🧠 Training Neural Network...")
        try:
            nn_model = MLPClassifier(
                hidden_layer_sizes=(512, 256, 128),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size='auto',
                learning_rate='constant',
                learning_rate_init=0.001,
                max_iter=300,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
            
            nn_model.fit(self.X_train, self.y_train)
            trained_models['neural_network'] = nn_model
            print("   ✅ Neural Network trained successfully")
            
        except Exception as e:
            print(f"   ❌ Neural Network training failed: {e}")
        
        print(f"\n✅ Base Models Trained: {len(trained_models)}")
        self.base_models = trained_models
        return len(trained_models) >= 2
    
    def evaluate_base_models(self):
        """Evaluate all base models"""
        print("\n📊 EVALUATING BASE MODELS")
        print("-" * 50)
        
        base_results = {}
        
        for name, model in self.base_models.items():
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
                
                improvement = ((f1 - self.baseline_f1) / self.baseline_f1) * 100
                
                print(f"   F1: {f1:.4f} ({f1*100:.2f}%)")
                print(f"   Precision: {precision:.4f}, Recall: {recall:.4f}")
                print(f"   vs Baseline: {improvement:+.2f}%")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        return base_results
    
    def advanced_weighted_voting(self, base_results):
        """Advanced weighted voting with optimization"""
        print("\n⚖️  ADVANCED WEIGHTED VOTING")
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
            recall = recall_score(self.y_val, ensemble_pred)
            
            # Penalize if recall too low
            if recall < 0.88:
                return -(f1 * 0.8)  # Penalty for low recall
            
            return -f1
        
        print("🔍 Optimizing ensemble weights with recall constraint...")
        initial_weights = np.array([0.4, 0.3, 0.2, 0.1])[:len(model_names)]  # Favor better models
        initial_weights = initial_weights / initial_weights.sum()
        
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
        bounds = [(0.05, 0.6) for _ in model_names]  # Prevent extreme weights
        
        try:
            result = minimize(objective, initial_weights, method='SLSQP', 
                             bounds=bounds, constraints=constraints)
            optimal_weights = result.x / result.x.sum()
        except:
            print("   ⚠️ Optimization failed, using performance-based weights")
            # Use F1-scores as weights
            f1_scores = [base_results[name]['f1_score'] for name in model_names]
            optimal_weights = np.array(f1_scores) / sum(f1_scores)
        
        # Apply optimal weights
        ensemble_prob = np.dot(prob_matrix, optimal_weights)
        ensemble_pred = (ensemble_prob >= 0.5).astype(int)
        
        f1 = f1_score(self.y_val, ensemble_pred)
        precision = precision_score(self.y_val, ensemble_pred)
        recall = recall_score(self.y_val, ensemble_pred)
        auc = roc_auc_score(self.y_val, ensemble_prob)
        
        improvement = ((f1 - self.baseline_f1) / self.baseline_f1) * 100
        
        results = {
            'method': 'Advanced Weighted Voting',
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'auc_roc': auc,
            'weights': dict(zip(model_names, optimal_weights)),
            'improvement_over_baseline': improvement
        }
        
        print(f"📊 Results: F1={f1:.4f} ({f1*100:.2f}%)")
        print(f"   Precision: {precision:.4f}, Recall: {recall:.4f}")
        print(f"   Improvement: {improvement:+.2f}%")
        
        print(f"\n🏋️ Optimal Weights:")
        for name, weight in zip(model_names, optimal_weights):
            print(f"   {name.title()}: {weight:.3f}")
        
        target_achieved = f1 >= self.target_f1
        recall_ok = recall >= 0.88
        print(f"\n🎯 Target Analysis:")
        print(f"   F1 ≥ 95.0%: {'✅ ACHIEVED' if target_achieved else '❌ NOT ACHIEVED'}")
        print(f"   Recall ≥ 88%: {'✅ MET' if recall_ok else '❌ VIOLATED'}")
        
        return results
    
    def sophisticated_stacking(self, base_results):
        """Sophisticated stacking with enhanced meta-features"""
        print("\n🏗️  SOPHISTICATED STACKING ENSEMBLE")
        print("-" * 50)
        
        if len(base_results) < 2:
            return None
        
        model_names = list(base_results.keys())
        
        # Enhanced meta-features
        meta_features = []
        
        for name in model_names:
            # Probabilities
            meta_features.append(base_results[name]['probabilities'])
            
            # Binary predictions
            meta_features.append(base_results[name]['predictions'].astype(float))
            
            # Confidence scores
            probs = base_results[name]['probabilities']
            confidence = np.abs(probs - 0.5) * 2
            meta_features.append(confidence)
            
            # Calibrated probabilities (sigmoid transformation)
            calibrated_probs = 1 / (1 + np.exp(-5 * (probs - 0.5)))
            meta_features.append(calibrated_probs)
        
        X_meta = np.column_stack(meta_features)
        
        print(f"📊 Enhanced meta-features: {X_meta.shape}")
        print(f"   Features per model: 4 (prob, pred, confidence, calibrated)")
        
        # Stratified cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        meta_predictions = np.zeros(len(self.y_val))
        meta_probabilities = np.zeros(len(self.y_val))
        
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_meta, self.y_val)):
            X_train_fold = X_meta[train_idx]
            X_val_fold = X_meta[val_idx]
            y_train_fold = self.y_val[train_idx]
            
            # Use regularized logistic regression
            meta_learner = LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight='balanced',
                C=0.1,  # Regularization
                penalty='l2'
            )
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
            'method': 'Sophisticated Stacking',
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'auc_roc': auc,
            'cv_mean_f1': np.mean(fold_scores),
            'cv_std_f1': np.std(fold_scores),
            'improvement_over_baseline': improvement
        }
        
        print(f"\n📊 Results: F1={f1:.4f} ({f1*100:.2f}%)")
        print(f"   Precision: {precision:.4f}, Recall: {recall:.4f}")
        print(f"   CV Mean: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
        print(f"   Improvement: {improvement:+.2f}%")
        
        target_achieved = f1 >= self.target_f1
        recall_ok = recall >= 0.88
        print(f"\n🎯 Target Analysis:")
        print(f"   F1 ≥ 95.0%: {'✅ ACHIEVED' if target_achieved else '❌ NOT ACHIEVED'}")
        print(f"   Recall ≥ 88%: {'✅ MET' if recall_ok else '❌ VIOLATED'}")
        
        return results
    
    def run_day2_optimization(self):
        """Run complete Day 2 practical optimization"""
        
        print("\n" + "="*80)
        print("🚀 PRACTICAL OPTIMIZATION PIPELINE")
        print("="*80)
        
        # Load data
        if not self.load_data():
            return False
        
        # Train base models
        if not self.train_base_models():
            return False
        
        # Evaluate base models
        base_results = self.evaluate_base_models()
        if len(base_results) < 2:
            print("❌ Need at least 2 base models for ensemble")
            return False
        
        # Advanced ensemble methods
        ensemble_results = {}
        
        weighted_results = self.advanced_weighted_voting(base_results)
        if weighted_results:
            ensemble_results['advanced_weighted'] = weighted_results
        
        stacking_results = self.sophisticated_stacking(base_results)
        if stacking_results:
            ensemble_results['sophisticated_stacking'] = stacking_results
        
        # Analysis
        success = self.analyze_results(base_results, ensemble_results)
        return success
    
    def analyze_results(self, base_results, ensemble_results):
        """Comprehensive results analysis"""
        print("\n" + "="*80)
        print("📊 COLLAB-001 DAY 2: COMPREHENSIVE RESULTS ANALYSIS")
        print("="*80)
        
        all_results = []
        
        # Base models
        print("📈 BASE MODEL PERFORMANCE:")
        print("-" * 50)
        for name, results in base_results.items():
            all_results.append({
                'Method': f"{name.title().replace('_', ' ')}",
                'F1-Score': results['f1_score'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'AUC-ROC': results['auc_roc'],
                'Type': 'Base Model'
            })
            improvement = ((results['f1_score'] - self.baseline_f1) / self.baseline_f1) * 100
            print(f"   {name.title().replace('_', ' ')}: F1={results['f1_score']:.4f} ({improvement:+.2f}%)")
        
        # Ensemble methods
        print(f"\n🔬 ENSEMBLE METHOD PERFORMANCE:")
        print("-" * 50)
        for name, results in ensemble_results.items():
            all_results.append({
                'Method': results['method'],
                'F1-Score': results['f1_score'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'AUC-ROC': results['auc_roc'],
                'Type': 'Ensemble'
            })
            improvement = results.get('improvement_over_baseline', 0)
            print(f"   {results['method']}: F1={results['f1_score']:.4f} ({improvement:+.2f}%)")
        
        # Create summary DataFrame
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('F1-Score', ascending=False)
        
        print(f"\n📊 COMPLETE PERFORMANCE RANKING:")
        print("-" * 80)
        print(results_df.to_string(index=False, float_format='%.4f'))
        
        # Best method analysis
        best_method = results_df.iloc[0]
        print(f"\n🏆 BEST PERFORMING METHOD:")
        print(f"   Method: {best_method['Method']}")
        print(f"   F1-Score: {best_method['F1-Score']:.4f} ({best_method['F1-Score']*100:.2f}%)")
        print(f"   Precision: {best_method['Precision']:.4f} ({best_method['Precision']*100:.2f}%)")
        print(f"   Recall: {best_method['Recall']:.4f} ({best_method['Recall']*100:.2f}%)")
        print(f"   AUC-ROC: {best_method['AUC-ROC']:.4f}")
        
        # Target achievement analysis
        target_achieved = best_method['F1-Score'] >= self.target_f1
        stretch_achieved = best_method['F1-Score'] >= 0.955
        recall_met = best_method['Recall'] >= 0.88
        
        print(f"\n🎯 FINAL TARGET ACHIEVEMENT ANALYSIS:")
        print(f"   Primary Target (F1≥95.0%): {'✅ ACHIEVED' if target_achieved else '❌ NOT ACHIEVED'}")
        print(f"   Stretch Target (F1≥95.5%): {'✅ ACHIEVED' if stretch_achieved else '❌ NOT ACHIEVED'}")
        print(f"   Recall Constraint (≥88%): {'✅ MET' if recall_met else '❌ VIOLATED'}")
        
        if not target_achieved:
            gap = (self.target_f1 - best_method['F1-Score']) * 100
            print(f"   Gap to primary target: {gap:.1f} percentage points")
        
        # Performance progression
        print(f"\n📈 PERFORMANCE PROGRESSION:")
        print(f"   Baseline (Original): {self.baseline_f1:.2%}")
        print(f"   Best Day 2 Result: {best_method['F1-Score']:.2%}")
        print(f"   Total Improvement: {(best_method['F1-Score'] - self.baseline_f1)*100:+.2f} percentage points")
        
        # Success assessment
        overall_success = target_achieved and recall_met
        
        # Save results
        self.save_results(results_df, best_method, overall_success)
        
        duration = (time.time() - self.start_time) / 60
        print(f"\n✅ COLLAB-001 DAY 2 COMPLETION:")
        print(f"   Duration: {duration:.1f} minutes")
        print(f"   Status: {'FULL SUCCESS ✅' if overall_success else 'PARTIAL SUCCESS ⚠️'}")
        print(f"   Next Phase: {'Day 3 - Production Prep' if overall_success else 'Day 3 - Final Optimization'}")
        
        return overall_success
    
    def save_results(self, results_df, best_method, success):
        """Save Day 2 results"""
        timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
        
        results_summary = {
            'timestamp': timestamp,
            'phase': 'COLLAB-001 Day 2 Practical',
            'duration_minutes': (time.time() - self.start_time) / 60,
            'baseline_f1': self.baseline_f1,
            'target_f1': self.target_f1,
            'models_trained': len(self.base_models),
            'best_method': {
                'name': best_method['Method'],
                'f1_score': best_method['F1-Score'],
                'precision': best_method['Precision'],
                'recall': best_method['Recall'],
                'auc_roc': best_method['AUC-ROC']
            },
            'target_achieved': best_method['F1-Score'] >= self.target_f1,
            'recall_constraint_met': best_method['Recall'] >= 0.88,
            'overall_success': success
        }
        
        results_path = f'models/collab001_day2_practical_results_{timestamp}.json'
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        print(f"\n💾 Day 2 Results saved: {results_path}")
        
        # Save detailed comparison
        csv_path = f'models/collab001_day2_practical_comparison_{timestamp}.csv'
        results_df.to_csv(csv_path, index=False)
        print(f"💾 Detailed comparison saved: {csv_path}")

def main():
    """Execute COLLAB-001 Day 2 Practical"""
    
    try:
        optimizer = COLLAB001Day2Practical()
        success = optimizer.run_day2_optimization()
        
        if success:
            print("\n🌟 COLLAB-001 DAY 2 SUCCESSFULLY COMPLETED!")
            print("🎯 95%+ F1-Score target achieved with advanced ensemble methods!")
            print("📋 Ready for Day 3: Production Preparation & Final Polish")
        else:
            print("\n⚠️ COLLAB-001 Day 2 completed with partial success")
            print("🔧 Day 3 will focus on final optimization techniques")
        
        return success
        
    except Exception as e:
        print(f"\n❌ Error in COLLAB-001 Day 2: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 