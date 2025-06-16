#!/usr/bin/env python3
"""
COLLAB-001 Day 3: Production Implementation & Validation
SMS Spam Detection Project - Strategic Excellence Phase

Author: Data Scientist (AI-Enhanced)  
Date: 16/06/2025 08:45:00
Objective: Execute ensemble frameworks and confirm 95%+ F1-Score achievement
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
import joblib

print("="*80)
print("🚀 COLLAB-001 DAY 3: PRODUCTION IMPLEMENTATION & VALIDATION")
print("="*80)
print(f"📅 Day 3 Start: 16/06/2025 08:45:00")
print(f"🎯 Mission: Execute ensemble frameworks and achieve 95%+ F1-Score")
print(f"⚡ Status: Production implementation of advanced ensemble methods")
print()

class COLLAB001Day3Production:
    """Production ensemble implementation and validation"""
    
    def __init__(self, target_f1=0.95, baseline_f1=0.9467):
        self.target_f1 = target_f1
        self.baseline_f1 = baseline_f1
        self.start_time = time.time()
        
        self.base_models = {}
        self.best_ensemble = None
        
        print(f"🎯 COLLAB-001 Day 3 Production Initialized")
        print(f"   Target F1-Score: {target_f1:.1%}")
        print(f"   Baseline to Beat: {baseline_f1:.2%}")
        
    def load_production_data(self):
        """Load data for production ensemble training"""
        print("\n📊 LOADING PRODUCTION DATA")
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
            
            self.y_train = (train_df['label'] == 1).astype(int)
            self.y_val = (val_df['label'] == 1).astype(int)
            self.y_test = (test_df['label'] == 1).astype(int)
            
            print(f"✅ Production Data Loaded:")
            print(f"   Training: {self.X_train.shape[0]:,} samples, {self.X_train.shape[1]:,} features")
            print(f"   Validation: {self.X_val.shape[0]:,} samples")
            print(f"   Test: {self.X_test.shape[0]:,} samples")
            print(f"   Spam Rate: {self.y_train.mean()*100:.1f}% (training)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def train_production_models(self):
        """Train production-ready base models"""
        print("\n🧠 TRAINING PRODUCTION BASE MODELS")
        print("-" * 50)
        
        models = {}
        
        # 1. Logistic Regression
        print("📊 Training Logistic Regression...")
        try:
            lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
            lr.fit(self.X_train, self.y_train)
            models['logistic'] = lr
            print("   ✅ Logistic Regression trained")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        
        # 2. Random Forest
        print("🌳 Training Random Forest...")
        try:
            rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
            rf.fit(self.X_train, self.y_train)
            models['random_forest'] = rf
            print("   ✅ Random Forest trained")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        
        # 3. Neural Network
        print("🧠 Training Neural Network...")
        try:
            nn = MLPClassifier(hidden_layer_sizes=(256, 128), random_state=42, max_iter=200)
            nn.fit(self.X_train, self.y_train)
            models['neural_network'] = nn
            print("   ✅ Neural Network trained")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        
        self.base_models = models
        print(f"\n✅ Production Models Trained: {len(models)}")
        return len(models) >= 2
    
    def evaluate_models(self):
        """Evaluate base models on validation set"""
        print("\n📊 EVALUATING BASE MODELS")
        print("-" * 50)
        
        results = {}
        
        for name, model in self.base_models.items():
            try:
                y_pred = model.predict(self.X_val)
                y_prob = model.predict_proba(self.X_val)[:, 1]
                
                f1 = f1_score(self.y_val, y_pred)
                precision = precision_score(self.y_val, y_pred)
                recall = recall_score(self.y_val, y_pred)
                
                results[name] = {
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'predictions': y_pred,
                    'probabilities': y_prob
                }
                
                print(f"📈 {name.title()}: F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}")
                
            except Exception as e:
                print(f"   ❌ {name} failed: {e}")
        
        return results
    
    def advanced_ensemble(self, base_results):
        """Execute advanced ensemble optimization"""
        print("\n⚖️  ADVANCED ENSEMBLE OPTIMIZATION")
        print("-" * 50)
        
        if len(base_results) < 2:
            return None
        
        model_names = list(base_results.keys())
        prob_matrix = np.column_stack([base_results[name]['probabilities'] for name in model_names])
        
        # Weighted voting optimization
        def objective(weights):
            weights = np.array(weights) / np.sum(weights)
            ensemble_prob = np.dot(prob_matrix, weights)
            ensemble_pred = (ensemble_prob >= 0.5).astype(int)
            return -f1_score(self.y_val, ensemble_pred)
        
        # Optimize weights
        f1_scores = [base_results[name]['f1_score'] for name in model_names]
        initial_weights = np.array(f1_scores) / sum(f1_scores)
        
        try:
            from scipy.optimize import minimize
            result = minimize(objective, initial_weights, method='SLSQP',
                            bounds=[(0.1, 0.6) for _ in model_names],
                            constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1})
            optimal_weights = result.x / result.x.sum()
        except:
            optimal_weights = initial_weights
        
        # Apply weights
        ensemble_prob = np.dot(prob_matrix, optimal_weights)
        ensemble_pred = (ensemble_prob >= 0.5).astype(int)
        
        f1 = f1_score(self.y_val, ensemble_pred)
        precision = precision_score(self.y_val, ensemble_pred)
        recall = recall_score(self.y_val, ensemble_pred)
        
        results = {
            'method': 'Advanced Weighted Ensemble',
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'weights': dict(zip(model_names, optimal_weights)),
            'predictions': ensemble_pred,
            'probabilities': ensemble_prob
        }
        
        print(f"📊 Ensemble Results:")
        print(f"   F1-Score: {f1:.4f} ({f1*100:.2f}%)")
        print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"   Recall: {recall:.4f} ({recall*100:.2f}%)")
        
        print(f"\n🏋️ Optimal Weights:")
        for name, weight in zip(model_names, optimal_weights):
            print(f"   {name.title()}: {weight:.3f}")
        
        target_achieved = f1 >= self.target_f1
        recall_ok = recall >= 0.88
        
        print(f"\n🎯 Target Analysis:")
        print(f"   F1 ≥ 95.0%: {'✅ ACHIEVED' if target_achieved else '❌ NOT ACHIEVED'}")
        print(f"   Recall ≥ 88%: {'✅ MET' if recall_ok else '❌ VIOLATED'}")
        
        if target_achieved and recall_ok:
            self.best_ensemble = results
        
        return results
    
    def test_set_validation(self):
        """Final validation on test set"""
        print("\n🧪 FINAL TEST SET VALIDATION")
        print("-" * 50)
        
        if not self.best_ensemble:
            print("❌ No validated ensemble available")
            return None
        
        # Get test predictions from base models
        test_probs = []
        weights = []
        
        for name, model in self.base_models.items():
            if name in self.best_ensemble['weights']:
                test_prob = model.predict_proba(self.X_test)[:, 1]
                test_probs.append(test_prob)
                weights.append(self.best_ensemble['weights'][name])
        
        if len(test_probs) < 2:
            print("❌ Insufficient models for test ensemble")
            return None
        
        # Apply ensemble to test set
        test_prob_matrix = np.column_stack(test_probs)
        weights = np.array(weights) / sum(weights)
        
        test_ensemble_prob = np.dot(test_prob_matrix, weights)
        test_ensemble_pred = (test_ensemble_prob >= 0.5).astype(int)
        
        # Evaluate on test set
        test_f1 = f1_score(self.y_test, test_ensemble_pred)
        test_precision = precision_score(self.y_test, test_ensemble_pred)
        test_recall = recall_score(self.y_test, test_ensemble_pred)
        
        print(f"📊 Final Test Results:")
        print(f"   F1-Score: {test_f1:.4f} ({test_f1*100:.2f}%)")
        print(f"   Precision: {test_precision:.4f} ({test_precision*100:.2f}%)")
        print(f"   Recall: {test_recall:.4f} ({test_recall*100:.2f}%)")
        
        final_success = test_f1 >= self.target_f1 and test_recall >= 0.88
        
        print(f"\n🎯 FINAL TARGET ACHIEVEMENT:")
        print(f"   F1 ≥ 95.0%: {'✅ ACHIEVED' if test_f1 >= self.target_f1 else '❌ NOT ACHIEVED'}")
        print(f"   Recall ≥ 88%: {'✅ MET' if test_recall >= 0.88 else '❌ VIOLATED'}")
        print(f"   Overall Success: {'✅ SUCCESS' if final_success else '❌ PARTIAL'}")
        
        return {
            'test_f1': test_f1,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'final_success': final_success
        }
    
    def save_production_models(self):
        """Save production models and ensemble"""
        print("\n💾 SAVING PRODUCTION MODELS")
        print("-" * 50)
        
        timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
        
        # Save base models
        for name, model in self.base_models.items():
            model_path = f'models/collab001_day3_{name}_production_{timestamp}.joblib'
            joblib.dump(model, model_path)
            print(f"💾 Saved: {model_path}")
        
        # Save ensemble configuration
        if self.best_ensemble:
            ensemble_config = {
                'timestamp': timestamp,
                'method': self.best_ensemble['method'],
                'weights': self.best_ensemble['weights'],
                'performance': {
                    'f1_score': self.best_ensemble['f1_score'],
                    'precision': self.best_ensemble['precision'],
                    'recall': self.best_ensemble['recall']
                }
            }
            
            config_path = f'models/collab001_day3_ensemble_config_{timestamp}.json'
            with open(config_path, 'w') as f:
                json.dump(ensemble_config, f, indent=2)
            print(f"💾 Ensemble config saved: {config_path}")
    
    def run_day3_production(self):
        """Execute complete Day 3 production pipeline"""
        
        print("\n" + "="*80)
        print("🚀 COLLAB-001 DAY 3: PRODUCTION EXECUTION")
        print("="*80)
        
        # Load data
        if not self.load_production_data():
            return False
        
        # Train models
        if not self.train_production_models():
            return False
        
        # Evaluate base models
        base_results = self.evaluate_models()
        if len(base_results) < 2:
            return False
        
        # Execute ensemble
        ensemble_results = self.advanced_ensemble(base_results)
        if not ensemble_results:
            return False
        
        # Test set validation
        test_results = self.test_set_validation()
        
        # Save models
        self.save_production_models()
        
        # Final analysis
        success = self.analyze_day3_results(base_results, ensemble_results, test_results)
        
        return success
    
    def analyze_day3_results(self, base_results, ensemble_results, test_results):
        """Analyze Day 3 production results"""
        print("\n" + "="*80)
        print("📊 COLLAB-001 DAY 3: FINAL RESULTS ANALYSIS")
        print("="*80)
        
        # Performance summary
        best_base_f1 = max([r['f1_score'] for r in base_results.values()])
        ensemble_f1 = ensemble_results['f1_score']
        test_f1 = test_results['test_f1'] if test_results else 0
        
        print("📈 PERFORMANCE SUMMARY:")
        print(f"   Best Base Model: {best_base_f1:.4f} ({best_base_f1*100:.2f}%)")
        print(f"   Ensemble (Validation): {ensemble_f1:.4f} ({ensemble_f1*100:.2f}%)")
        print(f"   Ensemble (Test): {test_f1:.4f} ({test_f1*100:.2f}%)")
        
        # Target achievement
        validation_success = ensemble_f1 >= self.target_f1
        test_success = test_f1 >= self.target_f1 if test_results else False
        overall_success = validation_success and test_success
        
        print(f"\n🎯 TARGET ACHIEVEMENT:")
        print(f"   Validation ≥95%: {'✅ ACHIEVED' if validation_success else '❌ MISSED'}")
        print(f"   Test Set ≥95%: {'✅ ACHIEVED' if test_success else '❌ MISSED'}")
        print(f"   Overall Success: {'✅ SUCCESS' if overall_success else '⚠️ PARTIAL'}")
        
        # Save final results
        final_results = {
            'timestamp': datetime.now().strftime('%d%m%Y_%H%M%S'),
            'phase': 'COLLAB-001 Day 3 Production',
            'duration_minutes': (time.time() - self.start_time) / 60,
            'target_f1': self.target_f1,
            'validation_f1': ensemble_f1,
            'test_f1': test_f1,
            'validation_success': validation_success,
            'test_success': test_success,
            'overall_success': overall_success,
            'models_trained': len(self.base_models)
        }
        
        results_path = f'models/collab001_day3_final_results_{final_results["timestamp"]}.json'
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        duration = (time.time() - self.start_time) / 60
        print(f"\n✅ COLLAB-001 DAY 3 COMPLETE:")
        print(f"   Duration: {duration:.1f} minutes")
        print(f"   Status: {'FULL SUCCESS' if overall_success else 'PARTIAL SUCCESS'}")
        print(f"💾 Results saved: {results_path}")
        
        return overall_success

def main():
    """Execute COLLAB-001 Day 3 Production Implementation"""
    
    try:
        producer = COLLAB001Day3Production()
        success = producer.run_day3_production()
        
        if success:
            print("\n🌟 COLLAB-001 DAY 3: MISSION ACCOMPLISHED!")
            print("🎯 95%+ F1-Score target achieved in production!")
            print("📋 Ready for production deployment!")
        else:
            print("\n⚠️ COLLAB-001 Day 3: Partial success achieved")
            print("🔧 Additional optimization may be needed")
        
        return success
        
    except Exception as e:
        print(f"\n❌ Error in COLLAB-001 Day 3: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 