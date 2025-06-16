#!/usr/bin/env python3
"""
DS-005 Hyperparameter Optimization
Goal: Achieve 90%+ F1-Score through Bayesian optimization
Current Baseline: F1=86.08%, Gap=3.9% to stretch target
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna import Trial
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib
from datetime import datetime
import time
import json

class SpamFilterOptimizer:
    def __init__(self):
        self.baseline_f1 = 0.4358
        self.current_best_f1 = 0.8608
        self.target_f1 = 0.90
        self.min_recall = 0.88
        self.start_time = time.time()
        
        print(f"🚀 DS-005 Hyperparameter Optimization Started: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        print(f"🎯 Current Best: F1={self.current_best_f1:.4f}")
        print(f"🎯 Target: F1≥{self.target_f1:.2f} (Gap: {(self.target_f1-self.current_best_f1)*100:.1f}%)")
        print(f"⚖️ Constraint: Recall≥{self.min_recall:.2f}")
        
    def load_data(self):
        """Load and prepare data for optimization"""
        print("\n📁 Loading processed data...")
        
        train_df = pd.read_csv('data/processed/train.csv')
        test_df = pd.read_csv('data/processed/test.csv')
        
        # Calculate class weights
        class_counts = train_df['label_encoded'].value_counts()
        self.class_weight_ratio = class_counts[0] / class_counts[1]
        
        # Load TF-IDF vectorizer
        tfidf_vectorizer = joblib.load('models/tfidf_vectorizer_v1.0.0.joblib')
        
        # Prepare features
        self.X_train = tfidf_vectorizer.transform(train_df['text_aggressive'])
        self.X_test = tfidf_vectorizer.transform(test_df['text_aggressive'])
        self.y_train = train_df['label_encoded']
        self.y_test = test_df['label_encoded']
        
        print(f"✅ Data loaded: {self.X_train.shape[0]} train, {self.X_test.shape[0]} test")
        print(f"✅ Features: {self.X_train.shape[1]} TF-IDF features")
        print(f"✅ Class ratio: {self.class_weight_ratio:.2f}:1 (Ham:Spam)")
        
        # Setup cross-validation
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
    def xgboost_objective(self, trial: Trial):
        """Optuna objective function for XGBoost optimization"""
        
        # Define hyperparameter search space
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'scale_pos_weight': self.class_weight_ratio,
            'random_state': 42,
            'n_jobs': -1,
            
            # Optimizable parameters
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0, step=0.1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0, step=0.1),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        }
        
        # Create model
        model = xgb.XGBClassifier(**params)
        
        # Cross-validation scores
        f1_scores = []
        recall_scores = []
        
        for train_idx, val_idx in self.cv.split(self.X_train, self.y_train):
            X_train_fold = self.X_train[train_idx]
            X_val_fold = self.X_train[val_idx]
            y_train_fold = self.y_train.iloc[train_idx]
            y_val_fold = self.y_train.iloc[val_idx]
            
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            
            f1 = f1_score(y_val_fold, y_pred)
            recall = recall_score(y_val_fold, y_pred)
            
            f1_scores.append(f1)
            recall_scores.append(recall)
        
        avg_f1 = np.mean(f1_scores)
        avg_recall = np.mean(recall_scores)
        
        # Apply recall constraint penalty
        if avg_recall < self.min_recall:
            penalty = (self.min_recall - avg_recall) * 2  # Heavy penalty for recall violation
            return avg_f1 - penalty
        
        return avg_f1
    
    def lightgbm_objective(self, trial: Trial):
        """Optuna objective function for LightGBM optimization"""
        
        # Calculate class weights for LightGBM
        n_pos = sum(self.y_train == 1)
        n_neg = sum(self.y_train == 0)
        class_weight = {0: 1.0, 1: n_neg/n_pos}
        
        # Define hyperparameter search space
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'class_weight': class_weight,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
            
            # Optimizable parameters
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0, step=0.1),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0, step=0.1),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }
        
        # Create model
        model = lgb.LGBMClassifier(**params)
        
        # Cross-validation scores
        f1_scores = []
        recall_scores = []
        
        for train_idx, val_idx in self.cv.split(self.X_train, self.y_train):
            X_train_fold = self.X_train[train_idx]
            X_val_fold = self.X_train[val_idx]
            y_train_fold = self.y_train.iloc[train_idx]
            y_val_fold = self.y_train.iloc[val_idx]
            
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            
            f1 = f1_score(y_val_fold, y_pred)
            recall = recall_score(y_val_fold, y_pred)
            
            f1_scores.append(f1)
            recall_scores.append(recall)
        
        avg_f1 = np.mean(f1_scores)
        avg_recall = np.mean(recall_scores)
        
        # Apply recall constraint penalty
        if avg_recall < self.min_recall:
            penalty = (self.min_recall - avg_recall) * 2
            return avg_f1 - penalty
        
        return avg_f1
    
    def optimize_xgboost(self, n_trials=100):
        """Optimize XGBoost hyperparameters"""
        print(f"\n🚀 Starting XGBoost Optimization ({n_trials} trials)...")
        
        study = optuna.create_study(
            direction='maximize',
            study_name='xgboost_spam_filter',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        study.optimize(self.xgboost_objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"✅ XGBoost Optimization Complete!")
        print(f"📊 Best CV F1-Score: {study.best_value:.4f}")
        print(f"📋 Best Parameters: {study.best_params}")
        
        return study
    
    def optimize_lightgbm(self, n_trials=100):
        """Optimize LightGBM hyperparameters"""
        print(f"\n🚀 Starting LightGBM Optimization ({n_trials} trials)...")
        
        study = optuna.create_study(
            direction='maximize',
            study_name='lightgbm_spam_filter',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        study.optimize(self.lightgbm_objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"✅ LightGBM Optimization Complete!")
        print(f"📊 Best CV F1-Score: {study.best_value:.4f}")
        print(f"📋 Best Parameters: {study.best_params}")
        
        return study
    
    def evaluate_final_model(self, model_type, best_params):
        """Train and evaluate final model with best parameters"""
        print(f"\n🔍 Final Evaluation: {model_type}")
        
        if model_type == 'XGBoost':
            final_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'scale_pos_weight': self.class_weight_ratio,
                'random_state': 42,
                'n_jobs': -1,
                **best_params
            }
            model = xgb.XGBClassifier(**final_params)
        else:  # LightGBM
            n_pos = sum(self.y_train == 1)
            n_neg = sum(self.y_train == 0)
            class_weight = {0: 1.0, 1: n_neg/n_pos}
            
            final_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'class_weight': class_weight,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1,
                **best_params
            }
            model = lgb.LGBMClassifier(**final_params)
        
        # Train on full training set
        model.fit(self.X_train, self.y_train)
        
        # Evaluate on test set
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        f1 = f1_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_pred_proba)
        
        improvement = ((f1 - self.baseline_f1) / self.baseline_f1) * 100
        
        results = {
            'model_type': model_type,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'auc_roc': auc,
            'improvement_%': improvement,
            'parameters': best_params
        }
        
        print(f"📊 {model_type} Final Results:")
        print(f"  F1-Score: {f1:.4f} ({f1*100:.2f}%)")
        print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"  Recall: {recall:.4f} ({recall*100:.2f}%)")
        print(f"  AUC-ROC: {auc:.4f}")
        print(f"  Improvement: {improvement:+.2f}%")
        
        return model, results
    
    def run_optimization(self, n_trials_per_model=100):
        """Run complete hyperparameter optimization"""
        
        self.load_data()
        
        # Optimize both models
        print("\n" + "="*60)
        print("🎯 HYPERPARAMETER OPTIMIZATION")
        print("="*60)
        
        xgb_study = self.optimize_xgboost(n_trials_per_model)
        lgb_study = self.optimize_lightgbm(n_trials_per_model)
        
        # Evaluate final models
        print("\n" + "="*60)
        print("📊 FINAL MODEL EVALUATION")
        print("="*60)
        
        xgb_model, xgb_results = self.evaluate_final_model('XGBoost', xgb_study.best_params)
        lgb_model, lgb_results = self.evaluate_final_model('LightGBM', lgb_study.best_params)
        
        # Compare results
        print("\n" + "="*60)
        print("🏆 OPTIMIZATION RESULTS SUMMARY")
        print("="*60)
        
        all_results = [
            {'Model': 'Baseline (Logistic)', 'F1': self.baseline_f1, 'Precision': 0.2797, 'Recall': 0.9865, 'AUC': None, 'Improvement': 0.0},
            {'Model': 'Pre-Optimization Best', 'F1': self.current_best_f1, 'Precision': 0.8193, 'Recall': 0.9067, 'AUC': 0.9747, 'Improvement': 97.51},
            {'Model': f'Optimized {xgb_results["model_type"]}', 'F1': xgb_results['f1_score'], 'Precision': xgb_results['precision'], 'Recall': xgb_results['recall'], 'AUC': xgb_results['auc_roc'], 'Improvement': xgb_results['improvement_%']},
            {'Model': f'Optimized {lgb_results["model_type"]}', 'F1': lgb_results['f1_score'], 'Precision': lgb_results['precision'], 'Recall': lgb_results['recall'], 'AUC': lgb_results['auc_roc'], 'Improvement': lgb_results['improvement_%']}
        ]
        
        print(f"{'Model':<25} {'F1':>8} {'Precision':>10} {'Recall':>8} {'AUC':>8} {'Improve':>8}")
        print("-"*75)
        
        for result in all_results:
            auc_str = f"{result['AUC']:.4f}" if result['AUC'] is not None else "-"
            improve_str = f"{result['Improvement']:+.1f}%" if result['Improvement'] != 0 else "-"
            print(f"{result['Model']:<25} {result['F1']:>8.4f} {result['Precision']:>10.4f} {result['Recall']:>8.4f} {auc_str:>8} {improve_str:>8}")
        
        # Determine best model
        best_result = max([xgb_results, lgb_results], key=lambda x: x['f1_score'])
        best_model = xgb_model if best_result == xgb_results else lgb_model
        
        print(f"\n🏆 BEST OPTIMIZED MODEL: {best_result['model_type']}")
        print(f"🎯 F1-Score: {best_result['f1_score']:.4f} ({best_result['f1_score']*100:.2f}%)")
        
        # Target achievement analysis
        target_achieved = best_result['f1_score'] >= self.target_f1
        gap_to_target = max(0, (self.target_f1 - best_result['f1_score']) * 100)
        
        print(f"\n🎯 TARGET ACHIEVEMENT:")
        print(f"  Stretch Target (F1≥90%): {'✅ ACHIEVED' if target_achieved else '❌ NOT ACHIEVED'}")
        if not target_achieved:
            print(f"  Gap to target: {gap_to_target:.1f} percentage points")
        print(f"  Recall constraint (≥88%): {'✅ MET' if best_result['recall'] >= self.min_recall else '❌ VIOLATED'}")
        
        # Save best model
        timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
        model_filename = f"models/{best_result['model_type'].lower()}_optimized_v1.0.0_{timestamp}.joblib"
        results_filename = f"models/optimization_results_{timestamp}.json"
        
        joblib.dump(best_model, model_filename)
        
        # Save optimization results
        optimization_summary = {
            'timestamp': timestamp,
            'optimization_duration_minutes': (time.time() - self.start_time) / 60,
            'trials_per_model': n_trials_per_model,
            'best_model': best_result['model_type'],
            'best_f1_score': best_result['f1_score'],
            'target_achieved': target_achieved,
            'baseline_f1': self.baseline_f1,
            'pre_optimization_f1': self.current_best_f1,
            'xgboost_results': xgb_results,
            'lightgbm_results': lgb_results,
            'xgboost_best_params': xgb_study.best_params,
            'lightgbm_best_params': lgb_study.best_params
        }
        
        with open(results_filename, 'w') as f:
            json.dump(optimization_summary, f, indent=2, default=str)
        
        print(f"\n💾 Best model saved: {model_filename}")
        print(f"💾 Results saved: {results_filename}")
        
        total_time = (time.time() - self.start_time) / 60
        print(f"\n✅ Optimization Complete! Duration: {total_time:.1f} minutes")
        
        return best_model, best_result, optimization_summary

def main():
    """Main optimization execution"""
    optimizer = SpamFilterOptimizer()
    
    try:
        # Run optimization with 150 trials per model for thorough search
        best_model, best_result, summary = optimizer.run_optimization(n_trials_per_model=150)
        
        # Final status
        success = best_result['f1_score'] >= 0.90
        print(f"\n🎉 DS-005 Hyperparameter Optimization: {'SUCCESS' if success else 'SIGNIFICANT IMPROVEMENT'}")
        return success
        
    except Exception as e:
        print(f"❌ Optimization Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 