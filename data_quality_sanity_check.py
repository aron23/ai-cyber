#!/usr/bin/env python3
"""
Data Quality Sanity Check - Pre-Advanced Methods Validation
==========================================================
CRITICAL: Comprehensive validation before neural network training
Ensures clean data quality for reliable 95-98% F1-Score targets
"""

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
import hashlib
from datetime import datetime
import json

class DataQualitySanityCheck:
    """Comprehensive pre-advanced-methods data quality validation"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        self.checks_passed = 0
        self.total_checks = 0
        self.critical_issues = []
        self.warnings = []
        self.recommendations = []
        
    def log_check(self, description, passed, critical=False, message=""):
        """Log individual check result"""
        self.total_checks += 1
        if passed:
            self.checks_passed += 1
            print(f"✅ {description}")
        else:
            print(f"❌ {description}")
            if critical:
                self.critical_issues.append(f"{description}: {message}")
            else:
                self.warnings.append(f"{description}: {message}")
        if message:
            print(f"   {message}")
    
    def check_file_existence(self):
        """Verify all required clean data files exist"""
        print("\n🔍 CHECKING FILE EXISTENCE")
        print("-" * 40)
        
        required_files = [
            'data/clean/train_clean.csv',
            'data/clean/val_clean.csv', 
            'data/clean/test_clean.csv',
            'data/clean/pipeline_documentation.txt'
        ]
        
        all_exist = True
        for file_path in required_files:
            exists = os.path.exists(file_path)
            if exists:
                size_mb = os.path.getsize(file_path) / (1024*1024)
                self.log_check(f"File exists: {file_path} ({size_mb:.2f} MB)", True)
            else:
                self.log_check(f"File missing: {file_path}", False, critical=True)
                all_exist = False
        
        return all_exist
    
    def load_clean_datasets(self):
        """Load and validate clean datasets"""
        print("\n📊 LOADING CLEAN DATASETS")
        print("-" * 40)
        
        try:
            train_df = pd.read_csv('data/clean/train_clean.csv')
            val_df = pd.read_csv('data/clean/val_clean.csv')
            test_df = pd.read_csv('data/clean/test_clean.csv')
            
            print(f"📈 Dataset shapes:")
            print(f"   Train: {train_df.shape} ({train_df.shape[0]:,} samples)")
            print(f"   Val:   {val_df.shape} ({val_df.shape[0]:,} samples)")
            print(f"   Test:  {test_df.shape} ({test_df.shape[0]:,} samples)")
            
            # Check expected columns
            required_cols = ['message', 'label']
            for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
                has_cols = all(col in df.columns for col in required_cols)
                self.log_check(f"{name} has required columns {required_cols}", has_cols, critical=True)
            
            return train_df, val_df, test_df, True
            
        except Exception as e:
            self.log_check("Load clean datasets", False, critical=True, message=str(e))
            return None, None, None, False
    
    def validate_data_integrity(self, train_df, val_df, test_df):
        """Comprehensive data integrity validation"""
        print("\n🔒 VALIDATING DATA INTEGRITY")
        print("-" * 40)
        
        # Check for missing values
        for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            missing_msgs = df['message'].isnull().sum()
            missing_labels = df['label'].isnull().sum()
            self.log_check(f"{name} no missing messages", missing_msgs == 0, critical=True)
            self.log_check(f"{name} no missing labels", missing_labels == 0, critical=True)
        
        # Check for empty messages
        for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            empty_msgs = (df['message'].str.strip().str.len() == 0).sum()
            self.log_check(f"{name} no empty messages", empty_msgs == 0, critical=True)
        
        # Check valid labels
        valid_labels = {'ham', 'spam'}
        for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            invalid_labels = set(df['label'].unique()) - valid_labels
            self.log_check(f"{name} has valid labels only", len(invalid_labels) == 0, 
                          critical=True, message=f"Invalid: {invalid_labels}" if invalid_labels else "")
        
        # Check message length distribution
        for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            msg_lengths = df['message'].str.len()
            avg_length = msg_lengths.mean()
            min_length = msg_lengths.min()
            max_length = msg_lengths.max()
            
            print(f"📏 {name} message lengths: avg={avg_length:.1f}, min={min_length}, max={max_length}")
            
            # Sanity checks
            self.log_check(f"{name} avg length reasonable (>30 chars)", avg_length > 30)
            self.log_check(f"{name} no extremely short messages (<5 chars)", min_length >= 5)
            self.log_check(f"{name} no extremely long messages (>1000 chars)", max_length <= 1000)
    
    def validate_zero_data_leakage(self, train_df, val_df, test_df):
        """CRITICAL: Verify zero data leakage between splits"""
        print("\n🚨 CRITICAL: ZERO DATA LEAKAGE VALIDATION")
        print("-" * 50)
        
        # Create message sets for overlap detection
        train_messages = set(train_df['message'].values)
        val_messages = set(val_df['message'].values)
        test_messages = set(test_df['message'].values)
        
        # Check overlaps
        train_val_overlap = len(train_messages.intersection(val_messages))
        train_test_overlap = len(train_messages.intersection(test_messages))
        val_test_overlap = len(val_messages.intersection(test_messages))
        
        print(f"🔍 Overlap Analysis:")
        print(f"   Train-Val overlap: {train_val_overlap} messages")
        print(f"   Train-Test overlap: {train_test_overlap} messages") 
        print(f"   Val-Test overlap: {val_test_overlap} messages")
        
        # CRITICAL checks
        self.log_check("ZERO Train-Val overlap", train_val_overlap == 0, critical=True)
        self.log_check("ZERO Train-Test overlap", train_test_overlap == 0, critical=True)
        self.log_check("ZERO Val-Test overlap", val_test_overlap == 0, critical=True)
        
        total_overlap = train_val_overlap + train_test_overlap + val_test_overlap
        if total_overlap == 0:
            print("🎉 SUCCESS: ZERO DATA LEAKAGE CONFIRMED!")
            return True
        else:
            print("🚨 CRITICAL ERROR: DATA LEAKAGE DETECTED!")
            return False
    
    def validate_class_distribution(self, train_df, val_df, test_df):
        """Validate consistent class distribution across splits"""
        print("\n⚖️ VALIDATING CLASS DISTRIBUTION")
        print("-" * 40)
        
        # Calculate spam rates
        train_spam_rate = (train_df['label'] == 'spam').mean()
        val_spam_rate = (val_df['label'] == 'spam').mean()
        test_spam_rate = (test_df['label'] == 'spam').mean()
        
        print(f"📊 Spam rates:")
        print(f"   Train: {train_spam_rate:.1%} ({(train_df['label'] == 'spam').sum():,} spam)")
        print(f"   Val:   {val_spam_rate:.1%} ({(val_df['label'] == 'spam').sum():,} spam)")
        print(f"   Test:  {test_spam_rate:.1%} ({(test_df['label'] == 'spam').sum():,} spam)")
        
        # Check consistency
        max_diff = max(abs(train_spam_rate - val_spam_rate),
                      abs(train_spam_rate - test_spam_rate),
                      abs(val_spam_rate - test_spam_rate))
        
        self.log_check("Class distribution consistent (<5% diff)", max_diff < 0.05, 
                      message=f"Max difference: {max_diff:.1%}")
        
        # Check reasonable spam rate (not too imbalanced)
        avg_spam_rate = np.mean([train_spam_rate, val_spam_rate, test_spam_rate])
        self.log_check("Reasonable spam rate (5-30%)", 0.05 <= avg_spam_rate <= 0.30,
                      message=f"Average spam rate: {avg_spam_rate:.1%}")
        
        return max_diff < 0.05
    
    def validate_feature_readiness(self, train_df, val_df, test_df):
        """Validate readiness for TF-IDF feature extraction"""
        print("\n🔤 VALIDATING FEATURE EXTRACTION READINESS")  
        print("-" * 45)
        
        # Test TF-IDF processing
        try:
            # Small test vectorizer
            test_vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1,2))
            test_features = test_vectorizer.fit_transform(train_df['message'].head(100))
            
            vocab_size = len(test_vectorizer.vocabulary_)
            sparsity = 1 - (test_features.nnz / np.prod(test_features.shape))
            
            self.log_check("TF-IDF processing successful", True)
            self.log_check("Reasonable vocabulary created", vocab_size > 10,
                          message=f"Vocabulary: {vocab_size} terms")
            self.log_check("Appropriate sparsity", 0.5 <= sparsity <= 0.99,
                          message=f"Sparsity: {sparsity:.3f}")
            
            # Memory estimation for full processing
            estimated_features = min(5000, vocab_size * 50)  # Conservative estimate
            estimated_memory_mb = (train_df.shape[0] * estimated_features * 8) / (1024*1024)
            
            print(f"💾 Memory estimation for full TF-IDF:")
            print(f"   Estimated features: {estimated_features:,}")
            print(f"   Estimated memory: {estimated_memory_mb:.1f} MB")
            
            self.log_check("Memory requirements reasonable (<500MB)", estimated_memory_mb < 500,
                          message=f"Estimated: {estimated_memory_mb:.1f} MB")
            
            return True
            
        except Exception as e:
            self.log_check("TF-IDF processing test", False, critical=True, message=str(e))
            return False
    
    def validate_neural_network_readiness(self, train_df, val_df, test_df):
        """Validate readiness for neural network training"""
        print("\n🧠 VALIDATING NEURAL NETWORK READINESS")
        print("-" * 42)
        
        # Dataset size analysis
        train_size = len(train_df)
        val_size = len(val_df)
        test_size = len(test_df)
        
        print(f"📊 Dataset size analysis:")
        print(f"   Training samples: {train_size:,}")
        print(f"   Validation samples: {val_size:,}")
        print(f"   Test samples: {test_size:,}")
        
        # Size adequacy checks
        self.log_check("Sufficient training data (>2000)", train_size > 2000,
                      message=f"Training: {train_size:,} samples")
        self.log_check("Sufficient validation data (>500)", val_size > 500,
                      message=f"Validation: {val_size:,} samples")
        self.log_check("Sufficient test data (>500)", test_size > 500,
                      message=f"Test: {test_size:,} samples")
        
        # Balance check for neural networks
        train_spam = (train_df['label'] == 'spam').sum()
        train_ham = (train_df['label'] == 'ham').sum()
        min_class_size = min(train_spam, train_ham)
        
        self.log_check("Adequate minority class samples (>300)", min_class_size > 300,
                      message=f"Minority class: {min_class_size:,} samples")
        
        # Computational readiness
        estimated_epochs = 50  # Conservative for neural networks
        estimated_time_minutes = (train_size * estimated_epochs) / 10000  # Rough estimate
        
        print(f"⏱️ Training time estimation:")
        print(f"   Estimated epochs: {estimated_epochs}")
        print(f"   Estimated time: {estimated_time_minutes:.1f} minutes")
        
        self.log_check("Reasonable training time (<60 min)", estimated_time_minutes < 60,
                      message=f"Estimated: {estimated_time_minutes:.1f} minutes")
        
        return True
    
    def generate_quality_score(self):
        """Generate overall data quality score"""
        if self.total_checks == 0:
            return 0
        
        base_score = (self.checks_passed / self.total_checks) * 100
        
        # Penalties for critical issues
        critical_penalty = len(self.critical_issues) * 20
        warning_penalty = len(self.warnings) * 5
        
        final_score = max(0, base_score - critical_penalty - warning_penalty)
        return final_score
    
    def run_comprehensive_sanity_check(self):
        """Run complete data quality sanity check"""
        print("=" * 80)
        print("🔍 DATA QUALITY SANITY CHECK - PRE-ADVANCED METHODS")
        print("=" * 80)
        print(f"📅 Timestamp: {self.timestamp}")
        print(f"🎯 Purpose: Validate clean data before neural network training")
        print(f"🚀 Target: Ensure foundation for 95-98% F1-Score achievement")
        print("=" * 80)
        
        # Run all validation checks
        files_ok = self.check_file_existence()
        
        if not files_ok:
            print("\n🚨 CRITICAL: Required files missing - cannot proceed")
            return self.generate_report()
        
        train_df, val_df, test_df, load_ok = self.load_clean_datasets()
        
        if not load_ok:
            print("\n🚨 CRITICAL: Cannot load datasets - cannot proceed")  
            return self.generate_report()
        
        # Core validation checks
        self.validate_data_integrity(train_df, val_df, test_df)
        leakage_ok = self.validate_zero_data_leakage(train_df, val_df, test_df)
        distribution_ok = self.validate_class_distribution(train_df, val_df, test_df)
        features_ok = self.validate_feature_readiness(train_df, val_df, test_df)
        nn_ready = self.validate_neural_network_readiness(train_df, val_df, test_df)
        
        # Generate final report
        report = self.generate_report()
        self.display_final_assessment(report)
        
        return report
    
    def generate_report(self):
        """Generate comprehensive quality report"""
        quality_score = self.generate_quality_score()
        
        report = {
            'timestamp': self.timestamp,
            'overall_quality_score': quality_score,
            'checks_passed': self.checks_passed,
            'total_checks': self.total_checks,
            'pass_rate': self.checks_passed / max(1, self.total_checks),
            'critical_issues': self.critical_issues,
            'warnings': self.warnings,
            'recommendations': self.recommendations,
            'ready_for_advanced_methods': quality_score >= 85 and len(self.critical_issues) == 0
        }
        
        return report
    
    def display_final_assessment(self, report):
        """Display final assessment and recommendations"""
        print("\n" + "=" * 80)
        print("📋 FINAL DATA QUALITY ASSESSMENT")
        print("=" * 80)
        
        score = report['overall_quality_score']
        if score >= 95:
            status_icon = "🟢"
            status_text = "EXCELLENT"
        elif score >= 85:
            status_icon = "🟡"
            status_text = "GOOD"
        elif score >= 70:
            status_icon = "🟠"
            status_text = "FAIR"
        else:
            status_icon = "🔴"
            status_text = "POOR"
        
        print(f"🎯 Overall Quality Score: {score:.1f}/100 {status_icon} {status_text}")
        print(f"✅ Checks Passed: {report['checks_passed']}/{report['total_checks']} ({report['pass_rate']:.1%})")
        
        if report['critical_issues']:
            print(f"\n🚨 CRITICAL ISSUES ({len(report['critical_issues'])}):")
            for issue in report['critical_issues']:
                print(f"   ❌ {issue}")
        
        if report['warnings']:
            print(f"\n⚠️ WARNINGS ({len(report['warnings'])}):")
            for warning in report['warnings']:
                print(f"   ⚠️ {warning}")
        
        print(f"\n🚀 READY FOR ADVANCED METHODS: {'✅ YES' if report['ready_for_advanced_methods'] else '❌ NO'}")
        
        if report['ready_for_advanced_methods']:
            print("\n🎉 SUCCESS: Data quality validation passed!")
            print("📈 Proceeding with neural networks and ensemble methods")
            print("🎯 Target: 95-98% F1-Score with world-class methodology")
        else:
            print("\n🚨 CANNOT PROCEED: Critical data quality issues detected")
            print("🔧 Action required: Fix issues before advanced methods")

def main():
    """Execute comprehensive data quality sanity check"""
    checker = DataQualitySanityCheck()
    report = checker.run_comprehensive_sanity_check()
    
    # Save report
    os.makedirs('reports', exist_ok=True)
    report_file = f"reports/data_quality_sanity_check_{datetime.now().strftime('%d%m%Y_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Report saved: {report_file}")
    
    return report['ready_for_advanced_methods']

if __name__ == "__main__":
    main() 