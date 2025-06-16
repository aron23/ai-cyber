#!/usr/bin/env python3
"""
DS-005 Phase 2: Neural Network Implementation
Goal: Achieve 90%+ F1-Score using PyTorch deep learning
Baseline to Beat: F1=86.08% (XGBoost/LightGBM)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn import functional as F

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns

class SpamFilterDataset(Dataset):
    """Custom Dataset for sparse TF-IDF features"""
    
    def __init__(self, X, y):
        # Convert sparse matrix to dense tensor
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y.values if hasattr(y, 'values') else y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SpamFilterNet(nn.Module):
    """Neural Network for Spam Classification
    
    Architecture optimized for sparse TF-IDF features:
    - Input: 5000 TF-IDF features
    - Hidden layers with dropout and batch normalization
    - Output: Binary classification (Ham/Spam)
    """
    
    def __init__(self, input_size=5000, hidden_sizes=[512, 256, 128], dropout_rate=0.3):
        super(SpamFilterNet, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        
        # Build layers dynamically
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            # Batch normalization
            layers.append(nn.BatchNorm1d(hidden_size))
            # Activation
            layers.append(nn.ReLU())
            # Dropout
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 2))  # Binary classification
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class SpamFilterTrainer:
    """Comprehensive trainer for spam filter neural networks"""
    
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.baseline_f1 = 0.4358
        self.current_best_f1 = 0.8608
        self.target_f1 = 0.90
        self.min_recall = 0.88
        self.start_time = time.time()
        
        print(f"🚀 DS-005 Neural Network Training Started: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        print(f"🖥️  Device: {self.device}")
        print(f"🎯 Current Best: F1={self.current_best_f1:.4f}")
        print(f"🎯 Target: F1≥{self.target_f1:.2f} (Gap: {(self.target_f1-self.current_best_f1)*100:.1f}%)")
        print(f"⚖️ Constraint: Recall≥{self.min_recall:.2f}")
    
    def load_data(self):
        """Load and prepare data for neural network training"""
        print("\n📁 Loading processed data...")
        
        train_df = pd.read_csv('data/processed/train.csv')
        test_df = pd.read_csv('data/processed/test.csv')
        val_df = pd.read_csv('data/processed/validation.csv')
        
        # Calculate class weights for imbalanced dataset
        class_counts = train_df['label_encoded'].value_counts()
        self.class_weight_ratio = class_counts[0] / class_counts[1]
        
        # Calculate class weights for loss function
        total_samples = len(train_df)
        weight_for_0 = total_samples / (2.0 * class_counts[0])  # Ham
        weight_for_1 = total_samples / (2.0 * class_counts[1])  # Spam
        self.class_weights = torch.FloatTensor([weight_for_0, weight_for_1]).to(self.device)
        
        # Load TF-IDF vectorizer
        tfidf_vectorizer = joblib.load('models/tfidf_vectorizer_v1.0.0.joblib')
        
        # Prepare features
        self.X_train = tfidf_vectorizer.transform(train_df['text_aggressive'])
        self.X_test = tfidf_vectorizer.transform(test_df['text_aggressive'])
        self.X_val = tfidf_vectorizer.transform(val_df['text_aggressive'])
        self.y_train = train_df['label_encoded']
        self.y_test = test_df['label_encoded'] 
        self.y_val = val_df['label_encoded']
        
        print(f"✅ Data loaded: {self.X_train.shape[0]} train, {self.X_test.shape[0]} test, {self.X_val.shape[0]} val")
        print(f"✅ Features: {self.X_train.shape[1]} TF-IDF features")
        print(f"✅ Class ratio: {self.class_weight_ratio:.2f}:1 (Ham:Spam)")
        print(f"✅ Class weights: [{self.class_weights[0]:.3f}, {self.class_weights[1]:.3f}]")
        
        # Create datasets
        self.train_dataset = SpamFilterDataset(self.X_train, self.y_train)
        self.test_dataset = SpamFilterDataset(self.X_test, self.y_test)
        self.val_dataset = SpamFilterDataset(self.X_val, self.y_val)
        
        return self.X_train.shape[1]  # Return input size
    
    def create_data_loaders(self, batch_size=64):
        """Create PyTorch data loaders"""
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"✅ Data loaders created: batch_size={batch_size}")
    
    def train_epoch(self, model, train_loader, criterion, optimizer):
        """Train for one epoch"""
        model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        f1 = f1_score(all_targets, all_predictions)
        recall = recall_score(all_targets, all_predictions)
        
        return avg_loss, f1, recall
    
    def validate_epoch(self, model, val_loader, criterion):
        """Validate for one epoch"""
        model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                total_loss += loss.item()
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of spam
        
        avg_loss = total_loss / len(val_loader)
        f1 = f1_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions)
        recall = recall_score(all_targets, all_predictions)
        auc = roc_auc_score(all_targets, all_probabilities)
        
        return avg_loss, f1, precision, recall, auc, all_predictions, all_probabilities
    
    def train_model(self, architecture_config, training_config):
        """Train neural network with given configuration"""
        
        # Create model
        model = SpamFilterNet(
            input_size=architecture_config['input_size'],
            hidden_sizes=architecture_config['hidden_sizes'],
            dropout_rate=architecture_config['dropout_rate']
        ).to(self.device)
        
        print(f"\n🧠 Model Architecture:")
        print(f"  Input size: {architecture_config['input_size']}")
        print(f"  Hidden layers: {architecture_config['hidden_sizes']}")
        print(f"  Dropout rate: {architecture_config['dropout_rate']}")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Create loss function with class weights
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        # Create optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        
        # Training loop
        train_losses, val_losses = [], []
        train_f1s, val_f1s = [], []
        val_recalls = []
        best_f1 = 0
        best_model_state = None
        patience_counter = 0
        
        print(f"\n🔄 Starting training for {training_config['epochs']} epochs...")
        
        for epoch in range(training_config['epochs']):
            # Train
            train_loss, train_f1, train_recall = self.train_epoch(model, self.train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_f1, val_precision, val_recall, val_auc, _, _ = self.validate_epoch(
                model, self.val_loader, criterion
            )
            
            # Learning rate scheduling
            scheduler.step(val_f1)
            
            # Record metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_f1s.append(train_f1)
            val_f1s.append(val_f1)
            val_recalls.append(val_recall)
            
            # Early stopping and best model saving
            if val_f1 > best_f1 and val_recall >= self.min_recall:
                best_f1 = val_f1
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                improvement_mark = "🌟"
            else:
                patience_counter += 1
                improvement_mark = "  "
            
            # Print progress
            if epoch % 5 == 0 or epoch == training_config['epochs'] - 1:
                print(f"Epoch {epoch:3d}: Train F1={train_f1:.4f}, Val F1={val_f1:.4f}, "
                      f"Val Recall={val_recall:.4f}, Val Loss={val_loss:.4f} {improvement_mark}")
            
            # Early stopping
            if patience_counter >= training_config['patience']:
                print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"✅ Best model loaded: F1={best_f1:.4f}")
        
        # Final evaluation on test set
        test_loss, test_f1, test_precision, test_recall, test_auc, test_preds, test_probs = self.validate_epoch(
            model, self.test_loader, criterion
        )
        
        results = {
            'test_f1': test_f1,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_auc': test_auc,
            'best_val_f1': best_f1,
            'training_epochs': epoch + 1,
            'architecture': architecture_config,
            'training_config': training_config,
            'training_history': {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_f1s': train_f1s,
                'val_f1s': val_f1s,
                'val_recalls': val_recalls
            }
        }
        
        return model, results
    
    def run_cross_validation(self, architecture_configs, training_config, n_folds=5):
        """Run cross-validation for multiple architectures"""
        
        print("\n" + "="*60)
        print("🎯 NEURAL NETWORK CROSS-VALIDATION")
        print("="*60)
        
        best_architecture = None
        best_cv_score = 0
        all_results = []
        
        for i, arch_config in enumerate(architecture_configs):
            print(f"\n📐 Architecture {i+1}/{len(architecture_configs)}: {arch_config['name']}")
            print(f"   Hidden layers: {arch_config['hidden_sizes']}")
            print(f"   Dropout: {arch_config['dropout_rate']}")
            
            # Prepare for cross-validation
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(self.X_train, self.y_train)):
                print(f"   Fold {fold+1}/{n_folds}...", end=" ")
                
                # Create fold datasets
                X_train_fold = self.X_train[train_idx]
                X_val_fold = self.X_train[val_idx]
                y_train_fold = self.y_train.iloc[train_idx]
                y_val_fold = self.y_train.iloc[val_idx]
                
                train_fold_dataset = SpamFilterDataset(X_train_fold, y_train_fold)
                val_fold_dataset = SpamFilterDataset(X_val_fold, y_val_fold)
                
                train_fold_loader = DataLoader(train_fold_dataset, batch_size=64, shuffle=True)
                val_fold_loader = DataLoader(val_fold_dataset, batch_size=64, shuffle=False)
                
                # Train model for this fold
                fold_model = SpamFilterNet(
                    input_size=arch_config['input_size'],
                    hidden_sizes=arch_config['hidden_sizes'],
                    dropout_rate=arch_config['dropout_rate']
                ).to(self.device)
                
                fold_criterion = nn.CrossEntropyLoss(weight=self.class_weights)
                fold_optimizer = optim.Adam(fold_model.parameters(), lr=training_config['learning_rate'])
                
                # Quick training for CV (fewer epochs)
                cv_epochs = min(20, training_config['epochs'] // 2)
                for epoch in range(cv_epochs):
                    fold_model.train()
                    for batch_X, batch_y in train_fold_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        fold_optimizer.zero_grad()
                        outputs = fold_model(batch_X)
                        loss = fold_criterion(outputs, batch_y)
                        loss.backward()
                        fold_optimizer.step()
                
                # Evaluate fold
                _, fold_f1, _, fold_recall, _, _, _ = self.validate_epoch(fold_model, val_fold_loader, fold_criterion)
                
                # Apply recall constraint
                if fold_recall >= self.min_recall:
                    fold_scores.append(fold_f1)
                    print(f"F1={fold_f1:.4f}")
                else:
                    fold_scores.append(0)  # Penalize if recall constraint violated
                    print(f"F1={fold_f1:.4f} (recall violation)")
            
            # Calculate average CV score
            avg_cv_score = np.mean(fold_scores)
            arch_config['cv_score'] = avg_cv_score
            
            print(f"   📊 Average CV F1: {avg_cv_score:.4f}")
            
            if avg_cv_score > best_cv_score:
                best_cv_score = avg_cv_score
                best_architecture = arch_config
                print(f"   🌟 New best architecture!")
            
            all_results.append({
                'architecture': arch_config,
                'cv_score': avg_cv_score,
                'fold_scores': fold_scores
            })
        
        print(f"\n🏆 Best Architecture: {best_architecture['name']}")
        print(f"🎯 Best CV F1: {best_cv_score:.4f}")
        
        return best_architecture, all_results
    
    def run_full_training(self):
        """Run complete neural network training pipeline"""
        
        # Load data and create data loaders
        input_size = self.load_data()
        self.create_data_loaders(batch_size=64)
        
        # Define architecture configurations to test
        architecture_configs = [
            {
                'name': 'Deep Wide',
                'input_size': input_size,
                'hidden_sizes': [1024, 512, 256],
                'dropout_rate': 0.3
            },
            {
                'name': 'Deep Narrow',
                'input_size': input_size,
                'hidden_sizes': [512, 256, 128, 64],
                'dropout_rate': 0.4
            },
            {
                'name': 'Moderate',
                'input_size': input_size,
                'hidden_sizes': [512, 256, 128],
                'dropout_rate': 0.3
            },
            {
                'name': 'Simple Deep',
                'input_size': input_size,
                'hidden_sizes': [256, 128],
                'dropout_rate': 0.2
            }
        ]
        
        # Training configuration
        training_config = {
            'epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'patience': 15
        }
        
        # Cross-validation to find best architecture
        best_architecture, cv_results = self.run_cross_validation(architecture_configs, training_config)
        
        # Train final model with best architecture
        print("\n" + "="*60)
        print("🎯 FINAL MODEL TRAINING")
        print("="*60)
        
        final_model, final_results = self.train_model(best_architecture, training_config)
        
        # Results analysis
        print("\n" + "="*60)
        print("📊 NEURAL NETWORK RESULTS")
        print("="*60)
        
        test_f1 = final_results['test_f1']
        test_precision = final_results['test_precision']
        test_recall = final_results['test_recall']
        test_auc = final_results['test_auc']
        
        improvement = ((test_f1 - self.baseline_f1) / self.baseline_f1) * 100
        vs_current_best = ((test_f1 - self.current_best_f1) / self.current_best_f1) * 100
        
        print(f"📊 Final Neural Network Results:")
        print(f"  F1-Score: {test_f1:.4f} ({test_f1*100:.2f}%)")
        print(f"  Precision: {test_precision:.4f} ({test_precision*100:.2f}%)")
        print(f"  Recall: {test_recall:.4f} ({test_recall*100:.2f}%)")
        print(f"  AUC-ROC: {test_auc:.4f}")
        print(f"  Improvement over baseline: {improvement:+.2f}%")
        print(f"  vs Current best (86.08%): {vs_current_best:+.2f}%")
        
        # Target achievement
        target_achieved = test_f1 >= self.target_f1
        recall_constraint_met = test_recall >= self.min_recall
        
        print(f"\n🎯 TARGET ACHIEVEMENT:")
        print(f"  Stretch Target (F1≥90%): {'✅ ACHIEVED' if target_achieved else '❌ NOT ACHIEVED'}")
        print(f"  Recall constraint (≥88%): {'✅ MET' if recall_constraint_met else '❌ VIOLATED'}")
        
        if not target_achieved:
            gap = (self.target_f1 - test_f1) * 100
            print(f"  Gap to target: {gap:.1f} percentage points")
        
        # Save model and results
        timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
        model_path = f'models/neural_network_v1.0.0_{timestamp}.pth'
        results_path = f'models/neural_network_results_{timestamp}.json'
        
        # Save model
        torch.save({
            'model_state_dict': final_model.state_dict(),
            'architecture': best_architecture,
            'results': final_results,
            'training_config': training_config
        }, model_path)
        
        # Save detailed results
        complete_results = {
            'timestamp': timestamp,
            'training_duration_minutes': (time.time() - self.start_time) / 60,
            'best_architecture': best_architecture,
            'cv_results': cv_results,
            'final_results': final_results,
            'target_achieved': target_achieved,
            'baseline_comparison': {
                'baseline_f1': self.baseline_f1,
                'current_best_f1': self.current_best_f1,
                'neural_network_f1': test_f1,
                'improvement_over_baseline_%': improvement,
                'improvement_over_current_best_%': vs_current_best
            }
        }
        
        with open(results_path, 'w') as f:
            json.dump(complete_results, f, indent=2, default=str)
        
        print(f"\n💾 Model saved: {model_path}")
        print(f"💾 Results saved: {results_path}")
        
        total_time = (time.time() - self.start_time) / 60
        print(f"\n✅ Neural Network Training Complete! Duration: {total_time:.1f} minutes")
        
        return final_model, final_results, target_achieved

def main():
    """Main neural network training execution"""
    
    try:
        # Check PyTorch availability
        print(f"🔍 PyTorch version: {torch.__version__}")
        print(f"🔍 CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🔍 CUDA device: {torch.cuda.get_device_name()}")
        
        # Initialize trainer
        trainer = SpamFilterTrainer()
        
        # Run full training pipeline
        model, results, success = trainer.run_full_training()
        
        print(f"\n🎉 DS-005 Neural Network Phase: {'SUCCESS' if success else 'SIGNIFICANT PROGRESS'}")
        return success
        
    except Exception as e:
        print(f"❌ Neural Network Training Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 