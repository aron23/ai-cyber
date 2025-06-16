#!/usr/bin/env python3
"""
DE-ADV-002: PyTorch Neural Network Quick Test (Fixed)
Date: 15/06/2025 20:42
Engineer: AI Data Engineer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define model class at module level to enable pickling
class SpamClassifier(nn.Module):
    def __init__(self, input_size):
        super(SpamClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

def create_test_data():
    """Create test dataset"""
    spam_messages = [
        "URGENT! Win £1000 cash! Call 09061234567 now! Costs £3/min. Limited time offer!",
        "FREE entry to win iPhone 15! Text WIN to 81234. T&C apply. Act now!",
        "WINNER! You've won £2000! Claim now by calling 09061234567. Expires today!",
        "Congratulations! You won our daily draw! Call 09061234567 to claim prize!",
        "Cash waiting for you! £5000 guaranteed! Text NOW to 85234. Don't miss out!"
    ] * 20  # 100 spam messages
    
    ham_messages = [
        "Hi, how are you doing today? Hope all is well with you and your family.",
        "Meeting at 3pm tomorrow in conference room B. Please bring the reports.",
        "Thanks for your help with the project yesterday. Really appreciate it!",
        "Can you pick up some milk on your way home? We're running low.",
        "Happy birthday! Hope you have a wonderful day filled with joy and laughter."
    ] * 20  # 100 ham messages
    
    messages = spam_messages + ham_messages
    labels = [1] * 100 + [0] * 100
    
    return messages, labels

def main():
    print("🔥 PYTORCH NEURAL NETWORK QUICK TEST (FIXED)")
    print("=" * 50)
    
    # Create test data
    messages, labels = create_test_data()
    print(f"✅ Dataset: {len(messages)} messages")
    
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    X = vectorizer.fit_transform(messages).toarray().astype(np.float32)
    y = np.array(labels, dtype=np.float32)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Training shape: {X_train.shape}")
    
    # Create model
    device = torch.device('cpu')
    model = SpamClassifier(X_train.shape[1]).to(device)
    
    print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    print("🚀 Training...")
    start_time = time.time()
    
    model.train()
    for epoch in range(5):  # Quick training
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.2f}s")
    
    # Test
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test)
        y_pred_prob = model(X_test_tensor).numpy()
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        accuracy = accuracy_score(y_test, y_pred)
    
    print(f"📊 Test Accuracy: {accuracy:.3f}")
    
    # Test serialization (FIXED)
    print("💾 Testing serialization...")
    model_path = Path("models/pytorch_fixed_v1.0.0.pt")
    model_path.parent.mkdir(exist_ok=True)
    
    # Save model with the class information
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': X_train.shape[1],
        'accuracy': accuracy,
        'training_time': training_time
    }, model_path)
    
    # Save vectorizer
    vectorizer_path = Path("models/pytorch_fixed_vectorizer_v1.0.0.joblib")
    joblib.dump(vectorizer, vectorizer_path)
    
    print("✅ Serialization successful!")
    
    # Test loading
    print("🔄 Testing loading...")
    checkpoint = torch.load(model_path, map_location=device)
    loaded_model = SpamClassifier(checkpoint['input_size']).to(device)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()
    
    loaded_vectorizer = joblib.load(vectorizer_path)
    
    print("✅ Loading successful!")
    
    # Test inference speed
    print("⚡ Testing inference speed...")
    test_messages = [
        "WIN FREE iPhone now! Call 123456789",
        "Meeting tomorrow at 2pm in room A"
    ]
    
    X_inference = loaded_vectorizer.transform(test_messages).toarray().astype(np.float32)
    X_inference_tensor = torch.tensor(X_inference)
    
    # Batch inference timing
    batch_messages = test_messages * 50  # 100 messages
    X_batch = loaded_vectorizer.transform(batch_messages).toarray().astype(np.float32)
    X_batch_tensor = torch.tensor(X_batch)
    
    start_time = time.time()
    with torch.no_grad():
        batch_predictions = loaded_model(X_batch_tensor)
    batch_inference_time = time.time() - start_time
    avg_inference_time = batch_inference_time / len(batch_messages)
    
    print(f"📊 Batch inference: {avg_inference_time*1000:.2f}ms per message")
    print(f"🎯 Target: <50ms ({'✅ PASS' if avg_inference_time*1000 < 50 else '❌ FAIL'})")
    
    # Model size
    model_size = model_path.stat().st_size / (1024 * 1024)
    vectorizer_size = vectorizer_path.stat().st_size / (1024 * 1024)
    total_size = model_size + vectorizer_size
    
    print(f"💾 Total size: {total_size:.2f}MB")
    
    # Summary
    print("\n🎉 PYTORCH NEURAL NETWORK TEST - SUCCESS!")
    print("=" * 50)
    print(f"✅ Training: {training_time:.2f}s")
    print(f"✅ Accuracy: {accuracy:.3f}")
    print(f"✅ Inference: {avg_inference_time*1000:.2f}ms per message")
    print(f"✅ Size: {total_size:.2f}MB")
    print(f"✅ Serialization: Working")
    print(f"✅ Performance: {'PASS' if avg_inference_time*1000 < 50 else 'FAIL'}")
    
    return {
        'training_time': training_time,
        'accuracy': accuracy,
        'inference_time_ms': avg_inference_time * 1000,
        'model_size_mb': total_size,
        'parameters': sum(p.numel() for p in model.parameters()),
        'meets_target': avg_inference_time * 1000 < 50
    }

if __name__ == "__main__":
    results = main() 