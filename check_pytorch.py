#!/usr/bin/env python3
"""
PyTorch Installation & Readiness Check for DS-005 Neural Networks
"""

def check_pytorch():
    print("🔍 Checking PyTorch installation and readiness...")
    
    try:
        import torch
        print(f"✅ PyTorch installed: version {torch.__version__}")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"🖥️  CUDA available: {cuda_available}")
        
        if cuda_available:
            device_name = torch.cuda.get_device_name()
            print(f"🚀 GPU device: {device_name}")
            device = "cuda"
        else:
            print("💻 Using CPU for training")
            device = "cpu"
        
        # Test basic tensor operations
        print("\n🧪 Testing tensor operations...")
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = x + y
        print(f"✅ Basic operations work: {z.shape}")
        
        # Test neural network modules
        print("\n🧠 Testing neural network modules...")
        import torch.nn as nn
        import torch.optim as optim
        
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        
        optimizer = optim.Adam(model.parameters())
        print("✅ Neural network modules working")
        
        # Check required packages
        print("\n📦 Checking additional dependencies...")
        
        try:
            import pandas as pd
            print(f"✅ pandas: {pd.__version__}")
        except ImportError:
            print("❌ pandas not available")
            return False
        
        try:
            import sklearn
            print(f"✅ scikit-learn: {sklearn.__version__}")
        except ImportError:
            print("❌ scikit-learn not available")
            return False
        
        try:
            import joblib
            print("✅ joblib available")
        except ImportError:
            print("❌ joblib not available")
            return False
        
        print(f"\n🎉 PyTorch environment ready for DS-005 Neural Networks!")
        print(f"🖥️  Recommended device: {device}")
        return True
        
    except ImportError as e:
        print(f"❌ PyTorch not installed or not working: {e}")
        print("\n💡 To install PyTorch, run:")
        print("   pip install torch torchvision torchaudio")
        return False
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    print("🚀 DS-005 PyTorch Readiness Check")
    print("=" * 40)
    
    ready = check_pytorch()
    
    if ready:
        print("\n🎯 READY TO PROCEED!")
        print("\n📋 Available neural network scripts:")
        print("1. pytorch_neural_network.py     - Streamlined & fast")
        print("2. neural_networks_pytorch.py    - Comprehensive with CV")
        print("3. ds005_neural_networks.py      - Full architecture testing")
        
        print("\n🚀 Recommended quick start:")
        print("   python pytorch_neural_network.py")
        
        print("\n⏱️  Expected runtime: 30-60 minutes")
        print("🎯 Target: F1 ≥ 90% (Current best: 86.08%)")
        
    else:
        print("\n⚠️  Please install PyTorch first, then re-run this check")
    
    return ready

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 