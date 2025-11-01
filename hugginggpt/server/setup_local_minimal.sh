#!/bin/bash

# Setup script for local JARVIS with Qwen 2.5 7B
# Downloads all models - you can prune heavy ones later

echo "🚀 Setting up local JARVIS environment..."
echo "This setup downloads all models - you can remove heavy ones later if needed!"

# 1. Install PyTorch with fixed versions first
echo "📦 Installing PyTorch with fixed versions..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 2. Install remaining requirements (no version constraints)
echo "📦 Installing remaining requirements (let pip resolve versions)..."
pip install -r requirements_modern.txt

# 2.5. Install xformers (optional, for better performance)
echo "🚀 Installing xformers (optional, for better performance)..."
pip install xformers --no-deps

# 3. Download Qwen 2.5 7B + standard and minimal models (excludes full-only models)
echo "⬇️  Downloading Qwen 2.5 7B + standard and minimal models..."
chmod +x download_models.sh
bash download_models.sh

# 4. Make scripts executable
chmod +x qwen_server.py
chmod +x run_gradio_demo_qwen.py

echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Start Qwen server:"
echo "   python qwen_server.py --model_path Qwen/Qwen2.5-7B-Instruct --port 8006 &"
echo ""
echo "2. Start JARVIS models server:"
echo "   python models_server.py --config configs/config_local_minimal.yaml &"
echo ""
echo "3. Start JARVIS (CLI mode):"
echo "   python awesome_chat.py --config configs/config_local_minimal.yaml --mode cli"
echo ""
echo "4. Start JARVIS (Gradio UI):"
echo "   python run_gradio_demo_qwen.py"
echo ""
echo "💡 All models run locally - no cloud APIs needed!"
echo "📦 Full model set downloaded - you can remove heavy models later if needed!"
