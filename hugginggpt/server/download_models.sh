#!/bin/bash

# Download models for JARVIS ["full", "standard"] and ["full", "standard", "minimal"] deployments
# Excludes ["full"] only models to reduce download size
# Includes Qwen 2.5 7B as the central LLM

echo "Setting up JARVIS environment with standard and minimal models..."

# Create models directory
mkdir -p models
cd models

# Central LLM - Qwen 2.5 7B (required for local inference)
echo "Downloading Qwen 2.5 7B (central LLM)..."
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir Qwen/Qwen2.5-7B-Instruct

# Models for ["full", "standard"] deployment (standard_pipes)
echo "Downloading standard models..."
huggingface-cli download openai/whisper-base --local-dir openai/whisper-base
huggingface-cli download microsoft/speecht5_asr --local-dir microsoft/speecht5_asr
huggingface-cli download Intel/dpt-large --local-dir Intel/dpt-large
huggingface-cli download facebook/detr-resnet-50-panoptic --local-dir facebook/detr-resnet-50-panoptic
huggingface-cli download facebook/detr-resnet-101 --local-dir facebook/detr-resnet-101
huggingface-cli download google/owlvit-base-patch32 --local-dir google/owlvit-base-patch32
huggingface-cli download impira/layoutlm-document-qa --local-dir impira/layoutlm-document-qa
huggingface-cli download ydshieh/vit-gpt2-coco-en --local-dir ydshieh/vit-gpt2-coco-en
huggingface-cli download dandelin/vilt-b32-finetuned-vqa --local-dir dandelin/vilt-b32-finetuned-vqa

# Models for ["full", "standard", "minimal"] deployment (controlnet_sd_pipes)
echo "Downloading ControlNet and Stable Diffusion models..."
huggingface-cli download lllyasviel/ControlNet --local-dir lllyasviel/ControlNet
huggingface-cli download lllyasviel/sd-controlnet-canny --local-dir lllyasviel/sd-controlnet-canny
huggingface-cli download lllyasviel/sd-controlnet-depth --local-dir lllyasviel/sd-controlnet-depth
huggingface-cli download lllyasviel/sd-controlnet-hed --local-dir lllyasviel/sd-controlnet-hed
huggingface-cli download lllyasviel/sd-controlnet-mlsd --local-dir lllyasviel/sd-controlnet-mlsd
huggingface-cli download lllyasviel/sd-controlnet-openpose --local-dir lllyasviel/sd-controlnet-openpose
huggingface-cli download lllyasviel/sd-controlnet-scribble --local-dir lllyasviel/sd-controlnet-scribble
huggingface-cli download lllyasviel/sd-controlnet-seg --local-dir lllyasviel/sd-controlnet-seg
huggingface-cli download runwayml/stable-diffusion-v1-5 --local-dir runwayml/stable-diffusion-v1-5

# Required datasets
echo "Downloading required datasets..."
huggingface-cli download Matthijs/cmu-arctic-xvectors --local-dir Matthijs/cmu-arctic-xvectors

echo "Model download complete!"
echo "Total disk usage:"
du -sh .

cd ..
echo "Setup complete! You can now run JARVIS with standard and minimal models."
echo "This includes Qwen 2.5 7B (central LLM), audio, vision, and ControlNet models."
echo "Total models: Qwen LLM + 9 standard models + 9 ControlNet models + 1 dataset."
