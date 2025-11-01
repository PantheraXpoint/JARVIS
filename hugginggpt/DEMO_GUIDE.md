# HuggingGPT (JARVIS) Complete Demo Guide

## 🎯 Quick Start

**For decision-only mode (testing model selection without execution):**
```bash
# 1. Start Qwen Server only
python3 qwen_server.py --model_path models/Qwen/Qwen2.5-7B-Instruct --port 8006 &

# 2. Start API server with test config
python3 awesome_chat.py --config configs/config_test_decisions.yaml --mode server

# 3. Test with API (returns decisions only)
curl -X POST http://localhost:8004/hugginggpt \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Detect objects in /examples/a.jpg"}], "return_decisions": true, ...}'
```

**For full demo with execution:**
```bash
# 1. Start Qwen Server
python3 qwen_server.py --model_path models/Qwen/Qwen2.5-7B-Instruct --port 8006 &

# 2. Start Models Server
python3 models_server.py --config configs/config_local_minimal.yaml

# 3. Start Gradio UI (in another terminal)
python3 run_gradio_demo_qwen.py
# Open http://localhost:7860
```

---

## 📋 Your Current Setup

### ✅ What You Have:
1. **Qwen 2.5 7B** - Local LLM (~15GB) running on port 8006
2. **Config files**:
   - `config_local_minimal.yaml` - Local-only mode
   - `config_qwen.yaml` - Hybrid mode (local + HuggingFace)
   - `config_test_decisions.yaml` - Decision-only testing
3. **Downloaded models** for various tasks (see Available Tasks section)

### 🎯 Current Status:
- ✅ **Qwen Server** - LLM coordinator running at http://localhost:8006
- 🔄 **Models Server** - Loads local models (port 8005) - Optional for decision-only mode
- 🔄 **Demo Interface** - Gradio UI (port 7860) or CLI mode

---

## 🚀 Running the Demo

### Option 1: Gradio Web UI (Recommended for Full Demo)

**Terminal 1: Start Qwen Server**
```bash
cd /home/panthera/JARVIS/hugginggpt/server
python3 qwen_server.py --model_path models/Qwen/Qwen2.5-7B-Instruct --port 8006 &
```

**Terminal 2: Start Models Server** (Required for execution, optional for decision-only)
```bash
docker exec -it icdcs bash
cd /home/panthera/JARVIS/hugginggpt/server
python3 models_server.py --config configs/config_local_minimal.yaml
```

**Terminal 3: Start Gradio UI**
```bash
docker exec -it icdcs bash
cd /home/panthera/JARVIS/hugginggpt/server
python3 run_gradio_demo_qwen.py
```

Open http://localhost:7860 in your browser!

### Option 2: CLI Mode

```bash
cd /home/panthera/JARVIS/hugginggpt/server
python3 awesome_chat.py --config configs/config_local_minimal.yaml --mode cli
```

### Option 3: API Server Mode

```bash
python3 awesome_chat.py --config configs/config_local_minimal.yaml --mode server
```

Then send POST requests to http://localhost:8004/hugginggpt

### Option 4: Decision-Only Mode (No Model Execution)

Perfect for testing LLM decision-making without running models:

```bash
# Only need Qwen server
python3 qwen_server.py --model_path models/Qwen/Qwen2.5-7B-Instruct --port 8006 &

# Start with test config (uses HuggingFace for availability checks only)
python3 awesome_chat.py --config configs/config_test_decisions.yaml --mode server
```

Send requests with `"return_decisions": true` to get model selection decisions without execution.

---

## ✅ Available Tasks

### Image Processing

**1. Image-to-Text (Image Captioning)**
- **Model**: `ydshieh/vit-gpt2-coco-en`
- **Usage**: `"Describe this image: /path/to/image.jpg"`

**2. Object Detection**
- **Model**: `facebook/detr-resnet-101`
- **Usage**: `"What objects are in this image: /path/to/image.jpg?"`

**3. Zero-Shot Object Detection**
- **Model**: `google/owlvit-base-patch32`
- **Usage**: `"Find cats in this image: /path/to/image.jpg"`

**4. Visual Question Answering (VQA)**
- **Model**: `dandelin/vilt-b32-finetuned-vqa`
- **Usage**: `"Looking at /path/to/image.jpg, what color is the car?"`

**5. Document Question Answering**
- **Model**: `impira/layoutlm-document-qa`
- **Usage**: `"What is the student amount in this document: /path/to/doc.jpg"`

**6. Image Segmentation (Panoptic)**
- **Model**: `facebook/detr-resnet-50-panoptic`
- **Usage**: `"Segment this image: /path/to/image.jpg"`

**7. Depth Estimation**
- **Model**: `Intel/dpt-large`
- **Usage**: `"Generate depth map for this image: /path/to/image.jpg"`

### ControlNet Tasks

**8. OpenPose Control** - Extracts human poses
**9. Canny Edge Control** - Detects edges using Canny algorithm
**10. HED Control** - Detects edges using HED
**11. MLSD Control** - Detects straight lines
**12. Midas Depth Control** - Generates depth maps
**13. Scribble Control** - Generates scribble-style sketches

### Text-to-Image with ControlNet

**14. Canny Text-to-Image** - Generate images from text + Canny edges
**15. Depth Text-to-Image** - Generate images from text + depth maps
**16. HED Text-to-Image** - Generate images from text + HED edges
**17. MLSD Text-to-Image** - Generate images from text + lines
**18. OpenPose Text-to-Image** - Generate images from text + poses
**19. Scribble Text-to-Image** - Generate images from text + scribbles
**20. Segmentation Text-to-Image** - Generate images from text + segmentation

### Audio Processing

**21. Speech-to-Text (ASR)**
- **Models**: `openai/whisper-base`, `microsoft/speecht5_asr`
- **Usage**: `"Transcribe this audio: /path/to/audio.wav"`

**22. Text-to-Speech** (requires `local_deployment: full`)
- **Model**: `espnet/kan-bayashi_ljspeech_vits`
- **Usage**: `"Read this text aloud: 'Hello world'"`

### Text Processing

**23. Named Entity Recognition**
**24. Summarization**
**25. Translation**
**26. Question Answering**
**27. Text Generation**
**28. Conversational**

### Video Processing

**29. Text-to-Video** (requires `local_deployment: full`)
- **Model**: `damo-vilab/text-to-video-ms-1.7b`

### Task Summary

| Category | Available Tasks | Total |
|----------|----------------|-------|
| **Vision** | Image-to-Text, Object Detection, VQA, Segmentation, Depth | 5 |
| **Audio** | Speech-to-Text | 1 |
| **ControlNet Controls** | OpenPose, Canny, HED, MLSD, Midas, Scribble | 6 |
| **Controlled Generation** | Canny/Depth/HED/MLSD/OpenPose/Scribble/Seg Text-to-Image | 7 |
| **Document Processing** | Document QA | 1 |
| **Text Processing** | NER, Summarization, Translation, QA, Generation | 5 |
| **Video** | Text-to-Video | 1 |
| **TOTAL** | | **26 Tasks** |

---

## 📝 Demo Examples You Can Try

### 🖼️ Image Tasks

**1. Image Captioning**
```
Describe this image: /hugginggpt/server/public/examples/a.jpg
```

**2. Object Detection**
```
What objects are in this image: /hugginggpt/server/public/examples/b.jpg?
```

**3. Visual Question Answering**
```
Looking at /hugginggpt/server/public/examples/c.jpg, what color is the car?
```

**4. Image Generation** (Text-to-Image)
```
Generate an image of a sunset over the ocean with palm trees
```

**5. ControlNet Image Generation**
```
Generate an image based on the pose in /hugginggpt/server/public/examples/d.jpg and the content of /hugginggpt/server/public/examples/e.jpg
```

### 🔊 Audio Tasks

**6. Speech Recognition**
```
Transcribe the audio I uploaded
```

**7. Text-to-Speech**
```
Read this text aloud: "Artificial intelligence is transforming the world"
```

### 📄 Document Tasks

**8. Document Q&A**
```
Given the document at /hugginggpt/server/public/examples/g.jpg, 
what is the student amount?
```

### 🎬 Multi-Modal Tasks

**9. Combined Image Tasks**
```
Look at /hugginggpt/server/public/examples/f.jpg, can you tell me 
how many objects are in the picture? Then generate a similar picture and video.
```

**10. Story with Image**
```
Tell me a joke about a cat and show me a picture of a cat
```

**11. Video Generation**
```
Show me a video of a boy running and also generate an image of it, then describe it with your voice
```

### 🎯 Real Examples from the Codebase

**Example 1: Count Objects**
- **User**: "Give me some pictures e1.jpg, e2.png, e3.jpg, help me count the number of sheep?"
- **JARVIS Plan**:
  1. Extract text from e1.jpg (image-to-text)
  2. Detect objects in e1.jpg (object-detection)
  3. Ask "How many sheep?" about the detected image (visual-Q&A)
  4. Repeat for e2.png and e3.jpg
  5. Count total sheep across all images

**Example 2: ControlNet Image Generation**
- **User**: "Given an image /example.jpg, first generate a HED image, then based on the HED image generate a new image where a girl is reading a book"
- **JARVIS Plan**:
  1. Extract pose from /example.jpg using HED control
  2. Generate new image using Stable Diffusion with ControlNet: "a girl is reading a book" + pose structure

**Example 3: Multi-Modal Content**
- **User**: "Show me a video and an image of 'a boy is running' and dub it"
- **JARVIS Plan**:
  1. Generate video: text-to-video("a boy is running")
  2. Generate speech: text-to-speech("a boy is running")
  3. Generate image: text-to-image("a boy is running")
  4. Combine video + audio

### 🎨 Image Examples Available

You have these example images ready to use:
- `/hugginggpt/server/public/examples/a.jpg`
- `/hugginggpt/server/public/examples/b.jpg`
- `/hugginggpt/server/public/examples/c.jpg`
- `/hugginggpt/server/public/examples/d.jpg`
- `/hugginggpt/server/public/examples/e.jpg`
- `/hugginggpt/server/public/examples/f.jpg`
- `/hugginggpt/server/public/examples/g.jpg`

---

## ⚙️ Configuration Files Explained

### `config_local_minimal.yaml`
- **LLM**: Qwen 2.5 7B at localhost:8006
- **Inference**: Local models only
- **Deployment**: Standard (image, object detection, etc.)
- **Best for**: Privacy, offline demo

### `config_qwen.yaml`
- **LLM**: Qwen 2.5 7B
- **Inference**: Hybrid (local models + HuggingFace API)
- **Deployment**: Minimal local, falls back to HuggingFace
- **Best for**: Maximum capability with minimal local storage

### `config_test_decisions.yaml`
- **LLM**: Qwen 2.5 7B
- **Inference**: HuggingFace only (for availability checks)
- **Purpose**: Decision-only mode testing (no model execution)
- **Best for**: Testing LLM decision-making without running models

### `config.default.yaml` (Original)
- **LLM**: OpenAI text-davinci-003 or GPT-4
- **Inference**: Hybrid mode
- **Deployment**: Full
- **Best for**: Original setup with OpenAI

---

## 🔧 Troubleshooting

### Issue: "HuggingFace token not found"
1. Get token from: https://huggingface.co/settings/tokens
2. Edit your config file:
   ```yaml
   huggingface:
     token: YOUR_TOKEN_HERE
   ```

### Issue: Models server fails to load
- **Solution 1**: Use HuggingFace API mode (no models_server needed)
  - Set `inference_mode: huggingface` in config
- **Solution 2**: Check GPU memory: `nvidia-smi`
- **Solution 3**: Verify models are downloaded in `models/` directory
- **Solution 4**: Use decision-only mode - doesn't require models_server

### Issue: "Port already in use"
- **Qwen server**: port 8006
- **Models server**: port 8005  
- **Demo server**: port 7860
- **Check**: `netstat -tulpn | grep -E '8005|8006|7860'`
- **Kill processes**: `docker exec icdcs pkill -f qwen_server.py`

### Issue: "Local inference endpoints not running"
**For decision-only mode**: This is OK! The warning appears but decision-only mode works without models_server.

**For full execution**: Start models_server.py first.

### Simple Solutions (When models_server fails)

**Option A: Use HuggingFace API Only**
```yaml
# In config_qwen.yaml or config_test_decisions.yaml
inference_mode: huggingface  # No local models needed
```
Then you don't need models_server.py at all!

**Option B: Test Qwen Server Directly**
```bash
curl http://localhost:8006/health
curl -X POST http://localhost:8006/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

**Option C: Use Decision-Only Mode**
- Only needs Qwen server
- Tests LLM decision-making
- No model execution required

---

## 📊 Understanding How It Works

HuggingGPT (JARVIS) works in 4 stages:

1. **Task Planning** - Qwen reads your request and breaks it into tasks
2. **Model Selection** - Qwen picks the best models for each task
3. **Execution** - Models run on your downloaded models (or HuggingFace API)
4. **Response** - Qwen summarizes everything and gives you the answer

**Example**: "What objects are in this image?"
1. ✅ Qwen identifies task: "object-detection"
2. ✅ Qwen selects: `facebook/detr-resnet-101` 
3. ✅ Model detects objects in image
4. ✅ Qwen tells you what objects were found

For detailed technical explanation, see `DECISION_FLOW_EXPLAINED.md`.

---

## 🔗 Useful Commands

```bash
# Check Qwen server status
curl http://localhost:8006/health

# Check GPU usage
nvidia-smi

# View logs
tail -f /home/panthera/JARVIS/hugginggpt/server/logs/debug.log

# Restart everything (in container)
docker exec icdcs pkill -f qwen_server.py
docker exec icdcs pkill -f models_server.py
docker exec icdcs pkill -f run_gradio

# Test decision-only mode
curl -X POST http://localhost:8004/hugginggpt \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Detect objects in /examples/a.jpg"}],
    "return_decisions": true,
    "api_key": "dummy",
    "api_type": "openai",
    "api_endpoint": "http://localhost:8006/v1"
  }'
```

---

## 💡 Tips for Best Results

1. **Be Specific**: "Generate an image of a cat" works better than "make a cat"
2. **Use Existing Images**: Reference the `/public/examples/` images
3. **Chain Tasks**: Ask JARVIS to do multiple things in one request
4. **Describe Images**: Ask "what's in this image" before asking complex questions
5. **Try ControlNet**: Use "pose", "depth", "canny" for better image control
6. **Test Decision-Only First**: Use `return_decisions: true` to see what models would be selected

---

## 📚 Next Steps

1. **For Setup/Reinstallation**: Follow this guide step-by-step
2. **For Testing Decisions**: Use `config_test_decisions.yaml` with `return_decisions: true`
3. **For Full Demo**: Start all 3 servers (Qwen + Models + Gradio)
4. **For Troubleshooting**: See Troubleshooting section above
5. **For Technical Details**: See `DECISION_FLOW_EXPLAINED.md`

Good luck with your demo! 🚀
