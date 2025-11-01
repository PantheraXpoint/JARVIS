# Decision-Making Flow in HuggingGPT

## 🎯 Overview

This document explains **how HuggingGPT makes decisions** from receiving a user request to selecting models **BEFORE actual execution**.

---

## 📊 Complete Decision Flow (Request → Model Selection)

### **Phase 1: Task Planning** (Steps 1-5)

**Step 1: Receive Request** 
- Location: `server()` → `/hugginggpt` route (Line 1221-1234)
- HTTP POST with user message

**Step 2: Extract Input**
- Location: `chat_huggingface()` (Line 1021)
- Extracts `messages[-1]["content"]` as user input

**Step 3: Parse Task - LLM Call #1** ⚡
- Location: `parse_task()` (Lines 317-348)
- **LLM Reasoning Process:**
  - **System Prompt** (`parse_task_tprompt`): Instructs LLM to "Think step by step about all the tasks needed"
  - **Few-Shot Examples** (`demos/demo_parse_task.json`): Shows LLM how to break down complex requests
  - **User Prompt** (`parse_task_prompt`): "The chat log [context] may contain resources. Now I input {input}. Pay attention to dependencies."
  
- **Chain of Thought:** The prompt explicitly says **"Think step by step"** - this triggers the LLM to reason through:
  1. What is the user asking for?
  2. What subtasks are needed?
  3. What are the dependencies between tasks? (e.g., Task B needs output from Task A)
  4. What resources (text/image/audio) does each task need?

- **Output Format:** LLM returns JSON array of tasks:
  ```json
  [
    {"task": "object-detection", "id": 0, "dep": [-1], "args": {"image": "photo.jpg"}},
    {"task": "visual-question-answering", "id": 1, "dep": [0], "args": {"image": "<GENERATED>-0", "text": "how many objects?"}}
  ]
  ```

**Step 4: Parse & Validate JSON**
- Location: Lines 1043-1059
- Cleans JSON, handles errors, validates structure

**Step 5: Task Post-Processing**
- `unfold()` (Line 1061): Expands tasks with multiple `<GENERATED>-X` references
- `fix_dep()` (Line 1062): Validates and fixes dependency graph

---

### **Phase 2: Model Selection** (Steps 6-10, per task)

**Step 6: Get Candidate Models**
- Location: `run_task()` or `run_task_decisions_only()` (Line 780)
- Queries `MODELS_MAP[task]` - built from `p0_models.jsonl` (Line 145-154)
- Gets top 10 candidates for the task type

**Step 7: Check Availability**
- Location: `get_avaliable_models()` (Lines 678-710)
- **Parallel Status Checks:**
  - Checks local server: `GET /status/<model_id>` (if `inference_mode != "huggingface"`)
  - Checks HuggingFace API: `GET /status/<model_id>` (if `inference_mode != "local"`)
- Returns lists: `{"local": [...], "huggingface": [...]}`

**Step 8-10: Model Selection Decision** 🧠

#### **Case A: Only 1 Model Available**
- **Location:** Lines 789-793
- **Auto-selection:** No LLM call needed
- **Reason:** "Only one model available."

#### **Case B: Multiple Models Available**
- **Location:** `choose_model()` (Lines 350-374) → **LLM Call #2**
- **LLM Reasoning Process:**
  - **System Prompt** (`choose_model_tprompt`): 
    > "Given the user request and parsed tasks, help select a suitable model. Focus on model description. Prefer local endpoints for speed."
  
  - **Few-Shot Examples:** `demos/demo_choose_model.json` (shows selection pattern)
  
  - **User Prompt** (`choose_model_prompt`):
    > "Please choose the most suitable model from {metas} for task {task}. Output JSON: {\"id\": \"id\", \"reason\": \"your detail reasons\"}."
  
  - **Input to LLM:**
    ```json
    {
      "input": "user's original request",
      "task": {"task": "object-detection", "id": 0, ...},
      "metas": [
        {
          "id": "facebook/detr-resnet-101",
          "inference endpoint": "local",
          "likes": 234,
          "description": "DETR model fine-tuned on COCO...",
          "tags": ["object-detection", "detection"]
        },
        ...
      ]
    }
    ```
  
  - **LLM Reasoning (Implicit Chain of Thought):**
    1. Reads model descriptions
    2. Compares capabilities to task requirements
    3. Considers endpoint location (local preferred)
    4. Considers popularity (likes)
    5. Returns selection with detailed reasoning
    
  - **Output:** 
    ```json
    {
      "id": "facebook/detr-resnet-101",
      "reason": "This model is specifically designed for object detection and is available locally for faster inference."
    }
    ```

#### **Case C: Special Cases (No LLM)**
- **ControlNet tasks:** Auto-selected (Line 754-767)
- **Simple NLP tasks:** Auto-selected to "ChatGPT" (Line 769-772)

---

## 🔄 Task Scheduling Loop Explained

**Purpose:** Execute tasks **in parallel** while respecting **dependencies**.

**Location:** Lines 1107-1131

**How It Works:**

```python
while True:
    # Check each task
    for task in tasks:
        dep = task["dep"]
        
        # Can run if:
        # 1. No dependencies (dep[0] == -1), OR
        # 2. All dependencies completed (their IDs are in results dict)
        if dep[0] == -1 or all_deps_completed(dep, results):
            # Start task in separate thread
            thread = Thread(target=run_task, args=(...))
            thread.start()
            tasks.remove(task)  # Remove from queue
    
    # If no new tasks started this iteration, wait
    if no_progress:
        time.sleep(0.5)  # Wait for running tasks
        retry += 1
        if retry > 160:
            break  # Timeout
    
    if len(tasks) == 0:
        break  # All tasks scheduled

# Wait for all threads to finish
for thread in threads:
    thread.join()
```

**Example Dependency Resolution:**
```
Task 0: object-detection (dep: [-1])      → Run immediately
Task 1: visual-qa (dep: [0])              → Wait for Task 0
Task 2: text-to-image (dep: [-1])         → Run immediately (parallel with Task 0)
Task 3: text-to-speech (dep: [2])          → Wait for Task 2
```

**Timeline:**
```
Time 0: Start Task 0, Task 2 (parallel)
Time 1: Task 0 completes → Start Task 1
Time 2: Task 2 completes → Start Task 3
Time 3: Task 1, Task 3 complete
```

**Why This Matters:** 
- **Decision-only mode** doesn't need this loop because we're not waiting for execution results
- **Normal mode** needs it to handle dependencies (e.g., Task B needs Task A's output image)

---

## 🎯 Decision-Only Mode Usage

### **What It Does:**
1. ✅ Receives user request
2. ✅ Parses tasks (LLM call #1)
3. ✅ Selects models for each task (LLM call #2 per task)
4. ❌ **Does NOT execute models**
5. ❌ **Does NOT generate final response**

### **How to Use:**

#### **Via API (HTTP POST):**
```bash
curl -X POST http://localhost:8000/hugginggpt \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Detect objects in /examples/a.jpg"}],
    "api_key": "your_key",
    "api_type": "openai",
    "api_endpoint": "http://localhost:8006/v1",
    "return_decisions": true
  }'
```

#### **Via CLI (Python):**
```python
from awesome_chat import chat_huggingface

messages = [{"role": "user", "content": "Detect objects in /examples/a.jpg"}]
result = chat_huggingface(
    messages, 
    api_key="...",
    api_type="openai",
    api_endpoint="http://localhost:8006/v1",
    return_decisions=True
)

print(json.dumps(result, indent=2))
```

### **Output Format:**

```json
{
  "decisions": {
    "input": "Detect objects in /examples/a.jpg",
    "planned_tasks": [
      {"task": "object-detection", "id": 0, "dep": [-1], "args": {"image": "/examples/a.jpg"}}
    ],
    "model_selections": {
      "0": {
        "task": {"task": "object-detection", "id": 0, ...},
        "selected_model": "facebook/detr-resnet-101",
        "hosted_on": "local",
        "reason": "This model is specifically designed for object detection...",
        "status": "selected_not_executed",
        "available_candidates": 3
      }
    },
    "summary": {
      "total_tasks": 1,
      "tasks_with_selections": 1,
      "tasks_unavailable": 0
    }
  },
  "raw_results": {
    "0": {
      "task": {...},
      "choose model result": {"id": "facebook/detr-resnet-101", "reason": "..."},
      "selected_model_id": "facebook/detr-resnet-101",
      "hosted_on": "local",
      "status": "selected_not_executed"
    }
  }
}
```

---

## 🧠 Decision-Making Deep Dive

### **Does HuggingGPT Use Chain of Thought?**

**Yes, but implicitly:**
1. **Task Planning:** System prompt says "**Think step by step** about all tasks needed"
2. **Model Selection:** LLM naturally reasons through:
   - Model capabilities vs. task requirements
   - Speed/stability trade-offs (local vs. remote)
   - Quality signals (likes, descriptions)

**Not explicit CoT like:**
- "First, I need to... Second, I should..."
- But LLM's reasoning emerges in the `reason` field

### **Decision Factors:**

1. **Task-Model Match:** Does model description fit the task?
2. **Endpoint Location:** Local preferred (configurable)
3. **Model Popularity:** `likes` count (from p0_models.jsonl)
4. **Tags/Categories:** Model tags match task type
5. **Availability:** Only models with active endpoints are considered

---

## 📝 Key Code Locations Summary

| Component | Function | Lines |
|-----------|----------|-------|
| **Task Planning** | `parse_task()` | 317-348 |
| **Model Selection** | `choose_model()` | 350-374 |
| **Availability Check** | `get_avaliable_models()` | 678-710 |
| **Decision-Only Execution** | `run_task_decisions_only()` | 720-836 |
| **Task Scheduling** | `chat_huggingface()` loop | 1107-1131 |
| **API Endpoint** | `/hugginggpt` route | 1221-1234 |

---

## 🔍 Prompt Templates Location

- **System Prompts:** `configs/*.yaml` → `tprompt.parse_task`, `tprompt.choose_model`
- **User Prompts:** `configs/*.yaml` → `prompt.parse_task`, `prompt.choose_model`
- **Few-Shot Examples:** `demos/demo_parse_task.json`, `demos/demo_choose_model.json`

---

## 🚀 Next Steps

To customize decision-making:
1. **Modify prompts** in your config file (`configs/config_qwen.yaml` or similar)
2. **Add examples** to `demos/demo_choose_model.json` for better LLM reasoning
3. **Adjust availability logic** in `get_avaliable_models()` (Line 678)
4. **Change selection criteria** in `choose_model()` prompt template

