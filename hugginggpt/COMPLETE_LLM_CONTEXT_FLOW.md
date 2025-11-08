# Complete LLM Context Flow: Deep Analysis

## Executive Summary

**Number of Prompt Types**: **3 types** (parse_task, choose_model, response_results)

**Total LLM Calls**:
- **awesome_chat.py (with execution)**: 1 + N + 1 = N+2 calls
- **decision_api.py (no execution)**: 1 + N + 0 = N+1 calls

**Context Chaining**: YES - outputs from previous steps become context for later steps

**External Context**: YES - conversation history, few-shot examples, model metadata

---

## File Locations Reference

### Main Files:
- **awesome_chat.py**: `hugginggpt/server/awesome_chat.py`
- **decision_api.py**: `hugginggpt/server/decision_api.py`

### Config Files:
- **Config (awesome_chat.py)**: `hugginggpt/server/configs/config.default.yaml`
- **Config (decision_api.py)**: `hugginggpt/server/configs/config_custom_decisions.yaml`

### Demo Files:
- **parse_task demos**: 
  - `hugginggpt/server/demos/demo_parse_task.json`
  - `hugginggpt/server/demos/demo_parse_task_custom.json`
- **choose_model demos**: 
  - `hugginggpt/server/demos/demo_choose_model.json`
  - `hugginggpt/server/demos/demo_choose_model_custom.json`
- **response_results demo**: 
  - `hugginggpt/server/demos/demo_response_results.json`

---

## Three Prompt Types

### 1. parse_task Prompt
**Purpose**: Parse user input into structured task array  
**Used Once**: At the beginning  
**Code Locations**:
- **awesome_chat.py**: `parse_task()` function (lines 319-350)
- **decision_api.py**: `parse_task()` function (lines 250-305)

### 2. choose_model Prompt
**Purpose**: Select best model for a specific task  
**Used N Times**: Once per task (if 3 tasks → 3 calls)  
**Code Locations**:
- **awesome_chat.py**: `choose_model()` function (lines 352-376)
- **decision_api.py**: `choose_model_for_task()` function (lines 358-417)

### 3. response_results Prompt
**Purpose**: Generate natural language response describing workflow  
**Used Once**: At the end (ONLY in awesome_chat.py)  
**Skipped in decision_api.py**: Replaced with template-based JSON generation  
**Code Locations**:
- **awesome_chat.py**: `response_results()` function (lines 379-400)
- **decision_api.py**: `generate_response_description()` function (lines 691-761) - **NO LLM call**

---

## Complete Flow with Context Details

### awesome_chat.py (WITH EXECUTION)

```
User Input: "Detect objects and faces in image.jpg"
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ LLM CALL #1: parse_task()                                       │
│ Prompt Type: parse_task                                         │
│ CODE: awesome_chat.py lines 319-350, decision_api.py lines 250-305│
│                                                                  │
│ CONTEXT INCLUDES:                                               │
│ 1. System Prompt (tprompt):                                     │
│    - Task structure instructions                                │
│    - Available task list                                        │
│    - Dependency rules                                           │
│    - Config: config.yaml line 31 (tprompt.parse_task)          │
│    - Loaded: line 143/72 (awesome_chat/decision_api)           │
│                                                                  │
│ 2. Few-shot Examples:                                           │
│    - 7+ complete examples from demo_parse_task.json            │
│    - Shows: user query → task array with dependencies          │
│    - Config: config.yaml line 38 (demos_or_presteps.parse_task)│
│    - Loaded: line 135/62 (awesome_chat/decision_api)           │
│    - Inserted: line 321-322/274-275                            │
│                                                                  │
│ 3. Conversation History (if multi-turn):                       │
│    - Previous user messages                                     │
│    - Previous assistant responses                               │
│    - Trimmed to fit context window                             │
│    - Code: lines 326-338/279-294 (context trimming loop)       │
│                                                                  │
│ 4. Current User Input:                                          │
│    - "Detect objects and faces in image.jpg"                   │
│    - Added: line 332/285 (messages.append)                     │
│                                                                  │
│ LLM REQUEST SENT:                                               │
│ - awesome_chat.py: line 350 (send_request)                     │
│ - decision_api.py: line 301 (send_llm_request)                 │
│                                                                  │
│ OUTPUT:                                                         │
│ [{task: "object-detection", id: 0, dep: [-1], ...},           │
│  {task: "face-detection", id: 1, dep: [0], ...}]              │
└─────────────────────────────────────────────────────────────────┘
    ↓
Post-processing (NO LLM):
- unfold(): awesome_chat.py lines 283-307, decision_api.py lines 308-331
- fix_dep(): awesome_chat.py lines 270-281, decision_api.py lines 333-355
- Called from: awesome_chat.py lines 1068-1069, decision_api.py lines 605-606
    ↓
Task Array: [Task 0, Task 1]
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ EXECUTION PHASE (SKIPPED IN DECISION-ONLY MODE)                │
│ CODE: awesome_chat.py lines 1113-1141                          │
│                                                                  │
│ For each task (with dependency ordering):                       │
│   - Check dependencies satisfied (line 1124)                   │
│   - Launch model inference (line 1126) or skip in decision-only│
│   - Collect results (inside run_task)                          │
│                                                                  │
│ DECISION-ONLY: awesome_chat.py lines 1080-1111                 │
│ DECISION-ONLY: decision_api.py lines 637-639                   │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ LLM CALL #2: choose_model() - FOR TASK 0                       │
│ Prompt Type: choose_model                                       │
│ CODE: awesome_chat.py lines 352-376, decision_api.py lines 358-417│
│                                                                  │
│ CONTEXT INCLUDES:                                               │
│ 1. System Prompt (tprompt):                                     │
│    - Model selection instructions                               │
│    - Criteria: description, likes, availability                │
│    - Config: config.yaml line 33 (tprompt.choose_model)        │
│    - Loaded: line 144/71 (awesome_chat/decision_api)           │
│    - Inserted: line 364/391                                    │
│                                                                  │
│ 2. Few-shot Examples:                                           │
│    - Examples from demo_choose_model.json                      │
│    - Shows: task + models → selection with reason             │
│    - Config: config.yaml line 39 (demos_or_presteps.choose_model)│
│    - Loaded: line 136/63 (awesome_chat/decision_api)           │
│    - Processed: lines 358-362/385-389                          │
│                                                                  │
│ 3. User's ORIGINAL Input (external context):                   │
│    - "Detect objects and faces in image.jpg"                   │
│    - (NOT the parsed tasks, but original user query)           │
│    - Embedded in prompt: line 353/380 (replace_slot)           │
│                                                                  │
│ 4. CURRENT Task Details (from parse_task output):              │
│    - Task type: "object-detection"                             │
│    - Task args: {"image": "image.jpg"}                         │
│    - Task id: 0                                                │
│    - Embedded in prompt: line 353-356/380-384                  │
│                                                                  │
│ 5. Candidate Models (external context from catalog/API):       │
│    - Model IDs                                                 │
│    - Descriptions (truncated to 100-200 chars)                 │
│    - Likes count                                               │
│    - Tags                                                      │
│    - Inference endpoint (local/huggingface/api)                │
│    - Built: awesome_chat.py lines 979-992, decision_api.py lines 368-377│
│    - Embedded in prompt: line 353-356/380-384                  │
│                                                                  │
│ LLM REQUEST SENT:                                               │
│ - awesome_chat.py: line 376 (send_request via choose_model)   │
│ - decision_api.py: line 399 (send_llm_request)                 │
│ - Called from: awesome_chat.py line 994/811, decision_api.py line 489│
│                                                                  │
│ OUTPUT:                                                         │
│ {"id": "model-A", "reason": "Best for object detection..."}   │
│ - Parsed: awesome_chat.py lines 996-1005, decision_api.py lines 405-417│
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ LLM CALL #3: choose_model() - FOR TASK 1                       │
│ Prompt Type: choose_model (SAME type as Call #2)               │
│                                                                  │
│ CONTEXT INCLUDES:                                               │
│ 1. System Prompt: SAME as Call #2                              │
│ 2. Few-shot Examples: SAME as Call #2                          │
│ 3. User's ORIGINAL Input: SAME ("Detect objects and faces...")│
│ 4. DIFFERENT Task Details:                                     │
│    - Task type: "face-detection"                               │
│    - Task args: {"image": "image.jpg"}                         │
│    - Task id: 1                                                │
│ 5. DIFFERENT Candidate Models:                                 │
│    - Models for face-detection task (different from Task 0)   │
│                                                                  │
│ NOTE: Does NOT include output from Call #2                     │
│       Each model selection is INDEPENDENT                       │
│                                                                  │
│ OUTPUT:                                                         │
│ {"id": "model-B", "reason": "Specialized for faces..."}       │
└─────────────────────────────────────────────────────────────────┘
    ↓
Model Execution Phase (if not skipped):
- Execute selected models
- Collect inference results
    ↓
Results: {
  0: {task: {...}, inference_result: {predicted: [...], generated_image: "..."}, ...},
  1: {task: {...}, inference_result: {predicted: [...]}, ...}
}
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ LLM CALL #4: response_results()                                │
│ Prompt Type: response_results (DIFFERENT from previous)        │
│ CODE: awesome_chat.py lines 379-400                            │
│                                                                  │
│ CONTEXT INCLUDES:                                               │
│ 1. System Prompt (tprompt):                                     │
│    - Response generation instructions                           │
│    - "Describe process and inference results"                  │
│    - Config: config.yaml line 35 (tprompt.response_results)    │
│    - Loaded: line 145 (awesome_chat.py)                        │
│    - Inserted: line 389                                        │
│                                                                  │
│ 2. Few-shot Examples:                                           │
│    - From demo_response_results.json                           │
│    - Shows: user input + processes → workflow description     │
│    - Config: config.yaml line 40 (demos_or_presteps.response_results)│
│    - Loaded: line 137 (awesome_chat.py)                        │
│    - Processed: lines 384-387                                  │
│                                                                  │
│ 3. User's ORIGINAL Input (external context):                   │
│    - "Detect objects and faces in image.jpg"                   │
│    - Embedded in prompt: lines 381-383                         │
│                                                                  │
│ 4. ALL EXECUTION RESULTS (from Call #2, #3 + execution):       │
│    - Task 0: model-A results (bounding boxes, labels, etc.)   │
│    - Task 1: model-B results (face detections, etc.)          │
│    - Includes: task details, chosen models, inference results  │
│    - Sorted results: line 380                                  │
│    - Embedded via {{processes}}: lines 384-387                 │
│                                                                  │
│ LLM REQUEST SENT:                                               │
│ - awesome_chat.py: line 400 (send_request)                     │
│ - Called from: line 1146                                       │
│                                                                  │
│ OUTPUT:                                                         │
│ "Based on the inference results, I performed the following     │
│  workflow: First, I used model-A to detect objects in the     │
│  image, finding [objects]. Then I used model-B to detect      │
│  faces, finding [faces]..."                                    │
│ - Returned: line 1146 (stored in 'response' variable)         │
└─────────────────────────────────────────────────────────────────┘
    ↓
Final Response (Natural Language)
- Returned to caller: line 1151 ({"message": response})
```

**Total LLM Calls**: 1 (parse) + 2 (choose per task) + 1 (response) = **4 calls**

---

### decision_api.py (NO EXECUTION)

```
User Input: "Detect objects and faces in image.jpg"
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ LLM CALL #1: parse_task()                                       │
│ IDENTICAL TO awesome_chat.py                                    │
│ (Same context, same structure)                                  │
└─────────────────────────────────────────────────────────────────┘
    ↓
Post-processing: unfold(), fix_dep(), semantic mapping (optional)
    ↓
Task Array: [Task 0, Task 1]
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ LLM CALL #2: choose_model_for_task() - FOR TASK 0             │
│ IDENTICAL TO awesome_chat.py                                    │
│ (Same context, same structure)                                  │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ LLM CALL #3: choose_model_for_task() - FOR TASK 1             │
│ IDENTICAL TO awesome_chat.py                                    │
│ (Same context, same structure)                                  │
└─────────────────────────────────────────────────────────────────┘
    ↓
Decision Results: {
  0: {task: {...}, selected_model: "model-A", reason: "...", status: "not_executed"},
  1: {task: {...}, selected_model: "model-B", reason: "...", status: "not_executed"}
}
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ NO LLM CALL: generate_response_description()                   │
│ Method: Template-based JSON generation (NO LLM)                │
│ CODE: decision_api.py lines 691-761                            │
│                                                                  │
│ INPUT (from previous steps):                                    │
│ - User input (from Call #1 input)                             │
│ - Tasks array (from Call #1 output)                           │
│ - Model selections (from Calls #2, #3 outputs)                │
│ - Passed in: line 666                                          │
│                                                                  │
│ PROCESSING (pure code, no LLM):                                │
│ - Calculate execution levels based on dependencies (line 711-718)│
│ - Build DAG edges (line 717-718)                               │
│ - Group tasks by level for parallel execution (lines 733-742) │
│ - Construct JSON structure (lines 747-758)                    │
│                                                                  │
│ OUTPUT:                                                         │
│ {                                                              │
│   "execution_config": {                                        │
│     "user_request": "...",                                     │
│     "total_tasks": 2,                                          │
│     "tasks": [...],                                            │
│     "dag": {"nodes": [...], "edges": [...]},                   │
│     "execution_order": [...]                                   │
│   }                                                            │
│ }                                                              │
│ - Returned: line 761 (json.dumps)                             │
│ - Saved to file: lines 669-677                                │
└─────────────────────────────────────────────────────────────────┘
    ↓
Final Output (Structured JSON)
```

**Total LLM Calls**: 1 (parse) + 2 (choose per task) + 0 (no response_results) = **3 calls**

---

## Call Stack: Where Each LLM Call is Invoked

### awesome_chat.py Call Stack:

```
User Request → HTTP /hugginggpt endpoint
    ↓
server() (line 1192) → chat() endpoint (line 1228)
    ↓
chat_huggingface() (line 1017)
    ↓
    ├─→ LLM CALL #1: parse_task() (line 1032)
    │                Function: lines 319-350
    │
    ├─→ unfold() (line 1068) - NO LLM
    ├─→ fix_dep() (line 1069) - NO LLM
    │
    ├─→ Execution Loop (lines 1113-1141) OR Decision Loop (lines 1080-1111)
    │   │
    │   ├─→ For each task:
    │       ├─→ LLM CALL #2, #3, ...: choose_model() (line 994 or 811)
    │       │                          Function: lines 352-376
    │       │                          Called from: run_task() or run_task_decisions_only()
    │       │
    │       └─→ model_inference() (line 1006) - SKIPPED in decision-only mode
    │
    └─→ LLM CALL #N+2: response_results() (line 1146) - ONLY in execution mode
                       Function: lines 379-400
                       SKIPPED in decision-only mode
```

### decision_api.py Call Stack:

```
User Request → HTTP /decisions endpoint
    ↓
decisions() (line 518)
    ↓
    ├─→ LLM CALL #1: parse_task() (line 539)
    │                Function: lines 250-305
    │
    ├─→ unfold() (line 605) - NO LLM
    ├─→ fix_dep() (line 606) - NO LLM
    ├─→ map_task_semantically() (line 631) - OPTIONAL LLM CALL
    │                            Function: lines 112-160
    │
    ├─→ process_decisions() (line 639)
    │   │   Function: lines 420-504
    │   │
    │   └─→ For each task (line 424):
    │       └─→ LLM CALL #2, #3, ...: choose_model_for_task() (line 489)
    │                                  Function: lines 358-417
    │
    └─→ NO LLM: generate_response_description() (line 666)
                Function: lines 691-761
                Pure template-based JSON generation
```

---

## Detailed Context Analysis for Each LLM Call

### LLM Call #1: parse_task()

**Prompt Type**: `parse_task`

**Function Definition**:
- **awesome_chat.py**: lines 319-350
- **decision_api.py**: lines 250-305

**Called From**:
- **awesome_chat.py**: line 1032 (from `chat_huggingface()`)
- **decision_api.py**: line 539 (from `decisions()` endpoint)

**Message Structure**:
```python
messages = [
    {"role": "system", "content": tprompt},
    # Few-shot examples:
    {"role": "user", "content": "Example query 1"},
    {"role": "assistant", "content": "[{task: ..., id: 0, dep: [-1], ...}]"},
    {"role": "user", "content": "Example query 2"},
    {"role": "assistant", "content": "[{task: ..., ...}]"},
    # ... 5-7 more examples ...
    # Optional: conversation history (multi-turn)
    {"role": "user", "content": "Previous user message"},
    {"role": "assistant", "content": "Previous response"},
    # Current request:
    {"role": "user", "content": "Current user input"}
]
```

**Context Components**:

1. **System Prompt (tprompt)**:
   - Source: `config["tprompt"]["parse_task"]`
   - Content: Task structure instructions, available tasks, dependency rules
   - Config location: `config.default.yaml` line 31 or `config_custom_decisions.yaml` line 52
   - Loaded at: awesome_chat.py line 143, decision_api.py line 72
   - Inserted at: awesome_chat.py line 322, decision_api.py line 275
   - External? NO - static from config
   - Same for all calls? YES

2. **Few-shot Examples**:
   - Source: `demo_parse_task.json` or `demo_parse_task_custom.json`
   - Content: 7+ complete examples (user query → task array)
   - Config location: `config.default.yaml` line 38 or `config_custom_decisions.yaml` line 61
   - Loaded at: awesome_chat.py line 135, decision_api.py line 62
   - Parsed at: awesome_chat.py line 321, decision_api.py line 274
   - External? NO - static from file
   - Same for all calls? YES (per config)

3. **Conversation History**:
   - Source: Previous messages in the session
   - Content: User queries and assistant responses from earlier turns
   - Trimming logic: awesome_chat.py lines 326-338, decision_api.py lines 279-294
   - Token counting: awesome_chat.py line 334, decision_api.py line 290
   - Context window check: awesome_chat.py line 335, decision_api.py line 291
   - External? YES - from user's conversation state
   - Same for all calls? NO - grows with conversation
   - Trimmed: YES - to fit context window (leaves 800 tokens for response)

4. **Current User Input**:
   - Source: User's current request
   - Content: The actual query (e.g., "Detect faces in image.jpg")
   - Embedded via: awesome_chat.py lines 328-332, decision_api.py lines 281-285
   - Template: `config["prompt"]["parse_task"]`
   - Template location: config.yaml line 42 (awesome_chat) or line 67 (decision_api)
   - External? YES - from user
   - Same for all calls? NO - unique to this turn

**Previous LLM Output Used?**: NO (first call)

**Output Structure**: JSON array of tasks
```json
[
  {"task": "task-name", "id": 0, "dep": [-1], "args": {"image": "..."}},
  {"task": "task-name", "id": 1, "dep": [0], "args": {"image": "..."}}
]
```

---

### LLM Call #2, #3, ... #N+1: choose_model()

**Prompt Type**: `choose_model` (SAME for all model selection calls)

**Message Structure**:
```python
messages = [
    {"role": "system", "content": tprompt},
    # Few-shot examples:
    {"role": "user", "content": "Choose model for task X from [models]"},
    {"role": "assistant", "content": '{"id": "model-id", "reason": "..."}'},
    # ... more examples ...
    # Current request:
    {"role": "user", "content": "Choose model for current task from {{metas}}"}
]
```

**Context Components**:

1. **System Prompt (tprompt)**:
   - Source: `config["tprompt"]["choose_model"]`
   - Content: Model selection criteria, instructions
   - External? NO - static from config
   - Same for all model selection calls? YES

2. **Few-shot Examples**:
   - Source: `demo_choose_model.json` or `demo_choose_model_custom.json`
   - Content: Examples of model selection with reasoning
   - External? NO - static from file
   - Same for all model selection calls? YES

3. **User's ORIGINAL Input** (embedded in user prompt):
   - Source: From Call #1 input (user's original query)
   - Content: "Detect faces in image.jpg" (the original request)
   - External? YES - carried forward from Call #1
   - Same for all model selection calls? YES (same original query)
   - **IMPORTANT**: This is the ORIGINAL user input, not the parsed tasks

4. **Current Task Details** (embedded in user prompt):
   - Source: From Call #1 OUTPUT (parse_task result)
   - Content: 
     - Task type: "face-detection"
     - Task args: {"image": "image.jpg"}
     - Task id: 1
   - External? NO - from previous LLM output (Call #1)
   - Same for all model selection calls? NO - different task per call
   - **KEY**: Output of Call #1 feeds into Calls #2, #3, ...

5. **Candidate Models Metadata**:
   - Source: External model catalog (p0_models.jsonl) or API
   - Content:
     - Model IDs
     - Descriptions (truncated to max_description_length)
     - Likes count
     - Tags
     - Inference endpoint
   - External? YES - from model database
   - Same for all model selection calls? NO - different models per task type
   - Example:
     ```python
     [
       {
         "id": "facebook/detr-resnet-50",
         "description": "DETR model for object detection...",
         "likes": 1234,
         "tags": ["object-detection", "computer-vision"],
         "inference endpoint": "huggingface"
       },
       # ... more models ...
     ]
     ```

**Previous LLM Output Used?**: 
- YES - Task details from Call #1 (parse_task)
- NO - Does NOT use model selections from other tasks (Call #2 doesn't see Call #3's result)

**Output Structure**: JSON object with model ID and reason
```json
{
  "id": "facebook/detr-resnet-50",
  "reason": "This model has high accuracy for object detection and is widely used..."
}
```

**Important**: Each model selection call is INDEPENDENT. Call #2 doesn't know about Call #3's selection.

---

### LLM Call #N+2: response_results() (ONLY in awesome_chat.py)

**Prompt Type**: `response_results` (DIFFERENT from previous calls)

**Message Structure**:
```python
messages = [
    {"role": "system", "content": tprompt},
    # Few-shot example:
    {"role": "user", "content": "{{input}}"},
    {"role": "assistant", "content": "Workflow intro: {{processes}}. Any demands?"},
    # Current request:
    {"role": "user", "content": "User's original input"}
]
```

**Context Components**:

1. **System Prompt (tprompt)**:
   - Source: `config["tprompt"]["response_results"]`
   - Content: "Describe process and inference results"
   - External? NO - static from config
   - Same as previous calls? NO - different prompt type

2. **Few-shot Example**:
   - Source: `demo_response_results.json`
   - Content: Shows how to describe workflow with results
   - External? NO - static from file
   - Example structure:
     ```json
     [
       {"role": "user", "content": "{{input}}"},
       {"role": "assistant", "content": "Before give you a response, I want to introduce my workflow for your request, which is shown in the following JSON data: {{processes}}. Do you have any demands regarding my response?"}
     ]
     ```

3. **User's ORIGINAL Input**:
   - Source: From Call #1 input
   - Content: "Detect faces in image.jpg"
   - External? YES - carried forward from Call #1
   - Same as in model selection? YES

4. **ALL EXECUTION RESULTS** (embedded in few-shot example via {{processes}}):
   - Source: Aggregated from:
     - Call #1 output (tasks)
     - Calls #2-#N+1 outputs (model selections)
     - Execution results (inference outputs)
   - Content: Array of result objects:
     ```python
     [
       {
         "task": {"task": "object-detection", "id": 0, "args": {...}},
         "inference result": {
           "predicted": [{"label": "person", "box": {...}}, ...],
           "generated image": "/images/xyz.jpg"
         },
         "choose model result": {
           "id": "facebook/detr-resnet-50",
           "reason": "Best for object detection..."
         }
       },
       # ... more results ...
     ]
     ```
   - External? NO - from ALL previous calls + execution
   - **KEY**: Outputs from Calls #1, #2, #3, ... AND execution results feed into this call

**Previous LLM Output Used?**: 
- YES - Tasks from Call #1
- YES - Model selections from Calls #2, #3, ...
- YES - Execution results (if executed)

**Output Structure**: Natural language text
```
Based on the inference results, I performed the following workflow:
First, I used the facebook/detr-resnet-50 model to detect objects in the image.
The model identified 3 people and 2 cars. Then I used the retina-face model
to detect faces in the detected people, finding 3 faces...
```

---

### decision_api.py: No Call #N+2

**Instead of LLM call**: `generate_response_description()` - Template-based

**Input**:
- User's original input (from Call #1)
- Tasks array (from Call #1)
- Model selections (from Calls #2, #3, ...)

**Processing** (NO LLM):
1. Calculate execution levels based on dependencies
2. Build DAG edges
3. Group tasks by level
4. Construct JSON

**Output**: Structured JSON (not natural language)

---

## Context Chaining: How Outputs Feed Forward

### Flow of Information:

```
Call #1 (parse_task)
    ↓ OUTPUT: Task array with dependencies
    ↓
    ├─→ Call #2 (choose_model for Task 0)
    │       ↓ OUTPUT: Selected model for Task 0
    │       ↓
    ├─→ Call #3 (choose_model for Task 1)
    │       ↓ OUTPUT: Selected model for Task 1
    │       ↓
    └─→ ... (more model selections)
            ↓
            ↓ ALL OUTPUTS AGGREGATED
            ↓
    ┌───────┴───────────────────────────────┐
    │                                       │
    ↓ awesome_chat.py                      ↓ decision_api.py
    Execute models                         Skip execution
    Collect inference results              Use selection results
    ↓                                      ↓
    Call #N+2 (response_results)          generate_response_description()
    INPUT: All tasks + selections + results   INPUT: All tasks + selections
    ↓                                      ↓
    Natural language response              Structured JSON
```

**Key Points**:
1. **Call #1 → Calls #2-#N+1**: Task details feed into model selection
2. **Calls #2-#N+1 are independent**: Don't see each other's outputs during selection
3. **All calls → Final step**: Everything aggregates at the end

---

## Same or Different Prompts?

### System Prompts (tprompt):

| Call | Prompt Type | Same as Others? |
|------|-------------|-----------------|
| #1 | parse_task | NO - unique |
| #2-#N+1 | choose_model | YES - all model selections use same |
| #N+2 | response_results | NO - unique |

### User Prompts (prompt):

| Call | Content | Same as Others? |
|------|---------|-----------------|
| #1 | User input + context | Unique |
| #2-#N+1 | User input + task + models | Same template, different values |
| #N+2 | User input + results | Unique |

### Few-shot Examples:

| Call | Demo File | Same as Others? |
|------|-----------|-----------------|
| #1 | demo_parse_task.json | NO |
| #2-#N+1 | demo_choose_model.json | YES - all use same |
| #N+2 | demo_response_results.json | NO |

---

## Output Structure Consistency

| Call | Output Format | Consistent Across Calls? |
|------|---------------|--------------------------|
| #1 | JSON array: `[{task:..., id:..., dep:..., args:...}, ...]` | N/A (only one parse call) |
| #2-#N+1 | JSON object: `{"id": "model-id", "reason": "..."}` | YES - all model selections return same structure |
| #N+2 | Natural language text | N/A (only one response call) |

**Key**: Model selection calls (#2-#N+1) all return the SAME structure.

---

## External Context Summary

### What External Context is Used?

1. **User Input** (all calls):
   - Original user query
   - Carries through all calls

2. **Conversation History** (Call #1 only):
   - Previous messages in session
   - Multi-turn conversations

3. **Model Catalog** (Calls #2-#N+1):
   - Model metadata from p0_models.jsonl or API
   - Descriptions, likes, tags, availability

4. **Few-shot Examples** (all calls):
   - Static examples from JSON files
   - Not technically "external" but separate from runtime data

5. **Config Prompts** (all calls):
   - System prompts from YAML config
   - Static configuration

### What Internal Context Flows Between Calls?

1. **Call #1 → Calls #2-#N+1**:
   - Task details (type, args, id)
   - Used to select appropriate models

2. **Calls #1-#N+1 → Call #N+2** (awesome_chat.py only):
   - All tasks
   - All model selections
   - All inference results (if executed)
   - Used to generate final response

---

## Key Differences: awesome_chat.py vs decision_api.py

| Aspect | awesome_chat.py | decision_api.py |
|--------|----------------|-----------------|
| **LLM Calls** | 1 + N + 1 = N+2 | 1 + N + 0 = N+1 |
| **Final Response** | LLM-generated natural language | Template-based JSON |
| **Context for Final Step** | Tasks + selections + execution results | Tasks + selections only |
| **Execution Phase** | YES - runs models | NO - skipped |
| **Output Format** | Friendly text for users | Structured config for execution server |

---

## Summary Table

| LLM Call | Prompt Type | Context Includes | Uses Previous Output? | External Context? | Output Format |
|----------|-------------|------------------|-----------------------|-------------------|---------------|
| **#1: parse_task** | parse_task | System prompt, few-shot examples, conversation history, user input | NO (first call) | YES (user input, history) | JSON array of tasks |
| **#2: choose_model (Task 0)** | choose_model | System prompt, few-shot examples, user input, Task 0 details, Task 0 models | YES (Task 0 from #1) | YES (user input, models from catalog) | JSON: model + reason |
| **#3: choose_model (Task 1)** | choose_model | System prompt, few-shot examples, user input, Task 1 details, Task 1 models | YES (Task 1 from #1) | YES (user input, models from catalog) | JSON: model + reason |
| **#N+2: response_results** (awesome_chat.py only) | response_results | System prompt, few-shot example, user input, ALL results | YES (all previous + execution) | YES (user input) | Natural language text |
| **generate_response_description** (decision_api.py only) | N/A - not LLM | Tasks, model selections | YES (all previous) | NO | Structured JSON |

---

## Conclusion

### Key Insights:

1. **Three Prompt Types**: parse_task, choose_model, response_results
   - Each has unique system prompt, user prompt template, and few-shot examples

2. **Context Chaining**: YES
   - Call #1 output feeds into Calls #2-#N+1
   - All calls feed into final step

3. **Independent Model Selection**: 
   - Each choose_model call is independent
   - Don't see other tasks' model selections during their own selection

4. **External Context**:
   - User input (all calls)
   - Conversation history (Call #1)
   - Model metadata (Calls #2-#N+1)
   - Few-shot examples (all calls)

5. **Main Difference**:
   - awesome_chat.py: Uses LLM for final natural language response
   - decision_api.py: Uses template code for final structured JSON

6. **Same Prompt Type, Different Values**:
   - All model selection calls use the same prompt type
   - Different task details and models per call

---

## Quick Reference: Line Numbers

### Core LLM Functions

| Function | awesome_chat.py | decision_api.py | Purpose |
|----------|----------------|-----------------|---------|
| `parse_task()` | Lines 319-350 | Lines 250-305 | LLM Call #1: Parse tasks |
| `choose_model()` | Lines 352-376 | Lines 358-417 | LLM Calls #2-N: Select models |
| `response_results()` | Lines 379-400 | N/A | LLM Call #N+1: Generate response |
| `generate_response_description()` | N/A | Lines 691-761 | NO LLM: Template JSON |

### Main Orchestrator

| Function | awesome_chat.py | decision_api.py | Purpose |
|----------|----------------|-----------------|---------|
| Main orchestrator | `chat_huggingface()` 1017-1154 | `decisions()` 518-688 | Entry point |
| Task parsing call | Line 1032 | Line 539 | Calls `parse_task()` |
| Post-processing | Lines 1068-1069 | Lines 605-606 | `unfold()`, `fix_dep()` |
| Model selection loop | Lines 1113-1141 (exec)<br>Lines 1080-1111 (decision) | Lines 637-639 → 420-504 | Loop through tasks |
| Final response | Line 1146 | Line 666 | LLM or template |

### Helper Functions

| Function | awesome_chat.py | decision_api.py | Purpose |
|----------|----------------|-----------------|---------|
| `unfold()` | Lines 283-307 | Lines 308-331 | Expand multiple GENERATED refs |
| `fix_dep()` | Lines 270-281 | Lines 333-355 | Fix dependencies |
| `run_task()` | Lines 842-1015 | N/A | Execute one task |
| `run_task_decisions_only()` | Lines 722-840 | N/A | Decision for one task |
| `process_decisions()` | N/A | Lines 420-504 | Decision loop |
| `map_task_semantically()` | N/A | Lines 112-160 | Optional LLM mapping |

### Config Loading

| Component | awesome_chat.py | decision_api.py | Config File |
|-----------|----------------|-----------------|-------------|
| System prompts (tprompt) | Lines 143-145 | Lines 70-72 | Lines 31-35 (config.yaml) |
| Few-shot demos | Lines 135-137 | Lines 62-64 | Lines 38-40 (config.yaml) |
| User prompt templates | Lines 139-141 | Lines 66-68 | Lines 42-46 (config.yaml) |

### LLM Request Sending

| Function | awesome_chat.py | decision_api.py | Purpose |
|----------|----------------|-----------------|---------|
| `send_request()` | Lines 191-215 | N/A | Send to LLM API |
| `send_llm_request()` | N/A | Lines 222-247 | Send to LLM API |

### Key Execution Points

| Action | awesome_chat.py | decision_api.py |
|--------|----------------|-----------------|
| Parse task called | Line 1032 | Line 539 |
| Choose model called | Line 994 (exec)<br>Line 811 (decision) | Line 489 |
| Response generation called | Line 1146 | Line 666 (no LLM) |
| Unfold called | Line 1068 | Line 605 |
| Fix dependencies called | Line 1069 | Line 606 |

### Config File Locations

| Component | Line (config.default.yaml) | Line (config_custom_decisions.yaml) |
|-----------|----------------------------|-------------------------------------|
| tprompt.parse_task | 31 | 52 |
| tprompt.choose_model | 33 | 54 |
| tprompt.response_results | 35 | 56 |
| demos_or_presteps.parse_task | 38 | 61 |
| demos_or_presteps.choose_model | 39 | 62 |
| demos_or_presteps.response_results | 40 | 63 |
| prompt.parse_task | 42 | 67 |
| prompt.choose_model | 44 | 69 |
| prompt.response_results | 46 | 71 |

---

**Document Version**: 1.0  
**Date**: 2025-11-01  
**Verified By**: Code analysis of all LLM call paths

