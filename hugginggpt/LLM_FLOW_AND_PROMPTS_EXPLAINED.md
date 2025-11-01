# LLM Flow and Prompt Templates: decision_api.py vs awesome_chat.py

## Executive Summary

**Prompt Templates**: Both use **3 prompt templates** (parse_task, choose_model, response_results)

**LLM Call Strategy**: **Few-shot learning** (not chain-of-thought)

**Number of LLM Calls**:
- **decision_api.py**: 2-3 calls (1 parse_task + 1 per task for choose_model + 0-1 optional semantic mapping)
- **awesome_chat.py**: 3 calls (1 parse_task + 1 per task for choose_model + 1 response_results)

**Key Difference**: `decision_api.py` replaces the final `response_results` LLM call with **template-based JSON generation** (no LLM).

---

## Complete Flow Breakdown

### decision_api.py Flow

```
User Input
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Parse Tasks (LLM Call #1)                            │
│ Function: parse_task()                                       │
│ Prompt Template: parse_task_tprompt + parse_task_prompt      │
│ Few-shot Examples: demo_parse_task_custom.json               │
│ Context: Previous conversation messages                      │
│ Output: JSON array of tasks                                 │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ POST-PROCESSING (No LLM):                                     │
│ - unfold(): Expand multiple <GENERATED> references          │
│ - fix_dep(): Fix dependencies                                │
│ - Replace <GENERATED> tags with original image paths        │
│ - map_task_semantically(): Optional LLM call if task names  │
│   don't match API (LLM Call #1.5 - conditional)             │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Choose Models (LLM Call #2, #3, #4... - one per task)│
│ Function: choose_model_for_task()                           │
│ Prompt Template: choose_model_tprompt + choose_model_prompt │
│ Few-shot Examples: demo_choose_model_custom.json            │
│ Context: User input + Task details + Model candidates      │
│ Output: Selected model ID + reason for each task            │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Generate Execution Config (NO LLM - Template-based) │
│ Function: generate_response_description()                   │
│ Method: Direct JSON construction from results                │
│ Output: Structured JSON with DAG and execution order        │
└─────────────────────────────────────────────────────────────┘
    ↓
Final Output (JSON)
```

---

## Prompt Templates and Context Details

### Template 1: parse_task

**Location in code**: `decision_api.py` lines 250-305

**Prompt Structure**:
```python
messages = [
    {"role": "system", "content": parse_task_tprompt},  # System prompt with task list
    ...few-shot examples from demo_parse_task_custom.json...,
    {"role": "user", "content": parse_task_prompt}       # User prompt with {{input}} and {{context}}
]
```

**System Prompt (tprompt)**:
```
#1 Task Planning Stage: The AI assistant can parse user input to several tasks: 
[{"task": task, "id": task_id, "dep": dependency_task_id, "args": {...}}]
...
The task MUST be selected from the following options: [TASKS_LIST_WILL_BE_REPLACED_DYNAMICALLY]
...
Think step by step about all the tasks needed to resolve the user's request.
```

**User Prompt Template**:
```
The chat log [ {{context}} ] may contain the resources I mentioned. 
Now I input { {{input}} }. 
Pay attention to the input and output types of tasks and the dependencies between tasks.
```

**Few-shot Examples Format** (from `demo_parse_task_custom.json`):
```json
[
    {
        "role": "user",
        "content": "Detect objects first, then detect faces in /examples/image1.jpg"
    },
    {
        "role": "assistant",
        "content": "[{\"task\": \"object-detection-general\", \"id\": 0, \"dep\": [-1], \"args\": {\"image\": \"/examples/image1.jpg\"}}, ...]"
    },
    ...
]
```

**Context Included**:
- Previous conversation messages (trimmed to fit context window)
- System prompt explaining task structure
- 7+ few-shot examples showing task parsing patterns
- Dynamic task list from API

**LLM Parameters**:
- Temperature: 0.1 (low, for consistency)
- Logit bias: Applied to highlight task parsing keywords
- Max tokens: Context-dependent (ensures 800 tokens remaining)

**Output Format**: Raw JSON string (must be parsed)

**Is this Chain-of-Thought?**: **NO** - This is **few-shot learning**. The LLM sees examples and directly generates the output format without intermediate reasoning steps.

---

### Template 2: choose_model (called once per task)

**Location in code**: `decision_api.py` lines 358-417

**Prompt Structure**:
```python
messages = [
    {"role": "system", "content": choose_model_tprompt},  # System prompt
    ...few-shot examples from demo_choose_model_custom.json...,
    {"role": "user", "content": choose_model_prompt}       # User prompt with {{input}}, {{task}}, {{metas}}
]
```

**System Prompt (tprompt)**:
```
#2 Model Selection Stage: Given the user request and the parsed tasks, 
the AI assistant helps the user to select a suitable model from a list of models 
to process the user request. The assistant should focus more on the description 
of the model and find the model that has the most potential to solve requests and tasks.
```

**User Prompt Template**:
```
Please choose the most suitable model from {{metas}} for the task {{task}}. 
The output must be in a strict JSON format: {"id": "id", "reason": "your detail reasons for the choice"}.
```

**Few-shot Examples Format** (from `demo_choose_model_custom.json`):
```json
[
    {
        "role": "user",
        "content": "Please choose the most suitable model from [model_list] for the task object-detection..."
    },
    {
        "role": "assistant",
        "content": "{\"id\": \"model-id\", \"reason\": \"This model is best because...\"}"
    },
    ...
]
```

**Context Included**:
- User's original input
- Specific task details (task type, args)
- List of candidate models with metadata:
  - Model ID
  - Description (truncated to 100 chars)
  - Likes count
  - Tags
  - Inference endpoint ("api")

**LLM Parameters**:
- Temperature: 0.1
- Logit bias: Stronger (5.0) to highlight model selection keywords

**Output Format**: JSON string `{"id": "model-id", "reason": "..."}`

**Is this Chain-of-Thought?**: **NO** - Direct few-shot selection.

**Number of Calls**: One call per task (if multiple tasks, multiple LLM calls)

---

### Template 3: map_task_semantically (conditional, optional)

**Location in code**: `decision_api.py` lines 112-160

**When Called**: Only if LLM parsed a task name that doesn't exist in the API task list

**Prompt Structure**:
```python
messages = [
    {"role": "system", "content": system_prompt},  # Simple mapping instruction
    {"role": "user", "content": user_prompt}      # Original request + parsed task + available tasks
]
```

**System Prompt**:
```
You are a task mapping assistant. Map the given task name to the most 
semantically similar task from the available list based on the user's original request.

Return ONLY the task ID from the list that best matches semantically. 
Return JSON format: {"task_id": "task-name"}.
```

**User Prompt**:
```
Original user request: "{user_input}"

Parsed task: "{parsed_task}"

Available API tasks: [task1, task2, ...]

Which task from the available list best matches the semantic meaning of 
"{parsed_task}" given the user's request? Return JSON: {"task_id": "task-name"}.
```

**Context Included**:
- User's original request (for semantic context)
- Parsed task name (may not match API exactly)
- List of available API task names

**LLM Parameters**:
- Temperature: 0.1
- No logit bias
- No few-shot examples (one-shot prompt)

**Output Format**: JSON string `{"task_id": "task-name"}`

**Is this Chain-of-Thought?**: **NO** - Simple one-shot mapping.

---

### Template 4: response_results (NOT USED in decision_api.py)

**decision_api.py**: This LLM call is **NOT used**. Instead, `generate_response_description()` directly constructs JSON.

**awesome_chat.py**: Uses this to generate natural language response.

**System Prompt** (if used):
```
#4 Response Generation Stage: With the task execution logs, 
the AI assistant needs to describe the process and inference results.
```

**User Prompt** (if used):
```
Yes. Please first think carefully and directly answer my request based on 
the inference results. ... Then please detail your workflow including the 
used models and inference results for my request in your friendly tone.
```

---

## awesome_chat.py Flow Comparison

```
User Input
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Parse Tasks (LLM Call #1) - IDENTICAL              │
│ Function: parse_task()                                      │
│ Same structure as decision_api.py                           │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ POST-PROCESSING (No LLM) - IDENTICAL                        │
│ - unfold(), fix_dep()                                       │
│ - No semantic mapping (uses p0_models.jsonl directly)      │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Choose Models (LLM Call #2, #3, #4... - one per task)│
│ Function: choose_model()                                    │
│ Same structure as decision_api.py                           │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Execute Models (NO LLM - actual inference)          │
│ Functions: model_inference(), run_task()                    │
│ Executes selected models                                    │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: Generate Final Response (LLM Call #5)              │
│ Function: response_results()                                │
│ Prompt Template: response_results_tprompt + response_results_prompt│
│ Few-shot Examples: demo_response_results.json              │
│ Context: User input + All execution results                 │
│ Output: Natural language description of workflow and results │
└─────────────────────────────────────────────────────────────┘
    ↓
Final Output (Natural Language Message)
```

---

## Key Differences

### 1. Final Response Generation

| | awesome_chat.py | decision_api.py |
|---|----------------|-----------------|
| **Method** | LLM call (`response_results()`) | Template-based JSON (`generate_response_description()`) |
| **Output** | Natural language text | Structured JSON with DAG |
| **Context** | All execution results | Planned tasks + model selections |
| **Purpose** | User-friendly description | Execution server config |

### 2. Task Mapping

| | awesome_chat.py | decision_api.py |
|---|----------------|-----------------|
| **Method** | Direct lookup in `p0_models.jsonl` | Semantic mapping LLM call (if needed) |
| **LLM Calls** | None for mapping | 0-1 conditional call |

### 3. Model Selection Context

| | awesome_chat.py | decision_api.py |
|---|----------------|-----------------|
| **Model Source** | `p0_models.jsonl` | Custom API endpoint |
| **Availability Check** | Live HuggingFace/local status | Assumes API-provided models are available |
| **Model Metadata** | Same structure | Same structure (from API) |

---

## Detailed LLM Call Breakdown

### Example: "Detect objects, faces, and emotions in image.jpg"

#### decision_api.py LLM Calls:

**Call #1: parse_task()**
```
Input Context:
- System: Task planning instructions + available task list
- Few-shot: 7 examples of task parsing
- User: "Detect objects, faces, and emotions in image.jpg"

Output: 
[
  {"task": "object-detection-general", "id": 0, "dep": [-1], "args": {"image": "image.jpg"}},
  {"task": "face-detection", "id": 1, "dep": [0], "args": {"image": "image.jpg"}},
  {"task": "emotion-classification", "id": 2, "dep": [1], "args": {"image": "image.jpg"}}
]
```

**Call #2: choose_model_for_task()** - for object-detection-general
```
Input Context:
- System: Model selection instructions
- Few-shot: Examples of model selection
- User: "Choose model from [candidate_models] for object-detection-general task"

Output: {"id": "model-id-1", "reason": "Best for general object detection..."}
```

**Call #3: choose_model_for_task()** - for face-detection
```
Input Context: (same structure)
Output: {"id": "model-id-2", "reason": "Specialized for face detection..."}
```

**Call #4: choose_model_for_task()** - for emotion-classification
```
Input Context: (same structure)
Output: {"id": "model-id-3", "reason": "Excellent emotion recognition..."}
```

**Total**: 4 LLM calls

**No LLM Call**: Final response generation (uses `generate_response_description()` template)

---

#### awesome_chat.py LLM Calls (same scenario):

**Calls #1-4**: Identical to decision_api.py (parse_task + 3 choose_model)

**Call #5: response_results()**
```
Input Context:
- System: Response generation instructions
- Few-shot: Examples of workflow descriptions
- User: "Describe the workflow and results for: Detect objects, faces, and emotions"

Output: 
"Based on the inference results, I performed the following workflow:
1. First, I used model-id-1 to detect objects in the image...
2. Then, I used model-id-2 to detect faces...
3. Finally, I used model-id-3 to classify emotions...
The results show: [actual results]..."
```

**Total**: 5 LLM calls

---

## Prompt Engineering Techniques Used

### 1. Few-shot Learning (Not Chain-of-Thought)

**What it is**: Providing examples of input-output pairs to guide LLM behavior.

**Used in**:
- `parse_task()`: Shows 7+ examples of user queries → task JSON
- `choose_model()`: Shows examples of model selection decisions

**Why not Chain-of-Thought?**:
- Chain-of-thought would show: "Step 1: I need to identify tasks... Step 2: The first task is..."
- Few-shot shows: Direct examples without explicit reasoning steps
- The LLM learns patterns from examples, not step-by-step reasoning

### 2. Logit Bias

**What it is**: Boosting probability of specific tokens/keywords.

**Used in**:
- `parse_task()`: Bias = 0.1 (light) on task parsing keywords
- `choose_model()`: Bias = 5.0 (strong) on model selection keywords

**Purpose**: Increase likelihood of valid task names and model IDs in output.

### 3. Dynamic Prompt Injection

**What it is**: Replacing placeholders with runtime values.

**Used in**:
- Task list: `[TASKS_LIST_WILL_BE_REPLACED_DYNAMICALLY]` → actual API tasks
- User input: `{{input}}` → actual user query
- Context: `{{context}}` → conversation history
- Model metadata: `{{metas}}` → candidate models

### 4. Context Window Management

**What it is**: Trimming conversation history to fit token limit.

**Method** (in `parse_task()`):
```python
start = 0
while start <= len(context):
    history = context[start:]
    # ... build messages ...
    if get_max_context_length(LLM) - num > 800:
        break  # Enough room for response
    start += 2  # Remove 2 messages at a time
```

**Purpose**: Keep system prompt + few-shot examples + recent context within token limits.

---

## Comparison Table: decision_api.py vs awesome_chat.py

| Aspect | decision_api.py | awesome_chat.py |
|--------|----------------|-----------------|
| **Total LLM Calls** | 2-3 (1 parse + N choose + 0-1 map) | 3 (1 parse + N choose + 1 response) |
| **Few-shot Learning** | ✅ Yes | ✅ Yes |
| **Chain-of-Thought** | ❌ No | ❌ No |
| **parse_task Prompt** | ✅ Same structure | ✅ Same structure |
| **choose_model Prompt** | ✅ Same structure | ✅ Same structure |
| **Final Response** | ❌ Template-based JSON | ✅ LLM-generated text |
| **Semantic Mapping** | ✅ Optional LLM call | ❌ None (direct lookup) |
| **Task List Source** | Dynamic (API) | Static (hardcoded) |
| **Model Source** | API endpoint | p0_models.jsonl |

---

## Prompt Template Files Location

**Config Files** (contain prompt templates):
- `configs/config_custom_decisions.yaml` (used by decision_api.py)
- `configs/config.default.yaml` (used by awesome_chat.py)

**Few-shot Example Files**:
- `demos/demo_parse_task_custom.json` (for decision_api.py)
- `demos/demo_choose_model_custom.json` (for decision_api.py)
- `demos/demo_parse_task.json` (for awesome_chat.py)
- `demos/demo_choose_model.json` (for awesome_chat.py)
- `demos/demo_response_results.json` (for awesome_chat.py only)

---

## Summary

**Number of Prompt Templates**: **3** (parse_task, choose_model, response_results)
- decision_api.py uses 2 (parse_task, choose_model)
- awesome_chat.py uses 3 (all of them)

**LLM Call Strategy**: **Few-shot learning** (not chain-of-thought)
- Examples guide behavior without explicit reasoning steps
- Direct input → output pattern learning

**Total LLM Calls**:
- decision_api.py: **2-4 calls** (depends on number of tasks and semantic mapping needs)
- awesome_chat.py: **3-5 calls** (depends on number of tasks, always includes final response)

**Key Mechanistic Difference**: 
- decision_api.py: Replaces final LLM response generation with template-based JSON
- awesome_chat.py: Uses LLM to generate natural language workflow description

**Core Algorithm**: Both use identical prompt structures and LLM calling patterns for task parsing and model selection. The only difference is the final response generation method.

---

**Document Version**: 1.0  
**Date**: 2025-11-01  
**Verified By**: Code analysis and prompt template inspection

