# Prompt Structure Update Summary

## Overview
Updated the prompt structure for `decision_api.py` to use a new pipeline-based DAG format from `demo_parse_tasks_sensys.json` instead of the previous simple task list format.

## Changes Made

### 1. Updated System Prompt (tprompt) in `config_custom_decisions.yaml`

**Location:** `tprompt.parse_task` section

**Key Changes:**
- Completely rewrote the system prompt to explain the new pipeline DAG structure
- Added detailed explanations for all 5 required fields in each task node:
  1. `id`: Task identifier (string)
  2. `inputs_from_upstreams`: Semantic description of input data
  3. `upstreams`: Task IDs that must execute before this task
  4. `outputs_for_downstreams`: Semantic description of output data
  5. `downstreams`: Task IDs that will execute after this task

**Structure Explained:**
- Every pipeline MUST start with a "source" node
- Every pipeline MUST end with a "pipeline-end" node
- Tasks form a Directed Acyclic Graph (DAG) with dependencies
- Detailed data flow patterns (source → object-detection → specialized tasks → pipeline-end)
- Execution rules and dependency guidelines

### 2. Updated User Prompt in `config_custom_decisions.yaml`

**Location:** `prompt.parse_task` section

**Key Changes:**
- New input format expects:
  - `objects_seen`: List of objects present in the scene
  - `input`: Scene description
- Prompt guides LLM to construct a complete DAG pipeline
- Emphasizes proper dependencies and the source → pipeline-end structure

### 3. Created New Demo File

**File:** `demos/demo_parse_task_sensys_formatted.json`

**Content:**
- Formatted 2 examples from `demo_parse_tasks_sensys.json` into few-shot learning format
- Each example shows:
  - **User message**: Scenario with objects and description
  - **Assistant message**: Complete task pipeline in JSON format

**Examples include:**
1. Traffic with fight scenario (8 tasks including source and pipeline-end)
2. Traffic with accident scenario (10 tasks including source and pipeline-end)

### 4. Updated Config to Use New Demo File

**Location:** `demos_or_presteps.parse_task` in config

Changed from:
```yaml
parse_task: demos/demo_parse_task_custom.json
```

To:
```yaml
parse_task: demos/demo_parse_task_sensys_formatted.json
```

### 5. Updated `decision_api.py` Code

**Location:** `parse_task()` function, line ~327

**Change:**
- Added `objects_seen` parameter to the `replace_slot()` call
- Set default value to `"infer from the scene description"` when not provided
- This allows the LLM to determine objects from the scene description

## New Format Structure

### Input Format (Scenario)
```json
{
  "objects-seen": ["car", "truck", "bus", "person", ...],
  "sample-description": "Description of the scene..."
}
```

### Output Format (Task Pipeline)
```json
[
  {
    "id": "source",
    "inputs_from_upstreams": ["none"],
    "upstreams": ["none"],
    "outputs_for_downstreams": ["image"],
    "downstreams": ["object-detection-general"]
  },
  {
    "id": "object-detection-general",
    "inputs_from_upstreams": ["image"],
    "upstreams": ["source"],
    "outputs_for_downstreams": ["bounding boxes"],
    "downstreams": ["vehicle-color-classification", "face-detection"]
  },
  ...
  {
    "id": "pipeline-end",
    "inputs_from_upstreams": ["various task outputs"],
    "upstreams": ["vehicle-color-classification", "face-detection"],
    "outputs_for_downstreams": ["none"],
    "downstreams": []
  }
]
```

## Important Notes

### 1. Output Format Compatibility
⚠️ **IMPORTANT:** The new pipeline format uses different field names than the old format:
- Old format: `{"task": "...", "id": 0, "dep": [-1], "args": {...}}`
- New format: `{"id": "task-name", "upstreams": [...], "downstreams": [...], ...}`

The current code may need a **conversion layer** to transform the new pipeline format back to the old format that the rest of `decision_api.py` expects. This wasn't implemented yet since you mentioned focusing on prompts first.

### 2. Task IDs
- **Old format:** Numeric IDs (0, 1, 2, 3, ...)
- **New format:** String IDs matching task names ("object-detection-general", "vehicle-color-classification", etc.)

### 3. Dependencies
- **Old format:** `dep` field with numeric task IDs (e.g., `[0, 1]`)
- **New format:** `upstreams` field with string task IDs (e.g., `["object-detection-general"]`)

### 4. Special Nodes
The new format requires two special nodes that didn't exist in the old format:
- **source**: Pipeline entry point (provides the input image)
- **pipeline-end**: Pipeline termination point (collects final outputs)

## Testing Recommendations

1. **Start the servers:**
   ```bash
   # Terminal 1: Start Qwen LLM server
   python qwen_server.py --model_path /path/to/qwen-2.5-7b-instruct
   
   # Terminal 2: Start decision API server
   python decision_api.py --config configs/config_custom_decisions.yaml
   ```

2. **Send a test request:**
   ```bash
   curl -X POST http://localhost:8004/decisions \
     -H "Content-Type: application/json" \
     -d '{
       "messages": [
         {
           "role": "user",
           "content": "Analyze a traffic scene with cars, trucks, and people. Detect vehicles, identify their colors and license plates, and detect faces of people."
         }
       ]
     }'
   ```

3. **Check the output:**
   - Look in `prompt_logs/request_TIMESTAMP/01_parse_task.json` to see the exact prompt sent to LLM
   - Check if the LLM returns the new pipeline format
   - Verify if the rest of the code can handle it (or if conversion is needed)

## Next Steps

### For Parse Task (Completed ✓)
- ✅ System prompt (tprompt) updated with detailed field explanations
- ✅ User prompt updated to accept scenario format
- ✅ Demo file created with pipeline examples
- ✅ Code updated to pass objects_seen parameter

### For Choose Model (To Be Done)
You mentioned wanting to update choose_model prompts as well. The current choose_model prompt is still using the old format. Let me know if you want to update this next.

### Potential Follow-up Work
1. **Add format conversion:** Convert new pipeline format → old format for downstream processing
2. **Update task processing:** Modify `process_decisions()` to work with new format
3. **Update choose_model prompts:** Adapt to work with pipeline task structure
4. **Add validation:** Ensure generated pipelines have required source/pipeline-end nodes

## Files Modified

1. `/home/panthera/JARVIS/hugginggpt/server/configs/config_custom_decisions.yaml`
   - Updated `tprompt.parse_task` (comprehensive system prompt)
   - Updated `prompt.parse_task` (user prompt with scenario format)
   - Updated `demos_or_presteps.parse_task` (points to new demo file)

2. `/home/panthera/JARVIS/hugginggpt/server/decision_api.py`
   - Updated `parse_task()` function to pass `objects_seen` parameter

3. `/home/panthera/JARVIS/hugginggpt/server/demos/demo_parse_task_sensys_formatted.json` (NEW)
   - Created with 2 formatted examples from sensys data

## Files Not Modified (But May Need Updates)

1. `decision_api.py` - `process_decisions()` function may need to handle new format
2. `decision_api.py` - `unfold()` and `fix_dep()` functions may need adaptation
3. Choose model prompts (if you want to update those next)

