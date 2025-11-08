# Dynamic Prompt Loading Implementation

## Overview
The `decision_api.py` has been updated to **dynamically load prompts from API at server startup** and automatically update the JSON files on disk.

## What Was Implemented

### 1. **API Fetching Functions**

#### `fetch_tasks_from_api()`
- Fetches available tasks from: `http://143.248.55.143:8081/query/context/tasks`
- Returns list of task dictionaries with: `id`, `name`, `description`, `inputs`, `outputs`

#### `fetch_examples_from_api()`
- Fetches examples from: `http://143.248.55.143:8081/query/get_examples_for_task_pipelines`
- Payload: `{"task_id": "", "scenario_id": "", "num_examples": 100}`
- Returns list of all example dictionaries

### 2. **Data Formatting Functions**

#### `format_tasks_for_system_prompt(tasks)`
- Formats tasks into the system prompt structure
- Matches exact format in `system_prompt_parse_task.json`:
  ```
  **AVAILABLE TASK TYPES** - You MUST select tasks ONLY from...
  
  1. **task-id**:
     - name: Task Name
     - description: Task description
     - inputs: ["input1", "input2"]
     - outputs: ["output1", "output2"]
  ```

#### `format_examples_for_demo(examples)`
- Formats 16 example-0s into demo file structure
- Matches exact format in `demo_parse_task_sensys_formatted.json`
- **Post-processing applied:**
  - Removes `"general"` key from `sample-reasoning`
  - Removes `"objects-seen"` from `scenario`
  - Preserves field order: `id`, `inputs_from_upstreams`, `upstreams`, `outputs_for_downstreams`, `downstreams`, `sample-reasoning`

### 3. **File Update Functions**

#### `update_system_prompt_file(tasks)`
- Updates `demos/system_prompt_parse_task.json`
- Replaces AVAILABLE TASK TYPES section (between start marker and "**UNDERSTANDING TASK FIELDS**:")
- Preserves all other content

#### `update_demo_file(examples)`
- Updates `demos/demo_parse_task_sensys_formatted.json`
- Regenerates complete file with:
  - Header/introduction text
  - 16 formatted examples (one from each category)
  - Analysis sections (placeholder text - can be manually updated later)

### 4. **Main Loading Function**

#### `load_prompts_dynamically()`
**Called at server startup** - performs these steps:

1. **Fetch tasks from API**
   - If successful: Updates `system_prompt_parse_task.json`
   - If failed: Logs warning, uses existing file

2. **Fetch examples from API**
   - If successful: Filters to 16 example-0s, updates `demo_parse_task_sensys_formatted.json`
   - If failed: Logs warning, uses existing file

3. **Load files into memory**
   - Loads all 3 prompt files into global variables:
     - `parse_task_tprompt` (system prompt)
     - `parse_task_demos_content` (demo examples)
     - `parse_task_prompt` (user prompt template)

4. **Return status**
   - Returns `True` if API data was used
   - Returns `False` if fallback to static files was used

### 5. **Runtime Usage**

#### In `parse_task()` function:
```python
# Uses the dynamically loaded prompts
dynamic_tprompt = parse_task_tprompt  # From API or static file
demo_content = parse_task_demos_content  # From API or static file

# Build user prompt ({{input}} is replaced)
prompt = replace_slot(parse_task_prompt, {"input": user_input, "context": ""})

# Concatenate in order (same as before)
unified_content = dynamic_tprompt + "\n\n" + demo_content + "\n\n" + prompt

# Send to LLM
messages = [{"role": "system", "content": unified_content}]
```

## Startup Behavior

### When server starts (`python decision_api.py`):

```
Decision API server initializing...
LLM endpoint: http://localhost:8006/v1/chat/completions
Model API: http://143.248.55.143:8081

================================================================================
LOADING PROMPTS DYNAMICALLY FROM API
================================================================================
Fetching tasks from API...
✓ Fetched 17 tasks from API
✓ Updated demos/system_prompt_parse_task.json with 17 tasks
✓ System prompt updated successfully

Fetching examples from API...
✓ Fetched 37 examples from API
✓ Demo file updated successfully with 16 examples

Loading prompt files into memory...
✓ Prompt files loaded into memory
================================================================================
✓ PROMPT LOADING COMPLETE - Using API data
================================================================================

Starting Decision API server on 0.0.0.0:8004...
```

### If API is unavailable (fallback):

```
Decision API server initializing...
...

================================================================================
LOADING PROMPTS DYNAMICALLY FROM API
================================================================================
Fetching tasks from API...
✗ Failed to fetch tasks from API, will use static file

Fetching examples from API...
✗ Failed to fetch examples from API, will use static file

Loading prompt files into memory...
✓ Prompt files loaded into memory
================================================================================
⚠ PROMPT LOADING COMPLETE - Using static files as fallback
================================================================================

Starting Decision API server on 0.0.0.0:8004...
```

## Benefits

### ✅ Dynamic Updates
- JSON files are automatically updated from API at startup
- No manual regeneration needed

### ✅ File Synchronization
- JSON files on disk always match what the LLM sees
- Can review/version control the prompt files
- `git diff` shows exactly what changed

### ✅ Fallback Mechanism
- If API is unreachable, uses existing static files
- Server still starts and functions normally
- No dependency on API availability for server startup

### ✅ Same Prompt Format
- Concatenation order unchanged: system → demos → user
- LLM receives identical structure as before
- Only the content is dynamically updated

## Files Modified

1. **`decision_api.py`** (lines 263-712):
   - Added 10 new functions for fetching, formatting, and updating
   - Modified `parse_task()` to use dynamic prompts
   - Modified startup code to call `load_prompts_dynamically()`

2. **`demos/system_prompt_parse_task.json`**:
   - Will be auto-updated with AVAILABLE TASK TYPES from API

3. **`demos/demo_parse_task_sensys_formatted.json`**:
   - Will be auto-updated with 16 example-0s from API

4. **`demos/user_prompt_parse_task.json`**:
   - No changes (already has `{{input}}` placeholder)

## Testing

Run the test script to verify setup:
```bash
cd /home/panthera/JARVIS/hugginggpt/server
python3 test_dynamic_loading.py
```

This checks:
- API endpoints are reachable
- Static files exist as fallback
- JSON files are valid

## Maintenance

### To update prompts manually:
1. Update the API data sources
2. Restart the server: `python decision_api.py`
3. The JSON files will be automatically regenerated

### To review what the LLM sees:
Simply inspect the 3 JSON files - they always match the runtime prompts:
- `demos/system_prompt_parse_task.json`
- `demos/demo_parse_task_sensys_formatted.json`
- `demos/user_prompt_parse_task.json`

## Notes

- Analysis sections in demo file use placeholder text (can be manually improved later)
- Post-processing ensures format matches exactly (no "general", no "objects-seen")
- Field order is preserved per requirements
- API timeout is 10s for tasks, 30s for examples
- Files are written with `indent=2`, `ensure_ascii=False` for readability
