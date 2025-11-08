# Prompt Reorganization Summary

## What Changed

Reorganized the prompt structure from inline YAML strings to **3 separate JSON files** for better organization and maintainability.

## New Structure

### 1. System Prompt: `demos/system_prompt_parse_task.json`
**Purpose:** Teaches the LLM about task types, rules, and patterns

**Content Organization:**
1. **AVAILABLE TASK TYPES** (First - provides context)
   - Dynamic placeholder: `[TASKS_LIST_WILL_BE_REPLACED_DYNAMICALLY]`
   - Gets replaced with actual tasks from API at runtime
   - Detailed descriptions of ALL 17 tasks:
     - source, object-detection-general, face-detection
     - vehicle-plate-detection, vehicle-damage-detection
     - protective-gear-detection, equipment-detection, fire-detection
     - cloth-color-classification, vehicle-color-classification
     - gender-classification, age-classification, emotion-classification
     - face-recognition, vehicle-make-classification
     - human-pose-detection, pipeline-end
   - Each task includes: Description, Inputs, Outputs, Usage guidelines

2. **EXECUTION FLOW RULES**
   - Must start with "source" node
   - Must end with "pipeline-end" node
   - Tasks form a DAG
   - Execution dependencies and parallelization rules

3. **DATA FLOW PATTERNS**
   - Pattern 1: Initial Detection (source → detectors)
   - Pattern 2: Vehicle Processing (object-detection → vehicle tasks)
   - Pattern 3: Person Processing (object-detection → person tasks)
   - Pattern 4: Face Analysis (face-detection → face tasks)
   - Pattern 5: Parallel Execution (independent tasks)
   - Pattern 6: Convergence (terminal tasks → pipeline-end)

4. **DEPENDENCY GUIDELINES WITH EXAMPLES**
   - Object Detection First (with example)
   - Sequential Vehicle Processing (with example pipeline)
   - Sequential Face Processing (with example pipeline)
   - Parallel Independent Tasks (with example)
   - Safety Scenario Dependencies (with example)

5. **TASK NODE FIELDS**
   - Explains the 5 required fields: id, inputs_from_upstreams, upstreams, outputs_for_downstreams, downstreams

6. **OUTPUT REQUIREMENTS**
   - Return only JSON array
   - No explanatory text
   - Empty array if cannot parse

### 2. Few-Shot Examples: `demos/demo_parse_task_sensys_formatted.json`
**Purpose:** Shows the LLM concrete examples of correct outputs

**Content:**
- Example 1: Traffic with fight scenario
  - User message: Scenario with objects and description
  - Assistant message: Complete pipeline JSON
- (Can add more examples as needed)

### 3. User Prompt: `demos/user_prompt_parse_task.json`
**Purpose:** Formats the actual user request with their scenario

**Content:**
- Template with placeholders: `{{objects_seen}}` and `{{input}}`
- Instructions for what to output
- Format: JSON object with "template" key

## Config File Changes

### Before:
```yaml
tprompt:
  parse_task: >-
    Very long inline string...

prompt:
  parse_task: >-
    Another inline string...

demos_or_presteps:
  parse_task: demos/demo_parse_task_custom.json
```

### After:
```yaml
tprompt:
  parse_task: demos/system_prompt_parse_task.json  # Points to JSON file

prompt:
  parse_task: demos/user_prompt_parse_task.json  # Points to JSON file

demos_or_presteps:
  parse_task: demos/demo_parse_task_sensys_formatted.json  # Already was a file
```

## Code Changes in `decision_api.py`

Added `load_prompt_value()` function to handle loading from JSON files:
- If value ends with `.json`, loads from file
- Extracts `template` or `content` field from JSON
- Otherwise uses value directly (backward compatible)

```python
def load_prompt_value(value):
    """Load prompt value - if it's a file path, load from file; otherwise use as-is"""
    if isinstance(value, str) and value.endswith('.json'):
        with open(value, "r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data.get("template", data.get("content", json.dumps(data)))
            return json.dumps(data)
    return value
```

## Key Improvements

### 1. Fixed Prompt Organization
- **AVAILABLE TASK TYPES now comes first** - provides context before examples
- **All 17 tasks documented** - prevents hallucination by explaining every task
- **No overlapping content** - cleaned up redundancy between sections

### 2. Better Separation of Concerns
- **System prompt** = Rules and knowledge
- **Few-shot examples** = Format demonstrations
- **User prompt** = Actual request formatting

### 3. Easier Maintenance
- Update task descriptions in one place
- Modify examples without touching system prompt
- Change user prompt format independently

### 4. Backward Compatible
- Old configs with inline strings still work
- Only new configs use JSON files

## Files Created/Modified

### Created:
1. `/home/panthera/JARVIS/hugginggpt/server/demos/system_prompt_parse_task.json`
2. `/home/panthera/JARVIS/hugginggpt/server/demos/user_prompt_parse_task.json`

### Modified:
1. `/home/panthera/JARVIS/hugginggpt/server/configs/config_custom_decisions.yaml`
   - Changed tprompt.parse_task to file path
   - Changed prompt.parse_task to file path

2. `/home/panthera/JARVIS/hugginggpt/server/decision_api.py`
   - Added load_prompt_value() function
   - Updated prompt loading to use new function

### Already Existed:
1. `/home/panthera/JARVIS/hugginggpt/server/demos/demo_parse_task_sensys_formatted.json`
   - Few-shot examples (user created earlier)

## How It Works at Runtime

1. **Server starts:**
   - Loads config from `config_custom_decisions.yaml`
   - Calls `load_prompt_value()` for tprompt.parse_task
   - Loads system prompt from `demos/system_prompt_parse_task.json`
   - Extracts "content" field from JSON

2. **Gets available tasks from API:**
   - Fetches task list: `["source", "object-detection-general", "face-detection", ...]`
   - Creates comma-separated string: `"source", "object-detection-general", "face-detection", ...`

3. **Replaces placeholder:**
   - Finds `[TASKS_LIST_WILL_BE_REPLACED_DYNAMICALLY]` in system prompt
   - Replaces with actual task list from API
   - Now LLM knows which tasks are currently available

4. **Builds messages array:**
   - Message 1: System prompt (with replaced task list)
   - Messages 2-N: Few-shot examples from `demo_parse_task_sensys_formatted.json`
   - Message N+1: User prompt from `user_prompt_parse_task.json` (with user's scenario)

5. **Sends to LLM:**
   - Qwen 2.5 7B processes the complete prompt
   - Returns JSON array of tasks in pipeline format

## Testing the New Structure

```bash
# Terminal 1: Start Qwen server
python qwen_server.py --model_path /path/to/qwen-2.5-7b-instruct --port 8006

# Terminal 2: Start decision API
python decision_api.py --config configs/config_custom_decisions.yaml --port 8004

# Terminal 3: Test request
curl -X POST http://localhost:8004/decisions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Analyze a traffic scene with vehicles and pedestrians. Detect vehicles, classify their colors and makes, detect license plates, and detect faces of people."
      }
    ]
  }'

# Check the generated prompts
cat prompt_logs/request_*/01_parse_task.json
```

## Next Steps

1. **Test the new prompt structure** with various scenarios
2. **Update choose_model prompts** similarly (if needed)
3. **Add more few-shot examples** to `demo_parse_task_sensys_formatted.json`
4. **Consider format conversion** if the rest of the code expects old format

