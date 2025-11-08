# Task Pipeline Prompt System - Implementation Summary

## Overview

Successfully restructured the prompt system for task pipeline generation with new scenario-based format, comprehensive field explanations, and testing capabilities.

---

## 🎯 Key Changes

### 1. **Category Mapping System**
   - **File Created**: `demos/category_mapping.json`
   - **Purpose**: Maps category indices to pipeline types for easy selection
   - **Content**: 16 categories, 37 total examples
   - **Categories**:
     - [0] business-normal (2 examples)
     - [1] business-with-fight (3 examples)
     - [2] business-with-fire (3 examples)
     - [3] business-with-human-fall (3 examples)
     - [4] campus-normal (3 examples)
     - [5] campus-with-fight (2 examples)
     - [6] campus-with-fire (2 examples)
     - [7] campus-with-human-fall (3 examples)
     - [8] factory-normal (2 examples)
     - [9] factory-with-fight (2 examples)
     - [10] factory-with-human-fall (2 examples)
     - [11] traffic-with-accident (2 examples)
     - [12] traffic-with-fight (2 examples)
     - [13] traffic-with-fire (2 examples)
     - [14] traffic-with-humans (2 examples)
     - [15] traffic-with-no-humans (2 examples)

---

### 2. **System Prompt Redesign** (`system_prompt_parse_task.json`)
   - **Format**: Option A - Show all tasks first, then explain fields with examples
   - **Structure**:
     1. **Available Task Types**: All 17 task types with 4 fields each (name, description, inputs, outputs)
     2. **Understanding Task Fields**: Detailed explanation of each field with multiple examples
     3. **Execution Flow Rules**: Pipeline structure, dependencies, input-output matching, common patterns
     4. **Output Requirements**: References to few-shot examples for format
   - **Removed**: Dynamic task list replacement, DATA FLOW PATTERNS, DEPENDENCY GUIDELINES (moved to examples)

---

### 3. **Demo File Restructure** (`demo_parse_task_sensys_formatted.json`)
   - **Added Comprehensive Explanations**:
     - **Scenario Fields**:
       - `objects-seen`: What objects are present and how it guides task selection
       - `sample-description`: Context and situation that determines task relevance
     - **Task Fields** (6 required fields):
       1. `id`: Task type identifier
       2. `inputs_from_upstreams`: Semantic input description
       3. `upstreams`: Task IDs this depends on
       4. `outputs_for_downstreams`: Semantic output description
       5. `downstreams`: Task IDs that depend on this
       6. `sample-reasoning`: WHY tasks chosen/excluded (learning only, not in output)
   - **Example Analysis**: Shows how scenario content leads to specific task pipeline choices
   - **Exclusion Logic**: Explains why certain tasks are NOT included (e.g., emotion-classification during fights)

---

### 4. **API Enhancements** (`decision_api.py`)

#### **New Helper Functions**:
   - `load_all_examples()`: Load examples from file or API
   - `load_category_mapping()`: Load category index mapping
   - `build_few_shot_examples()`: Select examples from specified categories
   - `format_few_shot_for_prompt()`: Format examples into prompt messages

#### **Updated `/decisions` Endpoint** (Option 1):
   - **Supports Two Formats**:
     
     **A. New Scenario Format**:
     ```json
     {
       "scenario": {
         "objects-seen": ["car", "truck", "person"],
         "sample-description": "Scene description..."
       }
     }
     ```
     
     **B. Legacy Messages Format** (backward compatible):
     ```json
     {
       "messages": [{"role": "user", "content": "..."}]
     }
     ```

#### **New `/decisions/test` Endpoint** (Option 2):
   - **Purpose**: Automated testing with category-based few-shot selection
   - **Request Format**:
     ```json
     {
       "few_shot_categories": [0, 2, 5],
       "num_few_shot_examples_per_category": 2,
       "test_category_index": 10,
       "test_example_index": 0
     }
     ```
   - **Process**:
     1. Loads examples from specified few-shot categories
     2. Extracts test scenario from specified category/example index
     3. Generates tasks using LLM
     4. Returns both generated tasks AND ground truth for comparison
   - **Response**:
     ```json
     {
       "test_info": {...},
       "test_scenario": {...},
       "generated_tasks": [...],
       "ground_truth_tasks": [...],
       "prompt_logs_directory": "..."
     }
     ```

#### **Updated `parse_task()` Function**:
   - Added `objects_seen` parameter
   - Handles both array and string formats for objects_seen
   - Properly formats scenario context for LLM

---

## 📋 API Usage Examples

### Option 1: Direct Scenario Input

```bash
curl -X POST http://localhost:8004/decisions \
  -H "Content-Type: application/json" \
  -d '{
    "scenario": {
      "objects-seen": ["car", "truck", "person"],
      "sample-description": "In a typical urban intersection, a crowd of people is gathered around a car, with some individuals appearing to be in a physical altercation."
    }
  }'
```

### Option 2: Testing Mode

```bash
curl -X POST http://localhost:8004/decisions/test \
  -H "Content-Type: application/json" \
  -d '{
    "few_shot_categories": [0, 1, 2],
    "num_few_shot_examples_per_category": 2,
    "test_category_index": 5,
    "test_example_index": 0
  }'
```

### Backward Compatible (Legacy)

```bash
curl -X POST http://localhost:8004/decisions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Detect objects in image.jpg"}
    ]
  }'
```

---

## 🔍 Category Reference

Use category indices for testing:

| Index | Category | Examples |
|-------|----------|----------|
| 0 | business-normal | 2 |
| 1 | business-with-fight | 3 |
| 2 | business-with-fire | 3 |
| 3 | business-with-human-fall | 3 |
| 4 | campus-normal | 3 |
| 5 | campus-with-fight | 2 |
| 6 | campus-with-fire | 2 |
| 7 | campus-with-human-fall | 3 |
| 8 | factory-normal | 2 |
| 9 | factory-with-fight | 2 |
| 10 | factory-with-human-fall | 2 |
| 11 | traffic-with-accident | 2 |
| 12 | traffic-with-fight | 2 |
| 13 | traffic-with-fire | 2 |
| 14 | traffic-with-humans | 2 |
| 15 | traffic-with-no-humans | 2 |

---

## 📁 Modified Files

1. **Created**:
   - `demos/category_mapping.json` - Category index mapping

2. **Updated**:
   - `demos/system_prompt_parse_task.json` - Redesigned system prompt with Option A format
   - `demos/demo_parse_task_sensys_formatted.json` - Comprehensive field explanations
   - `decision_api.py` - Added helper functions and new endpoints
   - `demos/user_prompt_parse_task.json` - Already correct, no changes needed

3. **Referenced**:
   - `demos/all_examples.json` - All 37 examples (already exists)
   - `demos/system_prompt_context.json` - Task definitions (already exists)

---

## ✅ Testing Checklist

1. **Test Option 1** (Direct Scenario):
   ```bash
   curl -X POST http://localhost:8004/decisions \
     -H "Content-Type: application/json" \
     -d '{"scenario": {"objects-seen": ["car", "person"], "sample-description": "A car accident with people nearby"}}'
   ```

2. **Test Option 2** (Testing Mode):
   ```bash
   curl -X POST http://localhost:8004/decisions/test \
     -H "Content-Type: application/json" \
     -d '{"few_shot_categories": [0, 1], "num_few_shot_examples_per_category": 1, "test_category_index": 11, "test_example_index": 0}'
   ```

3. **Test Backward Compatibility**:
   ```bash
   curl -X POST http://localhost:8004/decisions \
     -H "Content-Type: application/json" \
     -d '{"messages": [{"role": "user", "content": "Analyze traffic scene"}]}'
   ```

4. **Check Prompt Logs**:
   - Location: `prompt_logs/request_YYYYMMDD_HHMMSS/`
   - Files: `01_parse_task.json` (contains full prompt sent to LLM)

5. **Verify Category Mapping**:
   ```bash
   cat demos/category_mapping.json
   ```

---

## 🚀 Next Steps

1. Start the server:
   ```bash
   cd /home/panthera/JARVIS/hugginggpt/server
   python decision_api.py --config configs/config_custom_decisions.yaml --port 8004
   ```

2. Test both API modes with different scenarios

3. Compare generated tasks vs ground truth in testing mode

4. Adjust few-shot category selection based on test results

---

## 📝 Notes

- **Few-shot Flexibility**: Categories array allows selective example inclusion
- **Overlap Allowed**: Test category can overlap with few-shot categories
- **Auto-adjust**: If category has fewer examples than requested, uses all available
- **Backward Compatible**: Legacy message format still works
- **Prompt Logging**: All prompts saved to `prompt_logs/` for debugging
- **Ground Truth Comparison**: Testing mode provides both generated and expected outputs

---

## 🎓 Key Improvements

1. **Better Readability**: `content_array` format makes JSON files human-readable
2. **Comprehensive Explanations**: Every field explained with examples
3. **Flexible Testing**: Category-based few-shot selection
4. **Automated Comparison**: Ground truth vs generated for accuracy testing
5. **Backward Compatible**: Doesn't break existing functionality
6. **Well Documented**: Each component has clear purpose and examples

