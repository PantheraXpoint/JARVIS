# Quick Start Guide - Task Pipeline API

## Prerequisites

1. **Qwen Server Running**:
   ```bash
   # In terminal 1
   cd /home/panthera/JARVIS/hugginggpt/server
   python qwen_server.py
   ```

2. **Decision API Running**:
   ```bash
   # In terminal 2
   cd /home/panthera/JARVIS/hugginggpt/server
   python decision_api.py --config configs/config_custom_decisions.yaml --port 8004
   ```

---

## Usage Mode 1: Direct Scenario Input

### Example 1: Traffic Accident Scene

```bash
curl -X POST http://localhost:8004/decisions \
  -H "Content-Type: application/json" \
  -d '{
    "scenario": {
      "objects-seen": ["car", "truck", "person", "bicycle"],
      "sample-description": "A white car has collided with a bicycle, causing the cyclist to fall onto the road. The cyclist is lying on the ground near the bicycle. Vehicles are moving along their respective lanes, and pedestrians are walking on the sidewalks."
    }
  }'
```

**What it does**: Generates a task pipeline based on the scenario description.

### Example 2: Fire Emergency

```bash
curl -X POST http://localhost:8004/decisions \
  -H "Content-Type: application/json" \
  -d '{
    "scenario": {
      "objects-seen": ["person", "building"],
      "sample-description": "In a shopping mall, a fire has broken out in one of the stores, with visible flames and smoke. Shoppers are seen walking and standing nearby, some looking towards the fire."
    }
  }'
```

### Example 3: Factory Safety

```bash
curl -X POST http://localhost:8004/decisions \
  -H "Content-Type: application/json" \
  -d '{
    "scenario": {
      "objects-seen": ["person", "forklift", "equipment"],
      "sample-description": "A normal factory floor with workers performing tasks at designated workstations or operating machinery. People are walking in marked pedestrian walkways."
    }
  }'
```

---

## Usage Mode 2: Testing with Ground Truth

### Example 1: Test Traffic Accident Category

```bash
curl -X POST http://localhost:8004/decisions/test \
  -H "Content-Type: application/json" \
  -d '{
    "few_shot_categories": [0, 1, 2, 4],
    "num_few_shot_examples_per_category": 2,
    "test_category_index": 11,
    "test_example_index": 0
  }'
```

**Explanation**:
- Use examples from categories 0, 1, 2, 4 (business-normal, business-with-fight, business-with-fire, campus-normal)
- Take 2 examples from each category (total 8 few-shot examples)
- Test on category 11 (traffic-with-accident), example 0

**Output includes**:
- `generated_tasks`: What the LLM generates
- `ground_truth_tasks`: What it should generate
- Compare them for accuracy

### Example 2: Test Fire Detection

```bash
curl -X POST http://localhost:8004/decisions/test \
  -H "Content-Type: application/json" \
  -d '{
    "few_shot_categories": [0, 4, 8],
    "num_few_shot_examples_per_category": 2,
    "test_category_index": 2,
    "test_example_index": 1
  }'
```

**Explanation**:
- Use normal scenarios as few-shot (business-normal, campus-normal, factory-normal)
- Test on business-with-fire, example 1
- See if LLM correctly identifies fire-detection task is needed

### Example 3: Cross-Domain Testing

```bash
curl -X POST http://localhost:8004/decisions/test \
  -H "Content-Type: application/json" \
  -d '{
    "few_shot_categories": [0, 1, 4, 5, 8, 9],
    "num_few_shot_examples_per_category": 1,
    "test_category_index": 14,
    "test_example_index": 0
  }'
```

**Explanation**:
- Use diverse examples: business (normal + fight), campus (normal + fight), factory (normal + fight)
- Test on traffic-with-humans
- Evaluate cross-domain generalization

---

## Understanding the Response

### Option 1 Response (Direct Scenario):
```json
{
  "decisions": {
    "input": "scene description...",
    "planned_tasks": [...],
    "model_selections": {...},
    "summary": {...}
  },
  "execution_config_file": "path/to/config.json",
  "prompt_logs_directory": "path/to/logs/"
}
```

### Option 2 Response (Testing):
```json
{
  "test_info": {
    "test_category": "traffic-with-accident",
    "test_example_id": "traffic-with-accident-example-0",
    "few_shot_categories": [...],
    "num_few_shot_examples": 8
  },
  "test_scenario": {
    "objects_seen": [...],
    "description": "..."
  },
  "generated_tasks": [...],      // What LLM generated
  "ground_truth_tasks": [...],   // What it should be
  "prompt_logs_directory": "..."
}
```

---

## Category Reference Quick Table

| Index | Category | Use For |
|-------|----------|---------|
| 0-3 | business-* | Retail, restaurant scenes |
| 4-7 | campus-* | University, school scenes |
| 8-10 | factory-* | Industrial, warehouse scenes |
| 11-15 | traffic-* | Road, intersection scenes |

**Suffixes**:
- `normal`: No emergency
- `with-fight`: Physical altercation
- `with-fire`: Fire/smoke emergency
- `with-human-fall`: Person lying/fallen
- `with-accident`: Vehicle collision

---

## Tips for Testing

### 1. **Start with Similar Domains**
```bash
# Test campus scene with campus examples
curl -X POST http://localhost:8004/decisions/test \
  -H "Content-Type: application/json" \
  -d '{
    "few_shot_categories": [4, 7],
    "num_few_shot_examples_per_category": 2,
    "test_category_index": 5,
    "test_example_index": 0
  }'
```

### 2. **Test Cross-Domain Generalization**
```bash
# Test traffic with business/campus examples
curl -X POST http://localhost:8004/decisions/test \
  -H "Content-Type: application/json" \
  -d '{
    "few_shot_categories": [0, 4],
    "num_few_shot_examples_per_category": 2,
    "test_category_index": 14,
    "test_example_index": 0
  }'
```

### 3. **Test Emergency Detection**
```bash
# Use normal scenes, test emergency
curl -X POST http://localhost:8004/decisions/test \
  -H "Content-Type: application/json" \
  -d '{
    "few_shot_categories": [0, 4, 8],
    "num_few_shot_examples_per_category": 1,
    "test_category_index": 2,
    "test_example_index": 0
  }'
```

### 4. **Vary Few-Shot Count**
```bash
# Try 1, 2, or 3 examples per category
# See how it affects accuracy
```

---

## Debugging

### 1. **Check Prompt Logs**
```bash
# After making a request, check the prompt sent to LLM
cat prompt_logs/request_*/01_parse_task.json
```

### 2. **Verify Category Mapping**
```bash
# See all available categories
cat demos/category_mapping.json | jq '.categories[] | {index, name: .base_pipeline_id, count: .num_examples}'
```

### 3. **Test Health Endpoint**
```bash
curl http://localhost:8004/health
```

---

## Common Issues & Solutions

### Issue: "No examples available"
**Solution**: Ensure `demos/all_examples.json` exists. If not:
```bash
# Will auto-fetch from API on first request
# Or manually run:
curl -X POST "http://143.248.55.143:8081/query/get_examples_for_task_pipelines" \
  -H "Content-Type: application/json" \
  -d '{"task_id": "", "scenario_id": "", "num_examples": 100}' \
  > demos/all_examples.json
```

### Issue: "Category not found"
**Solution**: Check valid indices 0-15 in `demos/category_mapping.json`

### Issue: "LLM not responding"
**Solution**: Verify Qwen server is running on port 8006

---

## Advanced: Batch Testing

Create a test script to evaluate multiple scenarios:

```bash
#!/bin/bash
# test_accuracy.sh

for test_cat in {11..15}; do
  echo "Testing category $test_cat..."
  curl -X POST http://localhost:8004/decisions/test \
    -H "Content-Type: application/json" \
    -d "{
      \"few_shot_categories\": [0, 1, 4, 5, 8, 9],
      \"num_few_shot_examples_per_category\": 1,
      \"test_category_index\": $test_cat,
      \"test_example_index\": 0
    }" > results_cat_$test_cat.json
done
```

---

## Next Steps

1. Test both API modes to ensure they work
2. Compare generated vs ground truth tasks
3. Adjust few-shot category selection based on results
4. Experiment with different `num_few_shot_examples_per_category` values
5. Analyze prompt logs to understand LLM reasoning

---

Happy testing! 🚀

