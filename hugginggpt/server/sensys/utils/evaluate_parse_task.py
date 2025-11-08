#!/usr/bin/env python3
"""
Evaluation script for parse_task API endpoint.

This script:
1. Loads all_examples.json
2. Identifies the 16 example-0s used in demo_parse_task_sensys_formatted.json (few-shot examples)
3. For each remaining example (21 total: 37 - 16 = 21):
   - Checks if API response already exists (caching)
   - Calls the /decisions API with the scenario (if not cached)
   - Stores the output in prompt_logs/
   - Compares generated tasks with ground truth from all_examples.json
   - Compares only 5 structural fields: id, inputs_from_upstreams, upstreams, outputs_for_downstreams, downstreams
   - Excludes sample-reasoning field from comparison (including "general" key that may exist in ground truth)
   - Stores evaluation results in prompt_logs/

Evaluation Methods:
- Method 1 (Sample-level): Sample is correct only if ALL tasks match exactly
- Method 2 (Path-level): Extracts all paths from source to pipeline-end, compares paths individually
"""

import json
import requests
import os
import time
import glob
from typing import List, Dict, Any, Set, Tuple
from datetime import datetime
from collections import defaultdict

# Configuration
API_URL = "http://localhost:8004/decisions"
ALL_EXAMPLES_FILE = "../demos/all_examples.json"
PROMPT_LOGS_DIR = "../prompt_logs"

# The 16 example-0s used in demo_parse_task_sensys_formatted.json (few-shot examples)
FEW_SHOT_EXAMPLE_IDS = [
    "business-normal-example-0",
    "business-with-fight-example-0",
    "business-with-fire-example-0",
    "business-with-human-fall-example-0",
    "campus-normal-example-0",
    "campus-with-fight-example-0",
    "campus-with-fire-example-0",
    "campus-with-human-fall-example-0",
    "factory-normal-example-0",
    "factory-with-fight-example-0",
    "factory-with-human-fall-example-0",
    "traffic-with-accident-example-0",
    "traffic-with-fight-example-0",
    "traffic-with-fire-example-0",
    "traffic-with-humans-example-0",
    "traffic-with-no-humans-example-0"
]

# Fields to compare (excluding sample-reasoning, model id, and model-chosen-reason)
# Note: We exclude sample-reasoning because:
# 1. Ground truth (all_examples.json) may contain "general" key in sample-reasoning
# 2. Generated output does NOT contain "general" key (filtered out in decision_api.py)
# 3. We only compare the structural task fields, not the reasoning explanations
# Note: API now returns "models" array instead of "tasks", but we extract task info for comparison
# We ignore "id" (model ID) and "model-chosen-reason" fields, only compare task structure
COMPARISON_FIELDS = ["task_id", "inputs_from_upstreams", "upstreams", "outputs_for_downstreams", "downstreams"]


def load_all_examples() -> List[Dict[str, Any]]:
    """Load all examples from all_examples.json"""
    with open(ALL_EXAMPLES_FILE, "r") as f:
        return json.load(f)


def get_test_examples(all_examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get examples that are NOT in the few-shot set"""
    return [ex for ex in all_examples if ex.get("id") not in FEW_SHOT_EXAMPLE_IDS]


def extract_all_paths(tasks: List[Dict[str, Any]]) -> List[List[str]]:
    """Extract all possible paths from 'source' to 'pipeline-end' using DFS
    
    Args:
        tasks: List of task dictionaries with 'id' or 'task_id' and 'downstreams' fields
        
    Returns:
        List of paths, where each path is a list of task IDs in sequence
    """
    # Build adjacency list (task_id -> list of downstream task_ids)
    graph = {}
    task_ids = set()
    
    for task in tasks:
        # Handle both "id" (ground truth) and "task_id" (API response)
        task_id = task.get("task_id") if "task_id" in task else task.get("id")
        downstreams = task.get("downstreams", [])
        if task_id:
            task_ids.add(task_id)
            graph[task_id] = downstreams
    
    # Check if source and pipeline-end exist
    if "source" not in task_ids:
        return []
    if "pipeline-end" not in task_ids:
        return []
    
    # DFS to find all paths from source to pipeline-end
    all_paths = []
    
    def dfs(current: str, path: List[str], visited: Set[str]):
        """Depth-first search to find all paths"""
        if current == "pipeline-end":
            all_paths.append(path.copy())
            return
        
        if current in visited:
            # Avoid cycles
            return
        
        visited.add(current)
        
        # Explore all downstream tasks
        for downstream in graph.get(current, []):
            path.append(downstream)
            dfs(downstream, path, visited)
            path.pop()
        
        visited.remove(current)
    
    # Start DFS from source
    dfs("source", ["source"], set())
    
    return all_paths


def compare_paths(generated_paths: List[List[str]], ground_truth_paths: List[List[str]]) -> Dict[str, Any]:
    """Compare generated paths with ground truth paths
    
    Args:
        generated_paths: List of paths from generated tasks
        ground_truth_paths: List of paths from ground truth tasks
        
    Returns:
        Dictionary with:
        - matched_paths: Number of generated paths that match ground truth
        - total_ground_truth_paths: Total number of ground truth paths
        - accuracy: matched_paths / total_ground_truth_paths
        - missing_paths: Paths in ground truth but not in generated
        - extra_paths: Paths in generated but not in ground truth
    """
    # Convert paths to tuples for set operations
    gt_path_set = set(tuple(path) for path in ground_truth_paths)
    gen_path_set = set(tuple(path) for path in generated_paths)
    
    # Find matches
    matched_path_set = gt_path_set & gen_path_set
    matched_paths = len(matched_path_set)
    
    # Find missing and extra paths
    missing_path_set = gt_path_set - gen_path_set
    extra_path_set = gen_path_set - gt_path_set
    
    # Convert back to lists
    missing_paths = [list(path) for path in missing_path_set]
    extra_paths = [list(path) for path in extra_path_set]
    
    total_gt_paths = len(ground_truth_paths)
    accuracy = matched_paths / total_gt_paths if total_gt_paths > 0 else 0.0
    
    return {
        "matched_paths": matched_paths,
        "total_ground_truth_paths": total_gt_paths,
        "total_generated_paths": len(generated_paths),
        "accuracy": accuracy,
        "missing_paths": missing_paths,
        "extra_paths": extra_paths
    }


def normalize_task_for_comparison(task: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only the comparison fields and normalize them for comparison
    
    This function explicitly excludes sample-reasoning, model id, and model-chosen-reason
    from comparison. We only compare structural fields: task_id, inputs_from_upstreams,
    upstreams, outputs_for_downstreams, and downstreams.
    
    For API responses (models array): uses "task_id" field
    For ground truth (tasks array): uses "id" field and renames it to "task_id" for comparison
    """
    normalized = {}
    for field in COMPARISON_FIELDS:
        if field == "task_id":
            # Handle both "task_id" (API response) and "id" (ground truth)
            value = task.get("task_id") if "task_id" in task else task.get("id")
        else:
            value = task.get(field)
        
        # Sort lists for consistent comparison
        if isinstance(value, list):
            normalized[field] = sorted(value)
        else:
            normalized[field] = value
    return normalized


def compare_tasks(generated_tasks: List[Dict[str, Any]], ground_truth_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare generated tasks with ground truth tasks
    
    Note: Only compares structural fields (task_id, inputs_from_upstreams, upstreams,
    outputs_for_downstreams, downstreams). The sample-reasoning, model id, and
    model-chosen-reason fields are completely ignored.
    
    Args:
        generated_tasks: Tasks from API response (models array with task_id field)
        ground_truth_tasks: Tasks from all_examples.json (tasks array with id field)
    
    Returns:
        Dictionary with comparison results including:
        - exact_match: True if all tasks match exactly
        - total_tasks: Total number of tasks
        - matched_tasks: Number of tasks that match
        - mismatches: List of mismatched tasks
    """
    # Normalize both lists
    normalized_generated = [normalize_task_for_comparison(task) for task in generated_tasks]
    normalized_ground_truth = [normalize_task_for_comparison(task) for task in ground_truth_tasks]
    
    # Sort by task_id for consistent comparison
    normalized_generated.sort(key=lambda x: x.get("task_id", ""))
    normalized_ground_truth.sort(key=lambda x: x.get("task_id", ""))
    
    # Check if lengths match
    if len(normalized_generated) != len(normalized_ground_truth):
        return {
            "exact_match": False,
            "total_tasks": len(normalized_ground_truth),
            "generated_tasks": len(normalized_generated),
            "matched_tasks": 0,
            "mismatches": [{
                "reason": f"Task count mismatch: expected {len(normalized_ground_truth)}, got {len(normalized_generated)}"
            }]
        }
    
    # Compare each task
    matched_tasks = 0
    mismatches = []
    
    for i, (gen_task, gt_task) in enumerate(zip(normalized_generated, normalized_ground_truth)):
        if gen_task == gt_task:
            matched_tasks += 1
        else:
            # Find which fields differ
            differing_fields = []
            for field in COMPARISON_FIELDS:
                if gen_task.get(field) != gt_task.get(field):
                    differing_fields.append({
                        "field": field,
                        "generated": gen_task.get(field),
                        "ground_truth": gt_task.get(field)
                    })
            
            mismatches.append({
                "task_index": i,
                "task_id": gen_task.get("task_id", "unknown"),
                "differing_fields": differing_fields,
                "generated_task": gen_task,
                "ground_truth_task": gt_task
            })
    
    return {
        "exact_match": matched_tasks == len(normalized_ground_truth),
        "total_tasks": len(normalized_ground_truth),
        "matched_tasks": matched_tasks,
        "mismatches": mismatches
    }


def call_api(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Call the /decisions API with a scenario"""
    payload = {"scenario": scenario}
    
    try:
        response = requests.post(API_URL, json=payload, timeout=300)  # 5 minute timeout
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def save_api_output(example_id: str, api_response: Dict[str, Any], output_dir: str):
    """Save API response to file"""
    filename = os.path.join(output_dir, f"{example_id}_api_response.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(api_response, f, indent=2, ensure_ascii=False)
    print(f"  Saved API response to: {filename}")


def find_cached_response(example_id: str) -> Dict[str, Any]:
    """Check if API response already exists in any evaluation directory
    
    Args:
        example_id: The example ID to look for
        
    Returns:
        Cached API response if found, None otherwise
    """
    # Search in all evaluation_* directories
    pattern = os.path.join(PROMPT_LOGS_DIR, "evaluation_*", f"{example_id}_api_response.json")
    matching_files = glob.glob(pattern)
    
    if matching_files:
        # Use the most recent one (sorted by path, which includes timestamp)
        most_recent = sorted(matching_files)[-1]
        print(f"  Found cached response: {most_recent}")
        try:
            with open(most_recent, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"  Warning: Failed to load cached response: {e}")
            return None
    
    return None


def save_method1_log(eval_dir: str, results: List[Dict[str, Any]], timestamp: str):
    """Save Method 1 (Sample-level) accuracy log"""
    log_file = os.path.join(eval_dir, "method1_sample_level_accuracy.txt")
    
    total = len(results)
    correct = sum(1 for r in results 
                  if r.get("comparison") is not None and r["comparison"].get("exact_match", False))
    errors = sum(1 for r in results if r.get("status") == "error")
    incorrect = total - correct - errors
    accuracy = (correct / total * 100) if total > 0 else 0.0
    
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("METHOD 1: SAMPLE-LEVEL ACCURACY EVALUATION\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Evaluation Method: Sample-level (all tasks must match exactly)\n")
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total Samples: {total}\n")
        f.write(f"Correct Samples: {correct}\n")
        f.write(f"Incorrect Samples: {incorrect}\n")
        f.write(f"Error Samples: {errors}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("PER-SAMPLE BREAKDOWN\n")
        f.write("=" * 80 + "\n")
        f.write("\n")
        
        for i, result in enumerate(results, 1):
            example_id = result.get("example_id", "unknown")
            status = result.get("status", "unknown")
            
            f.write(f"[{i}/{total}] {example_id}\n")
            f.write("-" * 80 + "\n")
            
            if status == "error":
                f.write(f"Status: ERROR\n")
                f.write(f"Error: {result.get('error', 'Unknown error')}\n")
            else:
                comparison = result.get("comparison", {})
                exact_match = comparison.get("exact_match", False)
                matched_tasks = comparison.get("matched_tasks", 0)
                total_tasks = comparison.get("total_tasks", 0)
                
                f.write(f"Status: {'CORRECT ✓' if exact_match else 'INCORRECT ✗'}\n")
                f.write(f"Matched Tasks: {matched_tasks}/{total_tasks}\n")
                
                if not exact_match:
                    mismatches = comparison.get("mismatches", [])
                    f.write(f"Mismatches: {len(mismatches)}\n")
                    
                    if mismatches and isinstance(mismatches[0], dict) and "reason" in mismatches[0]:
                        # Task count mismatch
                        f.write(f"  - {mismatches[0]['reason']}\n")
                    else:
                        # Field-level mismatches
                        for mismatch in mismatches[:3]:  # Show first 3 mismatches
                            task_id = mismatch.get("task_id", "unknown")
                            differing_fields = mismatch.get("differing_fields", [])
                            f.write(f"  - Task '{task_id}': {len(differing_fields)} field(s) differ\n")
                        
                        if len(mismatches) > 3:
                            f.write(f"  - ... and {len(mismatches) - 3} more mismatches\n")
            
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"Method 1 accuracy log saved to: {log_file}")
    return accuracy


def save_method2_log(eval_dir: str, results: List[Dict[str, Any]], timestamp: str):
    """Save Method 2 (Path-level) accuracy log"""
    log_file = os.path.join(eval_dir, "method2_path_level_accuracy.txt")
    
    total = len(results)
    per_sample_accuracies = []
    total_matched = 0
    total_gt_paths = 0
    
    # Calculate per-sample accuracies
    for result in results:
        if result.get("status") == "success" and result.get("path_comparison"):
            path_comp = result["path_comparison"]
            accuracy = path_comp.get("accuracy", 0.0)
            per_sample_accuracies.append(accuracy)
            total_matched += path_comp.get("matched_paths", 0)
            total_gt_paths += path_comp.get("total_ground_truth_paths", 0)
    
    # Calculate average accuracy
    avg_accuracy = (sum(per_sample_accuracies) / len(per_sample_accuracies) * 100) if per_sample_accuracies else 0.0
    
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("METHOD 2: PATH-LEVEL ACCURACY EVALUATION\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Evaluation Method: Path-level (individual paths from source to pipeline-end)\n")
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total Samples: {total}\n")
        f.write(f"Samples with Valid Paths: {len(per_sample_accuracies)}\n")
        f.write(f"Total Ground Truth Paths: {total_gt_paths}\n")
        f.write(f"Total Matched Paths: {total_matched}\n")
        f.write(f"Average Path Accuracy: {avg_accuracy:.2f}%\n")
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("PER-SAMPLE BREAKDOWN\n")
        f.write("=" * 80 + "\n")
        f.write("\n")
        
        for i, result in enumerate(results, 1):
            example_id = result.get("example_id", "unknown")
            status = result.get("status", "unknown")
            
            f.write(f"[{i}/{total}] {example_id}\n")
            f.write("-" * 80 + "\n")
            
            if status == "error":
                f.write(f"Status: ERROR\n")
                f.write(f"Error: {result.get('error', 'Unknown error')}\n")
            else:
                path_comp = result.get("path_comparison")
                
                if not path_comp:
                    f.write(f"Status: NO PATHS (tasks do not form valid source->pipeline-end paths)\n")
                else:
                    matched = path_comp.get("matched_paths", 0)
                    total_gt = path_comp.get("total_ground_truth_paths", 0)
                    total_gen = path_comp.get("total_generated_paths", 0)
                    accuracy = path_comp.get("accuracy", 0.0)
                    
                    f.write(f"Ground Truth Paths: {total_gt}\n")
                    f.write(f"Generated Paths: {total_gen}\n")
                    f.write(f"Matched Paths: {matched}\n")
                    f.write(f"Path Accuracy: {accuracy * 100:.2f}%\n")
                    
                    missing_paths = path_comp.get("missing_paths", [])
                    extra_paths = path_comp.get("extra_paths", [])
                    
                    if missing_paths:
                        f.write(f"\nMissing Paths ({len(missing_paths)}):\n")
                        for path in missing_paths[:3]:  # Show first 3
                            f.write(f"  - {' -> '.join(path)}\n")
                        if len(missing_paths) > 3:
                            f.write(f"  - ... and {len(missing_paths) - 3} more\n")
                    
                    if extra_paths:
                        f.write(f"\nExtra Paths ({len(extra_paths)}):\n")
                        for path in extra_paths[:3]:  # Show first 3
                            f.write(f"  - {' -> '.join(path)}\n")
                        if len(extra_paths) > 3:
                            f.write(f"  - ... and {len(extra_paths) - 3} more\n")
            
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"Method 2 accuracy log saved to: {log_file}")
    return avg_accuracy


def run_evaluation():
    """Main evaluation function"""
    print("=" * 80)
    print("PARSE TASK EVALUATION")
    print("=" * 80)
    print()
    
    # Load all examples
    print("Loading all_examples.json...")
    all_examples = load_all_examples()
    print(f"  Loaded {len(all_examples)} total examples")
    
    # Get test examples (excluding few-shot)
    test_examples = get_test_examples(all_examples)
    print(f"  Found {len(test_examples)} test examples (excluding {len(FEW_SHOT_EXAMPLE_IDS)} few-shot examples)")
    print()
    
    # Create evaluation directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = os.path.join(PROMPT_LOGS_DIR, f"evaluation_{timestamp}")
    os.makedirs(eval_dir, exist_ok=True)
    print(f"Evaluation results will be saved to: {eval_dir}")
    print()
    
    # Store evaluation results
    evaluation_results = {
        "timestamp": timestamp,
        "total_test_examples": len(test_examples),
        "few_shot_examples": FEW_SHOT_EXAMPLE_IDS,
        "comparison_fields": COMPARISON_FIELDS,
        "results": []
    }
    
    # Process each test example
    for idx, example in enumerate(test_examples, 1):
        example_id = example.get("id", f"example-{idx}")
        scenario = example.get("scenario", {})
        ground_truth_tasks = example.get("tasks", [])
        
        print(f"[{idx}/{len(test_examples)}] Processing: {example_id}")
        print(f"  Scenario: {scenario.get('sample-description', '')[:80]}...")
        
        # Check for cached response
        api_response = find_cached_response(example_id)
        
        if api_response is None:
            # Call API
            print("  Calling API...")
            api_response = call_api(scenario)
            
            if "error" not in api_response:
                # Save API response
                save_api_output(example_id, api_response, eval_dir)
        else:
            print("  Using cached response")
        
        if "error" in api_response:
            print(f"  ERROR: {api_response['error']}")
            evaluation_results["results"].append({
                "example_id": example_id,
                "status": "error",
                "error": api_response["error"],
                "comparison": None,
                "path_comparison": None
            })
            continue
        
        # Extract generated tasks from models array (new format)
        # API now returns {"models": [...]} instead of {"tasks": [...]}
        generated_tasks = api_response.get("models", [])
        if not generated_tasks:
            # Fallback to old format for backward compatibility
            generated_tasks = api_response.get("tasks", [])
        print(f"  Generated {len(generated_tasks)} tasks")
        
        # Method 1: Compare tasks (sample-level)
        print("  Method 1: Comparing tasks (sample-level)...")
        comparison = compare_tasks(generated_tasks, ground_truth_tasks)
        
        if comparison["exact_match"]:
            print(f"    ✓ CORRECT: All {comparison['total_tasks']} tasks match!")
        else:
            print(f"    ✗ INCORRECT: {comparison['matched_tasks']}/{comparison['total_tasks']} tasks match")
        
        # Method 2: Extract and compare paths (path-level)
        print("  Method 2: Extracting paths (path-level)...")
        gt_paths = extract_all_paths(ground_truth_tasks)
        gen_paths = extract_all_paths(generated_tasks)
        
        print(f"    Ground truth paths: {len(gt_paths)}")
        print(f"    Generated paths: {len(gen_paths)}")
        
        if gt_paths and gen_paths:
            path_comparison = compare_paths(gen_paths, gt_paths)
            path_accuracy = path_comparison["accuracy"] * 100
            print(f"    Path accuracy: {path_accuracy:.2f}% ({path_comparison['matched_paths']}/{path_comparison['total_ground_truth_paths']} paths match)")
        else:
            path_comparison = None
            print(f"    WARNING: Could not extract paths (source or pipeline-end missing)")
        
        # Store result
        evaluation_results["results"].append({
            "example_id": example_id,
            "status": "success",
            "generated_task_count": len(generated_tasks),
            "ground_truth_task_count": len(ground_truth_tasks),
            "comparison": comparison,
            "path_comparison": path_comparison
        })
        
        print()
    
    # Save evaluation summary
    summary_file = os.path.join(eval_dir, "evaluation_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    print(f"Evaluation summary saved to: {summary_file}")
    
    # Save Method 1 accuracy log
    print()
    print("Generating Method 1 (Sample-level) accuracy log...")
    method1_accuracy = save_method1_log(eval_dir, evaluation_results["results"], timestamp)
    
    # Save Method 2 accuracy log
    print("Generating Method 2 (Path-level) accuracy log...")
    method2_accuracy = save_method2_log(eval_dir, evaluation_results["results"], timestamp)
    
    # Print final statistics
    print()
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    total = len(evaluation_results["results"])
    correct = sum(1 for r in evaluation_results["results"] 
                  if r.get("comparison") is not None and r["comparison"].get("exact_match", False))
    errors = sum(1 for r in evaluation_results["results"] if r.get("status") == "error")
    incorrect = total - correct - errors
    
    print(f"Total examples evaluated: {total}")
    print()
    print("METHOD 1 (Sample-level):")
    print(f"  ✓ Correct samples: {correct}")
    print(f"  ✗ Incorrect samples: {incorrect}")
    print(f"  ⚠ Error samples: {errors}")
    print(f"  Accuracy: {method1_accuracy:.2f}%")
    print()
    print("METHOD 2 (Path-level):")
    print(f"  Average Path Accuracy: {method2_accuracy:.2f}%")
    print()
    print(f"Detailed logs:")
    print(f"  - Method 1: {os.path.join(eval_dir, 'method1_sample_level_accuracy.txt')}")
    print(f"  - Method 2: {os.path.join(eval_dir, 'method2_path_level_accuracy.txt')}")
    print(f"  - Full results: {summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    run_evaluation()

