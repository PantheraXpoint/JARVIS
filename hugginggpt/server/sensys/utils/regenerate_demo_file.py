#!/usr/bin/env python3
"""
Regenerate demo_parse_task_sensys_formatted.json from all_examples.json

This script:
1. Reads all_examples.json and category_mapping.json
2. Extracts the 16 example-0s (one from each category)
3. Preserves the header/introduction text
4. Regenerates all examples with Scenario, Tasks, and Analysis sections
5. Maintains the exact same structure and field order as the current demo file

Usage:
    python regenerate_demo_file.py
"""

import json
import os
from typing import Dict, List, Any

# File paths
ALL_EXAMPLES_FILE = "../demos/all_examples.json"
CATEGORY_MAPPING_FILE = "../demos/category_mapping.json"
OUTPUT_FILE = "../demos/demo_parse_task_sensys_formatted.json"
CURRENT_DEMO_FILE = "../demos/demo_parse_task_sensys_formatted.json"  # For preserving header/analysis

# Category name mapping for example headers
CATEGORY_NAME_MAP = {
    "business-normal": "Business Normal",
    "business-with-fight": "Business With Fight",
    "business-with-fire": "Business With Fire",
    "business-with-human-fall": "Business With Human Fall",
    "campus-normal": "Campus Normal",
    "campus-with-fight": "Campus With Fight",
    "campus-with-fire": "Campus With Fire",
    "campus-with-human-fall": "Campus With Human Fall",
    "factory-normal": "Factory Normal",
    "factory-with-fight": "Factory With Fight",
    "factory-with-human-fall": "Factory With Human Fall",
    "traffic-with-accident": "Traffic With Accident",
    "traffic-with-fight": "Traffic With Fight",
    "traffic-with-fire": "Traffic With Fire",
    "traffic-with-humans": "Traffic With Humans",
    "traffic-with-no-humans": "Traffic With No Humans"
}


def load_json_file(filepath: str) -> Any:
    """Load JSON file"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def format_task_json(task: Dict[str, Any], indent: int = 2) -> List[str]:
    """Format a single task as JSON lines with proper indentation"""
    lines = []
    prefix = " " * indent
    
    lines.append(f"{prefix}{{")
    lines.append(f"{prefix}  \"id\": {json.dumps(task.get('id', ''))},")
    lines.append(f"{prefix}  \"inputs_from_upstreams\": {json.dumps(task.get('inputs_from_upstreams', []))},")
    lines.append(f"{prefix}  \"upstreams\": {json.dumps(task.get('upstreams', []))},")
    lines.append(f"{prefix}  \"outputs_for_downstreams\": {json.dumps(task.get('outputs_for_downstreams', []))},")
    lines.append(f"{prefix}  \"downstreams\": {json.dumps(task.get('downstreams', []))},")
    
    # Handle sample-reasoning (might be "sample-reasoning" or "sample-reasonings")
    sample_reasoning = task.get("sample-reasoning") or task.get("sample-reasonings", [])
    lines.append(f"{prefix}  \"sample-reasoning\": {json.dumps(sample_reasoning)}")
    
    lines.append(f"{prefix}}}")
    
    return lines


def format_tasks_array(tasks: List[Dict[str, Any]]) -> List[str]:
    """Format tasks array as JSON lines"""
    lines = ["["]
    
    for i, task in enumerate(tasks):
        task_lines = format_task_json(task, indent=2)
        # Add comma after all tasks except the last
        if i < len(tasks) - 1:
            task_lines[-1] = task_lines[-1] + ","
        lines.extend(task_lines)
    
    lines.append("]")
    return lines


def get_existing_analysis(base_pipeline_id: str, existing_content: List[str]) -> str:
    """Try to extract existing Analysis section for a category from the current demo file"""
    # Look for the example with this category
    category_name = CATEGORY_NAME_MAP.get(base_pipeline_id, base_pipeline_id)
    
    in_correct_example = False
    analysis_lines = []
    collecting = False
    
    for i, line in enumerate(existing_content):
        # Check if we're in the right example
        if f"**EXAMPLE" in line and category_name in line:
            in_correct_example = True
            continue
        
        # If we're in the right example, look for Analysis section
        if in_correct_example:
            if line.strip() == "Analysis:":
                collecting = True
                continue
            
            if collecting:
                # Stop when we hit the next example or empty line followed by next example
                if line.strip().startswith("**EXAMPLE"):
                    break
                if line.strip() == "" and i + 1 < len(existing_content):
                    if existing_content[i + 1].strip().startswith("**EXAMPLE"):
                        break
                
                analysis_lines.append(line)
    
    # Join and clean up
    if analysis_lines:
        analysis_text = "\n".join(analysis_lines).strip()
        # Remove trailing empty lines
        while analysis_text.endswith("\n"):
            analysis_text = analysis_text.rstrip("\n")
        return analysis_text
    
    return None


def generate_example_content(example: Dict[str, Any], example_num: int, 
                            base_pipeline_id: str, existing_analysis: str = None) -> List[str]:
    """Generate content for a single example"""
    lines = []
    
    # Example header
    category_name = CATEGORY_NAME_MAP.get(base_pipeline_id, base_pipeline_id)
    lines.append(f"**EXAMPLE {example_num} - {category_name}**:")
    lines.append("")
    
    # Scenario section (without objects-seen)
    scenario = example.get("scenario", {})
    lines.append("Scenario:")
    lines.append("{")
    sample_description = scenario.get("sample-description", "")
    lines.append(f"  \"sample-description\": {json.dumps(sample_description)}")
    lines.append("}")
    lines.append("")
    
    # Tasks section
    tasks = example.get("tasks", [])
    lines.append("Tasks:")
    task_lines = format_tasks_array(tasks)
    lines.extend(task_lines)
    lines.append("")
    
    # Analysis section
    lines.append("Analysis:")
    if existing_analysis:
        lines.append(existing_analysis)
    else:
        # Generate a placeholder analysis
        lines.append(f"[Analysis for {category_name} - Please update this section with detailed explanation of the scenario, objects present, task choices, and exclusions.]")
    lines.append("")
    
    return lines


def get_header_content() -> List[str]:
    """Get the header/introduction content (first 33 lines before examples)"""
    header_lines = [
        "Below are examples demonstrating how to construct task pipelines from scenario descriptions. Understanding the relationship between input scenarios and output task pipelines is crucial for successfully generating computer vision task execution graphs.",
        "",
        "**UNDERSTANDING THE INPUT-OUTPUT RELATIONSHIP**:",
        "",
        "Think of this as a translation task. You receive a scenario that describes a visual scene (the INPUT), and you need to generate a structured task pipeline (the OUTPUT) that processes that scene effectively. The scenario field is what you analyze, and the tasks field is what you produce based on that analysis.",
        "",
        "**INPUT FORMAT - The Scenario**:",
        "",
        "When you receive a scenario, it contains a single key field that guides your task selection:",
        "",
        "**sample-description** (string): This provides a detailed narrative description of the scene and what's happening. It tells you what objects are present in the scene (e.g., cars, trucks, people, fire) and gives you context about the situation - whether it's normal, an emergency, an accident, or something else. For instance, if the description mentions \"A fire is burning\", that indicates an emergency and you should include fire detection. If it says \"People engaged in physical altercation\", that's a fight scenario where emotion classification would be unreliable and should be excluded. If someone is \"lying motionless on ground\", that indicates potential injury and you should include pose detection. The context from this description determines which tasks are relevant and which should be excluded. You should infer what objects are present from the description itself.",
        "",
        "**OUTPUT FORMAT - The Tasks**:",
        "",
        "When you construct the task pipeline, each task must have exactly 6 required fields. Let me explain what each field means and how to use it:",
        "",
        "1. **id** (string): This is the task type identifier that you select from the available task types list. It must be one of the exact task IDs provided earlier, such as \"source\", \"object-detection-general\", \"face-detection\", or \"pipeline-end\".",
        "",
        "2. **inputs_from_upstreams** (array of strings): This semantically describes what type of data this task receives as input. It tells you what data flows into this task and must match the \"outputs_for_downstreams\" of the upstream tasks that feed into it. For example, the source task has [\"none\"] because it's the entry point, the first processing task typically has [\"image\"] from the source, and face detection would have [\"person bounding boxes\"] from object detection.",
        "",
        "3. **upstreams** (array of strings): This lists the specific task IDs that must complete before this task can run. It defines which tasks this task depends on and determines the execution order in the DAG. For instance, source has [\"none\"], object detection has [\"source\"], and face detection has [\"object-detection-general\"].",
        "",
        "4. **outputs_for_downstreams** (array of strings): This semantically describes what type of data this task produces. It tells you what data flows out of this task and must match the \"inputs_from_upstreams\" of any downstream tasks that consume its output. For example, source outputs [\"image\"], object detection outputs [\"bounding boxes\"], and face detection outputs [\"face bounding boxes\"].",
        "",
        "5. **downstreams** (array of strings): This lists the specific task IDs that will run after this task completes. It defines which tasks depend on this task's output and determines the execution flow in the DAG. For instance, the source task might have [\"object-detection-general\"], and face detection might have [\"gender-classification\", \"face-recognition\"].",
        "",
        "6. **sample-reasoning** (array of strings): This field explains your decision-making process. It tells why you chose certain downstream tasks, why you excluded others (especially important in special scenarios), and references specific parts of the scenario description. This field is REQUIRED in your output - you must include it for each task with meaningful explanations that demonstrate your understanding of the scenario and your reasoning for task selection.",
        "",
        "Now let's look at examples across different scenario types. Each example shows the scenario input, the corresponding task pipeline output, and an analysis explaining the key decisions.",
        ""
    ]
    return header_lines


def regenerate_demo_file():
    """Main function to regenerate the demo file"""
    print("=" * 80)
    print("REGENERATING DEMO FILE")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading data files...")
    all_examples = load_json_file(ALL_EXAMPLES_FILE)
    category_mapping = load_json_file(CATEGORY_MAPPING_FILE)
    
    print(f"  Loaded {len(all_examples)} examples from {ALL_EXAMPLES_FILE}")
    print(f"  Loaded {len(category_mapping['categories'])} categories from {CATEGORY_MAPPING_FILE}")
    print()
    
    # Try to load existing demo file to preserve Analysis sections
    existing_analyses = {}
    existing_content = []
    if os.path.exists(CURRENT_DEMO_FILE):
        print(f"Loading existing demo file to preserve Analysis sections...")
        try:
            existing_demo = load_json_file(CURRENT_DEMO_FILE)
            if isinstance(existing_demo, dict) and "content_array" in existing_demo:
                existing_content = existing_demo["content_array"]
                
                # Extract analyses for each category
                for category in category_mapping["categories"]:
                    base_pipeline_id = category["base_pipeline_id"]
                    analysis = get_existing_analysis(base_pipeline_id, existing_content)
                    if analysis:
                        existing_analyses[base_pipeline_id] = analysis
                        print(f"  Preserved Analysis for {base_pipeline_id}")
        except Exception as e:
            print(f"  Warning: Could not load existing file: {e}")
            print(f"  Will generate placeholder Analysis sections")
    
    print()
    
    # Build content array
    print("Generating demo content...")
    content_array = []
    
    # Add header
    content_array.extend(get_header_content())
    
    # Add examples (one example-0 from each category)
    categories = sorted(category_mapping["categories"], key=lambda x: x["index"])
    
    for idx, category in enumerate(categories, 1):
        base_pipeline_id = category["base_pipeline_id"]
        example_id = f"{base_pipeline_id}-example-0"
        
        # Find the example in all_examples
        example = None
        for ex in all_examples:
            if ex.get("id") == example_id:
                example = ex
                break
        
        if not example:
            print(f"  WARNING: Could not find {example_id}, skipping...")
            continue
        
        # Get existing analysis if available
        existing_analysis = existing_analyses.get(base_pipeline_id)
        
        # Generate example content
        example_lines = generate_example_content(
            example, 
            example_num=idx,
            base_pipeline_id=base_pipeline_id,
            existing_analysis=existing_analysis
        )
        content_array.extend(example_lines)
        
        print(f"  Generated example {idx}/16: {base_pipeline_id}")
    
    # Create output structure
    output_data = {
        "content_array": content_array
    }
    
    # Write output file
    print()
    print(f"Writing output to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Successfully regenerated {OUTPUT_FILE}")
    print(f"  Total lines: {len(content_array)}")
    print(f"  Examples generated: {len(categories)}")
    print()
    print("=" * 80)
    print("REGENERATION COMPLETE")
    print("=" * 80)
    print()
    print("NOTE: Please review the Analysis sections and update them if needed.")
    print("      The script preserves existing Analysis sections when possible.")


if __name__ == "__main__":
    regenerate_demo_file()

