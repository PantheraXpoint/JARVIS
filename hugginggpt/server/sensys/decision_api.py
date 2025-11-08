#!/usr/bin/env python3
"""
Simplified Decision-Only API for Custom Model Server
Fetches tasks and models from custom API endpoints instead of p0_models.jsonl
Only performs decision-making (model selection) without execution.

Maintains the SAME algorithm and logic as awesome_chat.py - only changes the model source.
"""

import json
import logging
import requests
import flask
from flask import request, jsonify, Response
from flask_cors import CORS
import waitress
import argparse
import yaml
import re
import copy
import os
import time

# Import token utilities (same as original)
try:
    from get_token_ids import get_token_ids_for_task_parsing, get_token_ids_for_choose_model
    HAS_TOKEN_UTILS = True
except ImportError:
    HAS_TOKEN_UTILS = False
    logger = logging.getLogger(__name__)
    logger.warning("Token utilities not available, using simplified token counting")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/config_custom_decisions.yaml")
parser.add_argument("--port", type=int, default=8004)
parser.add_argument("--host", type=str, default="0.0.0.0")
args = parser.parse_args()

# Load config
config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

# Detect method based on config file name or explicit config setting
# Paper 2 method if config file contains "paper2" or explicit method setting
USE_PAPER2_METHOD = "paper2" in args.config.lower() or config.get("method", "").lower() == "paper2"

# API Configuration
MODEL_API_BASE_URL = config.get("model_api_base_url", "http://143.248.55.143:8081")

# LLM endpoint - ensure it has the full path
llm_base = config.get("local", {}).get("endpoint", "http://localhost:8006")
if not llm_base.endswith("/v1/chat/completions"):
    if not llm_base.endswith("/v1"):
        LLM_ENDPOINT = f"{llm_base}/v1/chat/completions"
    else:
        LLM_ENDPOINT = f"{llm_base}/chat/completions"
else:
    LLM_ENDPOINT = llm_base

LLM_MODEL = config.get("model", "qwen-2.5-7b-instruct")
use_completion = config.get("use_completion", False)

# Load prompt templates and demos - supports both string and JSON file formats
def load_demos(filepath):
    """Load demo/few-shot examples, handling content_array format"""
    with open(filepath, "r") as f:
        content = f.read()
        # Try to parse as JSON to check for content_array format
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "content_array" in data:
                # Single dict with content_array - convert to list with one message
                # This is the new format for demo files
                content_str = "\n".join(data["content_array"])
                messages = [{"role": "user", "content": content_str}]
                return json.dumps(messages)
            elif isinstance(data, list):
                # List of messages (old conversational format)
                for msg in data:
                    if isinstance(msg, dict) and "content_array" in msg:
                        # Join array into single string with newlines
                        msg["content"] = "\n".join(msg["content_array"])
                        del msg["content_array"]
                return json.dumps(data)
        except:
            pass
        # If not JSON or no content_array, return as-is
        return content

# parse_task demos will be loaded dynamically - see load_prompts_dynamically()
# choose_model demos are loaded but empty (for future use)
# Paper 2 doesn't have choose_model in demos_or_presteps
if not USE_PAPER2_METHOD and "demos_or_presteps" in config and "choose_model" in config["demos_or_presteps"]:
    choose_model_demos_or_presteps = load_demos(config["demos_or_presteps"]["choose_model"])
else:
    choose_model_demos_or_presteps = ""

# Load user prompts (supports both direct string and JSON file)
def load_prompt_value(value):
    """Load prompt value - if it's a file path, load from file; otherwise use as-is"""
    if isinstance(value, str) and value.endswith('.json'):
        with open(value, "r") as f:
            data = json.load(f)
            # If it's a dict with 'template' key, return the template
            # If it's a dict with 'template_array' key, join into string
            # If it's a dict with 'content' key (system prompt format), return content
            # If it's a dict with 'content_array' key, join into string
            # Otherwise return the whole thing as JSON string
            if isinstance(data, dict):
                if "template_array" in data:
                    return "\n".join(data["template_array"])
                elif "content_array" in data:
                    return "\n".join(data["content_array"])
                else:
                    return data.get("template", data.get("content", json.dumps(data)))
            return json.dumps(data)
    return value

# Placeholder - will be loaded dynamically or from files
parse_task_prompt = None
choose_model_prompt = None

parse_task_tprompt = None
choose_model_tprompt = None

parse_task_demos_content = None  # Will be loaded dynamically
choose_model_demos_content = None  # Empty for now

# Token counting setup (same as original)
LLM_encoding = LLM_MODEL
if HAS_TOKEN_UTILS:
    task_parsing_highlight_ids = get_token_ids_for_task_parsing(LLM_encoding)
    choose_model_highlight_ids = get_token_ids_for_choose_model(LLM_encoding)
else:
    task_parsing_highlight_ids = []
    choose_model_highlight_ids = []

# Caching
TASKS_CACHE = None
TASKS_DICT_CACHE = None  # Cache for full task dictionaries with description, inputs, outputs
MODELS_CACHE = {}

# Flask app
app = flask.Flask(__name__)
CORS(app)

# Global variable to track current request timestamp for prompt logging
CURRENT_REQUEST_TIMESTAMP = None


# ============================================================================
# PROMPT LOGGING AND TIMING FUNCTIONS
# ============================================================================


def save_prompt_to_file(messages, prompt_type, task_id=None):
    """Save the complete prompt structure (messages array) to a JSON file
    
    Args:
        messages: The complete messages array sent to LLM
        prompt_type: Either "parse_task", "choose_model", or "parse_task_and_choose_models_paper2"
        task_id: Optional task ID for choose_model prompts
    """
    global CURRENT_REQUEST_TIMESTAMP
    
    if CURRENT_REQUEST_TIMESTAMP is None:
        CURRENT_REQUEST_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
    
    # Create prompt_logs directory
    prompt_dir = f"prompt_logs/request_{CURRENT_REQUEST_TIMESTAMP}"
    os.makedirs(prompt_dir, exist_ok=True)
    
    # Build filename
    if prompt_type == "parse_task":
        filename = f"{prompt_dir}/01_parse_task.json"
        txt_filename = f"{prompt_dir}/01_parse_task_unified.txt"
    elif prompt_type == "choose_model" and task_id is not None:
        filename = f"{prompt_dir}/02_choose_model_task_{task_id}.json"
        txt_filename = f"{prompt_dir}/02_choose_model_task_{task_id}_unified.txt"
    elif prompt_type == "parse_task_and_choose_models_paper2":
        filename = f"{prompt_dir}/01_unified_parse_task_and_choose_models_paper2.json"
        txt_filename = f"{prompt_dir}/01_unified_parse_task_and_choose_models_paper2_unified.txt"
    else:
        filename = f"{prompt_dir}/{prompt_type}.json"
        txt_filename = f"{prompt_dir}/{prompt_type}_unified.txt"
    
    # Save with metadata
    prompt_data = {
        "timestamp": CURRENT_REQUEST_TIMESTAMP,
        "prompt_type": prompt_type,
        "task_id": task_id,
        "messages": messages
    }
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(prompt_data, f, indent=2, ensure_ascii=False)
    
    # Also save a human-readable unified prompt
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write(f"UNIFIED PROMPT FOR: {prompt_type.upper()}\n")
        f.write(f"Timestamp: {CURRENT_REQUEST_TIMESTAMP}\n")
        if task_id:
            f.write(f"Task ID: {task_id}\n")
        f.write("="*80 + "\n\n")
        
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            f.write(f"{'='*80}\n")
            f.write(f"MESSAGE {i+1} - ROLE: {role}\n")
            f.write(f"{'='*80}\n")
            f.write(content)
            f.write("\n\n")
        
        f.write("="*80 + "\n")
        f.write("END OF UNIFIED PROMPT\n")
        f.write("="*80 + "\n")
    
    logger.info(f"Saved {prompt_type} prompt to: {filename} and {txt_filename}")


def save_timing_info(prompt_type, llm_inference_time_seconds, task_id=None):
    """Save LLM inference timing information to a txt file
    
    Args:
        prompt_type: Either "parse_task", "choose_model", or "parse_task_and_choose_models_paper2"
        llm_inference_time_seconds: Time taken for LLM inference in seconds
        task_id: Optional task ID for choose_model prompts
    """
    global CURRENT_REQUEST_TIMESTAMP
    
    if CURRENT_REQUEST_TIMESTAMP is None:
        CURRENT_REQUEST_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
    
    # Create prompt_logs directory
    prompt_dir = f"prompt_logs/request_{CURRENT_REQUEST_TIMESTAMP}"
    os.makedirs(prompt_dir, exist_ok=True)
    
    # Build filename
    if prompt_type == "parse_task":
        filename = f"{prompt_dir}/01_parse_task_timing.txt"
    elif prompt_type == "choose_model" and task_id is not None:
        filename = f"{prompt_dir}/02_choose_model_task_{task_id}_timing.txt"
    elif prompt_type == "parse_task_and_choose_models_paper2":
        filename = f"{prompt_dir}/01_unified_parse_task_and_choose_models_paper2_timing.txt"
    else:
        filename = f"{prompt_dir}/{prompt_type}_timing.txt"
    
    # Format timing information
    timing_info = f"""LLM Inference Timing Information
{'='*80}
Request Type: {prompt_type.upper()}
Timestamp: {CURRENT_REQUEST_TIMESTAMP}
"""
    if task_id:
        timing_info += f"Task ID: {task_id}\n"
    
    timing_info += f"""
{'='*80}
LLM Inference Time: {llm_inference_time_seconds:.4f} seconds
LLM Inference Time: {llm_inference_time_seconds * 1000:.2f} milliseconds

This timing measures:
- Start: When the prompt is sent to the LLM endpoint
- End: When the LLM response is received
- Duration: Time taken for LLM inference only (not including prompt preparation or response parsing)
{'='*80}
"""
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(timing_info)
    
    logger.info(f"Saved {prompt_type} timing info to: {filename}")


def save_all_choose_model_timing(task_timings):
    """Save unified timing information for all choose_model calls
    
    Args:
        task_timings: List of tuples (task_id, llm_inference_time_seconds)
    """
    global CURRENT_REQUEST_TIMESTAMP
    
    if CURRENT_REQUEST_TIMESTAMP is None:
        CURRENT_REQUEST_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
    
    # Create prompt_logs directory
    prompt_dir = f"prompt_logs/request_{CURRENT_REQUEST_TIMESTAMP}"
    os.makedirs(prompt_dir, exist_ok=True)
    
    filename = f"{prompt_dir}/02_choose_model_all_tasks_timing.txt"
    
    # Calculate summary statistics
    total_time = sum(t[1] for t in task_timings)
    avg_time = total_time / len(task_timings) if task_timings else 0.0
    num_tasks = len(task_timings)
    
    # Format timing information
    timing_info = f"""LLM Inference Timing Information - All Choose Model Calls
{'='*80}
Timestamp: {CURRENT_REQUEST_TIMESTAMP}
{'='*80}

SUMMARY:
- Total LLM Inference Time: {total_time:.4f} seconds ({total_time * 1000:.2f} milliseconds)
- Average Per Task: {avg_time:.4f} seconds ({avg_time * 1000:.2f} milliseconds)
- Number of Tasks: {num_tasks}

{'='*80}
PER-TASK BREAKDOWN:
{'='*80}

"""
    
    for task_id, llm_time in task_timings:
        timing_info += f"""Task: {task_id}
  LLM Inference Time: {llm_time:.4f} seconds ({llm_time * 1000:.2f} milliseconds)

"""
    
    timing_info += f"""{'='*80}
END OF TIMING REPORT
{'='*80}
"""
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(timing_info)
    
    logger.info(f"Saved unified choose_model timing info to: {filename}")


# ============================================================================
# DYNAMIC PROMPT LOADING - API Functions
# ============================================================================


def fetch_tasks_from_api():
    """Fetch available tasks from API with detailed information
    
    Returns:
        List of task dictionaries with id, name, description, inputs, outputs
    """
    try:
        logger.info("Fetching tasks from API...")
        response = requests.get(f"{MODEL_API_BASE_URL}/query/context/tasks", timeout=10)
        if response.status_code == 200:
            tasks = response.json()
            
            # API returns a dictionary where keys are task IDs and values are task objects
            # Convert to list of task objects for consistency
            if isinstance(tasks, dict):
                logger.info(f"✓ Fetched {len(tasks)} tasks from API (as dictionary)")
                # Convert dict to list of values (task objects)
                tasks = list(tasks.values())
            else:
                logger.info(f"✓ Fetched {len(tasks)} tasks from API (as list)")
            
            # Log sample task structure for debugging
            if tasks and len(tasks) > 0:
                sample_task = tasks[0]
                logger.info(f"Sample task structure: type={type(sample_task)}, keys={list(sample_task.keys()) if isinstance(sample_task, dict) else 'N/A'}")
                if isinstance(sample_task, dict):
                    logger.info(f"Sample task content: id={sample_task.get('id')}, name={sample_task.get('name')}, has_description={bool(sample_task.get('description'))}, has_inputs={bool(sample_task.get('inputs'))}, has_outputs={bool(sample_task.get('outputs'))}")
            
            return tasks
        else:
            logger.warning(f"Failed to fetch tasks from API: HTTP {response.status_code}")
            return None
    except Exception as e:
        logger.warning(f"Error fetching tasks from API: {e}")
        return None


def format_tasks_for_system_prompt(tasks):
    """Format tasks into the system prompt format
    
    Args:
        tasks: List of task dictionaries from API
        
    Returns:
        List of strings (content_array lines) for the AVAILABLE TASK TYPES section
    """
    if isinstance(tasks, dict):
        # Some APIs may wrap the list inside a dictionary
        if "tasks" in tasks and isinstance(tasks["tasks"], list):
            tasks = tasks["tasks"]
        else:
            # If dict but not list, treat values as entries
            tasks = list(tasks.values())

    if not tasks:
        logger.warning("No tasks provided to format for system prompt")
        return []

    lines = []
    lines.append("**AVAILABLE TASK TYPES** - You MUST select tasks ONLY from the following list (use these exact task IDs):")
    lines.append("")
    
    valid_count = 0
    for task in tasks:
        if not isinstance(task, dict):
            # Attempt to parse string entries as JSON
            if isinstance(task, str):
                try:
                    task = json.loads(task)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping task entry that is not a dict: {task}")
                    continue
            else:
                logger.warning(f"Skipping task entry that is not a dict: {task}")
                continue

        task_id = task.get("id")
        if not task_id:
            logger.warning(f"Skipping task entry without id: {task}")
            continue

        name = task.get("name", "")
        description = task.get("description", "")
        inputs = task.get("inputs", [])
        outputs = task.get("outputs", [])

        valid_count += 1
        idx = valid_count
        
        lines.append(f"{idx}. **{task_id}**:")
        lines.append(f"   - name: {name}")
        lines.append(f"   - description: {description}")
        lines.append(f"   - inputs: {json.dumps(inputs)}")
        lines.append(f"   - outputs: {json.dumps(outputs)}")
        lines.append("")
    
    if valid_count == 0:
        logger.warning("No valid task entries were formatted for system prompt")
        return []

    return lines


def fetch_examples_from_api():
    """Fetch examples from API
    
    Returns:
        List of example dictionaries
    """
    try:
        logger.info("Fetching examples from API...")
        payload = {
            "task_id": "",
            "scenario_id": "",
            "num_examples": 100
        }
        response = requests.post(
            f"{MODEL_API_BASE_URL}/query/get_examples_for_task_pipelines",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        if response.status_code == 200:
            examples = response.json()
            logger.info(f"✓ Fetched {len(examples)} examples from API")
            return examples
        else:
            logger.warning(f"Failed to fetch examples from API: HTTP {response.status_code}")
            return None
    except Exception as e:
        logger.warning(f"Error fetching examples from API: {e}")
        return None


def filter_example_0s(examples):
    """Filter to get only example-0s (16 examples, one from each category)
    
    Args:
        examples: List of all examples
        
    Returns:
        List of 16 example-0s sorted by category
    """
    # Category order (matching the current demo file)
    category_order = [
        "business-normal",
        "business-with-fight",
        "business-with-fire",
        "business-with-human-fall",
        "campus-normal",
        "campus-with-fight",
        "campus-with-fire",
        "campus-with-human-fall",
        "factory-normal",
        "factory-with-fight",
        "factory-with-human-fall",
        "traffic-with-accident",
        "traffic-with-fight",
        "traffic-with-fire",
        "traffic-with-humans",
        "traffic-with-no-humans"
    ]
    
    example_0s = []
    for category in category_order:
        example_id = f"{category}-example-0"
        for ex in examples:
            if ex.get("id") == example_id:
                example_0s.append(ex)
                break
    
    return example_0s


def clean_task_for_demo(task):
    """Clean a task to match the demo format (remove general, objects-seen, preserve field order)
    
    Args:
        task: Task dictionary
        
    Returns:
        Cleaned task dictionary with correct field order
    """
    # Get sample-reasoning and clean it
    sample_reasoning = task.get("sample-reasoning", {})
    if isinstance(sample_reasoning, dict):
        downstreams = task.get("downstreams", [])
        cleaned_reasoning = {
            k: v for k, v in sample_reasoning.items()
            if k != "general" and k in downstreams
        }
    else:
        cleaned_reasoning = {}
    
    # Build task with exact field order
    cleaned = {
        "id": task.get("id"),
        "inputs_from_upstreams": task.get("inputs_from_upstreams", []),
        "upstreams": task.get("upstreams", []),
        "outputs_for_downstreams": task.get("outputs_for_downstreams", []),
        "downstreams": task.get("downstreams", []),
        "sample-reasoning": cleaned_reasoning
    }
    
    return cleaned


def format_task_for_json_string(task, indent=2):
    """Format a single task as JSON string with proper indentation
    
    Args:
        task: Task dictionary
        indent: Number of spaces for base indentation
        
    Returns:
        List of strings (lines)
    """
    lines = []
    prefix = " " * indent
    
    lines.append(f"{prefix}{{")
    lines.append(f'{prefix}  "id": {json.dumps(task["id"])},')
    lines.append(f'{prefix}  "inputs_from_upstreams": {json.dumps(task["inputs_from_upstreams"])},')
    lines.append(f'{prefix}  "upstreams": {json.dumps(task["upstreams"])},')
    lines.append(f'{prefix}  "outputs_for_downstreams": {json.dumps(task["outputs_for_downstreams"])},')
    lines.append(f'{prefix}  "downstreams": {json.dumps(task["downstreams"])},')
    lines.append(f'{prefix}  "sample-reasoning": {json.dumps(task["sample-reasoning"])}')
    lines.append(f"{prefix}}}")
    
    return lines


def format_examples_for_demo(examples):
    """Format examples into the demo file format
    
    Args:
        examples: List of 16 example-0 dictionaries
        
    Returns:
        List of strings (content_array lines) for the examples section
    """
    # Category name mapping
    category_names = {
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
    
    lines = []
    
    for idx, example in enumerate(examples, 1):
        example_id = example.get("id", "")
        base_id = example_id.replace("-example-0", "")
        category_name = category_names.get(base_id, base_id)
        
        # Example header
        lines.append(f"**EXAMPLE {idx} - {category_name}**:")
        lines.append("")
        
        # Scenario (without objects-seen)
        scenario = example.get("scenario", {})
        sample_description = scenario.get("sample-description", "")
        lines.append("Scenario:")
        lines.append("{")
        lines.append(f'  "sample-description": {json.dumps(sample_description)}')
        lines.append("}")
        lines.append("")
        
        # Tasks
        tasks = example.get("tasks", [])
        cleaned_tasks = [clean_task_for_demo(t) for t in tasks]
        
        lines.append("Tasks:")
        lines.append("[")
        for i, task in enumerate(cleaned_tasks):
            task_lines = format_task_for_json_string(task, indent=2)
            if i < len(cleaned_tasks) - 1:
                task_lines[-1] = task_lines[-1] + ","
            lines.extend(task_lines)
        lines.append("]")
        lines.append("")
        
        # Analysis (placeholder - can be manually updated later)
        lines.append("Analysis:")
        lines.append(f"This example demonstrates task pipeline construction for {category_name.lower()} scenarios. The pipeline shows appropriate task selection based on the scenario description, with proper dependencies and data flow.")
        lines.append("")
    
    return lines


def get_demo_header_lines():
    """Get the header/introduction lines for the demo file"""
    return [
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
        '**sample-description** (string): This provides a detailed narrative description of the scene and what\'s happening. It tells you what objects are present in the scene (e.g., cars, trucks, people, fire) and gives you context about the situation - whether it\'s normal, an emergency, an accident, or something else. For instance, if the description mentions "A fire is burning", that indicates an emergency and you should include fire detection. If it says "People engaged in physical altercation", that\'s a fight scenario where emotion classification would be unreliable and should be excluded. If someone is "lying motionless on ground", that indicates potential injury and you should include pose detection. The context from this description determines which tasks are relevant and which should be excluded. You should infer what objects are present from the description itself.',
        "",
        "**OUTPUT FORMAT - The Tasks**:",
        "",
        "When you construct the task pipeline, each task must have exactly 6 required fields. Let me explain what each field means and how to use it:",
        "",
        '1. **id** (string): This is the task type identifier that you select from the available task types list. It must be one of the exact task IDs provided earlier, such as "source", "object-detection-general", "face-detection", or "pipeline-end".',
        "",
        '2. **inputs_from_upstreams** (array of strings): This semantically describes what type of data this task receives as input. It tells you what data flows into this task and must match the "outputs_for_downstreams" of the upstream tasks that feed into it. For example, the source task has ["none"] because it\'s the entry point, the first processing task typically has ["image"] from the source, and face detection would have ["person bounding boxes"] from object detection.',
        "",
        '3. **upstreams** (array of strings): This lists the specific task IDs that must complete before this task can run. It defines which tasks this task depends on and determines the execution order in the DAG. For instance, source has ["none"], object detection has ["source"], and face detection has ["object-detection-general"].',
        "",
        '4. **outputs_for_downstreams** (array of strings): This semantically describes what type of data this task produces. It tells you what data flows out of this task and must match the "inputs_from_upstreams" of any downstream tasks that consume its output. For example, source outputs ["image"], object detection outputs ["bounding boxes"], and face detection outputs ["face bounding boxes"].',
        "",
        '5. **downstreams** (array of strings): This lists the specific task IDs that will run after this task completes. It defines which tasks depend on this task\'s output and determines the execution flow in the DAG. For instance, the source task might have ["object-detection-general"], and face detection might have ["gender-classification", "face-recognition"].',
        "",
        '6. **sample-reasoning** (object/dictionary): This field explains your decision-making process. It must be an object (dictionary) where keys are task IDs from the downstreams array (MANDATORY - you must explain why you chose each downstream task). The values are strings explaining why you chose certain downstream tasks, why you excluded others (especially important in special scenarios), and references to specific parts of the scenario description. This field is REQUIRED in your output - you must include it for each task with meaningful explanations that demonstrate your understanding of the scenario and your reasoning for task selection. Do NOT include a "general" key - only use task IDs from the downstreams array as keys.',
        "",
        "Now let's look at examples across different scenario types. Each example shows the scenario input, the corresponding task pipeline output, and an analysis explaining the key decisions.",
        ""
    ]


def update_system_prompt_file(tasks):
    """Update system_prompt_parse_task.json with new tasks from API
    
    Args:
        tasks: List of task dictionaries from API
    """
    try:
        system_prompt_file = config["tprompt"]["parse_task"]
        
        # Read existing file
        with open(system_prompt_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Get the content array
        content_array = data.get("content_array", [])
        
        # Find where AVAILABLE TASK TYPES section starts (line index 5)
        # and where it ends (before "**UNDERSTANDING TASK FIELDS**:")
        start_idx = None
        end_idx = None
        
        for i, line in enumerate(content_array):
            if "**AVAILABLE TASK TYPES**" in line:
                start_idx = i
            if start_idx is not None and "**UNDERSTANDING TASK FIELDS**:" in line:
                end_idx = i
                break
        
        if start_idx is None or end_idx is None:
            logger.error("Could not find AVAILABLE TASK TYPES section markers in system prompt")
            return False
        
        # Generate new task lines
        new_task_lines = format_tasks_for_system_prompt(tasks)
        if not new_task_lines:
            logger.warning("No formatted task lines available for system prompt update")
            return False

        # Replace the section
        new_content_array = content_array[:start_idx] + new_task_lines + content_array[end_idx:]
        
        # Update data
        data["content_array"] = new_content_array
        
        # Write back to file
        with open(system_prompt_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Updated {system_prompt_file} with {len(tasks)} tasks")
        return True
        
    except Exception as e:
        logger.error(f"Error updating system prompt file: {e}")
        return False


def update_demo_file(examples):
    """Update demo_parse_task_sensys_formatted.json with new examples from API
    
    Args:
        examples: List of 16 example-0 dictionaries
    """
    try:
        demo_file = config["demos_or_presteps"]["parse_task"]
        
        # Generate header + examples
        content_array = get_demo_header_lines()
        example_lines = format_examples_for_demo(examples)
        content_array.extend(example_lines)
        
        # Create output structure
        output_data = {
            "content_array": content_array
        }
        
        # Write to file
        with open(demo_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Updated {demo_file} with {len(examples)} examples")
        return True
        
    except Exception as e:
        logger.error(f"Error updating demo file: {e}")
        return False


def load_prompts_dynamically():
    """Load prompts dynamically from API and update JSON files
    
    This function:
    1. Fetches tasks from API and updates system_prompt_parse_task.json
    2. Fetches examples from API and updates demo_parse_task_sensys_formatted.json
    3. Loads the updated files into memory for runtime use
    4. Loads choose_model prompts from JSON files
    
    Returns:
        bool: True if successful, False if failed (will use static files as fallback)
    """
    global parse_task_tprompt, parse_task_demos_content, parse_task_prompt
    global choose_model_tprompt, choose_model_demos_content, choose_model_prompt
    
    logger.info("=" * 80)
    logger.info("LOADING PROMPTS DYNAMICALLY FROM API")
    logger.info("=" * 80)
    
    success = True
    
    # Step 1: Fetch and update tasks
    global TASKS_DICT_CACHE
    tasks = fetch_tasks_from_api()
    if tasks:
        # Cache the full task dictionaries for later use
        TASKS_DICT_CACHE = tasks
        logger.info(f"✓ Cached {len(tasks)} task dictionaries")
        
        if update_system_prompt_file(tasks):
            logger.info("✓ System prompt updated successfully")
        else:
            logger.warning("✗ Failed to update system prompt file")
            success = False
    else:
        logger.warning("✗ Failed to fetch tasks from API, will use static file")
        success = False
    
    # Step 2: Fetch and update examples
    all_examples = fetch_examples_from_api()
    if all_examples:
        example_0s = filter_example_0s(all_examples)
        if len(example_0s) == 16:
            if update_demo_file(example_0s):
                logger.info(f"✓ Demo file updated successfully with {len(example_0s)} examples")
            else:
                logger.warning("✗ Failed to update demo file")
                success = False
        else:
            logger.warning(f"✗ Expected 16 example-0s, got {len(example_0s)}")
            success = False
    else:
        logger.warning("✗ Failed to fetch examples from API, will use static file")
        success = False
    
    # Step 3: Load the updated (or existing) files into memory
    logger.info("Loading prompt files into memory...")
    try:
        # Load parse_task prompts
        parse_task_tprompt = load_prompt_value(config["tprompt"]["parse_task"])
        # Paper 2 doesn't have prompt.parse_task, only prompt.choose_model
        if "prompt" in config and "parse_task" in config["prompt"]:
            parse_task_prompt = load_prompt_value(config["prompt"]["parse_task"])
        else:
            parse_task_prompt = None  # Paper 2 doesn't need this
        
        # Load demo content for parse_task
        demo_file = config["demos_or_presteps"]["parse_task"]
        with open(demo_file, "r", encoding="utf-8") as f:
            demo_data = json.load(f)
            if isinstance(demo_data, dict) and "content_array" in demo_data:
                parse_task_demos_content = "\n".join(demo_data["content_array"])
            else:
                logger.warning("Unexpected demo file format")
                parse_task_demos_content = ""
        
        # Load choose_model prompts (only for Paper 1; Paper 2 uses prompt.choose_model)
        if "tprompt" in config and "choose_model" in config["tprompt"]:
            choose_model_tprompt = load_prompt_value(config["tprompt"]["choose_model"])
        else:
            choose_model_tprompt = None  # Paper 2 doesn't need this
        
        if "prompt" in config and "choose_model" in config["prompt"]:
            choose_model_prompt = load_prompt_value(config["prompt"]["choose_model"])
        else:
            choose_model_prompt = None
        
        # Load demo content for choose_model (empty for now)
        if "demos_or_presteps" in config and "choose_model" in config["demos_or_presteps"]:
            choose_demo_file = config["demos_or_presteps"]["choose_model"]
            with open(choose_demo_file, "r", encoding="utf-8") as f:
                choose_demo_data = json.load(f)
                if isinstance(choose_demo_data, dict) and "content_array" in choose_demo_data:
                    choose_model_demos_content = "\n".join(choose_demo_data["content_array"])
                else:
                    choose_model_demos_content = ""
        else:
            choose_model_demos_content = ""
        
        logger.info("✓ Prompt files loaded into memory")
        
    except Exception as e:
        logger.error(f"Error loading prompt files: {e}")
        success = False
    
    logger.info("=" * 80)
    if success:
        logger.info("✓ PROMPT LOADING COMPLETE - Using API data")
    else:
        logger.info("⚠ PROMPT LOADING COMPLETE - Using static files as fallback")
    logger.info("=" * 80)
    logger.info("")
    
    return success


# ============================================================================
# TASK AND MODEL API FUNCTIONS
# ============================================================================


def get_available_tasks():
    """Fetch available tasks from custom API, with fallback to static list"""
    global TASKS_CACHE
    if TASKS_CACHE is not None:
        return TASKS_CACHE
    
    try:
        response = requests.get(f"{MODEL_API_BASE_URL}/query/context/tasks", timeout=5)
        if response.status_code == 200:
            tasks = response.json()
            # API returns a dictionary where keys are task IDs and values are task objects
            # Convert to list of task objects for consistency
            if isinstance(tasks, dict):
                TASKS_CACHE = list(tasks.values())
                logger.info(f"Loaded {len(TASKS_CACHE)} tasks from API (converted from dict)")
            else:
                TASKS_CACHE = tasks
                logger.info(f"Loaded {len(TASKS_CACHE)} tasks from API (as list)")
            return TASKS_CACHE
        else:
            logger.warning(f"Failed to fetch tasks from API: HTTP {response.status_code}, using fallback")
    except Exception as e:
        logger.warning(f"Error fetching tasks from API: {e}, using fallback")
    
    # Fallback: Use static task list matching the system prompt
    TASKS_CACHE = [
        "source",
        "object-detection-general",
        "face-detection",
        "vehicle-plate-detection",
        "vehicle-damage-detection",
        "protective-gear-detection",
        "equipment-detection",
        "fire-detection",
        "cloth-color-classification",
        "vehicle-color-classification",
        "gender-classification",
        "age-classification",
        "emotion-classification",
        "face-recognition",
        "vehicle-make-classification",
        "human-pose-detection",
        "pipeline-end"
    ]
    logger.info(f"Using fallback task list with {len(TASKS_CACHE)} tasks")
    return TASKS_CACHE


def get_models_for_task(task_id):
    """Fetch models for a specific task from custom API"""
    if task_id in MODELS_CACHE:
        return MODELS_CACHE[task_id]
    
    try:
        response = requests.post(
            f"{MODEL_API_BASE_URL}/query/get_models_for_task",
            headers={"Content-Type": "application/json"},
            json={"task_id": task_id},
            timeout=5
        )
        if response.status_code == 200:
            models = response.json()
            MODELS_CACHE[task_id] = models
            logger.debug(f"Loaded {len(models)} models for task {task_id}")
            return models
        else:
            logger.warning(f"Failed to fetch models for task {task_id}: HTTP {response.status_code}")
            return []
    except Exception as e:
        logger.warning(f"Error fetching models for task {task_id}: {e}")
        return []


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def replace_slot(text, entries):
    """Same as original - replace {{slot}} with values"""
    for key, value in entries.items():
        if not isinstance(value, str):
            value = str(value)
        text = text.replace("{{" + key +"}}", value.replace('"', "'").replace('\n', ""))
    return text


def find_json(s):
    """Same as original - extract JSON from response"""
    s = s.replace("\'", "\"")
    start = s.find("{")
    end = s.rfind("}")
    res = s[start:end+1]
    res = res.replace("\n", "")
    return res


def extract_valid_tasks_from_truncated_json(task_str):
    """Try to extract valid tasks from a truncated JSON response
    
    This function attempts to recover valid task objects from incomplete JSON
    by finding all complete task objects before the truncation point.
    
    Returns:
        List of valid task dictionaries, or None if extraction fails
    """
    try:
        # First, try to find the JSON array start
        array_start = task_str.find('[')
        if array_start == -1:
            return None
        
        # Find all complete task objects by parsing character by character
        # We track brace depth to find complete objects
        tasks = []
        brace_count = 0
        task_start = -1
        in_string = False
        escape_next = False
        
        i = array_start + 1
        while i < len(task_str):
            char = task_str[i]
            
            if escape_next:
                escape_next = False
                i += 1
                continue
            
            if char == '\\':
                escape_next = True
                i += 1
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                i += 1
                continue
            
            if in_string:
                i += 1
                continue
            
            if char == '{':
                if brace_count == 0:
                    task_start = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and task_start != -1:
                    # Found a complete task object
                    try:
                        task_json = task_str[task_start:i+1]
                        task_obj = json.loads(task_json)
                        if isinstance(task_obj, dict) and "id" in task_obj:
                            tasks.append(task_obj)
                    except json.JSONDecodeError:
                        pass  # Skip invalid task objects
                    task_start = -1
            elif char == ']' and brace_count == 0:
                # End of array (complete JSON)
                break
            
            i += 1
        
        # If we found at least one valid task, return them
        return tasks if tasks else None
        
    except Exception as e:
        logger.warning(f"Failed to extract tasks from truncated JSON: {e}")
        return None

def field_extract(s, field):
    """Same as original - extract field from string"""
    try:
        field_rep = re.compile(f'{field}.*?:.*?"(.*?)"', re.IGNORECASE)
        extracted = field_rep.search(s).group(1).replace("\"", "\'")
    except:
        field_rep = re.compile(f'{field}:\ *"(.*?)"', re.IGNORECASE)
        extracted = field_rep.search(s).group(1).replace("\"", "\'")
    return extracted

def get_id_reason(choose_str):
    """Same as original - extract id and reason from response"""
    reason = field_extract(choose_str, "reason")
    id = field_extract(choose_str, "id")
    choose = {"id": id, "reason": reason}
    return id.strip(), reason.strip(), choose


# ============================================================================
# LLM REQUEST AND PARSING FUNCTIONS
# ============================================================================


def send_llm_request(messages, temperature=0.1, logit_bias=None, max_tokens=None):
    """Send request to LLM endpoint - SAME structure as original send_request"""
    try:
        data = {
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": temperature
        }
        if logit_bias:
            data["logit_bias"] = logit_bias
        if max_tokens:
            data["max_tokens"] = max_tokens
        
        # Increase timeout for long prompts (system + demos + user prompt can be very long)
        response = requests.post(
            LLM_ENDPOINT,
            headers={"Content-Type": "application/json"},
            json=data,
            timeout=120  # 2 minutes for long prompts with many examples
        )
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"].strip()
        logger.error(f"LLM request failed: HTTP {response.status_code}")
        return None
    except Exception as e:
        logger.error(f"Error calling LLM: {e}")
        return None


# ============================================================================
# TASK PARSING FUNCTION (parse_task step)
# ============================================================================


def parse_task(user_input, context=[], objects_seen=None, api_key=None, api_type=None, api_endpoint=None):
    """Parse user input into tasks using LLM
    
    Args:
        user_input: The scene description
        context: Previous conversation context (for multi-turn dialogs)
        objects_seen: List of objects present in scene (or None to infer)
        
    Returns:
        String containing JSON array of tasks
    """
    global parse_task_tprompt, parse_task_demos_content, parse_task_prompt
    
    # Use dynamically loaded prompts
    dynamic_tprompt = parse_task_tprompt or ""
    demo_content = parse_task_demos_content or ""
    prompt_template = parse_task_prompt or ""
    if not prompt_template:
        logger.error("parse_task_prompt template is empty; cannot build prompt")
        return {"error": "parse_task prompt template not loaded"}
    
    # Build prompt with scenario format (context is not used in unified format)
    # Note: objects-seen is no longer included in the scenario prompt
    prompt = replace_slot(prompt_template, {
            "input": user_input,
        "context": ""  # Context not needed in unified format
    })
    
    # Merge ALL three files into ONE unified system message:
    # 1. System prompt (dynamic_tprompt)
    # 2. Demo examples (demo_content)
    # 3. User prompt with scenario already filled in (prompt)
    unified_content = dynamic_tprompt
    if demo_content:
        unified_content += "\n\n" + demo_content
    if unified_content:
        unified_content += "\n\n" + prompt
    else:
        unified_content = prompt
    
    # Single unified message - all content is in the system message
    # The user prompt with scenario is already included above, so no need for separate user message
    messages = [
        {"role": "system", "content": unified_content}
    ]
    
    # Call with logit_bias
    logit_bias = None
    if HAS_TOKEN_UTILS and task_parsing_highlight_ids:
        logit_bias = {item: config.get("logit_bias", {}).get("parse_task", 0.1) for item in task_parsing_highlight_ids}
    
    # Save prompt structure before sending to LLM
    save_prompt_to_file(messages, "parse_task")
    
    # Set max_tokens to prevent truncation (8192 should be enough for task parsing)
    # If the LLM server doesn't support this, it will ignore it
    # Measure LLM inference time
    start_time = time.time()
    response = send_llm_request(messages, temperature=0.1, logit_bias=logit_bias, max_tokens=8192)
    end_time = time.time()
    llm_inference_time = end_time - start_time
    
    # Save timing information
    save_timing_info("parse_task", llm_inference_time)
    
    if not response:
        return {"error": "Failed to get LLM response"}
    
    return response  # Return string, let caller parse


# ============================================================================
# FUTURE: Choose Model Flow (Currently Not Used - For Later Implementation)
# ============================================================================
# These functions are preserved for future use when implementing the choose_model step.
# They are currently commented out in the /decisions endpoint.


def choose_model_for_task(scenario_description, task, task_info, models, api_key=None, api_type=None, api_endpoint=None):
    """Use LLM to select best model with new prompt format
    
    Args:
        scenario_description: The original scenario description from user input
        task: The task dictionary (contains task_id, etc.)
        task_info: Task info dictionary with id, name, description, inputs, outputs (same format as parse_task)
        models: List of available models for this task
        
    Returns:
        Dictionary with "id" (model ID) and "reason" (explanation)
    """
    global choose_model_tprompt, choose_model_demos_content, choose_model_prompt
    
    if not models:
        return {"id": "none", "reason": "No models available for this task"}
    
    # Log what task_info we received
    logger.info(f"choose_model_for_task received task_info: {task_info}")
    logger.info(f"  task_info keys: {list(task_info.keys()) if isinstance(task_info, dict) else 'N/A'}")
    logger.info(f"  task_info.get('description'): {task_info.get('description')}")
    logger.info(f"  task_info.get('inputs'): {task_info.get('inputs')}")
    logger.info(f"  task_info.get('outputs'): {task_info.get('outputs')}")
    
    # Format task info the same way as parse_task (matching the format in system_prompt_parse_task.json)
    task_id = task_info.get("id", task.get("id", "unknown"))
    task_name = task_info.get("name", task_id)
    task_description = task_info.get("description", "No description available")
    task_inputs = task_info.get("inputs", [])
    task_outputs = task_info.get("outputs", [])
    
    logger.info(f"Formatted task info: id={task_id}, name={task_name}, description={task_description[:50] if task_description else 'N/A'}, inputs={task_inputs}, outputs={task_outputs}")
    
    # Format to match parse_task format: "**task-id**:\n   - name: ...\n   - description: ...\n   - inputs: ...\n   - outputs: ..."
    task_info_str = f"**{task_id}**:\n   - name: {task_name}\n   - description: {task_description}\n   - inputs: {json.dumps(task_inputs)}\n   - outputs: {json.dumps(task_outputs)}"
    
    # Format available models for the prompt
    models_info_lines = []
    for idx, model in enumerate(models, 1):
        model_id = model.get("id", "unknown")
        model_name = model.get("name", "")
        model_desc = model.get("description", "")
        
        models_info_lines.append(f"{idx}. **{model_id}**:")
        models_info_lines.append(f"   - name: {model_name}")
        models_info_lines.append(f"   - description: {model_desc}")
        
        # Add any additional metadata if available
        if "meta" in model and model["meta"]:
            meta = model["meta"]
            if "tags" in meta and meta["tags"]:
                models_info_lines.append(f"   - tags: {meta['tags']}")
        
        models_info_lines.append("")  # Empty line between models
    
    available_models_str = "\n".join(models_info_lines)
    
    # Use dynamically loaded prompts (same pattern as parse_task)
    dynamic_tprompt = choose_model_tprompt or ""
    demo_content = choose_model_demos_content or ""
    prompt_template = choose_model_prompt or ""
    
    if not prompt_template:
        logger.error("choose_model_prompt template is empty; cannot build prompt")
        return {"id": models[0].get("id", "none"), "reason": "Prompt template not loaded"}
    
    # Replace placeholders in system prompt
    system_content = dynamic_tprompt.replace("{{scenario_description}}", scenario_description)
    system_content = system_content.replace("{{task_info}}", task_info_str)
    system_content = system_content.replace("{{available_models}}", available_models_str)
    
    # User prompt template (no placeholders needed - just final instruction)
    user_prompt = prompt_template
    
    # Merge ALL three files into ONE unified system message (same pattern as parse_task):
    # 1. System prompt (dynamic_tprompt with placeholders replaced)
    # 2. Demo examples (demo_content)
    # 3. User prompt (final instruction)
    unified_content = system_content
    if demo_content:
        unified_content += "\n\n" + demo_content
    if unified_content:
        unified_content += "\n\n" + user_prompt
    else:
        unified_content = user_prompt
    
    # Build single SYSTEM message (no USER message)
    messages = [
        {"role": "system", "content": unified_content}
    ]
    
    # Save prompt structure before sending to LLM
    save_prompt_to_file(messages, "choose_model", task_id=task_id)
    
    # Call with logit_bias
    logit_bias = None
    if HAS_TOKEN_UTILS and choose_model_highlight_ids:
        logit_bias = {item: config.get("logit_bias", {}).get("choose_model", 5) for item in choose_model_highlight_ids}
    
    # Measure LLM inference time
    start_time = time.time()
    choose_str = send_llm_request(messages, temperature=0.1, logit_bias=logit_bias)
    end_time = time.time()
    llm_inference_time = end_time - start_time
    
    if not choose_str:
        # Fallback
        return {"id": models[0].get("id", "none"), "reason": "Selected first available model (LLM selection failed)"}
    
    # Parse JSON response - handle markdown code blocks and other formatting
    # First, try to strip markdown code blocks if present
    cleaned_str = choose_str.strip()
    if cleaned_str.startswith("```"):
        # Remove markdown code block markers
        lines = cleaned_str.split("\n")
        # Remove first line if it's ```json or ```
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned_str = "\n".join(lines).strip()
    
    try:
        choose = json.loads(cleaned_str)
        model_id = choose.get("id", "")
        reason = choose.get("reason", "")
        if not model_id:
            raise ValueError("Missing 'id' field in response")
        return {"id": model_id, "reason": reason, "llm_time": llm_inference_time}
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"JSON parsing failed: {e}, trying fallback extraction. Response: {choose_str[:200]}")
        # Fallback: try to extract JSON using find_json
        try:
            extracted_json = find_json(cleaned_str)
            choose = json.loads(extracted_json)
            model_id = choose.get("id", "")
            reason = choose.get("reason", "")
            if model_id:
                return {"id": model_id, "reason": reason, "llm_time": llm_inference_time}
        except Exception as e2:
            logger.warning(f"Fallback extraction also failed: {e2}")
        
        # Last resort: use regex extraction
        try:
            model_id, reason, choose = get_id_reason(cleaned_str)
            return {"id": model_id, "reason": reason, "llm_time": llm_inference_time}
        except Exception as e3:
            logger.error(f"All JSON parsing methods failed: {e3}")
            # Return first available model as fallback
            return {"id": models[0].get("id", "none"), "reason": f"JSON parsing failed, selected first model. Error: {str(e3)}", "llm_time": llm_inference_time}


def parse_task_and_choose_models_paper2(user_input, context=[], objects_seen=None, api_key=None, api_type=None, api_endpoint=None):
    """Paper 2 Method: Unified prompt with single LLM call for both task parsing and model selection
    
    Args:
        user_input: The scene description
        context: Previous conversation context (for multi-turn dialogs)
        objects_seen: List of objects present in scene (or None to infer)
        
    Returns:
        Dictionary with "tasks" array in format: [{"id": task_id, "model_id": model_id, ...}]
    """
    global parse_task_tprompt, parse_task_demos_content, parse_task_prompt
    global choose_model_tprompt, choose_model_demos_content, choose_model_prompt
    
    # Load Stage 1 prompts (tprompt and demos)
    parse_dynamic_tprompt = parse_task_tprompt or ""
    parse_demo_content = parse_task_demos_content or ""
    
    if not parse_dynamic_tprompt:
        logger.error("parse_task tprompt is empty; cannot build prompt")
        return {"error": "parse_task tprompt not loaded"}
    
    # Step 1: Build Stage 1 section (system prompt + demos + scenario)
    unified_content = parse_dynamic_tprompt
    if parse_demo_content:
        unified_content += "\n\n" + parse_demo_content
    
    # Add scenario description
    unified_content += "\n\nGiven the following scenario, generate a complete task execution pipeline.\n\nScenario:\n{\n  \"sample-description\": \"" + user_input.replace('"', "'") + "\"\n}\n\nBased on the objects present and the scene description, construct a task pipeline (DAG) that:\n1. Starts with the \"source\" node\n2. Includes all necessary processing tasks from the available task types\n3. Ends with the \"pipeline-end\" node\n4. Has proper dependencies (follow the dependency guidelines and data flow patterns)"
    
    # Step 2: Get all available tasks and fetch models for ALL of them
    all_tasks = get_available_tasks()
    logger.info(f"Fetching models for {len(all_tasks)} tasks for unified prompt...")
    
    # Build a dictionary: task_id -> list of models
    all_models_by_task = {}
    all_tasks_info = TASKS_DICT_CACHE if TASKS_DICT_CACHE else []
    
    # Build task_info_dict for formatting
    task_info_dict = {}
    for t in all_tasks_info:
        if isinstance(t, dict) and "id" in t:
            task_info_dict[t["id"]] = t
    
    # Also handle case where all_tasks is a list of strings
    for task_item in all_tasks:
        if isinstance(task_item, str):
            task_id = task_item
        elif isinstance(task_item, dict):
            task_id = task_item.get("id", "")
        else:
            continue
        
        if not task_id:
            continue
        
        # Fetch models for this task
        models = get_models_for_task(task_id)
        if models:
            all_models_by_task[task_id] = models
            logger.debug(f"  Fetched {len(models)} models for task: {task_id}")
    
    logger.info(f"Fetched models for {len(all_models_by_task)} tasks")
    
    # Step 3: Format models for ALL tasks
    all_models_section = []
    all_models_section.append("**AVAILABLE MODELS FOR ALL TASKS**:")
    all_models_section.append("")
    all_models_section.append("Below are the available models for each task type. You will need to:")
    all_models_section.append("1. Select the relevant tasks for the scenario (from your brainstorming in Stage 1)")
    all_models_section.append("2. For each selected task, choose the most suitable model from its available models")
    all_models_section.append("3. Output the final result in the required format")
    all_models_section.append("")
    
    # Group models by task
    for task_id in sorted(all_models_by_task.keys()):
        models = all_models_by_task[task_id]
        if not models:
            continue
        
        # Get task info for context (optional, for better understanding)
        task_info = task_info_dict.get(task_id, {})
        task_name = task_info.get("name", task_id) if task_info else task_id
        
        all_models_section.append(f"**Task: {task_id}** ({task_name}):")
        all_models_section.append("")
        
        for idx, model in enumerate(models, 1):
            model_id = model.get("id", "unknown")
            model_name = model.get("name", "")
            model_desc = model.get("description", "")
            
            all_models_section.append(f"  {idx}. **{model_id}**:")
            all_models_section.append(f"     - name: {model_name}")
            all_models_section.append(f"     - description: {model_desc}")
            all_models_section.append("")
        
        all_models_section.append("")
    
    all_models_str = "\n".join(all_models_section)
    
    # Add models section
    unified_content += "\n\n" + all_models_str
    
    # Step 4: Load Stage 2 prompt (model selection instructions and output requirements)
    choose_model_prompt_file = config.get("prompt", {}).get("choose_model", "demos/user_prompt_choose_model_paper2.json")
    stage2_content = load_prompt_value(choose_model_prompt_file)
    
    unified_content += "\n\n" + stage2_content
    
    # Build single unified message
    messages = [
        {"role": "system", "content": unified_content}
    ]
    
    # Combine logit biases from both stages
    logit_bias = {}
    if HAS_TOKEN_UTILS:
        if task_parsing_highlight_ids:
            for item in task_parsing_highlight_ids:
                logit_bias[item] = config.get("logit_bias", {}).get("parse_task", 0.1)
        if choose_model_highlight_ids:
            for item in choose_model_highlight_ids:
                # Use choose_model bias, but don't override if already set
                if item not in logit_bias:
                    logit_bias[item] = config.get("logit_bias", {}).get("choose_model", 5)
    
    # Save prompt structure before sending to LLM
    save_prompt_to_file(messages, "parse_task_and_choose_models_paper2")
    
    # Set max_tokens to handle long unified prompt (16384 should be enough)
    # Measure LLM inference time
    start_time = time.time()
    response = send_llm_request(messages, temperature=0.1, logit_bias=logit_bias if logit_bias else None, max_tokens=16384)
    end_time = time.time()
    llm_inference_time = end_time - start_time
    
    # Save timing information
    save_timing_info("parse_task_and_choose_models_paper2", llm_inference_time)
    
    if not response:
        return {"error": "Failed to get LLM response"}
    
    # Parse response - should be JSON with "models" array
    try:
        # Clean up response (remove markdown code blocks if present)
        cleaned_str = response.strip()
        if cleaned_str.startswith("```"):
            lines = cleaned_str.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned_str = "\n".join(lines).strip()
        
        # Try to parse as JSON
        result = json.loads(cleaned_str)
        
        # Validate structure - expect "tasks" array
        if "tasks" not in result:
            logger.warning("Response missing 'tasks' key, trying to extract...")
            # Maybe LLM returned just the array?
            if isinstance(result, list):
                result = {"tasks": result}
            else:
                return {"error": "Invalid response format: missing 'tasks' key"}
        
        # Convert old format to new format if needed (for backward compatibility during transition)
        if "models" in result and "tasks" not in result:
            logger.warning("Response has 'models' key, converting to 'tasks' format...")
            tasks = []
            for model_entry in result["models"]:
                task_entry = {
                    "id": model_entry.get("task_id", model_entry.get("id", "unknown")),
                    "model_id": model_entry.get("id", "none"),
                    "inputs_from_upstreams": model_entry.get("inputs_from_upstreams", []),
                    "upstreams": model_entry.get("upstreams", []),
                    "outputs_for_downstreams": model_entry.get("outputs_for_downstreams", []),
                    "downstreams": model_entry.get("downstreams", []),
                    "model-chosen-reason": model_entry.get("model-chosen-reason", ""),
                    "sample-reasoning": model_entry.get("sample-reasoning", {})
                }
                tasks.append(task_entry)
            result = {"tasks": tasks}
        
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        logger.error(f"Response (first 500 chars): {response[:500]}")
        # Try to extract JSON using find_json
        try:
            extracted_json = find_json(cleaned_str)
            result = json.loads(extracted_json)
            if "tasks" not in result:
                if isinstance(result, list):
                    result = {"tasks": result}
                elif "models" in result:
                    # Convert old format
                    tasks = []
                    for model_entry in result["models"]:
                        task_entry = {
                            "id": model_entry.get("task_id", model_entry.get("id", "unknown")),
                            "model_id": model_entry.get("id", "none"),
                            "inputs_from_upstreams": model_entry.get("inputs_from_upstreams", []),
                            "upstreams": model_entry.get("upstreams", []),
                            "outputs_for_downstreams": model_entry.get("outputs_for_downstreams", []),
                            "downstreams": model_entry.get("downstreams", []),
                            "model-chosen-reason": model_entry.get("model-chosen-reason", ""),
                            "sample-reasoning": model_entry.get("sample-reasoning", {})
                        }
                        tasks.append(task_entry)
                    result = {"tasks": tasks}
                else:
                    return {"error": "Invalid response format after extraction"}
            return result
        except Exception as e2:
            logger.error(f"Failed to extract JSON: {e2}")
            return {"error": f"Failed to parse LLM response: {str(e)}"}


def process_decisions(user_input, tasks, api_key=None, api_type=None, api_endpoint=None):
    """Process tasks and make model selection decisions - SAME logic as run_task_decisions_only"""
    results = {}
    
    for task in tasks:
        task_id = task.get("id", 0)
        task_type = task.get("task")
        args = task.get("args", {})
        
        # SAME as original: Handle args resolution (lines 735-740)
        for resource in ["image", "audio"]:
            if resource in args and len(args[resource]) > 0 and not args[resource].startswith("http"):
                path = args[resource]
                if not path.startswith("public/") and not path.startswith("/"):
                    path = f"public/{path}"
                args[resource] = path
        
        # SAME as original: Handle control tasks (lines 742-751)
        if "-text-to-image" in task_type and "text" not in args:
            control = task_type.split("-")[0]
            if control == "seg":
                task_type = "image-segmentation"
                task["task"] = task_type
            elif control == "depth":
                task_type = "depth-estimation"
                task["task"] = task_type
            else:
                task_type = f"{control}-control"
        
        task["args"] = args
        
        # SAME as original: Handle special tasks (lines 756-774)
        if task_type.endswith("-text-to-image") or task_type.endswith("-control"):
            # ControlNet tasks - would need local deployment
            choose = {"id": "N/A", "reason": "ControlNet requires local deployment"}
            results[task_id] = {"task": task, "choose model result": choose, "status": "unavailable"}
            continue
            
        elif task_type in ["summarization", "translation", "conversational", "text-generation", "text2text-generation"]:
            choose = {"id": "ChatGPT", "reason": "ChatGPT performs well on some NLP tasks as well."}
            results[task_id] = {
                "task": task,
                "choose model result": choose,
                "selected_model_id": "ChatGPT",
                "hosted_on": "api",
                "status": "selected_not_executed",
                "available_candidates_count": 1
            }
            continue
        
        # Get models for this task from custom API
        # Models returned from API are considered available (API handles filtering)
        models = get_models_for_task(task_type)
        
        if not models:
            choose = {"id": "N/A", "reason": f"No models available for task {task_type}"}
            results[task_id] = {"task": task, "choose model result": choose, "status": "unavailable"}
            continue
        
        # SAME as original: Select model (lines 796-825)
        # All models from API are available (no separate availability check needed)
        if len(models) == 1:
            best_model_id = models[0].get("id", "N/A")
            reason = "Only one model available."
            choose = {"id": best_model_id, "reason": reason, "hosted_on": "api"}
        else:
            # Limit to top candidates (same as original: config["num_candidate_models"])
            candidates = models[:config.get("num_candidate_models", 5)]
            # Use LLM to choose (SAME as original)
            choose = choose_model_for_task(user_input, task, candidates, api_key, api_type, api_endpoint)
        
        # SAME as original: Store decision (lines 827-838)
        selected_model_id = choose.get("id", "N/A")
        hosted_on_val = choose.get("hosted_on", "api")
        
        results[task_id] = {
            "task": task,
            "choose model result": choose,
            "selected_model_id": selected_model_id,
            "hosted_on": hosted_on_val,
            "status": "selected_not_executed",
            "available_candidates_count": len(models)
        }
    
    return results


# ============================================================================
# FLASK API ENDPOINTS
# ============================================================================


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    tasks = get_available_tasks()
    return jsonify({
        "status": "healthy",
        "tasks_loaded": len(tasks) if tasks else 0,
        "api_base_url": MODEL_API_BASE_URL
    })


@app.route('/decisions', methods=['POST'])
def decisions():
    """Main endpoint for decision-only mode
    
    Supports two modes:
    1. Direct scenario input (new format):
       {
         "scenario": {
           "objects-seen": ["car", "person", ...],
           "sample-description": "Scene description..."
         }
       }
    
    2. Legacy messages format (backward compatibility):
       {
         "messages": [{"role": "user", "content": "..."}]
       }
    """
    try:
        # Reset timestamp for this request (for prompt logging)
        global CURRENT_REQUEST_TIMESTAMP
        CURRENT_REQUEST_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
        
        data = request.get_json()
        
        # Check if this is new scenario format (Option 1)
        if "scenario" in data:
            scenario = data["scenario"]
            objects_seen = scenario.get("objects-seen", [])
            description = scenario.get("sample-description", "")
            
            if not description:
                return jsonify({"error": "Missing sample-description in scenario"}), 400
            
            # Build user input from scenario
            user_input = description
            context = []
            
            # Load demo examples from file (using the formatted demo file)
            demo_messages = load_demos(config["demos_or_presteps"]["parse_task"])
            
            logger.info(f"Processing scenario request: objects={objects_seen}, description={description[:100]}...")
        
        # Legacy format (backward compatibility)
        else:
            messages = data.get("messages", [])
            
            if not messages:
                return jsonify({"error": "No messages or scenario provided"}), 400
            
            # Get user input from last message
            user_input = messages[-1].get("content", "")
            if not user_input:
                return jsonify({"error": "Empty user input"}), 400
            
            # Get context (previous messages)
            context = messages[:-1] if len(messages) > 1 else []
            
            # Use default objects_seen placeholder
            objects_seen = None
            
            logger.info(f"Processing request: {user_input}")
        
        # Check which method to use
        if USE_PAPER2_METHOD:
            # Paper 2: Single unified LLM call for both task parsing and model selection
            logger.info("Using Paper 2 method: Unified prompt with single LLM call")
            result = parse_task_and_choose_models_paper2(user_input, context, objects_seen=objects_seen)
            
            if isinstance(result, dict) and "error" in result:
                return jsonify({"error": result["error"]}), 500
            
            # Result should already be in the correct format {"tasks": [...]}
            if "tasks" not in result:
                return jsonify({"error": "Invalid response format from Paper 2 method"}), 500
            
            # Log prompt directory location
            prompt_dir = f"prompt_logs/request_{CURRENT_REQUEST_TIMESTAMP}"
            logger.info(f"Prompt logs saved to: {prompt_dir}/")
            
            # Return unified output
            try:
                json_str = json.dumps(result, indent=2, ensure_ascii=False)
                return Response(json_str, mimetype='application/json', status=200)
            except (TypeError, ValueError) as json_err:
                logger.error(f"JSON serialization error: {json_err}")
                return jsonify({"error": f"JSON serialization failed: {str(json_err)}"}), 500
        
        # Paper 1: Original method with separate calls
        logger.info("Using Paper 1 method: Separate task parsing and model selection calls")
        
        # Step 1: Parse tasks with objects_seen parameter
        task_str = parse_task(user_input, context, objects_seen=objects_seen)
        
        if isinstance(task_str, dict) and "error" in task_str:
            return jsonify({"error": task_str["error"]}), 500
        
        if not task_str or task_str.strip() == "":
            return jsonify({"error": "Failed to get task parsing response"}), 500
        
        task_str = task_str.strip()
        
        # SAME as original: Clean up JSON (lines 1041-1051)
        task_str = task_str.replace(' }} , ', '}}, ')
        task_str = task_str.replace(' }} ,', '}},')
        task_str = task_str.replace('}} ,', '}},')
        task_str = re.sub(r',\s*([}\]])', r'\1', task_str)
        
        try:
            tasks = json.loads(task_str)
        except json.JSONDecodeError as e:
            error_msg = str(e)
            
            # Handle "Extra data" error - JSON is valid but has extra content after it
            if "Extra data" in error_msg:
                logger.warning(f"JSON response has extra data after valid JSON: {e}")
                # Try to find the end of the first valid JSON array
                try:
                    # Find the closing bracket of the array
                    bracket_count = 0
                    end_pos = -1
                    for i, char in enumerate(task_str):
                        if char == '[':
                            bracket_count += 1
                        elif char == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                end_pos = i + 1
                                break
        
                    if end_pos > 0:
                        # Extract just the array part
                        array_str = task_str[:end_pos]
                        tasks = json.loads(array_str)
                        logger.info(f"Successfully parsed JSON array by removing extra data")
                    else:
                        raise ValueError("Could not find end of JSON array")
                except Exception as e2:
                    logger.warning(f"Failed to extract valid array from extra data: {e2}, trying truncated extraction...")
                    # Fall through to truncated extraction
                    extracted_tasks = extract_valid_tasks_from_truncated_json(task_str)
                    if extracted_tasks and len(extracted_tasks) > 0:
                        logger.info(f"Successfully extracted {len(extracted_tasks)} valid tasks from response")
                        tasks = extracted_tasks
                    else:
                        logger.error(f"Failed to extract valid tasks: {e2}")
                        return jsonify({"error": f"LLM response had extra data and could not be parsed. Error: {error_msg}"}), 500
            
            # Check if it's a truncation error (unterminated string)
            elif "Unterminated string" in error_msg or "Expecting" in error_msg:
                logger.warning(f"JSON response appears truncated: {e}")
                logger.info("Attempting to extract valid tasks from truncated response...")
                
                # Try to extract valid tasks from truncated JSON
                extracted_tasks = extract_valid_tasks_from_truncated_json(task_str)
                if extracted_tasks and len(extracted_tasks) > 0:
                    logger.info(f"Successfully extracted {len(extracted_tasks)} valid tasks from truncated response")
                    tasks = extracted_tasks
                else:
                    logger.error(f"Failed to extract valid tasks from truncated JSON: {e}")
                    logger.error(f"Truncated response (first 500 chars): {task_str[:500]}")
                    return jsonify({"error": f"LLM response was truncated and could not be parsed. Error: {error_msg}"}), 500
            else:
                logger.error(f"Failed to parse tasks: {e}, response: {task_str[:500]}")
                return jsonify({"error": f"Invalid task JSON: {error_msg}"}), 500
        except Exception as e:
            logger.error(f"Unexpected error parsing tasks: {e}, response: {task_str[:500]}")
            return jsonify({"error": f"Unexpected error parsing tasks: {str(e)}"}), 500
        
        if not isinstance(tasks, list):
            tasks = []
        
        # Helper function to clean task to only have the 6 required fields
        def clean_task_to_6_fields(task):
            """Extract only the 6 required fields from a task object in the correct order
            Order matches all_examples.json: id, inputs_from_upstreams, upstreams, 
            outputs_for_downstreams, downstreams, sample-reasoning
            Python 3.7+ preserves dict insertion order, so regular dict works fine
            """
            # Clean sample-reasoning: remove "general" key if present, only keep task IDs from downstreams
            sample_reasoning = task.get("sample-reasoning", {})
            if isinstance(sample_reasoning, dict):
                downstreams = task.get("downstreams", [])
                cleaned_reasoning = {
                    k: v for k, v in sample_reasoning.items()
                    if k != "general" and k in downstreams
                }
            else:
                cleaned_reasoning = {}
            
            # Explicitly construct dict in the exact order to match all_examples.json
            # Python 3.7+ preserves insertion order for regular dicts
            cleaned = {
                "id": task.get("id"),
                "inputs_from_upstreams": task.get("inputs_from_upstreams", []),
                "upstreams": task.get("upstreams", []),
                "outputs_for_downstreams": task.get("outputs_for_downstreams", []),
                "downstreams": task.get("downstreams", []),
                "sample-reasoning": cleaned_reasoning
            }
            return cleaned
        
        # Fix: LLM generates tasks with "id" field, but code expects "task" field for internal processing
        # Set task["task"] = task["id"] since in our format, id IS the task type
        # This is needed for process_decisions and semantic mapping, but will be removed from final output
        for task in tasks:
            if "id" in task and "task" not in task:
                task["task"] = task["id"]
        
        if task_str == "[]" or len(tasks) == 0:
            return jsonify({
                        "tasks": []
                    })
        
        # Handle single chitchat task (these don't need model selection)
        if len(tasks) == 1 and tasks[0].get("task") in ["summarization", "translation", "conversational", "text-generation", "text2text-generation"]:
            # These are handled by ChatGPT/LLM directly, not models
            task = tasks[0]
            task_entry = {
                "id": task.get("id"),
                "model_id": "ChatGPT",
                "inputs_from_upstreams": task.get("inputs_from_upstreams", []),
                "upstreams": task.get("upstreams", []),
                "outputs_for_downstreams": task.get("outputs_for_downstreams", []),
                "downstreams": task.get("downstreams", []),
                "model-chosen-reason": "ChatGPT performs well on NLP tasks",
                "sample-reasoning": task.get("sample-reasoning", {})
            }
            try:
                response_data = {"tasks": [task_entry]}
                json_str = json.dumps(response_data, indent=2, ensure_ascii=False)
                return Response(json_str, mimetype='application/json', status=200)
            except (TypeError, ValueError) as json_err:
                logger.error(f"JSON serialization error: {json_err}")
                logger.error(f"Task entry: {task_entry}")
                return jsonify({"error": f"JSON serialization failed: {str(json_err)}"}), 500
        
        # Step 2: Choose models for each task
        logger.info(f"Starting model selection for {len(tasks)} tasks...")
        
        # Get tasks info for formatting (use cached task dictionaries from startup)
        global TASKS_DICT_CACHE
        all_tasks_info = TASKS_DICT_CACHE
        if not all_tasks_info:
            # Fallback: try fetching from API
            logger.warning("TASKS_DICT_CACHE not available, fetching from API...")
            all_tasks_info = fetch_tasks_from_api()
            if all_tasks_info:
                TASKS_DICT_CACHE = all_tasks_info
            else:
                # Last resort: try get_available_tasks() which might have cached data
                all_tasks_info = get_available_tasks()
                logger.warning("Using fallback task source for task info")
        
        # Build a dict for quick lookup - ensure we only use dict items
        task_info_dict = {}
        for t in all_tasks_info:
            if isinstance(t, dict) and "id" in t:
                task_id = t["id"]
                task_info_dict[task_id] = t
                logger.debug(f"Added task to dict: {task_id} - has_description={bool(t.get('description'))}, has_inputs={bool(t.get('inputs'))}, has_outputs={bool(t.get('outputs'))}")
            elif isinstance(t, str):
                # If it's just a string (task ID), create a minimal dict
                task_info_dict[t] = {"id": t, "name": t, "description": "No description available", "inputs": [], "outputs": []}
                logger.debug(f"Added string task to dict: {t} (minimal info)")
        
        full_info_count = sum(1 for v in task_info_dict.values() if isinstance(v, dict) and v.get('description') and v.get('description') != 'No description available')
        logger.info(f"Built task_info_dict with {len(task_info_dict)} tasks (with full info: {full_info_count})")
        
        # Log a sample task to verify structure
        if task_info_dict:
            sample_id = list(task_info_dict.keys())[0]
            sample_task = task_info_dict[sample_id]
            logger.info(f"Sample task in dict: {sample_id} = {sample_task}")
        
        # Track timing for each choose_model call
        choose_model_timings = []
        
        # Build unified output with models
        unified_models = []
        
        for task in tasks:
            task_id = task.get("id")
            
            logger.info(f"  Processing task: {task_id}")
            
            # Skip source and pipeline-end (no model needed)
            if task_id in ["source", "pipeline-end"]:
                reason = f"{task_id.replace('-', ' ').title()} task doesn't need a model"
                
                # Build unified task entry
                task_entry = {
                    "id": task_id,
                    "model_id": "none",
                    "inputs_from_upstreams": task.get("inputs_from_upstreams", []),
                    "upstreams": task.get("upstreams", []),
                    "outputs_for_downstreams": task.get("outputs_for_downstreams", []),
                    "downstreams": task.get("downstreams", []),
                    "model-chosen-reason": reason,
                    "sample-reasoning": task.get("sample-reasoning", {})
                }
                unified_models.append(task_entry)
                logger.info(f"    Skipped (no model needed): {task_id}")
                continue
            
            # Get task info for prompt (same source as parse_task)
            task_info = task_info_dict.get(task_id, {})
            if not task_info or task_info == {}:
                logger.warning(f"    Task info not found for {task_id} in task_info_dict (available keys: {list(task_info_dict.keys())[:5]}...), using minimal info")
                task_info = {"id": task_id, "name": task_id, "description": "No description available", "inputs": [], "outputs": []}
            # Ensure task_info has id field
            if "id" not in task_info:
                task_info["id"] = task_id
            
            # Log task info for debugging
            logger.info(f"    Task info for {task_id}: name={task_info.get('name')}, description={task_info.get('description', 'N/A')[:50]}, inputs={task_info.get('inputs', [])}, outputs={task_info.get('outputs', [])}")
            
            # Get available models for this task
            models = get_models_for_task(task_id)
            
            if not models or len(models) == 0:
                logger.warning(f"    No models available for task: {task_id}")
                task_entry = {
                    "id": task_id,
                    "model_id": "none",
                    "inputs_from_upstreams": task.get("inputs_from_upstreams", []),
                    "upstreams": task.get("upstreams", []),
                    "outputs_for_downstreams": task.get("outputs_for_downstreams", []),
                    "downstreams": task.get("downstreams", []),
                    "model-chosen-reason": f"No models available for task {task_id}",
                    "sample-reasoning": task.get("sample-reasoning", {})
                }
                unified_models.append(task_entry)
                continue
            
            logger.info(f"    Found {len(models)} available models")
            
            # Call LLM to choose model
            choice_result = choose_model_for_task(
                scenario_description=user_input,
                task=task,
                task_info=task_info,
                models=models
            )
            
            model_id = choice_result.get("id", "none")
            reason = choice_result.get("reason", "No reason provided")
            llm_time = choice_result.get("llm_time", 0.0)
            
            # Track timing
            choose_model_timings.append((task_id, llm_time))
            
            logger.info(f"    Selected model: {model_id}")
            
            # Build unified task entry
            task_entry = {
                "id": task_id,
                "model_id": model_id,
                "inputs_from_upstreams": task.get("inputs_from_upstreams", []),
                "upstreams": task.get("upstreams", []),
                "outputs_for_downstreams": task.get("outputs_for_downstreams", []),
                "downstreams": task.get("downstreams", []),
                "model-chosen-reason": reason,
                "sample-reasoning": task.get("sample-reasoning", {})
            }
            unified_models.append(task_entry)
        
        # Save unified timing information
        if choose_model_timings:
            save_all_choose_model_timing(choose_model_timings)
        
        # Log prompt directory location
        prompt_dir = f"prompt_logs/request_{CURRENT_REQUEST_TIMESTAMP}"
        logger.info(f"Prompt logs saved to: {prompt_dir}/")
        
        # Return unified output with tasks array
        # Use manual JSON serialization to ensure field order is preserved
        try:
            response_data = {"tasks": unified_models}
            json_str = json.dumps(response_data, indent=2, ensure_ascii=False)
            return Response(json_str, mimetype='application/json', status=200)
        except (TypeError, ValueError) as json_err:
            logger.error(f"JSON serialization error: {json_err}")
            logger.error(f"Unified tasks: {unified_models}")
            return jsonify({"error": f"JSON serialization failed: {str(json_err)}"}), 500
        
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def generate_response_description(user_input, tasks, results):
    """Generate JSON execution config for sending to execution server
    Returns structured execution plan with DAG and scheduling information"""
    
    # Build execution plan with dependency analysis
    task_list = []
    execution_levels = {}  # Group tasks by execution level
    dag_edges = []  # DAG edges (dependencies)
    
    for task_id, result in sorted(results.items()):
        task = result.get("task", {})
        task_type = task.get("task", "unknown")
        selected_model = result.get("selected_model_id", "N/A")
        args = task.get("args", {})
        deps = task.get("dep", [-1])
        
        # Filter out -1 from dependencies (means no deps)
        dependencies = [d for d in deps if d != -1]
        
        # Calculate execution level based on dependencies
        if len(dependencies) == 0:
            level = 0  # No dependencies - runs first
        else:
            # Level = max(dependency levels) + 1
            level = max([execution_levels.get(dep, {}).get("execution_level", 0) for dep in dependencies]) + 1
            # Add DAG edges
            for dep in dependencies:
                dag_edges.append({"from": dep, "to": task_id})
        
        task_config = {
            "task_id": task_id,
            "task_type": task_type,
            "model_id": selected_model,
            "model_reason": result.get("choose model result", {}).get("reason", ""),
            "args": args,
            "dependencies": dependencies,
            "execution_level": level
        }
        task_list.append(task_config)
        execution_levels[task_id] = task_config
    
    # Build execution order by level (showing which tasks can run in parallel)
    if task_list:
        max_level = max([t["execution_level"] for t in task_list])
        execution_order = []
        for level in range(max_level + 1):
            level_tasks = [t["task_id"] for t in task_list if t["execution_level"] == level]
            execution_order.append({
                "level": level,
                "parallel": len(level_tasks) > 1,
                "tasks": level_tasks
            })
    else:
        execution_order = []
    
    # Build complete execution config
    execution_config = {
        "execution_config": {
            "user_request": user_input,
            "total_tasks": len(task_list),
            "tasks": task_list,
            "dag": {
                "nodes": [t["task_id"] for t in task_list],
                "edges": dag_edges
            },
            "execution_order": execution_order
        }
    }
    
    # Return JSON string
    return json.dumps(execution_config, indent=2)


if __name__ == '__main__':
    # Load prompts dynamically from API on startup
    logger.info(f"Decision API server initializing...")
    logger.info(f"LLM endpoint: {LLM_ENDPOINT}")
    logger.info(f"Model API: {MODEL_API_BASE_URL}")
    logger.info("")
    
    # Fetch and update prompts from API
    load_prompts_dynamically()
    
    # Start server
    logger.info(f"Starting Decision API server on {args.host}:{args.port}...")
    waitress.serve(app, host=args.host, port=args.port)

