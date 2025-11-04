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
from flask import request, jsonify
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
    from get_token_ids import get_token_ids_for_task_parsing, get_token_ids_for_choose_model, count_tokens, get_max_context_length
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
parser.add_argument("--config", type=str, default="configs/config_test_decisions.yaml")
parser.add_argument("--port", type=int, default=8004)
parser.add_argument("--host", type=str, default="0.0.0.0")
args = parser.parse_args()

# Load config
config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

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
            if isinstance(data, list):
                # Process each message to handle content_array
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

parse_task_demos_or_presteps = load_demos(config["demos_or_presteps"]["parse_task"])
choose_model_demos_or_presteps = load_demos(config["demos_or_presteps"]["choose_model"])
response_results_demos_or_presteps = load_demos(config["demos_or_presteps"]["response_results"])

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

parse_task_prompt = load_prompt_value(config["prompt"]["parse_task"])
choose_model_prompt = config["prompt"]["choose_model"]
response_results_prompt = config["prompt"]["response_results"]

parse_task_tprompt = load_prompt_value(config["tprompt"]["parse_task"])
choose_model_tprompt = config["tprompt"]["choose_model"]
response_results_tprompt = config["tprompt"]["response_results"]

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
MODELS_CACHE = {}

# Flask app
app = flask.Flask(__name__)
CORS(app)

# Global variable to track current request timestamp for prompt logging
CURRENT_REQUEST_TIMESTAMP = None


def save_prompt_to_file(messages, prompt_type, task_id=None):
    """Save the complete prompt structure (messages array) to a JSON file
    
    Args:
        messages: The complete messages array sent to LLM
        prompt_type: Either "parse_task" or "choose_model"
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
    elif prompt_type == "choose_model" and task_id is not None:
        filename = f"{prompt_dir}/02_choose_model_task_{task_id}.json"
    else:
        filename = f"{prompt_dir}/{prompt_type}.json"
    
    # Save with metadata
    prompt_data = {
        "timestamp": CURRENT_REQUEST_TIMESTAMP,
        "prompt_type": prompt_type,
        "task_id": task_id,
        "messages": messages
    }
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(prompt_data, f, indent=2, ensure_ascii=False)
    
    logger.debug(f"Saved {prompt_type} prompt to: {filename}")


def get_available_tasks():
    """Fetch available tasks from custom API"""
    global TASKS_CACHE
    if TASKS_CACHE is not None:
        return TASKS_CACHE
    
    try:
        response = requests.get(f"{MODEL_API_BASE_URL}/query/context/tasks", timeout=5)
        if response.status_code == 200:
            TASKS_CACHE = response.json()
            logger.info(f"Loaded {len(TASKS_CACHE)} tasks from API: {TASKS_CACHE}")
            return TASKS_CACHE
        else:
            logger.error(f"Failed to fetch tasks: HTTP {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Error fetching tasks from API: {e}")
        return []


def map_task_semantically(parsed_task, available_tasks, user_input):
    """Use LLM to semantically map parsed task to closest API task based on user input"""
    if parsed_task in available_tasks:
        return parsed_task
    
    # Use LLM to find semantically closest task from API
    tasks_list = ", ".join([f'"{task}"' for task in available_tasks])
    
    system_prompt = """You are a task mapping assistant. Map the given task name to the most semantically similar task from the available list based on the user's original request.

Return ONLY the task ID from the list that best matches semantically. Return JSON format: {"task_id": "task-name"}."""

    user_prompt = f"""Original user request: "{user_input}"

Parsed task: "{parsed_task}"

Available API tasks: {tasks_list}

Which task from the available list best matches the semantic meaning of "{parsed_task}" given the user's request? Return JSON: {{"task_id": "task-name"}}."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = send_llm_request(messages, temperature=0.1)
    if not response:
        logger.warning(f"Failed to semantically map task '{parsed_task}'")
        return parsed_task
    
    # Parse response
    try:
        # Extract JSON
        json_match = re.search(r'\{[^{}]*"task_id"[^{}]*\}', response)
        if json_match:
            response = json_match.group()
        response = re.sub(r',\s*([}\]])', r'\1', response)
        result = json.loads(response)
        mapped_task = result.get("task_id", "").strip().strip('"').strip("'")
        
        if mapped_task in available_tasks:
            logger.info(f"Semantically mapped '{parsed_task}' -> '{mapped_task}' (user: '{user_input}')")
            return mapped_task
        else:
            logger.warning(f"Mapped task '{mapped_task}' not in available tasks: {available_tasks}")
            return parsed_task
    except Exception as e:
        logger.warning(f"Failed to parse semantic mapping response: {e}, response: {response}")
        return parsed_task


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

def send_llm_request(messages, temperature=0.1, logit_bias=None):
    """Send request to LLM endpoint - SAME structure as original send_request"""
    try:
        data = {
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": temperature
        }
        if logit_bias:
            data["logit_bias"] = logit_bias
        
        response = requests.post(
            LLM_ENDPOINT,
            headers={"Content-Type": "application/json"},
            json=data,
            timeout=30
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


def parse_task(user_input, context=[], objects_seen=None, api_key=None, api_type=None, api_endpoint=None):
    """Parse user input into tasks using LLM
    
    Args:
        user_input: The scene description
        context: Previous conversation context (for multi-turn dialogs)
        objects_seen: List of objects present in scene (or None to infer)
        
    Returns:
        String containing JSON array of tasks
    """
    available_tasks = get_available_tasks()
    if not available_tasks:
        return {"error": "No tasks available from API"}
    
    # No dynamic task list replacement needed anymore - system prompt is static
    dynamic_tprompt = parse_task_tprompt
    
    # Load few-shot examples
    demos_or_presteps = parse_task_demos_or_presteps
    messages = json.loads(demos_or_presteps)
    messages.insert(0, {"role": "system", "content": dynamic_tprompt})

    # Context window trimming logic
    start = 0
    while start <= len(context):
        history = context[start:]
        
        # Handle objects_seen parameter
        if objects_seen is not None:
            objects_str = ", ".join(objects_seen) if isinstance(objects_seen, list) else str(objects_seen)
        else:
            objects_str = "infer from the scene description"
        
        # Build prompt with scenario format
        prompt = replace_slot(parse_task_prompt, {
            "input": user_input,
            "context": history,
            "objects_seen": objects_str
        })
        messages.append({"role": "user", "content": prompt})
        
        # Token counting (if available)
        if HAS_TOKEN_UTILS:
            history_text = "<im_end>\nuser<im_start>".join([m["content"] for m in messages])
            num = count_tokens(LLM_encoding, history_text)
            if get_max_context_length(LLM_MODEL) - num > 800:
                break
        messages.pop()
        start += 2
    
    # Call with logit_bias
    logit_bias = None
    if HAS_TOKEN_UTILS and task_parsing_highlight_ids:
        logit_bias = {item: config.get("logit_bias", {}).get("parse_task", 0.1) for item in task_parsing_highlight_ids}
    
    # Save prompt structure before sending to LLM
    save_prompt_to_file(messages, "parse_task")
    
    response = send_llm_request(messages, temperature=0.1, logit_bias=logit_bias)
    if not response:
        return {"error": "Failed to get LLM response"}
    
    return response  # Return string, let caller parse


def unfold(tasks):
    """SAME as original - unfold tasks with multiple GENERATED references"""
    flag_unfold_task = False
    try:
        for task in tasks:
            for key, value in task.get("args", {}).items():
                if isinstance(value, str) and "<GENERATED>" in value:
                    generated_items = value.split(",")
                    if len(generated_items) > 1:
                        flag_unfold_task = True
                        for item in generated_items:
                            new_task = copy.deepcopy(task)
                            dep_task_id = int(item.split("-")[1])
                            new_task["dep"] = [dep_task_id]
                            new_task["args"][key] = item
                            tasks.append(new_task)
                        tasks.remove(task)
    except Exception as e:
        logger.debug(f"unfold task failed: {e}")
    
    if flag_unfold_task:
        logger.debug(f"unfold tasks: {tasks}")
    
    return tasks

def fix_dep(tasks):
    """SAME as original - Fix task dependencies (lines 270-281)
    BUT PRESERVES manually set dependencies for logical ordering"""
    for task in tasks:
        args = task.get("args", {})
        # Preserve existing dependencies (may be set by LLM for logical ordering)
        existing_deps = task.get("dep", [])
        if not isinstance(existing_deps, list):
            existing_deps = []
        
        task["dep"] = existing_deps.copy()  # Start with existing deps
        
        # Add dependencies from <GENERATED> tags
        for k, v in args.items():
            if isinstance(v, str) and "<GENERATED>" in v:
                dep_task_id = int(v.split("-")[1])
                if dep_task_id not in task["dep"]:
                    task["dep"].append(dep_task_id)
        
        # If no dependencies set (neither logical nor GENERATED), use default
        if len(task["dep"]) == 0:
            task["dep"] = [-1]
    return tasks


def choose_model_for_task(user_input, task_command, models, api_key=None, api_type=None, api_endpoint=None):
    """Use LLM to select best model - EXACT SAME LOGIC as original choose_model"""
    if not models:
        return {"id": "N/A", "reason": "No models available for this task"}
    
    # Models passed here are already filtered/limited - use them directly
    # (Same as original: candidates are pre-filtered before calling choose_model)
    candidates = models
    
    # Build model info EXACTLY as original (lines 799-809 in awesome_chat.py)
    cand_models_info = [
        {
            "id": model.get("id", "unknown"),
            "inference endpoint": "api",  # All from custom API
            "likes": model.get("likes", 0),
            "description": model.get("description", "")[:config.get("max_description_length", 100)],
            "tags": model.get("meta", {}).get("tags") if model.get("meta") else None,
        }
        for model in candidates
    ]
    
    # SAME as original: Use prompt templates and few-shot examples
    prompt = replace_slot(choose_model_prompt, {
        "input": user_input,
        "task": task_command,
        "metas": cand_models_info,
    })
    demos_or_presteps = replace_slot(choose_model_demos_or_presteps, {
        "input": user_input,
        "task": task_command,
        "metas": cand_models_info
    })
    messages = json.loads(demos_or_presteps)
    messages.insert(0, {"role": "system", "content": choose_model_tprompt})
    messages.append({"role": "user", "content": prompt})
    
    # Save prompt structure before sending to LLM (include task_id for filename)
    task_id = task_command.get("id", "unknown")
    save_prompt_to_file(messages, "choose_model", task_id=task_id)
    
    # SAME as original: Call with logit_bias
    logit_bias = None
    if HAS_TOKEN_UTILS and choose_model_highlight_ids:
        logit_bias = {item: config.get("logit_bias", {}).get("choose_model", 5) for item in choose_model_highlight_ids}
    
    choose_str = send_llm_request(messages, temperature=0.1, logit_bias=logit_bias)
    if not choose_str:
        # Fallback
        return {"id": candidates[0].get("id", "N/A"), "reason": "Selected first available model (LLM selection failed)"}
    
    # SAME as original: Parse JSON (lines 813-825)
    try:
        choose = json.loads(choose_str)
        reason = choose.get("reason", "")
        best_model_id = choose.get("id", "")
        choose["hosted_on"] = "api"
        return choose
    except Exception as e:
        logger.warning(f"the response [ {choose_str} ] is not a valid JSON, try to find the model id and reason in the response.")
        choose_str = find_json(choose_str)
        best_model_id, reason, choose = get_id_reason(choose_str)
        choose["hosted_on"] = "api"
        choose["reason"] = reason
        return choose


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


def load_all_examples():
    """Load all examples from local file or API"""
    # Try to load from local file first (faster)
    try:
        with open("demos/all_examples.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        pass
    
    # If local file doesn't exist, fetch from API
    try:
        response = requests.post(
            f"{MODEL_API_BASE_URL}/query/get_examples_for_task_pipelines",
            headers={"Content-Type": "application/json"},
            json={"task_id": "", "scenario_id": "", "num_examples": 100},
            timeout=10
        )
        if response.status_code == 200:
            examples = response.json()
            # Save to local file for future use
            with open("demos/all_examples.json", "w") as f:
                json.dump(examples, f, indent=2)
            return examples
        else:
            logger.error(f"Failed to fetch examples from API: HTTP {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Error fetching examples from API: {e}")
        return []


def load_category_mapping():
    """Load category mapping from file"""
    try:
        with open("demos/category_mapping.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("Category mapping file not found. Run the category mapping script first.")
        return None


def build_few_shot_examples(category_indices, num_examples_per_category, all_examples):
    """Build few-shot examples from specified categories
    
    Args:
        category_indices: List of category indices to use
        num_examples_per_category: Number of examples to take from each category
        all_examples: List of all available examples
        
    Returns:
        List of few-shot examples formatted for the prompt
    """
    category_mapping = load_category_mapping()
    if not category_mapping:
        return []
    
    # Group examples by base_pipeline_id
    examples_by_category = {}
    for example in all_examples:
        base_id = example.get("base_pipeline_id")
        if base_id not in examples_by_category:
            examples_by_category[base_id] = []
        examples_by_category[base_id].append(example)
    
    # Build few-shot examples
    few_shot_examples = []
    for cat_idx in category_indices:
        # Find category by index
        category = next((c for c in category_mapping["categories"] if c["index"] == cat_idx), None)
        if not category:
            logger.warning(f"Category index {cat_idx} not found")
            continue
        
        base_id = category["base_pipeline_id"]
        examples = examples_by_category.get(base_id, [])
        
        # Take up to num_examples_per_category (or all if fewer available)
        num_to_take = min(num_examples_per_category, len(examples))
        for example in examples[:num_to_take]:
            few_shot_examples.append(example)
    
    return few_shot_examples


def format_few_shot_for_prompt(few_shot_examples):
    """Format few-shot examples into prompt messages
    
    Args:
        few_shot_examples: List of example objects with scenario and tasks
        
    Returns:
        List of message dictionaries (user/assistant pairs)
    """
    messages = []
    
    for example in few_shot_examples:
        scenario = example.get("scenario", {})
        tasks = example.get("tasks", [])
        
        # Build user message (scenario input)
        objects_seen = scenario.get("objects-seen", [])
        description = scenario.get("sample-description", "")
        
        user_content = f"Given the following scenario, generate a complete task execution pipeline.\n\n"
        user_content += f"Scenario:\n"
        user_content += f"Objects present in scene: {', '.join(objects_seen)}\n"
        user_content += f"Scene description: {description}\n\n"
        user_content += "Based on the objects present and the scene description, construct a task pipeline (DAG) with proper dependencies. "
        user_content += "Output ONLY the JSON array of task objects with all 6 required fields."
        
        messages.append({"role": "user", "content": user_content})
        
        # Build assistant message (tasks output)
        # Remove sample-reasoning field from tasks (not part of output)
        clean_tasks = []
        for task in tasks:
            clean_task = {k: v for k, v in task.items() if k != "sample-reasoning"}
            clean_tasks.append(clean_task)
        
        assistant_content = json.dumps(clean_tasks, indent=2)
        messages.append({"role": "assistant", "content": assistant_content})
    
    return messages


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
        except Exception as e:
            logger.error(f"Failed to parse tasks: {e}, response: {task_str}")
            return jsonify({"error": f"Invalid task JSON: {task_str}"}), 500
        
        if not isinstance(tasks, list):
            tasks = []
        
        if task_str == "[]" or len(tasks) == 0:
            return jsonify({
                "decisions": {
                    "input": user_input,
                    "planned_tasks": [],
                    "model_selections": {},
                    "summary": {
                        "total_tasks": 0,
                        "tasks_with_selections": 0,
                        "tasks_unavailable": 0
                    }
                },
                "message": "No tasks could be parsed from input"
            })
        
        # SAME as original: Handle single chitchat task (line 1063-1066)
        if len(tasks) == 1 and tasks[0]["task"] in ["summarization", "translation", "conversational", "text-generation", "text2text-generation"]:
            # These are handled by ChatGPT/LLM directly, not models
            return jsonify({
                "decisions": {
                    "input": user_input,
                    "planned_tasks": tasks,
                    "model_selections": {
                        tasks[0]["id"]: {
                            "task": tasks[0],
                            "selected_model": "ChatGPT",
                            "hosted_on": "api",
                            "reason": "ChatGPT performs well on NLP tasks",
                            "status": "selected_not_executed",
                            "available_candidates": 1
                        }
                    },
                    "summary": {
                        "total_tasks": 1,
                        "tasks_with_selections": 1,
                        "tasks_unavailable": 0
                    }
                }
            })
        
        # SAME as original: Post-process tasks (lines 1068-1069)
        tasks = unfold(tasks)  # Handle multiple GENERATED references
        tasks = fix_dep(tasks)  # Fix dependencies
        
        # Fix: Replace any <GENERATED> tags with original image path
        # (Tasks don't generate images, they all use the same original image)
        original_image_path = None
        for task in tasks:
            image_path = task.get("args", {}).get("image", "")
            if image_path and "<GENERATED>" not in image_path:
                original_image_path = image_path
                break
        
        if original_image_path:
            for task in tasks:
                args = task.get("args", {})
                if "image" in args and "<GENERATED>" in args["image"]:
                    logger.info(f"Replacing <GENERATED> tag in task {task.get('id')} with original image path: {original_image_path}")
                    args["image"] = original_image_path
        
        # Map task names semantically to API task names
        # If LLM parsed a task name that doesn't match API exactly, use semantic mapping
        available_tasks = get_available_tasks()
        for task in tasks:
            original_task = task.get("task")
            if original_task not in available_tasks:
                # Use LLM to semantically map to closest API task based on user input
                mapped_task = map_task_semantically(original_task, available_tasks, user_input)
                if mapped_task in available_tasks:
                    task["task"] = mapped_task
                else:
                    logger.warning(f"Could not map task '{original_task}' to any API task")
        
        # Step 2: Make decisions (SAME as run_task_decisions_only logic)
        # Note: All models from API are considered available (API handles filtering)
        results = process_decisions(user_input, tasks, api_key=None, api_type=None, api_endpoint=None)
        
        # Format output
        decisions_output = {
            "input": user_input,
            "planned_tasks": tasks,
            "model_selections": {},
            "summary": {
                "total_tasks": len(tasks),
                "tasks_with_selections": len([r for r in results.values() if r.get("status") == "selected_not_executed"]),
                "tasks_unavailable": len([r for r in results.values() if r.get("status") == "unavailable"])
            }
        }
        
        # SAME as original: Format output (lines 1101-1109)
        for task_id, result in sorted(results.items()):
            decisions_output["model_selections"][task_id] = {
                "task": result["task"],
                "selected_model": result.get("selected_model_id"),  # SAME as original line 1104
                "hosted_on": result.get("hosted_on"),
                "reason": result.get("choose model result", {}).get("reason"),  # SAME as original line 1106
                "status": result.get("status"),
                "available_candidates": result.get("available_candidates_count")  # SAME as original line 1108
            }
        
        # Generate response description (similar to awesome_chat.py line 1146)
        # For decision-only mode, describe the planned workflow instead of actual results
        response_text = generate_response_description(user_input, tasks, results)
        
        # Save execution config to file for easy viewing
        output_dir = "execution_configs"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/execution_plan_{CURRENT_REQUEST_TIMESTAMP}.json"
        with open(filename, "w") as f:
            f.write(response_text)
        logger.info(f"Execution config saved to: {filename}")
        
        # Log prompt directory location
        prompt_dir = f"prompt_logs/request_{CURRENT_REQUEST_TIMESTAMP}"
        logger.info(f"Prompt logs saved to: {prompt_dir}/")
        
        return jsonify({
            "decisions": decisions_output, 
            "raw_results": results,
            "message": response_text,  # JSON execution config
            "execution_config_file": filename,  # Path to saved file
            "prompt_logs_directory": prompt_dir  # Path to prompt logs
        })
        
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/decisions/test', methods=['POST'])
def decisions_test():
    """Testing endpoint with automatic few-shot selection and test scenario extraction
    
    Request format:
    {
      "few_shot_categories": [0, 2, 5],           # Array of category indices to use for few-shot
      "num_few_shot_examples_per_category": 2,   # Number of examples per category
      "test_category_index": 10,                 # Category index for testing
      "test_example_index": 0                    # Example index within that category (0-based)
    }
    
    This will:
    1. Load examples from specified few-shot categories
    2. Extract the test scenario from the specified category and example index
    3. Generate tasks for that test scenario
    4. Return both the generated tasks and the ground truth for comparison
    """
    try:
        # Reset timestamp for this request
        global CURRENT_REQUEST_TIMESTAMP
        CURRENT_REQUEST_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
        
        data = request.get_json()
        
        # Validate required parameters
        few_shot_categories = data.get("few_shot_categories", [])
        num_examples_per_category = data.get("num_few_shot_examples_per_category", 1)
        test_category_index = data.get("test_category_index")
        test_example_index = data.get("test_example_index", 0)
        
        if not isinstance(few_shot_categories, list):
            return jsonify({"error": "few_shot_categories must be an array"}), 400
        
        if test_category_index is None:
            return jsonify({"error": "test_category_index is required"}), 400
        
        # Load all examples and category mapping
        all_examples = load_all_examples()
        if not all_examples:
            return jsonify({"error": "Failed to load examples"}), 500
        
        category_mapping = load_category_mapping()
        if not category_mapping:
            return jsonify({"error": "Failed to load category mapping"}), 500
        
        logger.info(f"Testing mode: few_shot_cats={few_shot_categories}, test_cat={test_category_index}, test_ex={test_example_index}")
        
        # Build few-shot examples (excluding test example)
        few_shot_examples_data = build_few_shot_examples(few_shot_categories, num_examples_per_category, all_examples)
        
        # Get test example
        test_category = next((c for c in category_mapping["categories"] if c["index"] == test_category_index), None)
        if not test_category:
            return jsonify({"error": f"Test category index {test_category_index} not found"}), 400
        
        # Find test example from all_examples
        test_examples_in_category = [ex for ex in all_examples if ex.get("base_pipeline_id") == test_category["base_pipeline_id"]]
        
        if test_example_index >= len(test_examples_in_category):
            return jsonify({
                "error": f"Test example index {test_example_index} out of range. Category '{test_category['base_pipeline_id']}' has {len(test_examples_in_category)} examples"
            }), 400
        
        test_example = test_examples_in_category[test_example_index]
        test_scenario = test_example.get("scenario", {})
        test_ground_truth = test_example.get("tasks", [])
        
        # Format few-shot examples for prompt
        few_shot_messages = format_few_shot_for_prompt(few_shot_examples_data)
        
        # Build system prompt
        system_prompt_content = load_prompt_value(config["tprompt"]["parse_task"])
        
        # Build complete prompt with few-shot examples
        messages = [{"role": "system", "content": system_prompt_content}]
        messages.extend(few_shot_messages)
        
        # Add test scenario as user input
        objects_seen = test_scenario.get("objects-seen", [])
        description = test_scenario.get("sample-description", "")
        
        user_prompt_template = load_prompt_value(config["prompt"]["parse_task"])
        test_user_prompt = replace_slot(user_prompt_template, {
            "input": description,
            "context": "",
            "objects_seen": ", ".join(objects_seen)
        })
        
        messages.append({"role": "user", "content": test_user_prompt})
        
        # Save prompt structure for debugging
        save_prompt_to_file(messages, "parse_task")
        
        # Call LLM to generate tasks
        logit_bias = None
        if HAS_TOKEN_UTILS and task_parsing_highlight_ids:
            logit_bias = {item: config.get("logit_bias", {}).get("parse_task", 0.1) for item in task_parsing_highlight_ids}
        
        response = send_llm_request(messages, temperature=0.1, logit_bias=logit_bias)
        if not response:
            return jsonify({"error": "Failed to get LLM response"}), 500
        
        # Parse generated tasks
        response = response.strip()
        response = response.replace(' }} , ', '}}, ')
        response = response.replace(' }} ,', '}},')
        response = response.replace('}} ,', '}},')
        response = re.sub(r',\s*([}\]])', r'\1', response)
        
        try:
            generated_tasks = json.loads(response)
        except Exception as e:
            logger.error(f"Failed to parse generated tasks: {e}, response: {response}")
            return jsonify({"error": f"Invalid task JSON: {response}"}), 500
        
        if not isinstance(generated_tasks, list):
            generated_tasks = []
        
        # Clean ground truth (remove sample-reasoning)
        clean_ground_truth = []
        for task in test_ground_truth:
            clean_task = {k: v for k, v in task.items() if k != "sample-reasoning"}
            clean_ground_truth.append(clean_task)
        
        # Return results with comparison
        return jsonify({
            "test_info": {
                "test_category": test_category["base_pipeline_id"],
                "test_category_index": test_category_index,
                "test_example_index": test_example_index,
                "test_example_id": test_example.get("id"),
                "few_shot_categories": [
                    {
                        "index": cat_idx,
                        "name": next((c["base_pipeline_id"] for c in category_mapping["categories"] if c["index"] == cat_idx), "unknown")
                    }
                    for cat_idx in few_shot_categories
                ],
                "num_few_shot_examples": len(few_shot_examples_data)
            },
            "test_scenario": {
                "objects_seen": objects_seen,
                "description": description
            },
            "generated_tasks": generated_tasks,
            "ground_truth_tasks": clean_ground_truth,
            "prompt_logs_directory": f"prompt_logs/request_{CURRENT_REQUEST_TIMESTAMP}/"
        })
        
    except Exception as e:
        logger.error(f"Error in testing mode: {e}", exc_info=True)
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
    # Load tasks on startup
    tasks = get_available_tasks()
    logger.info(f"Decision API server starting...")
    logger.info(f"Loaded {len(tasks) if tasks else 0} tasks from API")
    logger.info(f"LLM endpoint: {LLM_ENDPOINT}")
    logger.info(f"Model API: {MODEL_API_BASE_URL}")
    
    waitress.serve(app, host=args.host, port=args.port)

