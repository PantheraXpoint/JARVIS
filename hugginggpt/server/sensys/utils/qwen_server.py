#!/usr/bin/env python3
"""
Qwen 2.5 7B Local LLM Server for JARVIS
Compatible with OpenAI API format
"""

import os
import json
import logging
import argparse
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, request, jsonify
from flask_cors import CORS
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenServer:
    def __init__(self, model_path: str, device: str = "auto", max_memory: str = "12GB"):
        self.model_path = model_path
        self.device = device
        self.max_memory = max_memory
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load Qwen 2.5 7B model with memory optimization"""
        logger.info(f"Loading Qwen 2.5 7B from {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # Configure device and memory
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                # Use memory mapping for large models
                torch_dtype = torch.float16
            else:
                self.device = "cpu"
                torch_dtype = torch.float32
        else:
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Load model with memory optimization (official Qwen method)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",  # Official Qwen recommendation
            device_map="auto",   # Official Qwen recommendation
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        logger.info(f"Model loaded successfully on {self.device}")
    
    def generate_response(self, messages: List[Dict[str, str]], max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """Generate response using Qwen 2.5 7B with official chat template"""
        try:
            # Use Qwen's official apply_chat_template method
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize input
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response (official method)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"

# Flask app setup
app = Flask(__name__)
CORS(app)

# Global model instance
qwen_server = None

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI-compatible chat completions endpoint"""
    try:
        data = request.get_json()
        
        messages = data.get('messages', [])
        max_tokens = data.get('max_tokens', 1024)
        temperature = data.get('temperature', 0.7)
        
        # Generate response
        response_text = qwen_server.generate_response(messages, max_tokens, temperature)
        
        # Format response in OpenAI style
        response = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "qwen-2.5-7b-instruct",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(qwen_server.tokenizer.encode(str(messages))),
                "completion_tokens": len(qwen_server.tokenizer.encode(response_text)),
                "total_tokens": len(qwen_server.tokenizer.encode(str(messages))) + len(qwen_server.tokenizer.encode(response_text))
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in chat_completions: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/v1/completions', methods=['POST'])
def completions():
    """OpenAI-compatible completions endpoint"""
    try:
        data = request.get_json()
        
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 1024)
        temperature = data.get('temperature', 0.7)
        
        # Convert prompt to messages format
        messages = [{"role": "user", "content": prompt}]
        
        # Generate response
        response_text = qwen_server.generate_response(messages, max_tokens, temperature)
        
        # Format response in OpenAI style
        response = {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "qwen-2.5-7b-instruct",
            "choices": [{
                "index": 0,
                "text": response_text,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(qwen_server.tokenizer.encode(prompt)),
                "completion_tokens": len(qwen_server.tokenizer.encode(response_text)),
                "total_tokens": len(qwen_server.tokenizer.encode(prompt)) + len(qwen_server.tokenizer.encode(response_text))
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in completions: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model": "qwen-2.5-7b-instruct"})

def main():
    parser = argparse.ArgumentParser(description='Qwen 2.5 7B Server for JARVIS')
    parser.add_argument('--model_path', type=str, required=True, help='Path to Qwen 2.5 7B model')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8006, help='Port to bind to')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--max_memory', type=str, default='12GB', help='Maximum memory to use')
    
    args = parser.parse_args()
    
    # Initialize model
    global qwen_server
    qwen_server = QwenServer(
        model_path=args.model_path,
        device=args.device,
        max_memory=args.max_memory
    )
    
    logger.info(f"Starting Qwen 2.5 7B server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == '__main__':
    main()
