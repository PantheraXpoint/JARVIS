#!/usr/bin/env python3
"""
Script to update all_examples.json from the remote API

This script:
1. Queries the API endpoint to get examples for task pipelines
2. Saves the response to demos/all_examples.json
3. Validates the response format

Usage:
    python3 update_all_examples.py
"""

import json
import requests
import sys
from pathlib import Path

# Configuration
API_URL = "http://143.248.55.143:8081/query/get_examples_for_task_pipelines"
API_PAYLOAD = {
    "task_id": "",
    "scenario_id": "",
    "num_examples": 100
}
OUTPUT_FILE = "../demos/all_examples.json"


def update_all_examples():
    """Fetch examples from API and save to file"""
    print("=" * 80)
    print("UPDATING ALL_EXAMPLES.JSON")
    print("=" * 80)
    print()
    
    print(f"Fetching examples from API...")
    print(f"  URL: {API_URL}")
    print(f"  Payload: {API_PAYLOAD}")
    print()
    
    try:
        # Make API request
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            json=API_PAYLOAD,
            timeout=60
        )
        
        # Check for HTTP errors
        response.raise_for_status()
        
        # Parse JSON response
        try:
            data = response.json()
        except json.JSONDecodeError as e:
            print(f"✗ ERROR: Failed to parse JSON response")
            print(f"  Response text: {response.text[:500]}")
            sys.exit(1)
        
        # Validate response is a list
        if not isinstance(data, list):
            print(f"✗ ERROR: Expected list response, got {type(data)}")
            print(f"  Response: {data}")
            sys.exit(1)
        
        print(f"✓ Successfully fetched {len(data)} examples")
        
        # Create output directory if it doesn't exist
        output_path = Path(OUTPUT_FILE)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to file with pretty formatting
        print(f"\nSaving to {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Successfully saved {len(data)} examples to {OUTPUT_FILE}")
        print()
        print("=" * 80)
        print("UPDATE COMPLETE")
        print("=" * 80)
        
    except requests.exceptions.ConnectionError:
        print(f"✗ ERROR: Could not connect to API server")
        print(f"  Please check if the server is accessible at {API_URL}")
        sys.exit(1)
    
    except requests.exceptions.Timeout:
        print(f"✗ ERROR: Request timed out")
        print(f"  The server took too long to respond")
        sys.exit(1)
    
    except requests.exceptions.HTTPError as e:
        print(f"✗ ERROR: HTTP {e.response.status_code}")
        print(f"  Response: {e.response.text[:500]}")
        sys.exit(1)
    
    except Exception as e:
        print(f"✗ ERROR: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    update_all_examples()

