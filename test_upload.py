#!/usr/bin/env python3
"""Quick test to verify file upload and preview flow"""
import requests
import json

BASE_URL = "http://127.0.0.1:8000/api"

# Test upload
print("Testing file upload...")
with open("/tmp/test.txt", "w") as f:
    f.write("Test content")

with open("/tmp/test.txt", "rb") as f:
    files = {"file": ("test.txt", f, "text/plain")}
    data = {}
    response = requests.post(f"{BASE_URL}/documents/ingest", files=files, data=data)
    print(f"Upload response: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 200:
        result = response.json()
        notebook_id = result.get("notebook_id")
        print(f"\nNotebook ID: {notebook_id}")
        
        # Test list
        print("\nTesting document list...")
        list_response = requests.get(f"{BASE_URL}/documents/list", params={"notebook_id": notebook_id})
        print(f"List response: {list_response.status_code}")
        print(f"Response: {json.dumps(list_response.json(), indent=2)}")
        
        if list_response.status_code == 200:
            docs = list_response.json().get("documents", [])
            if docs:
                doc = docs[0]
                source_path = doc.get("source_path")
                print(f"\nSource path from metadata: {source_path}")
                
                # Test preview
                print("\nTesting preview...")
                preview_response = requests.get(
                    f"{BASE_URL}/documents/preview",
                    params={"notebook_id": notebook_id, "source_path": source_path}
                )
                print(f"Preview response: {preview_response.status_code}")
                if preview_response.status_code != 200:
                    print(f"Error: {preview_response.text}")

