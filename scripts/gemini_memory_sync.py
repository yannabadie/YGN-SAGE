import os
import sys
from google import genai
from google.genai import types

def get_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY environment variable not set.")
        sys.exit(1)
    return genai.Client(api_key=api_key)

def upload_canonical_memory():
    """Uploads the canonical knowledge base to the official Gemini File API."""
    client = get_client()
    file_path = "docs/plans/comprehensive_knowledge_transfer.md"
    
    if not os.path.exists(file_path):
        print(f"ERROR: {file_path} not found.")
        return None
        
    print(f"📡 Uploading canonical memory ({file_path}) to Gemini File API...")
    uploaded_file = client.files.upload(
        file=file_path,
        config={'display_name': 'YGN-SAGE_Canonical_Memory'}
    )
    print(f"✅ Success! File URI: {uploaded_file.uri}")
    print(f"   File Name: {uploaded_file.name}")
    
    # Save the URI reference locally so we can query it later
    with open("docs/plans/.gemini_file_ref", "w") as f:
        f.write(uploaded_file.name)
        
    return uploaded_file.name

def query_canonical_memory(query: str):
    """Queries the uploaded document using Gemini 2.5 Pro."""
    client = get_client()
    
    ref_path = "docs/plans/.gemini_file_ref"
    if not os.path.exists(ref_path):
        print("No canonical memory uploaded yet. Run 'upload' first.")
        return
        
    with open(ref_path, "r") as f:
        file_name = f.read().strip()
        
    print(f"🧠 Querying Official Exocortex (File: {file_name})...")
    print(f"   Question: {query}\n")
    
    response = client.models.generate_content(
        model='gemini-2.5-pro',
        contents=[
            types.Part.from_uri(file_uri=client.files.get(name=file_name).uri, mime_type="text/markdown"),
            query
        ]
    )
    
    print("--- RESPONSE ---")
    print(response.text)
    print("----------------")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gemini_memory_sync.py [upload|query] [query_text]")
        sys.exit(1)
        
    action = sys.argv[1]
    
    if action == "upload":
        upload_canonical_memory()
    elif action == "query":
        if len(sys.argv) < 3:
            print("Please provide a query string.")
        else:
            query_canonical_memory(sys.argv[2])
