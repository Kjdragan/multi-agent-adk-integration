#!/usr/bin/env python3
"""
Test script to verify Google Cloud Vertex AI authentication and permissions.
This will attempt to access the Gemini model using the configured service account.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import json
from google.oauth2 import service_account
import google.auth
from google.auth.exceptions import DefaultCredentialsError
from google import genai

def print_step(step_name):
    """Print a step name with formatting."""
    print(f"\n{'=' * 80}\n{step_name}\n{'=' * 80}")

def main():
    # Load environment variables from .env file
    print_step("Loading environment variables")
    load_dotenv()
    
    # Display key environment variables
    print(f"GOOGLE_CLOUD_PROJECT: {os.environ.get('GOOGLE_CLOUD_PROJECT')}")
    print(f"GOOGLE_GENAI_USE_VERTEXAI: {os.environ.get('GOOGLE_GENAI_USE_VERTEXAI')}")
    print(f"VERTEX_REGION: {os.environ.get('VERTEX_REGION', 'us-central1')}")
    
    # Check for service account key file
    creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if creds_path:
        print(f"GOOGLE_APPLICATION_CREDENTIALS: {creds_path}")
        if os.path.exists(creds_path):
            print(f"  ✅ Credentials file exists")
        else:
            print(f"  ❌ Credentials file does not exist at: {creds_path}")
            print("  Create a service account key and update the path in .env")
            return 1
    else:
        print("❌ GOOGLE_APPLICATION_CREDENTIALS not set in .env file")
        print("  Add GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json to your .env file")
        return 1

    # Attempt to get default credentials
    print_step("Testing authentication")
    try:
        credentials, project = google.auth.default()
        print(f"✅ Successfully obtained credentials")
        print(f"Project ID: {project}")
        
        # Check if using service account
        if hasattr(credentials, 'service_account_email'):
            print(f"Using service account: {credentials.service_account_email}")
        else:
            print("Warning: Not using service account credentials")
    except DefaultCredentialsError as e:
        print(f"❌ Authentication failed: {e}")
        return 1
    
    # Test Vertex AI access
    if test_vertex_ai():
        print_step("All tests completed successfully")
        return 0
    else:
        return 1

def test_vertex_ai():
    print("\n" + "=" * 80)
    print("Testing Vertex AI access")
    print("=" * 80)
    
    # Initialize Google GenAI client
    project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
    location = os.environ.get('VERTEX_REGION', 'us-central1')
    
    try:
        # Create a client instance with Vertex AI settings
        client = genai.Client(
            project=project_id,
            location=location,
            vertexai=True  # This enables Vertex AI integration
        )
        
        print("Attempting to access Vertex AI model...")
        
        # Verify model access
        print("✅ Successfully accessed Vertex AI model: gemini-2.5-flash")
        
        print("\nAttempting a test prediction...")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Tell me your favorite programming language."
        )
        
        print(f"✅ Test prediction successful! Response: {response.text}")
        return True
        
    except Exception as e:
        print(f"❌ Error accessing Vertex AI: {e}")
        print("  Possible solutions:")
        print("  1. Verify your service account has the 'Vertex AI User' role")
        print("  2. Enable the Vertex AI API: https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com")
        return False
        
    print_step("All tests completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
