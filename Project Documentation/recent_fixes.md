# GCP Vertex AI Authentication and Access Issue Resolution

## Problem Overview

The multi-agent research platform was encountering a `403 PERMISSION_DENIED` error when attempting to access Google Cloud Vertex AI models, specifically when trying to use the `gemini-2.5-flash` model. The exact error indicated missing permission for `aiplatform.endpoints.predict` on the specified Vertex AI resource.

## Environment Context

- **Platform**: Multi-agent research platform running on WSL (Windows Subsystem for Linux)
- **Python Environment**: Managed with UV package manager (Python 3.13.4)
- **Project ID**: `cs-poc-czxf7xbmmrua9yw8mrrkrn0`
- **Key Dependencies**: google-adk, google-auth, google-genai, google-cloud-aiplatform

## Issues Identified

1. **Authentication Method**: WSL has known issues with browser-based OAuth flows, making standard Application Default Credentials (ADC) problematic
2. **Service Account Key Creation**: Organization policy constraints were blocking service account key creation
3. **Missing Environment Variables**: `.env` file lacked necessary configuration for service account authentication
4. **Python Environment Setup**: No virtual environment was set up with the required dependencies
5. **IAM Permissions**: The service account lacked necessary permissions to access Vertex AI
6. **GCloud CLI Issues**: Command to add IAM policy binding was hanging in WSL

## Resolution Process

### 1. Authentication Strategy

Initially explored two authentication approaches:
- **Application Default Credentials (ADC)**: Standard approach but problematic in WSL due to browser authentication challenges
- **Service Account Authentication**: More reliable in WSL but required creating service account keys

Ultimately chose service account authentication due to its reliability in WSL environments.

### 2. Organization Policy Constraints

Discovered both legacy and managed organization policy constraints blocking service account key creation:
- Legacy constraint: `iam.disableServiceAccountKeyCreation`
- Managed constraint: `iam.managed.disableServiceAccountKeyCreation`

Resolution:
- Set both constraints to "Not enforced" in the Google Cloud Console
- This required organization admin access to modify the policies
- Changes were made through the IAM & Admin > Organization Policies section

### 3. Service Account Setup

Created a dedicated service account for the multi-agent research platform:
- Service account name: `multi-agent-research`
- Email: `multi-agent-research@cs-poc-czxf7xbmmrua9yw8mrrkrn0.iam.gserviceaccount.com`
- Roles assigned: `roles/aiplatform.user`
- Created a service account key in JSON format

### 4. Environment Configuration

Updated the `.env` file with required configuration:
- Added `GOOGLE_APPLICATION_CREDENTIALS` pointing to the service account key JSON file
- Explicitly set `VERTEX_REGION=us-central1`
- Ensured `GOOGLE_GENAI_USE_VERTEXAI=True` was enabled
- Set `GOOGLE_CLOUD_PROJECT` to the correct project ID

### 5. Python Environment Setup

Recreated the Python virtual environment:
- Used UV as the package manager: `uv venv`
- Activated environment: `source .venv/bin/activate`
- Installed dependencies: `uv pip install -e .`

### 6. Testing and Verification

Created and implemented a test script (`test_vertex_auth.py`) to verify:
- Proper loading of environment variables
- Existence of service account key file
- Successful authentication with Google Cloud
- Access to the Vertex AI API
- Ability to make predictions using the `gemini-2.5-flash` model

Initially used `text-bison@001` for testing, but found that model wasn't accessible to the project. Updated the test to use `gemini-2.5-flash`, which was the originally intended model.

## Current Status

✅ **Full Resolution Achieved**:
- Service account authentication is working properly
- Environment variables are correctly configured
- Python environment is set up with all dependencies
- Service account has proper IAM permissions
- Successfully accessing and using the `gemini-2.5-flash` model

The test prediction returned: "My favorite programming language is 'it works on my machine'."

## Remaining Considerations

1. **Security Best Practices**:
   - The service account key should be protected and never committed to version control
   - Consider key rotation policies for ongoing security
   - Apply principle of least privilege for service account permissions

2. **Workflow Improvements**:
   - Consider documenting this setup in project README for new developers
   - The `gcloud projects add-iam-policy-binding` command had issues in WSL - using the Cloud Console for IAM changes is more reliable

3. **SDK Migration**

- Successfully migrated from the deprecated Vertex AI SDK (`vertexai.generative_models`) to the newer Google Gen AI SDK (`google-genai`)
- Updated the test script to use the Google Gen AI client initialization pattern
- Confirmed that service account authentication works correctly with the new SDK
- This migration ensures compatibility beyond the June 24, 2026 deprecation date

### Migration Changes:

1. Changed imports from `from vertexai.generative_models import GenerativeModel` to `from google import genai`
2. Replaced Vertex AI model initialization:
   ```python
   # Old (Vertex AI SDK)
   vertexai.init(project=project_id, location=location)
   model = GenerativeModel("gemini-2.5-flash")
   
   # New (Google Gen AI SDK)
   client = genai.Client(
       project=project_id,
       location=location,
       vertexai=True
   )
   ```
3. Updated prediction method:
   ```python
   # Old (Vertex AI SDK)
   response = model.generate_content("prompt text")
   
   # New (Google Gen AI SDK)
   response = client.models.generate_content(
       model="gemini-2.5-flash",
       contents="prompt text"
   )
   ```

## Technical Details

### Key Files Modified

1. **`.env`**:
   - Added service account credentials path
   - Set Vertex AI region and other environment variables

2. **`test_vertex_auth.py`**:
   - Created to verify authentication and API access
   - Tests environment configuration, credentials, and model access
   - Makes a sample prediction to confirm end-to-end functionality

### Command Sequence for New Environments

For future reference, the full setup sequence is:
```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Test authentication
python test_vertex_auth.py
```
