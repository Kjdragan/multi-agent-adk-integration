Deploy to Cloud Run¶ Cloud Run is a fully managed platform that enables you to
run your code directly on top of Google's scalable infrastructure.

To deploy your agent, you can use either the adk deploy cloud_run command
(recommended for Python), or with gcloud run deploy command through Cloud Run.

Agent sample¶ For each of the commands, we will reference a the Capital Agent
sample defined on the LLM agent page. We will assume it's in a directory (eg:
capital_agent).

To proceed, confirm that your agent code is configured as follows:

Python Java Agent code is in a file called agent.py within your agent directory.
Your agent variable is named root_agent. **init**.py is within your agent
directory and contains from . import agent.

Environment variables¶ Set your environment variables as described in the Setup
and Installation guide.

export GOOGLE_CLOUD_PROJECT=your-project-id export
GOOGLE_CLOUD_LOCATION=us-central1 # Or your preferred location export
GOOGLE_GENAI_USE_VERTEXAI=True (Replace your-project-id with your actual GCP
project ID)

Deployment commands¶

Python - adk CLI Python - gcloud CLI Java - gcloud CLI gcloud CLI¶
Alternatively, you can deploy using the standard gcloud run deploy command with
a Dockerfile. This method requires more manual setup compared to the adk command
but offers flexibility, particularly if you want to embed your agent within a
custom FastAPI application.

Ensure you have authenticated with Google Cloud (gcloud auth login and gcloud
config set project <your-project-id>).

Project Structure¶ Organize your project files as follows:

your-project-directory/ ├── capital_agent/ │ ├── **init**.py │ └── agent.py #
Your agent code (see "Agent sample" tab) ├── main.py # FastAPI application entry
point ├── requirements.txt # Python dependencies └── Dockerfile # Container
build instructions Create the following files (main.py, requirements.txt,
Dockerfile) in the root of your-project-directory/.

Code files¶ This file sets up the FastAPI application using get_fast_api_app()
from ADK:

main.py

import os

import uvicorn from google.adk.cli.fast_api import get_fast_api_app

# Get the directory where main.py is located

AGENT_DIR = os.path.dirname(os.path.abspath(**file**))

# Example session DB URL (e.g., SQLite)

SESSION_DB_URL = "sqlite:///./sessions.db"

# Example allowed origins for CORS

ALLOWED_ORIGINS = ["http://localhost", "http://localhost:8080", "*"]

# Set web=True if you intend to serve a web interface, False otherwise

SERVE_WEB_INTERFACE = True

# Call the function to get the FastAPI app instance

# Ensure the agent directory name ('capital_agent') matches your agent folder

app = get_fast_api_app( agents_dir=AGENT_DIR,
session_service_uri=SESSION_DB_URL, allow_origins=ALLOWED_ORIGINS,
web=SERVE_WEB_INTERFACE, )

# You can add more FastAPI routes or configurations below if needed

# Example:

# @app.get("/hello")

# async def read_root():

# return {"Hello": "World"}

if **name** == "**main**": # Use the PORT environment variable provided by Cloud
Run, defaulting to 8080 uvicorn.run(app, host="0.0.0.0",
port=int(os.environ.get("PORT", 8080))) Note: We specify agent_dir to the
directory main.py is in and use os.environ.get("PORT", 8080) for Cloud Run
compatibility.

List the necessary Python packages:

requirements.txt

google_adk

# Add any other dependencies your agent needs

Define the container image:

Dockerfile

FROM python:3.13-slim WORKDIR /app

COPY requirements.txt . RUN pip install --no-cache-dir -r requirements.txt

RUN adduser --disabled-password --gecos "" myuser &&\
chown -R myuser:myuser /app

COPY . .

USER myuser

ENV PATH="/home/myuser/.local/bin:$PATH"

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"] Defining
Multiple Agents¶ You can define and deploy multiple agents within the same Cloud
Run instance by creating separate folders in the root of
your-project-directory/. Each folder represents one agent and must define a
root_agent in its configuration.

Example structure:

your-project-directory/ ├── capital_agent/ │ ├── **init**.py │ └── agent.py #
contains `root_agent` definition ├── population_agent/ │ ├── **init**.py │ └──
agent.py # contains `root_agent` definition └── ... Deploy using gcloud¶
Navigate to your-project-directory in your terminal.

gcloud run deploy capital-agent-service\
--source .\
--region $GOOGLE_CLOUD_LOCATION\
--project $GOOGLE_CLOUD_PROJECT \
--allow-unauthenticated \
--set-env-vars="GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT,GOOGLE_CLOUD_LOCATION=$GOOGLE_CLOUD_LOCATION,GOOGLE_GENAI_USE_VERTEXAI=$GOOGLE_GENAI_USE_VERTEXAI"

# Add any other necessary environment variables your agent might need

capital-agent-service: The name you want to give your Cloud Run service.
--source .: Tells gcloud to build the container image from the Dockerfile in the
current directory. --region: Specifies the deployment region. --project:
Specifies the GCP project. --allow-unauthenticated: Allows public access to the
service. Remove this flag for private services. --set-env-vars: Passes necessary
environment variables to the running container. Ensure you include all variables
required by ADK and your agent (like API keys if not using Application Default
Credentials). gcloud will build the Docker image, push it to Google Artifact
Registry, and deploy it to Cloud Run. Upon completion, it will output the URL of
your deployed service.

For a full list of deployment options, see the gcloud run deploy reference
documentation.

Testing your agent¶ Once your agent is deployed to Cloud Run, you can interact
with it via the deployed UI (if enabled) or directly with its API endpoints
using tools like curl. You'll need the service URL provided after deployment.

UI Testing API Testing (curl) UI Testing¶ If you deployed your agent with the UI
enabled:

adk CLI: You included the --with_ui flag during deployment. gcloud CLI: You set
SERVE_WEB_INTERFACE = True in your main.py. You can test your agent by simply
navigating to the Cloud Run service URL provided after deployment in your web
browser.

# Example URL format

# https://your-service-name-abc123xyz.a.run.app

The ADK dev UI allows you to interact with your agent, manage sessions, and view
execution details directly in the browser.

To verify your agent is working as intended, you can:

Select your agent from the dropdown menu. Type a message and verify that you
receive an expected response from your agent. If you experience any unexpected
behavior, check the Cloud Run console logs.
