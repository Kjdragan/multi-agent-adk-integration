import asyncio
import os
import dotenv

# Import ADK and Gemini components
from google.genai import types
from google.adk import agents
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
import google.generativeai as genai

# Load environment variables from the .env file
dotenv.load_dotenv()

# Configure the Gemini client with your API key
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)


def get_policy_information(policy: str) -> dict:
    """A tool to find information about a specific company policy."""
    print(f"Tool Call - get_policy_information called for policy: {policy}")

    # Mock database of company policies
    policy_db = {
        "office floor policy": "Employees are expected to maintain a clean and organized workspace. This includes keeping the floor free of clutter, spills, and debris. Regular cleaning and tidying up are mandatory.",
        "email etiquette policy": "Emails should be professional and respectful. Avoid using all caps, excessive punctuation, or informal language. Always sign off with a proper greeting and include your full name.",
        "data security policy": "All sensitive data must be protected. This includes using strong passwords, encrypting data, and reporting any security breaches immediately. Access to sensitive information is restricted to authorized personnel only.",
        "remote work policy": "Remote work is allowed for approved positions. Employees must have a dedicated workspace, use company-issued equipment, and adhere to all company policies while working remotely.",
        "confidentiality policy": "Employees must keep all company information confidential. This includes non-disclosure agreements (NDAs) and the protection of proprietary information. Sharing sensitive information with unauthorized parties is strictly prohibited.",
        "leave policy": "Employees are entitled to a certain number of paid and unpaid days per year. Sick leave, vacation leave, and other types of leave are subject to company policies and must be requested in advance.",
        "work hour policy": "The standard work hours are from 9:00 AM to 5:00 PM, Monday through Friday. Overtime is subject to approval and must be documented, and employees are expected to be punctual and available during their scheduled work hours."
    }

    if policy in policy_db:
        return policy_db[policy]
    else:
        return {"status": "error", "error_message": f"Sorry, I don't have information for '{policy}'."}

# Initialize the in-memory session service
session_service = InMemorySessionService()

# Define constants for identifying the interaction context
APP_NAME = "employee-policy-app"
USER_ID = "user-1"
SESSION_ID = "session-1"
AGENT_MODEL = "gemini-1.5-flash-latest"

# Initiate the Agent
policy_agent = agents.Agent(
    name="policy_agent_v1",
    model=AGENT_MODEL,
    description="Provides information regarding a specific policy.",
    instruction="""You are a helpful policy assistant.
Your primary goal is to provide information about company policies and procedures.
When the user asks about a specific policy or procedure,
you MUST use the 'get_policy_information' tool to find the information.
Analyze the tool's response: if the status is 'error', inform the user politely about the error message.
If the status is 'success', present the policy 'details' clearly and concisely to the user.
Only use the tool when a policy or procedure is mentioned in the user's request.
""",
    tools=[get_policy_information],
)

# Create the specific session where the conversation will happen
session = session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID
)
print(f"Session created: App='{APP_NAME}', User='{USER_ID}', Session='{SESSION_ID}'")

# Runner orchestrates the agent execution loop.
runner = Runner(
    agent=policy_agent,
    app_name=APP_NAME,
    session_service=session_service
)
print(f"Runner created for agent '{runner.agent.name}'.")


async def call_agent_async(query: str):
    """Helper function to call the agent and handle streaming responses."""
    print(f"\nUser Query - {query}")

    content = types.Content(role="user", parts=[types.Part(text=query)])

    final_response_text = "Agent did not produce a final response."

    async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
        elif event.actions and event.actions.escalate:
            final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
            break

    print(f"Agent Response - {final_response_text}")


async def run_conversation():
    """Runs a series of queries against the agent."""
    await call_agent_async("What is the remote work policy?")
    await call_agent_async("Who will win between thor vs wolverine?")
    await call_agent_async("what is the work hour policy?")


if __name__ == "__main__":
    try:
        asyncio.run(run_conversation())
    except KeyboardInterrupt:
        print("\nConversation ended.")
