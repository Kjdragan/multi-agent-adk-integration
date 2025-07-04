The Session Object¶
When a user starts interacting with your agent, the SessionService creates a Session object (google.adk.sessions.Session). This object acts as the container holding everything related to that one specific chat thread. Here are its key properties:

Identification (id, appName, userId): Unique labels for the conversation.
id: A unique identifier for this specific conversation thread, essential for retrieving it later. A SessionService object can handle multiple Session(s). This field identifies which particular session object are we referring to. For example, "test_id_modification".
app_name: Identifies which agent application this conversation belongs to. For example, "id_modifier_workflow".
userId: Links the conversation to a particular user.
History (events): A chronological sequence of all interactions (Event objects – user messages, agent responses, tool actions) that have occurred within this specific thread.
Session State (state): A place to store temporary data relevant only to this specific, ongoing conversation. This acts as a scratchpad for the agent during the interaction. We will cover how to use and manage state in detail in the next section.
Activity Tracking (lastUpdateTime): A timestamp indicating the last time an event occurred in this conversation thread.
Example: Examining Session Properties¶

Python
Java

 from google.adk.sessions import InMemorySessionService, Session

 # Create a simple session to examine its properties
 temp_service = InMemorySessionService()
 example_session = await temp_service.create_session(
     app_name="my_app",
     user_id="example_user",
     state={"initial_key": "initial_value"} # State can be initialized
 )

 print(f"--- Examining Session Properties ---")
 print(f"ID (`id`):                {example_session.id}")
 print(f"Application Name (`app_name`): {example_session.app_name}")
 print(f"User ID (`user_id`):         {example_session.user_id}")
 print(f"State (`state`):           {example_session.state}") # Note: Only shows initial state here
 print(f"Events (`events`):         {example_session.events}") # Initially empty
 print(f"Last Update (`last_update_time`): {example_session.last_update_time:.2f}")
 print(f"---------------------------------")

 # Clean up (optional for this example)
 temp_service = await temp_service.delete_session(app_name=example_session.app_name,
                             user_id=example_session.user_id, session_id=example_session.id)
 print("The final status of temp_service - ", temp_service)

(Note: The state shown above is only the initial state. State updates happen via events, as discussed in the State section.)

Managing Sessions with a SessionService¶
As seen above, you don't typically create or manage Session objects directly. Instead, you use a SessionService. This service acts as the central manager responsible for the entire lifecycle of your conversation sessions.

Its core responsibilities include:

Starting New Conversations: Creating fresh Session objects when a user begins an interaction.
Resuming Existing Conversations: Retrieving a specific Session (using its ID) so the agent can continue where it left off.
Saving Progress: Appending new interactions (Event objects) to a session's history. This is also the mechanism through which session state gets updated (more in the State section).
Listing Conversations: Finding the active session threads for a particular user and application.
Cleaning Up: Deleting Session objects and their associated data when conversations are finished or no longer needed.
SessionService Implementations¶
ADK provides different SessionService implementations, allowing you to choose the storage backend that best suits your needs:

InMemorySessionService

How it works: Stores all session data directly in the application's memory.
Persistence: None. All conversation data is lost if the application restarts.
Requires: Nothing extra.
Best for: Quick development, local testing, examples, and scenarios where long-term persistence isn't required.

Python
Java

 from google.adk.sessions import InMemorySessionService
 session_service = InMemorySessionService()

VertexAiSessionService

How it works: Uses Google Cloud's Vertex AI infrastructure via API calls for session management.
Persistence: Yes. Data is managed reliably and scalably via Vertex AI Agent Engine.
Requires:
A Google Cloud project (pip install vertexai)
A Google Cloud storage bucket that can be configured by this step.
A Reasoning Engine resource name/ID that can setup following this tutorial.
Best for: Scalable production applications deployed on Google Cloud, especially when integrating with other Vertex AI features.

Python
Java

# Requires: pip install google-adk[vertexai]
# Plus GCP setup and authentication
from google.adk.sessions import VertexAiSessionService

PROJECT_ID = "your-gcp-project-id"
LOCATION = "us-central1"
# The app_name used with this service should be the Reasoning Engine ID or name
REASONING_ENGINE_APP_NAME = "projects/your-gcp-project-id/locations/us-central1/reasoningEngines/your-engine-id"

session_service = VertexAiSessionService(project=PROJECT_ID, location=LOCATION)
# Use REASONING_ENGINE_APP_NAME when calling service methods, e.g.:
# session_service = await session_service.create_session(app_name=REASONING_ENGINE_APP_NAME, ...)

DatabaseSessionService

python_only

How it works: Connects to a relational database (e.g., PostgreSQL, MySQL, SQLite) to store session data persistently in tables.
Persistence: Yes. Data survives application restarts.
Requires: A configured database.
Best for: Applications needing reliable, persistent storage that you manage yourself.

from google.adk.sessions import DatabaseSessionService
# Example using a local SQLite file:
db_url = "sqlite:///./my_agent_data.db"
session_service = DatabaseSessionService(db_url=db_url)
Choosing the right SessionService is key to defining how your agent's conversation history and temporary data are stored and persist.

The Session Lifecycle¶
Session lifecycle

Here’s a simplified flow of how Session and SessionService work together during a conversation turn:

Start or Resume: Your application's Runner uses the SessionService to either create_session (for a new chat) or get_session (to retrieve an existing one).
Context Provided: The Runner gets the appropriate Session object from the appropriate service method, providing the agent with access to the corresponding Session's state and events.
Agent Processing: The user prompts the agent with a query. The agent analyzes the query and potentially the session state and events history to determine the response.
Response & State Update: The agent generates a response (and potentially flags data to be updated in the state). The Runner packages this as an Event.
Save Interaction: The Runner calls sessionService.append_event(session, event) with the session and the new event as the arguments. The service adds the Event to the history and updates the session's state in storage based on information within the event. The session's last_update_time also get updated.
Ready for Next: The agent's response goes to the user. The updated Session is now stored by the SessionService, ready for the next turn (which restarts the cycle at step 1, usually with the continuation of the conversation in the current session).
End Conversation: When the conversation is over, your application calls sessionService.delete_session(...) to clean up the stored session data if it is no longer required.
This cycle highlights how the SessionService ensures conversational continuity by managing the history and state associated with each Session object.

State: The Session's Scratchpad¶
Within each Session (our conversation thread), the state attribute acts like the agent's dedicated scratchpad for that specific interaction. While session.events holds the full history, session.state is where the agent stores and updates dynamic details needed during the conversation.

What is session.state?¶
Conceptually, session.state is a collection (dictionary or Map) holding key-value pairs. It's designed for information the agent needs to recall or track to make the current conversation effective:

Personalize Interaction: Remember user preferences mentioned earlier (e.g., 'user_preference_theme': 'dark').
Track Task Progress: Keep tabs on steps in a multi-turn process (e.g., 'booking_step': 'confirm_payment').
Accumulate Information: Build lists or summaries (e.g., 'shopping_cart_items': ['book', 'pen']).
Make Informed Decisions: Store flags or values influencing the next response (e.g., 'user_is_authenticated': True).
Key Characteristics of State¶
Structure: Serializable Key-Value Pairs

Data is stored as key: value.
Keys: Always strings (str). Use clear names (e.g., 'departure_city', 'user:language_preference').
Values: Must be serializable. This means they can be easily saved and loaded by the SessionService. Stick to basic types in the specific languages (Python/ Java) like strings, numbers, booleans, and simple lists or dictionaries containing only these basic types. (See API documentation for precise details).
⚠️ Avoid Complex Objects: Do not store non-serializable objects (custom class instances, functions, connections, etc.) directly in the state. Store simple identifiers if needed, and retrieve the complex object elsewhere.
Mutability: It Changes

The contents of the state are expected to change as the conversation evolves.
Persistence: Depends on SessionService

Whether state survives application restarts depends on your chosen service:
InMemorySessionService: Not Persistent. State is lost on restart.
DatabaseSessionService / VertexAiSessionService: Persistent. State is saved reliably.
Note

The specific parameters or method names for the primitives may vary slightly by SDK language (e.g., session.state['current_intent'] = 'book_flight' in Python, session.state().put("current_intent", "book_flight) in Java). Refer to the language-specific API documentation for details.

Organizing State with Prefixes: Scope Matters¶
Prefixes on state keys define their scope and persistence behavior, especially with persistent services:

No Prefix (Session State):

Scope: Specific to the current session (id).
Persistence: Only persists if the SessionService is persistent (Database, VertexAI).
Use Cases: Tracking progress within the current task (e.g., 'current_booking_step'), temporary flags for this interaction (e.g., 'needs_clarification').
Example: session.state['current_intent'] = 'book_flight'
user: Prefix (User State):

Scope: Tied to the user_id, shared across all sessions for that user (within the same app_name).
Persistence: Persistent with Database or VertexAI. (Stored by InMemory but lost on restart).
Use Cases: User preferences (e.g., 'user:theme'), profile details (e.g., 'user:name').
Example: session.state['user:preferred_language'] = 'fr'
app: Prefix (App State):

Scope: Tied to the app_name, shared across all users and sessions for that application.
Persistence: Persistent with Database or VertexAI. (Stored by InMemory but lost on restart).
Use Cases: Global settings (e.g., 'app:api_endpoint'), shared templates.
Example: session.state['app:global_discount_code'] = 'SAVE10'
temp: Prefix (Temporary Session State):

Scope: Specific to the current session processing turn.
Persistence: Never Persistent. Guaranteed to be discarded, even with persistent services.
Use Cases: Intermediate results needed only immediately, data you explicitly don't want stored.
Example: session.state['temp:raw_api_response'] = {...}
How the Agent Sees It: Your agent code interacts with the combined state through the single session.state collection (dict/ Map). The SessionService handles fetching/merging state from the correct underlying storage based on prefixes.

How State is Updated: Recommended Methods¶
State should always be updated as part of adding an Event to the session history using session_service.append_event(). This ensures changes are tracked, persistence works correctly, and updates are thread-safe.

1. The Easy Way: output_key (for Agent Text Responses)

This is the simplest method for saving an agent's final text response directly into the state. When defining your LlmAgent, specify the output_key:


Python
Java

from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService, Session
from google.adk.runners import Runner
from google.genai.types import Content, Part

# Define agent with output_key
greeting_agent = LlmAgent(
    name="Greeter",
    model="gemini-2.0-flash", # Use a valid model
    instruction="Generate a short, friendly greeting.",
    output_key="last_greeting" # Save response to state['last_greeting']
)

# --- Setup Runner and Session ---
app_name, user_id, session_id = "state_app", "user1", "session1"
session_service = InMemorySessionService()
runner = Runner(
    agent=greeting_agent,
    app_name=app_name,
    session_service=session_service
)
session = await session_service.create_session(app_name=app_name,
                                    user_id=user_id,
                                    session_id=session_id)
print(f"Initial state: {session.state}")

# --- Run the Agent ---
# Runner handles calling append_event, which uses the output_key
# to automatically create the state_delta.
user_message = Content(parts=[Part(text="Hello")])
for event in runner.run(user_id=user_id,
                        session_id=session_id,
                        new_message=user_message):
    if event.is_final_response():
      print(f"Agent responded.") # Response text is also in event.content

# --- Check Updated State ---
updated_session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
print(f"State after agent run: {updated_session.state}")
# Expected output might include: {'last_greeting': 'Hello there! How can I help you today?'}

Behind the scenes, the Runner uses the output_key to create the necessary EventActions with a state_delta and calls append_event.

2. The Standard Way: EventActions.state_delta (for Complex Updates)

For more complex scenarios (updating multiple keys, non-string values, specific scopes like user: or app:, or updates not tied directly to the agent's final text), you manually construct the state_delta within EventActions.


Python
Java

from google.adk.sessions import InMemorySessionService, Session
from google.adk.events import Event, EventActions
from google.genai.types import Part, Content
import time

# --- Setup ---
session_service = InMemorySessionService()
app_name, user_id, session_id = "state_app_manual", "user2", "session2"
session = await session_service.create_session(
    app_name=app_name,
    user_id=user_id,
    session_id=session_id,
    state={"user:login_count": 0, "task_status": "idle"}
)
print(f"Initial state: {session.state}")

# --- Define State Changes ---
current_time = time.time()
state_changes = {
    "task_status": "active",              # Update session state
    "user:login_count": session.state.get("user:login_count", 0) + 1, # Update user state
    "user:last_login_ts": current_time,   # Add user state
    "temp:validation_needed": True        # Add temporary state (will be discarded)
}

# --- Create Event with Actions ---
actions_with_update = EventActions(state_delta=state_changes)
# This event might represent an internal system action, not just an agent response
system_event = Event(
    invocation_id="inv_login_update",
    author="system", # Or 'agent', 'tool' etc.
    actions=actions_with_update,
    timestamp=current_time
    # content might be None or represent the action taken
)

# --- Append the Event (This updates the state) ---
await session_service.append_event(session, system_event)
print("`append_event` called with explicit state delta.")

# --- Check Updated State ---
updated_session = await session_service.get_session(app_name=app_name,
                                            user_id=user_id,
                                            session_id=session_id)
print(f"State after event: {updated_session.state}")
# Expected: {'user:login_count': 1, 'task_status': 'active', 'user:last_login_ts': <timestamp>}
# Note: 'temp:validation_needed' is NOT present.

3. Via CallbackContext or ToolContext (Recommended for Callbacks and Tools)

Modifying state within agent callbacks (e.g., on_before_agent_call, on_after_agent_call) or tool functions is best done using the state attribute of the CallbackContext or ToolContext provided to your function.

callback_context.state['my_key'] = my_value
tool_context.state['my_key'] = my_value
These context objects are specifically designed to manage state changes within their respective execution scopes. When you modify context.state, the ADK framework ensures that these changes are automatically captured and correctly routed into the EventActions.state_delta for the event being generated by the callback or tool. This delta is then processed by the SessionService when the event is appended, ensuring proper persistence and tracking.

This method abstracts away the manual creation of EventActions and state_delta for most common state update scenarios within callbacks and tools, making your code cleaner and less error-prone.

For more comprehensive details on context objects, refer to the Context documentation.


Python
Java

# In an agent callback or tool function
from google.adk.agents import CallbackContext # or ToolContext

def my_callback_or_tool_function(context: CallbackContext, # Or ToolContext
                                 # ... other parameters ...
                                ):
    # Update existing state
    count = context.state.get("user_action_count", 0)
    context.state["user_action_count"] = count + 1

    # Add new state
    context.state["temp:last_operation_status"] = "success"

    # State changes are automatically part of the event's state_delta
    # ... rest of callback/tool logic ...

What append_event Does:

Adds the Event to session.events.
Reads the state_delta from the event's actions.
Applies these changes to the state managed by the SessionService, correctly handling prefixes and persistence based on the service type.
Updates the session's last_update_time.
Ensures thread-safety for concurrent updates.
⚠️ A Warning About Direct State Modification¶
Avoid directly modifying the session.state collection (dictionary/Map) on a Session object that was obtained directly from the SessionService (e.g., via session_service.get_session() or session_service.create_session()) outside of the managed lifecycle of an agent invocation (i.e., not through a CallbackContext or ToolContext). For example, code like retrieved_session = await session_service.get_session(...); retrieved_session.state['key'] = value is problematic.

State modifications within callbacks or tools using CallbackContext.state or ToolContext.state are the correct way to ensure changes are tracked, as these context objects handle the necessary integration with the event system.

Why direct modification (outside of contexts) is strongly discouraged:

Bypasses Event History: The change isn't recorded as an Event, losing auditability.
Breaks Persistence: Changes made this way will likely NOT be saved by DatabaseSessionService or VertexAiSessionService. They rely on append_event to trigger saving.
Not Thread-Safe: Can lead to race conditions and lost updates.
Ignores Timestamps/Logic: Doesn't update last_update_time or trigger related event logic.
Recommendation: Stick to updating state via output_key, EventActions.state_delta (when manually creating events), or by modifying the state property of CallbackContext or ToolContext objects when within their respective scopes. These methods ensure reliable, trackable, and persistent state management. Use direct access to session.state (from a SessionService-retrieved session) only for reading state.

Best Practices for State Design Recap¶
Minimalism: Store only essential, dynamic data.
Serialization: Use basic, serializable types.
Descriptive Keys & Prefixes: Use clear names and appropriate prefixes (user:, app:, temp:, or none).
Shallow Structures: Avoid deep nesting where possible.
Standard Update Flow: Rely on append_event.




Memory: Long-Term Knowledge with MemoryService¶
python_only

We've seen how Session tracks the history (events) and temporary data (state) for a single, ongoing conversation. But what if an agent needs to recall information from past conversations or access external knowledge bases? This is where the concept of Long-Term Knowledge and the MemoryService come into play.

Think of it this way:

Session / State: Like your short-term memory during one specific chat.
Long-Term Knowledge (MemoryService): Like a searchable archive or knowledge library the agent can consult, potentially containing information from many past chats or other sources.
The MemoryService Role¶
The BaseMemoryService defines the interface for managing this searchable, long-term knowledge store. Its primary responsibilities are:

Ingesting Information (add_session_to_memory): Taking the contents of a (usually completed) Session and adding relevant information to the long-term knowledge store.
Searching Information (search_memory): Allowing an agent (typically via a Tool) to query the knowledge store and retrieve relevant snippets or context based on a search query.
MemoryService Implementations¶
ADK provides different ways to implement this long-term knowledge store:

InMemoryMemoryService

How it works: Stores session information in the application's memory and performs basic keyword matching for searches.
Persistence: None. All stored knowledge is lost if the application restarts.
Requires: Nothing extra.
Best for: Prototyping, simple testing, scenarios where only basic keyword recall is needed and persistence isn't required.

from google.adk.memory import InMemoryMemoryService
memory_service = InMemoryMemoryService()
VertexAiRagMemoryService

How it works: Leverages Google Cloud's Vertex AI RAG (Retrieval-Augmented Generation) service. It ingests session data into a specified RAG Corpus and uses powerful semantic search capabilities for retrieval.
Persistence: Yes. The knowledge is stored persistently within the configured Vertex AI RAG Corpus.
Requires: A Google Cloud project, appropriate permissions, necessary SDKs (pip install google-adk[vertexai]), and a pre-configured Vertex AI RAG Corpus resource name/ID.
Best for: Production applications needing scalable, persistent, and semantically relevant knowledge retrieval, especially when deployed on Google Cloud.

# Requires: pip install google-adk[vertexai]
# Plus GCP setup, RAG Corpus, and authentication
from google.adk.memory import VertexAiRagMemoryService

# The RAG Corpus name or ID
RAG_CORPUS_RESOURCE_NAME = "projects/your-gcp-project-id/locations/us-central1/ragCorpora/your-corpus-id"
# Optional configuration for retrieval
SIMILARITY_TOP_K = 5
VECTOR_DISTANCE_THRESHOLD = 0.7

memory_service = VertexAiRagMemoryService(
    rag_corpus=RAG_CORPUS_RESOURCE_NAME,
    similarity_top_k=SIMILARITY_TOP_K,
    vector_distance_threshold=VECTOR_DISTANCE_THRESHOLD
)
How Memory Works in Practice¶
The typical workflow involves these steps:

Session Interaction: A user interacts with an agent via a Session, managed by a SessionService. Events are added, and state might be updated.
Ingestion into Memory: At some point (often when a session is considered complete or has yielded significant information), your application calls memory_service.add_session_to_memory(session). This extracts relevant information from the session's events and adds it to the long-term knowledge store (in-memory dictionary or RAG Corpus).
Later Query: In a different (or the same) session, the user might ask a question requiring past context (e.g., "What did we discuss about project X last week?").
Agent Uses Memory Tool: An agent equipped with a memory-retrieval tool (like the built-in load_memory tool) recognizes the need for past context. It calls the tool, providing a search query (e.g., "discussion project X last week").
Search Execution: The tool internally calls memory_service.search_memory(app_name, user_id, query).
Results Returned: The MemoryService searches its store (using keyword matching or semantic search) and returns relevant snippets as a SearchMemoryResponse containing a list of MemoryResult objects (each potentially holding events from a relevant past session).
Agent Uses Results: The tool returns these results to the agent, usually as part of the context or function response. The agent can then use this retrieved information to formulate its final answer to the user.
Example: Adding and Searching Memory¶
This example demonstrates the basic flow using the InMemory services for simplicity.

Full Code

import asyncio
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService, Session
from google.adk.memory import InMemoryMemoryService # Import MemoryService
from google.adk.runners import Runner
from google.adk.tools import load_memory # Tool to query memory
from google.genai.types import Content, Part

# --- Constants ---
APP_NAME = "memory_example_app"
USER_ID = "mem_user"
MODEL = "gemini-2.0-flash" # Use a valid model

# --- Agent Definitions ---
# Agent 1: Simple agent to capture information
info_capture_agent = LlmAgent(
    model=MODEL,
    name="InfoCaptureAgent",
    instruction="Acknowledge the user's statement.",
    # output_key="captured_info" # Could optionally save to state too
)

# Agent 2: Agent that can use memory
memory_recall_agent = LlmAgent(
    model=MODEL,
    name="MemoryRecallAgent",
    instruction="Answer the user's question. Use the 'load_memory' tool "
                "if the answer might be in past conversations.",
    tools=[load_memory] # Give the agent the tool
)

# --- Services and Runner ---
session_service = InMemorySessionService()
memory_service = InMemoryMemoryService() # Use in-memory for demo

runner = Runner(
    # Start with the info capture agent
    agent=info_capture_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service # Provide the memory service to the Runner
)

# --- Scenario ---

# Turn 1: Capture some information in a session
print("--- Turn 1: Capturing Information ---")
session1_id = "session_info"
session1 = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session1_id)
user_input1 = Content(parts=[Part(text="My favorite project is Project Alpha.")], role="user")

# Run the agent
final_response_text = "(No final response)"
async for event in runner.run_async(user_id=USER_ID, session_id=session1_id, new_message=user_input1):
    if event.is_final_response() and event.content and event.content.parts:
        final_response_text = event.content.parts[0].text
print(f"Agent 1 Response: {final_response_text}")

# Get the completed session
completed_session1 = await runner.session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session1_id)

# Add this session's content to the Memory Service
print("\n--- Adding Session 1 to Memory ---")
memory_service = await memory_service.add_session_to_memory(completed_session1)
print("Session added to memory.")

# Turn 2: In a *new* (or same) session, ask a question requiring memory
print("\n--- Turn 2: Recalling Information ---")
session2_id = "session_recall" # Can be same or different session ID
session2 = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session2_id)

# Switch runner to the recall agent
runner.agent = memory_recall_agent
user_input2 = Content(parts=[Part(text="What is my favorite project?")], role="user")

# Run the recall agent
print("Running MemoryRecallAgent...")
final_response_text_2 = "(No final response)"
async for event in runner.run_async(user_id=USER_ID, session_id=session2_id, new_message=user_input2):
    print(f"  Event: {event.author} - Type: {'Text' if event.content and event.content.parts and event.content.parts[0].text else ''}"
        f"{'FuncCall' if event.get_function_calls() else ''}"
        f"{'FuncResp' if event.get_function_responses() else ''}")
    if event.is_final_response() and event.content and event.content.parts:
        final_response_text_2 = event.content.parts[0].text
        print(f"Agent 2 Final Response: {final_response_text_2}")
        break # Stop after final response

# Expected Event Sequence for Turn 2:
# 1. User sends "What is my favorite project?"
# 2. Agent (LLM) decides to call `load_memory` tool with a query like "favorite project".
# 3. Runner executes the `load_memory` tool, which calls `memory_service.search_memory`.
# 4. `InMemoryMemoryService` finds the relevant text ("My favorite project is Project Alpha.") from session1.
# 5. Tool returns this text in a FunctionResponse event.
# 6. Agent (LLM) receives the function response, processes the retrieved text.
# 7. Agent generates the final answer (e.g., "Your favorite project is Project Alpha.").
