Generative AI on Vertex AI

Release Notes
The Generative AI module in the Vertex AI SDK is deprecated and will no longer be available after June 24, 2026. The Google Gen AI SDK contains all the capabilities of the Vertex AI SDK, and supports many additional capabilities.

Use this migration guide to convert Python, Java, JavaScript, and Go code using the Vertex AI SDK to the Google Gen AI SDK.

Key changes
The following namespaces in the Vertex AI SDK are in the deprecation phase. SDK releases after June 24, 2026 won't include theses modules. Use the equivalent namespaces from the Google Gen AI SDK, which has full feature parity with the deprecated modules and packages.

Vertex AI SDK	Impacted code	Google Gen AI SDK replacement
google-cloud-aiplatform	Removed modules:
vertexai.generative_models
vertexai.language_models
vertexai.vision_models
vertexai.caching
vertexai.tuning
google-genai
cloud.google.com/go/vertexai/genai	Removed package:
vertex.genai
google.golang.org/genai
@google-cloud/vertexai	Removed modules:
vertexai.generative_models
vertexai.chat_session
vertexai.functions
@google/genai
com.google.cloud:google-cloud-vertexai	Removed package:
com.google.cloud.vertexai.generativeai
com.google.genai:google-genai
Code migration
Use the following sections to migrate specific code snippets from the Vertex AI SDK to the Google Gen AI SDK.

Note: The examples may omit imports, dependencies, and other boilerplate code to improve readability.
Installation
Replace the Vertex AI SDK dependency with the Google Gen AI SDK dependency.

Before

Python
Java
JavaScript
Go


pip install -U -q "google-cloud-aiplatform"
After

Python
Java
JavaScript
Go


pip install -U -q "google-genai"
Context caching
Context caching involves storing and reusing frequently used portions of model prompts for similar requests. Replace the Vertex AI SDK implementation with the Google Gen AI SDK dependency.

Before

Python
Java
JavaScript
Go
Imports



from google.cloud import aiplatform
import vertexai
import datetime
Create



vertexai.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_LOCATION)

cache_content = vertexai.caching.CachedContent.create(
  model_name=MODEL_NAME,
  system_instruction='Please answer my question formally',
  contents=['user content'],
  ttl=datetime.timedelta(days=1),
)
Get



vertexai.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_LOCATION)
cache_content = vertexai.caching.CachedContent.get(cached_content_name="projects/{project}/locations/{location}/cachedContents/{cached_content}")
Delete



cache_content.delete()
Update



cache_content.update(ttl=datetime.timedelta(days=2))
List



cache_contents = vertexai.caching.CachedContent.list()
After

Python
Java
JavaScript
Go
Imports



from google import genai
from google.genai.types import Content, CreateCachedContentConfig, HttpOptions, Part
Create



client = genai.Client(http_options=HttpOptions(api_version="v1"))

content_cache = client.caches.create(
    model="gemini-2.5-flash",
    config=CreateCachedContentConfig(
        contents=contents,
        system_instruction=system_instruction,
        display_name="example-cache",
        ttl="86400s",
    ),
)
Get



content_cache_list = client.caches.list()

# Access individual properties of a ContentCache object(s)
for content_cache in content_cache_list:
    print(f"Cache `{content_cache.name}` for model `{content_cache.model}`")
    print(f"Last updated at: {content_cache.update_time}")
    print(f"Expires at: {content_cache.expire_time}")
Delete



client.caches.delete(name=cache_name)
Update



content_cache = client.caches.update(
    name=cache_name, config=UpdateCachedContentConfig(ttl="36000s")
)
List



cache_contents = client.caches.list(config={'page_size': 2})
Configuration and system instructions
Configuration defines parameters that control model behavior, and system instructions provide guiding directives to steer model responses towards a specific persona, style, or task. Replace the configuration and system instructions from the Vertex AI SDK with the following code that uses the Google Gen AI SDK.

Before

Python
Java
JavaScript
Go


model = generative_models.GenerativeModel(
  GEMINI_MODEL_NAME,
  system_instruction=[
    "Talk like a pirate.",
    "Don't use rude words.",
  ],
)
response = model.generate_content(
  contents="Why is sky blue?",
  generation_config=generative_models.GenerationConfig(
    temperature=0,
    top_p=0.95,
    top_k=20,
    candidate_count=1,
    max_output_tokens=100,
    stop_sequences=["STOP!"],
    response_logprobs=True,
    logprobs=3,
  ),
  safety_settings={
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
  },
)
After

Python
Java
JavaScript
Go


from google.genai import types

response = client.models.generate_content(
  model='gemini-2.5-flash',
  contents='high',
  config=types.GenerateContentConfig(
    system_instruction='I say high, you say low',
    max_output_tokens=3,
    temperature=0.3,
    response_logprobs=True,
    logprobs=3,
  ),
)
Embeddings
Embeddings are numerical vector representations of text, images, or video that capture their semantic or visual meaning and relationships in a high-dimensional space. Replace the embedding implementation from the Vertex AI SDK with the following code that uses the Google Gen AI SDK.

Before

Python
Java
JavaScript
Go


model = vertexai.vision_models.MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
image = Image.load_from_file("image.png")
video = Video.load_from_file("video.mp4")

embeddings = model.get_embeddings(
  # One of image, video or contextual_text is required.
  image=image,
  video=video,
  contextual_text="Hello world",
)
image_embedding = embeddings.image_embedding
video_embeddings = embeddings.video_embeddings
text_embedding = embeddings.text_embedding
After

Python
Java
JavaScript
Go


from google.genai.types import EmbedContentConfig

client = genai.Client()
response = client.models.embed_content(
  model="gemini-embedding-001",
  contents="How do I get a driver's license/learner's permit?",
  config=EmbedContentConfig(
    task_type="RETRIEVAL_DOCUMENT",  # Optional
    output_dimensionality=3072,  # Optional
    title="Driver's License",  # Optional
  ),
)
Function calling
Function calling enables a model to identify when to invoke an external tool or API and then generate structured data containing the necessary function and arguments for execution. Replace the function calling implementation with the Vertex AI SDK with the following code that uses the Google Gen AI SDK.

Before

Python
Java
JavaScript
Go


get_current_weather_func = generative_models.FunctionDeclaration(
  name="get_current_weather",
  description="Get the current weather in a given location",
  parameters=_REQUEST_FUNCTION_PARAMETER_SCHEMA_STRUCT,
)

weather_tool = generative_models.Tool(
  function_declarations=[get_current_weather_func],
)

model = generative_models.GenerativeModel(
  GEMINI_MODEL_NAME,
  tools=[weather_tool],
)

chat = model.start_chat()

response1 = chat.send_message("What is the weather like in Boston?")
assert (
  response1.candidates[0].content.parts[0].function_call.name
  == "get_current_weather"
)
response2 = chat.send_message(
  generative_models.Part.from_function_response(
    name="get_current_weather",
    response={
      "content": {"weather": "super nice"},
    },
  ),
)
assert response2.text
After

Python
Java
JavaScript
Go


from google.genai import types

def get_current_weather(location: str) -> str:
  """Returns the current weather.

  Args:
    location: The city and state, e.g. San Francisco, CA
  """
  return 'sunny'

response = client.models.generate_content(
  model='gemini-2.5-flash',
  contents='What is the weather like in Boston?',
  config=types.GenerateContentConfig(tools=[get_current_weather]),
)
Grounding
Grounding is the process of providing a model with external, domain-specific information to improve response accuracy, relevance, and consistency. Replace the grounding implementation with the Vertex AI SDK with the following code that uses the Google Gen AI SDK.

Before

Python
Java
JavaScript
Go


model = generative_models.GenerativeModel(GEMINI_MODEL_NAME)
google_search_retriever_tool = (
  generative_models.Tool.from_google_search_retrieval(
    generative_models.grounding.GoogleSearchRetrieval()
  )
)
response = model.generate_content(
  "Why is sky blue?",
  tools=[google_search_retriever_tool],
  generation_config=generative_models.GenerationConfig(temperature=0),
)
After

Python
Java
JavaScript
Go


from google.genai import types
from google.genai import Client

client = Client(
  vertexai=True,
  project=GOOGLE_CLOUD_PROJECT,
  location=GOOGLE_CLOUD_LOCATION
)

response = client.models.generate_content(
  model='gemini-2.5-flash-exp',
  contents='Why is the sky blue?',
  config=types.GenerateContentConfig(
  tools=[types.Tool(google_search=types.GoogleSearch())]),
)
Safety settings
Safety settings are configurable parameters that allow users to manage model responses by filtering or blocking content related to specific harmful categories, such as hate speech, sexual content, or violence. Replace the safety settings implementation with the Vertex AI SDK with the following code that uses the Google Gen AI SDK.

Before

Python
Java
JavaScript
Go


model = generative_models.GenerativeModel(
  GEMINI_MODEL_NAME,
  system_instruction=[
    "Talk like a pirate.",
    "Don't use rude words.",
  ],
)
response = model.generate_content(
  contents="Why is sky blue?",
  generation_config=generative_models.GenerationConfig(
    temperature=0,
    top_p=0.95,
    top_k=20,
    candidate_count=1,
    max_output_tokens=100,
    stop_sequences=["STOP!"],
    response_logprobs=True,
    logprobs=3,
  ),
  safety_settings={
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
  },
)
After

Python
Java
JavaScript
Go


from google.genai import types

response = client.models.generate_content(
  model='gemini-2.5-flash',
  contents='Say something bad.',
  config=types.GenerateContentConfig(
    safety_settings=[
      types.SafetySetting(
        category='HARM_CATEGORY_HATE_SPEECH',
        threshold='BLOCK_ONLY_HIGH',
      )
    ]
  ),
)
Chat sessions
Chat sessions are conversational interactions where the model maintains context over multiple turns by recalling previous messages and using them to inform current responses. Replace the implementation from the Vertex AI SDK with the following code that uses the Google Gen AI SDK.

Before

Python
Java
JavaScript
Go


model = GenerativeModel(
  "gemini-2.5-flash",
  # You can specify tools when creating a model to avoid having to send them with every request.
  tools=[weather_tool],
  tool_config=tool_config,
)
chat = model.start_chat()
print(chat.send_message("What is the weather like in Boston?"))
print(chat.send_message(
  Part.from_function_response(
    name="get_current_weather",
    response={
      "content": {"weather_there": "super nice"},
      }
  ),
))
After

Python
Java
JavaScript
Go
Synchronous



chat = client.chats.create(model='gemini-2.5-flash')
response = chat.send_message('tell me a story')
print(response.text)
response = chat.send_message('summarize the story you told me in 1 sentence')
print(response.text)
Asynchronous



chat = client.aio.chats.create(model='gemini-2.5-flash')
response = await chat.send_message('tell me a story')
print(response.text)
Synchronous streaming



chat = client.chats.create(model='gemini-2.5-flash')
for chunk in chat.send_message_stream('tell me a story'):
    print(chunk.text, end='')
Asynchronous streaming



chat = client.aio.chats.create(model='gemini-2.5-flash')
async for chunk in await chat.send_message_stream('tell me a story'):
    print(chunk.text, end='') # end='' is optional, for demo purposes.
Multimodal inputs
Multimodal inputs refers to the ability of a model to process and understand information from data types beyond text, such as images, audio, and video. Replace the implementation with the Vertex AI SDK with the following code that uses the Google Gen AI SDK.

Before

Python
Java
JavaScript
Go


from vertexai.generative_models import GenerativeModel, Image
vision_model = GenerativeModel("gemini-2.5-flash-vision")

# Local image
image = Image.load_from_file("image.jpg")
print(vision_model.generate_content(["What is shown in this image?", image]))

# Image from Cloud Storage
image_part = generative_models.Part.from_uri("gs://download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg", mime_type="image/jpeg")
print(vision_model.generate_content([image_part, "Describe this image?"]))

# Text and video
video_part = Part.from_uri("gs://cloud-samples-data/video/animals.mp4", mime_type="video/mp4")
print(vision_model.generate_content(["What is in the video? ", video_part]))
After

Python
Java
JavaScript
Go


from google import genai
from google.genai.types import HttpOptions, Part

client = genai.Client(http_options=HttpOptions(api_version="v1"))
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        Part.from_uri(
            file_uri="gs://cloud-samples-data/generative-ai/video/ad_copy_from_video.mp4",
            mime_type="video/mp4",
        ),
        "What is in the video?",
    ],
)
print(response.text)
Text generation
Text generation is the process by which a model produces human-like written content based on a given prompt. Replace the implementation with the Vertex AI SDK with the following code that uses the Google Gen AI SDK.

Synchronous generation
Before

Python
Java
JavaScript
Go


response = model.generate_content(
  "Why is sky blue?",
  generation_config=generative_models.GenerationConfig(temperature=0),
)
assert response.text
After

Python
Java
JavaScript
Go


response = client.models.generate_content(
  model='gemini-2.5-flash', contents='Why is the sky blue?'
)
print(response.text)
Asynchronous generation
Before

Python
Java
JavaScript
Go


response = await model.generate_content_async(
  "Why is sky blue?",
  generation_config=generative_models.GenerationConfig(temperature=0),
)
After

Python
Java
JavaScript
Go


response = await client.aio.models.generate_content(
  model='gemini-2.5-flash', contents='Tell me a story in 300 words.'
)

print(response.text)
Streaming
Before

Python
Java
JavaScript
Go
Synchronous streaming



stream = model.generate_content(
  "Why is sky blue?",
  stream=True,
  generation_config=generative_models.GenerationConfig(temperature=0),
)
for chunk in stream:
  assert (
    chunk.text
    or chunk.candidates[0].finish_reason
    is generative_models.FinishReason.STOP
  )
Asynchronous streaming



async_stream = await model.generate_content_async(
  "Why is sky blue?",
  stream=True,
  generation_config=generative_models.GenerationConfig(temperature=0),
)
async for chunk in async_stream:
  assert (
    chunk.text
    or chunk.candidates[0].finish_reason
    is generative_models.FinishReason.STOP
  )
After

Python
Java
JavaScript
Go
Synchronous streaming



for chunk in client.models.generate_content_stream(
  model='gemini-2.5-flash', contents='Tell me a story in 300 words.'
):
  print(chunk.text, end='')
Asynchronous streaming



async for chunk in await client.aio.models.generate_content_stream(
  model='gemini-2.5-flash', contents='Tell me a story in 300 words.'
):
  print(chunk.text, end='')
Image generation
Image generation is the process by which a models creates images from textual descriptions or other input modalities. Replace the implementation with the Vertex AI SDK with the following code that uses the Google Gen AI SDK.

Before

Python
Java
JavaScript
Go


model = ImageGenerationModel.from_pretrained("imagegeneration@002")
response = model.generate_images(
    prompt="Astronaut riding a horse",
    # Optional:
    number_of_images=1,
    seed=0,
)
response[0].show()
response[0].save("image1.png")
After

Python
Java
JavaScript
Go


from google.genai import types

# Generate Image
response1 = client.models.generate_images(
    model='imagen-3.0-generate-002',
    prompt='An umbrella in the foreground, and a rainy night sky in the background',
    config=types.GenerateImagesConfig(
        number_of_images=1,
        include_rai_reason=True,
        output_mime_type='image/jpeg',
    ),
)
response1.generated_images[0].image.show()
Controlled generation
Controlled generation refers to the process of guiding model output to adhere to specific constraints, formats, styles, or attributes, rather than generating free-form text. Replace the implementation with the Vertex AI SDK with the following code that uses the Google Gen AI SDK.

Before

Python
Java
JavaScript
Go


_RESPONSE_SCHEMA_STRUCT = {
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
        },
    },
    "required": ["location"],
}

response = model.generate_content(
    contents="Why is sky blue? Respond in JSON Format.",
    generation_config=generative_models.GenerationConfig(
        ...
        response_schema=_RESPONSE_SCHEMA_STRUCT,
    ),
)
After

Python
Java
JavaScript
Go


response_schema = {
  "type": "ARRAY",
  "items": {
    "type": "OBJECT",
    "properties": {
      "recipe_name": {"type": "STRING"},
      "ingredients": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["recipe_name", "ingredients"],
  },
}

prompt = """
  List a few popular cookie recipes.
"""

response = client.models.generate_content(
  model="gemini-2.5-flash",
  contents=prompt,
  config={
    "response_mime_type": "application/json",
    "response_schema": response_schema,
  },
)
Count tokens
Tokens are the fundamental units of text (letters, words, phrases) that models process, analyze, and generate. To count or compute tokens in a response, replace the implementation with the Vertex AI SDK with the following code that uses the Google Gen AI SDK.

Before

Python
Java
JavaScript
Go


content = ["Why is sky blue?", "Explain it like I'm 5."]

response = model.count_tokens(content)
After

Python
Java
JavaScript
Go
Count Tokens



response = client.models.count_tokens(
    model='gemini-2.5-flash',
    contents='why is the sky blue?',
)
print(response)
Compute tokens



response = client.models.compute_tokens(
    model='gemini-2.5-flash',
    contents='why is the sky blue?',
)
