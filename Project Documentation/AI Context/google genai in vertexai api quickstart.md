his quickstart shows you how to install the Google Gen AI SDK for your language of choice and then make your first API request. The samples vary slightly based on whether you're using an API key or application default credentials (ADC) for authentication.

Choose your authentication method:

API key ADC
Before you begin
Configure application default credentials if you haven't yet.

Install the SDK and set up your environment
On your local machine, click one of the following tabs to install the SDK for your programming language.

Gen AI SDK for Python
Gen AI SDK for Go
Gen AI SDK for Node.js
More
Install and update the Gen AI SDK for Python by running this command.



pip install --upgrade google-genai
Set environment variables:



# Replace the `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` values
# with appropriate values for your project.
export GOOGLE_CLOUD_PROJECT=GOOGLE_CLOUD_PROJECT
export GOOGLE_CLOUD_LOCATION=global
export GOOGLE_GENAI_USE_VERTEXAI=True
Make your first request
Use the generateContent method to send a request to the Gemini API in Vertex AI:

Python
Go
Node.js
Java
REST



from google import genai
from google.genai.types import HttpOptions

client = genai.Client(http_options=HttpOptions(api_version="v1"))
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="How does AI work?",
)
print(response.text)
# Example response:
# Okay, let's break down how AI works. It's a broad field, so I'll focus on the ...
#
# Here's a simplified overview:
# ...
Generate images
Gemini can generate and process images conversationally. You can prompt Gemini with text, images, or a combination of both to achieve various image-related tasks, such as image generation and editing. The following code demonstrates how to generate an image based on a descriptive prompt:

You must include responseModalities: ["TEXT", "IMAGE"] in your configuration. Image-only output is not supported with these models.

Python



from google import genai
from google.genai.types import GenerateContentConfig, Modality
from PIL import Image
from io import BytesIO

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.0-flash-preview-image-generation",
    contents=(
        "Generate an image of the Eiffel tower with fireworks in the background."
    ),
    config=GenerateContentConfig(response_modalities=[Modality.TEXT, Modality.IMAGE]),
)
for part in response.candidates[0].content.parts:
    if part.text:
        print(part.text)
    elif part.inline_data:
        image = Image.open(BytesIO((part.inline_data.data)))
        image.save("example-image.png")
# Example response:
#   A beautiful photograph captures the iconic Eiffel Tower in Paris, France,
#   against a backdrop of a vibrant and dynamic fireworks display. The tower itself...
Image understanding
Gemini can understand images as well. The following code uses the image generated in the previous section and uses a different model to infer information about the image:

Python
Go
Node.js
Java



from google import genai
from google.genai.types import HttpOptions, Part

client = genai.Client(http_options=HttpOptions(api_version="v1"))
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        "What is shown in this image?",
        Part.from_uri(
            file_uri="gs://cloud-samples-data/generative-ai/image/scones.jpg",
            mime_type="image/jpeg",
        ),
    ],
)
print(response.text)
# Example response:
# The image shows a flat lay of blueberry scones arranged on parchment paper. There are ...
Code execution
The Gemini API in Vertex AI code execution feature enables the model to generate and run Python code and learn iteratively from the results until it arrives at a final output. Vertex AI provides code execution as a tool, similar to function calling. You can use this code execution capability to build applications that benefit from code-based reasoning and that produce text output. For example:

Python
Go



from google import genai
from google.genai.types import (
    HttpOptions,
    Tool,
    ToolCodeExecution,
    GenerateContentConfig,
)

client = genai.Client(http_options=HttpOptions(api_version="v1"))
model_id = "gemini-2.5-flash"

code_execution_tool = Tool(code_execution=ToolCodeExecution())
response = client.models.generate_content(
    model=model_id,
    contents="Calculate 20th fibonacci number. Then find the nearest palindrome to it.",
    config=GenerateContentConfig(
        tools=[code_execution_tool],
        temperature=0,
    ),
)
print("# Code:")
print(response.executable_code)
print("# Outcome:")
print(response.code_execution_result)

# Example response:
# # Code:
# def fibonacci(n):
#     if n <= 0:
#         return 0
#     elif n == 1:
#         return 1
#     else:
#         a, b = 0, 1
#         for _ in range(2, n + 1):
#             a, b = b, a + b
#         return b
#
# fib_20 = fibonacci(20)
# print(f'{fib_20=}')
#
# # Outcome:
# fib_20=6765
