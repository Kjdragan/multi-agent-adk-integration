Generative AI on Vertex AI
Documentation
Was this helpful?

Send feedbackStructured output

bookmark_border
Release Notes
To see an example of structured output, run the "Intro to structured output" Jupyter notebook in one of the following environments:

Open in Colab | Open in Colab Enterprise | Open in Vertex AI Workbench user-managed notebooks | View on GitHub

You can guarantee that a model's generated output always adheres to a specific schema so that you receive consistently formatted responses. For example, you might have an established data schema that you use for other tasks. If you have the model follow the same schema, you can directly extract data from the model's output without any post-processing.

To specify the structure of a model's output, define a response schema, which works like a blueprint for model responses. When you submit a prompt and include the response schema, the model's response always follows your defined schema.

You can control generated output when using the following models:

Vertex AI Model Optimizer science
Gemini 2.5 Pro
Gemini 2.5 Flash
Gemini 2.0 Flash
Gemini 2.0 Flash-Lite
Note: Using structured output on tuned Gemini models can result in decreased model quality.
Example use cases
One use case for applying a response schema is to ensure that a model's response produces valid JSON and conforms to your schema. Generative model outputs can have some degree of variability, so including a response schema ensures that you always receive valid JSON. Consequently, your downstream tasks can reliably expect valid JSON input from generated responses.

Another example is to constrain how a model can respond. For example, you can have a model annotate text with user-defined labels, not with labels that the model produces. This constraint is useful when you expect a specific set of labels such as positive or negative and don't want to receive a mixture of other labels that the model might generate like good, positive, negative, or bad.

Considerations
The following considerations discuss potential limitations if you plan on using a response schema:

You must use the API to define and use a response schema. There's no console support.
The size of your response schema counts towards the input token limit.
Only certain output formats are supported, such as application/json or text/x.enum. For more information, see the responseMimeType parameter in the Gemini API reference.
Structured output supports a subset of the Vertex AI schema reference. For more information, see Supported schema fields.
A complex schema can result in an InvalidArgument: 400 error. Complexity might come from long property names, long array length limits, enums with many values, objects with lots of optional properties, or a combination of these factors.

If you get this error with a valid schema, make one or more of the following changes to resolve the error:

Shorten property names or enum names.
Flatten nested arrays.
Reduce the number of properties with constraints, such as numbers with minimum and maximum limits.
Reduce the number of properties with complex constraints, such as properties with complex formats like date-time.
Reduce the number of optional properties.
Reduce the number of valid values for enums.
Supported schema fields
Structured output supports the following fields from the Vertex AI schema. If you use an unsupported field, Vertex AI can still handle your request but ignores the field.

anyOf
enum
format
items
maximum
maxItems
minimum
minItems
nullable
properties
propertyOrdering*
required

* propertyOrdering is specifically for structured output and not part of the Vertex AI schema. This field defines the order in which properties are generated. The listed properties must be unique and must be valid keys in the properties dictionary.

For the format field, Vertex AI supports the following values: date, date-time, duration, and time. The description and format of each value is described in the OpenAPI Initiative Registry

Before you begin
Define a response schema to specify the structure of a model's output, the field names, and the expected data type for each field. Use only the supported fields as listed in the Considerations section. All other fields are ignored.

Include your response schema as part of the responseSchema field only. Don't duplicate the schema in your input prompt. If you do, the generated output might be lower in quality.

For sample schemas, see the Example schemas and model responses section.

Model behavior and response schema
When a model generates a response, it uses the field name and context from your prompt. As such, we recommend that you use a clear structure and unambiguous field names so that your intent is clear.

By default, fields are optional, meaning the model can populate the fields or skip them. You can set fields as required to force the model to provide a value. If there's insufficient context in the associated input prompt, the model generates responses mainly based on the data it was trained on.

If you aren't seeing the results you expect, add more context to your input prompts or revise your response schema. For example, review the model's response without structured output to see how the model responds. You can then update your response schema that better fits the model's output.

Send a prompt with a response schema
To see an example of a response schema and structured output, run the "Introduction to structured output" Jupyter notebook in one of the following environments:

Open in Colab | Open in Colab Enterprise | Open in Vertex AI Workbench user-managed notebooks | View on GitHub

By default, all fields are optional, meaning a model might generate a response to a field. To force the model to always generate a response to a field, set the field as required.

Python
Go
REST
Install


pip install --upgrade google-genai
To learn more, see the SDK reference documentation.

Set environment variables to use the Gen AI SDK with Vertex AI:



# Replace the `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` values
# with appropriate values for your project.
export GOOGLE_CLOUD_PROJECT=GOOGLE_CLOUD_PROJECT
export GOOGLE_CLOUD_LOCATION=global
export GOOGLE_GENAI_USE_VERTEXAI=True



from google import genai
from google.genai.types import HttpOptions

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

client = genai.Client(http_options=HttpOptions(api_version="v1"))
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config={
        "response_mime_type": "application/json",
        "response_schema": response_schema,
    },
)

print(response.text)
# Example output:
# [
#     {
#         "ingredients": [
#             "2 1/4 cups all-purpose flour",
#             "1 teaspoon baking soda",
#             "1 teaspoon salt",
#             "1 cup (2 sticks) unsalted butter, softened",
#             "3/4 cup granulated sugar",
#             "3/4 cup packed brown sugar",
#             "1 teaspoon vanilla extract",
#             "2 large eggs",
#             "2 cups chocolate chips",
#         ],
#         "recipe_name": "Chocolate Chip Cookies",
#     }
# ]
Example schemas for JSON output
The following sections demonstrate a variety of sample prompts and response schemas. A sample model response is also included after each code sample.

Forecast the weather for each day of the week in an array
Classify a product with a well-defined enum
Forecast the weather for each day of the week
The following example outputs a forecast object for each day of the week that includes an array of properties such as the expected temperature and humidity level for the day. Some properties are set to nullable so the model can return a null value when it doesn't have enough context to generate a meaningful response. This strategy helps reduce hallucinations.


Python
Go
Install


pip install --upgrade google-genai
To learn more, see the SDK reference documentation.

Set environment variables to use the Gen AI SDK with Vertex AI:



# Replace the `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` values
# with appropriate values for your project.
export GOOGLE_CLOUD_PROJECT=GOOGLE_CLOUD_PROJECT
export GOOGLE_CLOUD_LOCATION=global
export GOOGLE_GENAI_USE_VERTEXAI=True



from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions

response_schema = {
    "type": "OBJECT",
    "properties": {
        "forecast": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "Day": {"type": "STRING", "nullable": True},
                    "Forecast": {"type": "STRING", "nullable": True},
                    "Temperature": {"type": "INTEGER", "nullable": True},
                    "Humidity": {"type": "STRING", "nullable": True},
                    "Wind Speed": {"type": "INTEGER", "nullable": True},
                },
                "required": ["Day", "Temperature", "Forecast", "Wind Speed"],
            },
        }
    },
}

prompt = """
    The week ahead brings a mix of weather conditions.
    Sunday is expected to be sunny with a temperature of 77°F and a humidity level of 50%. Winds will be light at around 10 km/h.
    Monday will see partly cloudy skies with a slightly cooler temperature of 72°F and the winds will pick up slightly to around 15 km/h.
    Tuesday brings rain showers, with temperatures dropping to 64°F and humidity rising to 70%.
    Wednesday may see thunderstorms, with a temperature of 68°F.
    Thursday will be cloudy with a temperature of 66°F and moderate humidity at 60%.
    Friday returns to partly cloudy conditions, with a temperature of 73°F and the Winds will be light at 12 km/h.
    Finally, Saturday rounds off the week with sunny skies, a temperature of 80°F, and a humidity level of 40%. Winds will be gentle at 8 km/h.
"""

client = genai.Client(http_options=HttpOptions(api_version="v1"))
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config=GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=response_schema,
    ),
)

print(response.text)
# Example output:
# {"forecast": [{"Day": "Sunday", "Forecast": "sunny", "Temperature": 77, "Wind Speed": 10, "Humidity": "50%"},
#   {"Day": "Monday", "Forecast": "partly cloudy", "Temperature": 72, "Wind Speed": 15},
#   {"Day": "Tuesday", "Forecast": "rain showers", "Temperature": 64, "Wind Speed": null, "Humidity": "70%"},
#   {"Day": "Wednesday", "Forecast": "thunderstorms", "Temperature": 68, "Wind Speed": null},
#   {"Day": "Thursday", "Forecast": "cloudy", "Temperature": 66, "Wind Speed": null, "Humidity": "60%"},
#   {"Day": "Friday", "Forecast": "partly cloudy", "Temperature": 73, "Wind Speed": 12},
#   {"Day": "Saturday", "Forecast": "sunny", "Temperature": 80, "Wind Speed": 8, "Humidity": "40%"}]}
Classify a product
The following example includes enums where the model must classify an object's type and condition from a list of given values.


Python
Go
Node.js
Java
Install


pip install --upgrade google-genai
To learn more, see the SDK reference documentation.

Set environment variables to use the Gen AI SDK with Vertex AI:



# Replace the `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` values
# with appropriate values for your project.
export GOOGLE_CLOUD_PROJECT=GOOGLE_CLOUD_PROJECT
export GOOGLE_CLOUD_LOCATION=global
export GOOGLE_GENAI_USE_VERTEXAI=True



from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions

client = genai.Client(http_options=HttpOptions(api_version="v1"))
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What type of instrument is an oboe?",
    config=GenerateContentConfig(
        response_mime_type="text/x.enum",
        response_schema={
            "type": "STRING",
            "enum": ["Percussion", "String", "Woodwind", "Brass", "Keyboard"],
        },
    ),
)

print(response.text)
# Example output:
# Woodwind
