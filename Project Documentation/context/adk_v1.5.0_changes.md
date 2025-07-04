# Google ADK v1.5.0 API Changes

This document outlines key API changes between previous versions of Google ADK and the current v1.5.0 version, with a focus on imports, classes, and functionality that required modifications in our Multi-Agent Research Platform.

## Memory Service Changes

### Content and Part Structure

In ADK v1.5.0, the structure for creating and handling content has been modified:

#### Previous Versions
```python
from google.genai.types import TextPart, Content

content = Content(parts=[TextPart(text="Some text")])
```

#### ADK v1.5.0
```python
from google.genai.types import Part, Content

content = Content(parts=[Part(text="Some text")])
```

### Memory Response Structure

The response structure for memory searches has changed significantly:

#### Previous Versions
```python
from google.adk.memory import SearchMemoryResponseEntry

# In search_memory implementation:
return [SearchMemoryResponseEntry(
    text="Memory content",
    relevance_score=0.85
)]
```

#### ADK v1.5.0
```python
from google.adk.memory import MemoryEntry

# In search_memory implementation:
return [MemoryEntry(
    text="Memory content",
    relevance_score=0.85
)]
```

## Context Module Changes

### ReadonlyContext Import Path

The import path for ReadonlyContext has changed:

#### Previous Versions
```python
from google.adk.agents import ReadonlyContext
```

#### ADK v1.5.0
```python
from google.adk.agents.readonly_context import ReadonlyContext
```

## Tool Configuration Changes

### ToolRegistry vs ToolsConfig

In ADK v1.5.0, ToolsConfig has been replaced by ToolRegistry:

#### Previous Versions
```python
from google.adk.tools import ToolsConfig

def create_mcp_server(tools_config: ToolsConfig):
    # Implementation
```

#### ADK v1.5.0
```python
from google.adk.tools import ToolRegistry

def create_mcp_server(tools_config: ToolRegistry):
    # Implementation
```

## API Usage Pattern Changes

### Memory Storage and Retrieval

ADK v1.5.0 introduces a more structured approach to memory storage and retrieval:

```python
# Storing memory
await memory_service.store(
    text="Information to store",
    metadata={
        "type": "conversation_summary",
        "importance": "high",
        "source": "user_interaction"
    }
)

# Retrieving memory
results = await memory_service.search_memory(
    query="Information to find",
    limit=5
)
# results is now a list of MemoryEntry objects
```

### Working with Sessions and Content

The pattern for creating and managing content in sessions has been standardized:

```python
from google.genai.types import Content, Part

user_message = Content(parts=[Part(text="User message")], role="user")

# Pass this to runner
async for event in runner.run_async(
    user_id="user123",
    session_id="session123", 
    new_message=user_message
):
    # Process events
    pass
```

## Common Issues and Fixes

1. **Import Errors**: Many modules have been restructured, requiring updates to import statements. Always check the latest import paths in the official documentation.

2. **Type Annotations**: Update parameter and return type annotations in functions that interact with ADK classes.

3. **Deprecated Classes**: Replace deprecated classes like `SearchMemoryResponseEntry` with their new equivalents (`MemoryEntry`).

4. **Tool Configurations**: Update tool configuration patterns to use `ToolRegistry` instead of `ToolsConfig`.

## Future-Proofing Recommendations

1. Use fully qualified imports rather than wildcard imports to make future migration easier.

2. Keep dependency versions pinned in requirements.txt to avoid unexpected API changes.

3. Write unit tests that explicitly verify API compatibility with the target ADK version.

4. Document API usage patterns in a centralized location for easier updates when new ADK versions are released.

## References

- [Google ADK Python API Documentation](https://ai.google.dev/api/python/google-ai-adk/)
- [ADK Migration Guide](https://ai.google.dev/docs/migration) (Check for the latest version)
