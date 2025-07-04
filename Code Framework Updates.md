# Code Framework Updates

This document tracks API and framework changes in dependencies used by the Multi-Agent Research Platform. It serves as a reference for why specific code modifications were made and helps maintain compatibility as SDK versions evolve.

## Google ADK (Agent Development Kit)

### Version 1.5.0 (2025-06-25)

#### Memory API Changes

The Google ADK Memory API underwent significant restructuring in version 1.0.0 and has remained stable through version 1.5.0. Notable changes include:

1. **Class Renaming and Structure**:
   - Old: `SearchMemoryResponseEntry` used for individual memory entries
   - New: `MemoryEntry` class with a different structure for representing memory content

2. **Response Format**:
   - Old: `SearchMemoryResponse` with a `results` attribute containing a list of `SearchMemoryResponseEntry` objects
   - New: `SearchMemoryResponse` with a `memories` attribute containing a list of `MemoryEntry` objects

3. **Content Structure**:
   - Old: Direct text content in memory entries
   - New: Structured `Content` objects using `types.Content` with `parts` containing `TextPart` objects

4. **Import Changes**:
   - `SearchMemoryResponse` must be imported directly from base module
   - `MemoryEntry` must be imported from `google.adk.memory.memory_entry`
   - Additional import needed for `types` module to create proper content objects

#### Code Changes Made

The following changes were implemented to adapt to the new API:

1. Updated imports to match new module structure
2. Changed response construction to use the new `memories` attribute instead of `results`
3. Created proper content objects using `types.Content` and `types.TextPart`
4. Updated memory entry construction to use the new `MemoryEntry` class with the correct attribute structure

## Python Package Structure

### Directory Naming Conventions

1. **Issue**: Package with hyphens in directory name causing import errors
   - Python modules must use underscores, not hyphens
   - Original directory: `multi-agent-research-platform`
   - Updated directory: `multi_agent_research_platform`

2. **Code Changes**:
   - Renamed directory structure to use underscores
   - Updated import statements to reference the new directory structure
   - Maintained original name in pyproject.toml while ensuring setuptools finds packages correctly

## Package Dependencies

### Version Compatibility

Maintaining version compatibility with various packages can be challenging. Current key dependencies include:

- google-adk>=1.5.0
- google-auth>=2.40.3
- google-genai>=1.24.0
- fastapi>=0.115.14

Changes to these package APIs should be documented here as they are discovered.
