# Memory Service ADK v1.5.0 Compatibility Plan

## Notes
- ADK v1.5.0 API changed: Content should be constructed with a list of `Part` objects, not `TextPart`.
- Correct import for `Part` is from `google.genai.types`.
- Correct import for `ReadonlyContext` is from `google.adk.agents.readonly_context`.
- Multiple files needed import path corrections for ADK compatibility.
- Found additional ReadonlyContext import issue in context/patterns.py (fixed)
- Fixed incorrect ToolsConfig import in mcp/factory.py (now uses ToolRegistry)
- Remaining ToolsConfig references in mcp/factory.py need replacing
- Pydantic V1 @validator deprecation warnings present, but not blocking
- Reviewed ADK 1.5 context and memory documentation; approach is correct

## Task List
- [x] Investigate ADK Content and Part types and correct imports
- [x] Update Content/Part usage in memory service code
- [x] Fix ReadonlyContext import in context/managers.py
- [x] Fix ReadonlyContext import in context/wrappers.py
- [x] Fix ReadonlyContext import in context/patterns.py
- [x] Replace all ToolsConfig references in mcp/factory.py with ToolRegistry
- [x] Run tests after ADK import fixes
- [ ] Diagnose and fix test collection errors (1 error remains)

## Current Goal
Diagnose and fix test collection errors
