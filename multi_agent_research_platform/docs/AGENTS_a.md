# Agent System Documentation (Current Implementation)

This comprehensive guide documents the **actual agent system implementation** as it exists in the Multi-Agent Research Platform, including agent types, capabilities, orchestration strategies, and factory patterns.

## 🤖 Agent System Overview

The Multi-Agent Research Platform employs a sophisticated agent architecture built on Google ADK v1.5.0 with advanced Gemini 2.5 integration, thread-safe registry management, and intelligent orchestration capabilities.

### Agent Architecture (Actual Implementation)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agent Registry (Thread-Safe)                │
│              src/agents/base.py:AgentRegistry                   │
│    Features: RLock(), Performance Tracking, Capability Index   │
├─────────────────────────────────────────────────────────────────┤
│                Agent Orchestrator (9 Strategies)               │
│             src/agents/orchestrator.py                         │
│    Features: Adaptive Selection, Timeout Control, Metrics      │
├─────────────┬─────────────────┬─────────────────┬──────────────┤
│ LLM Agents  │ Workflow Agents │ Custom Agents   │ Agent Factory │
│ (9 roles)   │ (5 types)       │ (12 types)      │ (Templates)   │
│ Gemini 2.5  │ Multi-step      │ Domain Specific │ Service Wiring│
│ Thinking    │ Orchestration   │ Specialized     │ Team Creation │
├─────────────┴─────────────────┴─────────────────┴──────────────┤
│              Base Agent (src/agents/base.py:Agent)             │
│               14 Capabilities, Service Integration             │
├─────────────────────────────────────────────────────────────────┤
│              Google ADK v1.5.0 Integration                     │
│            Tool Access, Context Management, Memory             │
└─────────────────────────────────────────────────────────────────┘
```

## 🎭 Agent Types (Actual Implementation)

### 1. LLM Agents (Gemini 2.5 Integration)

**Location**: `src/agents/llm_agent.py:LLMAgent`

LLM Agents are powered by Google's Gemini 2.5 models with advanced thinking capabilities, structured output, and intelligent model selection.

#### Available Roles (9 Specialized Types)

##### 🔬 RESEARCHER
**Capabilities**: REASONING, RESEARCH, ANALYSIS, MEMORY_ACCESS, COMMUNICATION, TOOL_USE, CONTEXT_MANAGEMENT
```python
researcher = factory.create_llm_agent(
    role=LLMRole.RESEARCHER,
    enable_thinking=True,
    auto_optimize_model=True
)
```
**Specialized For**:
- Web search and information gathering using MCP servers
- Source verification and fact-checking
- Literature review and analysis
- Multi-source data synthesis

##### 📊 ANALYST  
**Capabilities**: REASONING, ANALYSIS, DECISION_MAKING, FACT_CHECKING, COMMUNICATION, TOOL_USE, CONTEXT_MANAGEMENT
```python
analyst = factory.create_llm_agent(
    role=LLMRole.ANALYST,
    enable_structured_output=True,
    output_schema_type="analysis"
)
```
**Specialized For**:
- Statistical analysis and interpretation
- Trend identification and forecasting
- Data visualization recommendations
- Insight generation from complex datasets

##### 🔗 SYNTHESIZER
**Capabilities**: REASONING, SYNTHESIS, CONTENT_GENERATION, ANALYSIS, COMMUNICATION, TOOL_USE, CONTEXT_MANAGEMENT
```python
synthesizer = factory.create_llm_agent(
    role=LLMRole.SYNTHESIZER,
    temperature=0.3,  # Lower for factual synthesis
    enable_memory=True
)
```
**Specialized For**:
- Multi-source information integration
- Conflict resolution between sources
- Comprehensive summary creation
- Knowledge consolidation

##### 🎯 CRITIC
**Capabilities**: REASONING, ANALYSIS, FACT_CHECKING, DECISION_MAKING, COMMUNICATION, TOOL_USE, CONTEXT_MANAGEMENT
```python
critic = factory.create_llm_agent(
    role=LLMRole.CRITIC,
    enable_thinking=True,
    thinking_budget=2000  # More thinking for critical analysis
)
```
**Specialized For**:
- Quality assessment and review
- Constructive feedback generation
- Error identification and correction
- Bias detection and improvement recommendations

##### 📋 PLANNER
**Capabilities**: REASONING, PLANNING, DECISION_MAKING, EXECUTION, COMMUNICATION, TOOL_USE, CONTEXT_MANAGEMENT
```python
planner = factory.create_llm_agent(
    role=LLMRole.PLANNER,
    enable_structured_output=True,
    output_schema_type="planning"
)
```
**Specialized For**:
- Task breakdown and decomposition
- Resource allocation planning
- Timeline and milestone creation
- Risk assessment and mitigation planning

##### 💬 COMMUNICATOR
**Capabilities**: REASONING, CONTENT_GENERATION, SYNTHESIS, COMMUNICATION, TOOL_USE, CONTEXT_MANAGEMENT
```python
communicator = factory.create_llm_agent(
    role=LLMRole.COMMUNICATOR,
    temperature=0.7,  # Higher for creative communication
    priority_speed=True
)
```
**Specialized For**:
- Clear, effective communication
- Presentations and stakeholder engagement
- Multi-audience content adaptation
- Technical translation to business language

##### 🎨 CREATIVE
**Capabilities**: REASONING, CONTENT_GENERATION, SYNTHESIS, COMMUNICATION, TOOL_USE, CONTEXT_MANAGEMENT
```python
creative = factory.create_llm_agent(
    role=LLMRole.CREATIVE,
    temperature=0.9,  # Higher for creativity
    enable_cost_optimization=False  # Use best model
)
```
**Specialized For**:
- Creative content generation
- Innovative solution development
- Brainstorming and ideation
- Original approach development

##### 🎓 SPECIALIST
**Capabilities**: REASONING, ANALYSIS, DECISION_MAKING, RESEARCH, COMMUNICATION, TOOL_USE, CONTEXT_MANAGEMENT
```python
specialist = factory.create_llm_agent(
    role=LLMRole.SPECIALIST,
    model=GeminiModel.PRO,  # Best model for expertise
    enable_thinking=True
)
```
**Specialized For**:
- Deep domain expertise
- Technical accuracy and precision
- Industry-specific insights
- Expert-level analysis and recommendations

##### 🌟 GENERALIST
**Capabilities**: REASONING, ANALYSIS, RESEARCH, SYNTHESIS, CONTENT_GENERATION, PLANNING, COMMUNICATION, TOOL_USE, CONTEXT_MANAGEMENT
```python
generalist = factory.create_llm_agent(
    role=LLMRole.GENERALIST,
    auto_optimize_model=True,  # Adapt to any task
    enable_thinking=True
)
```
**Specialized For**:
- General-purpose assistance
- Cross-domain tasks
- Initial task assessment
- Backup agent capabilities

### 2. Workflow Agents (Process Orchestration)

**Location**: `src/agents/workflow_agent.py:WorkflowAgent`

Workflow Agents manage complex, multi-step processes with dependency management and coordination.

#### Workflow Types (5 Implementations)

##### 🔄 Sequential Workflow
```python
sequential_agent = factory.create_workflow_agent(
    workflow_type="sequential",
    preserve_context=True
)
```
**Process**: Step 1 → Step 2 → Step 3 → Result
**Best For**: Linear processes, dependent tasks, methodical analysis

##### ⚡ Parallel Workflow  
```python
parallel_agent = factory.create_workflow_agent(
    workflow_type="parallel",
    max_concurrent_steps=5
)
```
**Process**: Task A + Task B + Task C → Aggregation → Result
**Best For**: Independent subtasks, time-sensitive analysis, performance optimization

##### 🔀 Conditional Workflow
```python
conditional_agent = factory.create_workflow_agent(
    workflow_type="conditional",
    decision_criteria="context_dependent"
)
```
**Process**: Start → Decision → Path A/B → Result
**Best For**: Adaptive processes, context-dependent analysis, decision trees

##### 🔁 Iterative Workflow
```python
iterative_agent = factory.create_workflow_agent(
    workflow_type="iterative",
    max_iterations=5,
    convergence_threshold=0.9
)
```
**Process**: Initial → Process → Review → Refine/Complete → Result
**Best For**: Quality improvement, incremental refinement, optimization

##### 🌀 Recursive Workflow
```python
recursive_agent = factory.create_workflow_agent(
    workflow_type="recursive",
    max_depth=3
)
```
**Process**: Problem → Decompose → Solve Subproblems → Combine → Result
**Best For**: Complex problem decomposition, hierarchical analysis

### 3. Custom Agents (Domain Specialists)

**Location**: `src/agents/custom_agent.py:CustomAgent`

Custom Agents are specialized for specific domains and use cases with tailored capabilities.

#### Available Types (12 Implementations)

##### 🎓 Domain Expert
```python
domain_expert = factory.create_custom_agent(
    agent_type=CustomAgentType.DOMAIN_EXPERT,
    domain="artificial_intelligence",
    specialization="machine_learning"
)
```

##### ✅ Fact Checker
```python
fact_checker = factory.create_custom_agent(
    agent_type=CustomAgentType.FACT_CHECKER,
    verification_sources=["perplexity", "tavily"]
)
```

##### 📈 Data Analyst
```python
data_analyst = factory.create_custom_agent(
    agent_type=CustomAgentType.DATA_ANALYST,
    enable_code_execution=True
)
```

##### 🔍 Code Reviewer
```python
code_reviewer = factory.create_custom_agent(
    agent_type=CustomAgentType.CODE_REVIEWER,
    languages=["python", "javascript", "typescript"]
)
```

##### 🎨 Content Creator, 🌐 Translator, 📝 Summarizer, 🔬 Researcher, 📊 Analyst, ✏️ Writer, 🎯 Critic, 🏛️ Moderator

## 🎯 Agent Capabilities (14 Core Capabilities)

### Capability Enum (Actual Implementation)
**Location**: `src/agents/base.py:AgentCapability`

```python
class AgentCapability(str, Enum):
    REASONING = "reasoning"                    # Logical reasoning and analysis
    RESEARCH = "research"                      # Information gathering and research
    ANALYSIS = "analysis"                      # Data and content analysis
    SYNTHESIS = "synthesis"                    # Information synthesis and summarization
    PLANNING = "planning"                      # Task planning and decomposition
    EXECUTION = "execution"                    # Task execution and orchestration
    COMMUNICATION = "communication"            # Agent-to-agent communication
    LEARNING = "learning"                      # Learning from interactions
    TOOL_USE = "tool_use"                     # Using external tools and APIs
    MEMORY_ACCESS = "memory_access"            # Accessing and storing memories
    CONTEXT_MANAGEMENT = "context_management"  # Managing conversation context
    FACT_CHECKING = "fact_checking"            # Verifying information accuracy
    CONTENT_GENERATION = "content_generation"  # Creating new content
    DECISION_MAKING = "decision_making"        # Making informed decisions
```

### Capability Matrix (Actual Implementation)

| Agent Role | REASONING | RESEARCH | ANALYSIS | SYNTHESIS | CONTENT_GEN | FACT_CHECK | PLANNING | MEMORY | TOOLS |
|------------|-----------|----------|----------|-----------|-------------|------------|----------|--------|-------|
| RESEARCHER | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| ANALYST | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ |
| SYNTHESIZER | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |
| CRITIC | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ |
| PLANNER | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ |
| COMMUNICATOR | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |
| CREATIVE | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |
| SPECIALIST | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| GENERALIST | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ |

**Legend**: ✅ Has Capability • ❌ No Capability

## 🎬 Orchestration Strategies (9 Implementations)

**Location**: `src/agents/orchestrator.py:AgentOrchestrator`

### Strategy Implementations

#### 1. 🎯 ADAPTIVE Strategy (Intelligent Selection)
```python
result = await orchestrator.orchestrate_task(
    task="Complex research analysis",
    strategy=OrchestrationStrategy.ADAPTIVE
)
```
**Decision Logic**:
- Task complexity analysis (length, keywords, requirements)
- Agent availability assessment
- Performance history evaluation
- Automatic strategy selection

#### 2. 🏆 SINGLE_BEST Strategy
```python
result = await orchestrator.orchestrate_task(
    task="Simple factual question",
    strategy=OrchestrationStrategy.SINGLE_BEST
)
```
**Selection Criteria**:
- Performance score (40%)
- Capability match (40%)
- Current workload (20%)

#### 3. ⚡ PARALLEL_ALL Strategy
```python
result = await orchestrator.orchestrate_task(
    task="Comprehensive analysis",
    strategy=OrchestrationStrategy.PARALLEL_ALL
)
```
**Features**:
- Concurrent execution with all capable agents
- Timeout protection with proper task cancellation
- Result aggregation from successful agents
- Configurable max agents (default: 5)

#### 4. 🤝 CONSENSUS Strategy  
```python
result = await orchestrator.orchestrate_task(
    task="Important decision requiring validation",
    strategy=OrchestrationStrategy.CONSENSUS
)
```
**Process**:
- Execute with multiple diverse agents
- Cross-validate results
- Build consensus through comparison
- Calculate consensus score
- Configurable consensus threshold (default: 0.7)

#### 5. 🔄 PIPELINE Strategy
```python
result = await orchestrator.orchestrate_task(
    task="Multi-stage research workflow",
    strategy=OrchestrationStrategy.PIPELINE
)
```
**Process**: Research → Analysis → Synthesis → Review → Final Result

#### 6. 🏁 COMPETITIVE Strategy
```python
result = await orchestrator.orchestrate_task(
    task="Creative problem solving",
    strategy=OrchestrationStrategy.COMPETITIVE
)
```
**Features**:
- Multiple agents work independently
- Quality-based result evaluation
- Best result selection
- Performance feedback

#### 7. 🔄 SEQUENTIAL Strategy
```python
result = await orchestrator.orchestrate_task(
    task="Step-by-step analysis",
    strategy=OrchestrationStrategy.SEQUENTIAL
)
```
**Process**: Agent A → Agent B → Agent C → Final Result

#### 8. 🏗️ HIERARCHICAL Strategy
```python
result = await orchestrator.orchestrate_task(
    task="Complex project coordination",
    strategy=OrchestrationStrategy.HIERARCHICAL
)
```
**Structure**: Lead Agent coordinates multiple sub-agents

#### 9. 🤝 COLLABORATIVE Strategy
```python
result = await orchestrator.orchestrate_task(
    task="Team-based research project",
    strategy=OrchestrationStrategy.COLLABORATIVE
)
```
**Features**: Real-time agent coordination and information sharing

## 🏭 Agent Factory (Actual Implementation)

**Location**: `src/agents/factory.py:AgentFactory`

### Agent Creation Patterns

#### Individual Agent Creation
```python
from src.agents import AgentFactory

factory = AgentFactory(
    logger=logger,
    session_service=session_service,
    memory_service=memory_service,
    mcp_orchestrator=mcp_orchestrator
)

# Create LLM agent with auto-optimization
research_agent = factory.create_llm_agent(
    role=LLMRole.RESEARCHER,
    auto_optimize_model=True,
    enable_thinking=True,
    priority_cost=False  # Use best model
)

# CRITICAL: Activate agent before orchestration
await research_agent.activate()

# Create custom domain expert
ai_expert = factory.create_custom_agent(
    agent_type=CustomAgentType.DOMAIN_EXPERT,
    domain="artificial_intelligence",
    custom_config={
        "specialization": "machine_learning",
        "experience_level": "expert",
        "focus_areas": ["nlp", "computer_vision", "deep_learning"]
    }
)
```

#### Agent Teams (Predefined Suites)
```python
# Research team with optimal configuration
research_team = factory.create_agent_suite(
    suite_type=AgentSuite.RESEARCH_TEAM
)
# Creates: RESEARCHER + ANALYST + SYNTHESIZER

# Content creation team
content_team = factory.create_agent_suite(
    suite_type=AgentSuite.CONTENT_CREATION
)
# Creates: CREATIVE + COMMUNICATOR + CRITIC

# Data analysis team
data_team = factory.create_agent_suite(
    suite_type=AgentSuite.DATA_ANALYSIS
)
# Creates: DATA_ANALYST + FACT_CHECKER + CRITIC
```

### Available Agent Suites (8 Predefined Teams)

1. **RESEARCH_TEAM**: Comprehensive research capabilities
2. **CONTENT_CREATION**: Content development and optimization
3. **DATA_ANALYSIS**: Statistical analysis and validation
4. **FACT_CHECKING**: Verification and validation specialists
5. **GENERAL_PURPOSE**: Balanced general capabilities
6. **DOMAIN_EXPERTS**: Specialized domain knowledge
7. **WORKFLOW_AUTOMATION**: Process and workflow management
8. **QA_SPECIALISTS**: Question-answering optimization

## 🔧 Advanced Configuration (Gemini 2.5 Integration)

### LLM Agent Configuration (Actual Implementation)
```python
@dataclass
class LLMAgentConfig:
    # Model selection
    model: Optional[GeminiModel] = None  # Auto-select if None
    role: LLMRole = LLMRole.GENERALIST
    
    # Generation parameters
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 0.95
    top_k: int = 64
    
    # Advanced Gemini 2.5 features
    enable_thinking: bool = True
    thinking_budget: Optional[int] = None  # -1 for model control
    enable_structured_output: bool = False
    output_schema_type: str = "analysis"
    
    # Model selection preferences
    priority_speed: bool = False
    priority_cost: bool = False
    auto_optimize_model: bool = True
    enable_cost_optimization: bool = False
```

### Intelligent Model Selection
```python
# Automatic model selection based on task complexity
complexity = analyze_task_complexity(task_description)
# TaskComplexity.SIMPLE → Gemini 2.5 Flash-Lite
# TaskComplexity.MEDIUM → Gemini 2.5 Flash  
# TaskComplexity.COMPLEX → Gemini 2.5 Flash (high budget)
# TaskComplexity.CRITICAL → Gemini 2.5 Pro

generation_config = select_model_for_task(
    task=task_description,
    preferences=agent_config.get_model_preferences()
)
```

### Performance Monitoring
```python
# Agent performance metrics (actual implementation)
metrics = agent.get_performance_metrics()
# Returns:
{
    "total_tasks": 42,
    "successful_tasks": 38,
    "failed_tasks": 4,
    "success_rate_percent": 90.5,
    "average_response_time_ms": 2847.3,
    "total_tokens_used": 156789,
    "conversation_length": 12
}

# Registry status (thread-safe)
status = AgentRegistry.get_registry_status()
# Returns:
{
    "total_agents": 15,
    "agents_by_type": {"llm": 9, "custom": 4, "workflow": 2},
    "agents_by_capability": {"research": 5, "analysis": 8, ...},
    "active_agents": 12
}
```

## 🎯 Advanced Features (Current Implementation)

### Memory Integration
```python
# Store results in agent memory
await agent.store_memory(
    content="Research findings about quantum computing applications",
    metadata={
        "task_type": "research",
        "domain": "technology",
        "quality_score": 0.95
    }
)

# Retrieve relevant memories
relevant_memories = await agent.retrieve_memory(
    query="quantum computing performance",
    limit=5
)
```

### MCP Server Integration
```python
# Use MCP servers through agents
result = await agent.use_mcp_server(
    server_name="perplexity",
    operation="search",
    parameters={
        "query": "latest AI research developments",
        "strategy": "HYBRID_VALIDATION"
    }
)

# Advanced analysis with source verification
analysis = await agent.analyze_with_sources(
    content="Claims about AI performance improvements",
    analysis_type="comprehensive"
)
# Returns: analysis + source verification + confidence score
```

### Conversation Management
```python
# Conversational interaction with context
response = await agent.chat(
    message="What are the implications of this research?",
    maintain_context=True  # Includes conversation history
)

# Clear conversation history
agent.clear_conversation_history()
```

## 📊 Best Practices (Implementation-Based)

### Agent Selection Guidelines

1. **Task-Capability Matching**:
   ```python
   # Find agents with specific capabilities
   capable_agents = AgentRegistry.find_capable_agents(
       required_capabilities=[
           AgentCapability.RESEARCH,
           AgentCapability.FACT_CHECKING
       ]
   )
   ```

2. **Performance-Based Selection**:
   ```python
   # Use orchestrator for intelligent selection
   result = await orchestrator.orchestrate_task(
       task=complex_task,
       strategy=OrchestrationStrategy.ADAPTIVE  # Auto-selects best approach
   )
   ```

3. **Cost-Performance Optimization**:
   ```python
   # Cost-optimized agent for simple tasks
   cost_agent = factory.create_llm_agent(
       role=LLMRole.GENERALIST,
       priority_cost=True,
       enable_cost_optimization=True
   )
   
   # Performance-optimized agent for critical tasks
   performance_agent = factory.create_llm_agent(
       role=LLMRole.SPECIALIST,
       model=GeminiModel.PRO,
       priority_cost=False
   )
   ```

### Configuration Recommendations

1. **Development Environment**:
   ```python
   dev_config = LLMAgentConfig(
       enable_thinking=True,
       temperature=0.3,
       auto_optimize_model=True,
       priority_speed=True
   )
   ```

2. **Production Environment**:
   ```python
   prod_config = LLMAgentConfig(
       enable_thinking=True,
       enable_structured_output=True,
       priority_cost=False,
       enable_cost_optimization=False
   )
   ```

### Error Handling and Reliability
```python
# Proper error handling pattern
try:
    result = await agent.execute_task(task)
    if result.success:
        # Process successful result
        process_result(result.result)
    else:
        # Handle agent failure
        logger.error(f"Agent task failed: {result.error}")
        
except Exception as e:
    # Handle unexpected errors
    logger.error(f"Unexpected error: {e}")
```

## 🔍 Integration Patterns

### Service Integration
```python
# Agents automatically integrate with services
agent = factory.create_llm_agent(
    role=LLMRole.RESEARCHER,
    # Services injected automatically
)
# Agent has access to: session_service, memory_service, 
# artifact_service, mcp_orchestrator, logger
```

### Tool Access
```python
# ADK tool access
adk_result = await agent.use_adk_tool(
    tool_name="google_search",
    parameters={"query": "AI research trends"}
)

# MCP server access
mcp_result = await agent.use_mcp_server(
    server_name="tavily",
    operation="search",
    parameters={"query": "quantum computing news"}
)
```

---

This agent documentation reflects the **actual implementation** of the Multi-Agent Research Platform agent system as it exists today, providing accurate information for developers working with the platform's sophisticated multi-agent capabilities.