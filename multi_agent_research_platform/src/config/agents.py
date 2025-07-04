"""
Agent configuration models.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator

from .base import BaseConfig


class AgentType(str, Enum):
    """Types of agents supported by the platform."""
    LLM = "llm"
    WORKFLOW_SEQUENTIAL = "workflow_sequential"
    WORKFLOW_PARALLEL = "workflow_parallel"
    WORKFLOW_LOOP = "workflow_loop"
    CUSTOM = "custom"


class ContextType(str, Enum):
    """Types of context usage for agents."""
    INVOCATION = "invocation"      # Full InvocationContext
    CALLBACK = "callback"          # CallbackContext
    TOOL = "tool"                  # ToolContext
    READONLY = "readonly"          # ReadonlyContext


class ModelProvider(str, Enum):
    """Supported model providers."""
    GOOGLE_GEMINI = "google_gemini"
    GOOGLE_VERTEX = "google_vertex"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class AgentConfig(BaseModel):
    """Base configuration for all agents."""
    model_config = ConfigDict(extra="ignore")
    
    # Basic identification
    name: str = Field(description="Unique agent name")
    agent_type: AgentType = Field(description="Type of agent")
    description: str = Field(description="Agent description for routing decisions")
    
    # Context management
    primary_context_type: ContextType = Field(
        default=ContextType.INVOCATION,
        description="Primary context type this agent uses"
    )
    
    # Execution settings
    enabled: bool = Field(default=True, description="Whether agent is enabled")
    timeout_seconds: int = Field(default=300, description="Agent execution timeout")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    
    # State management
    state_prefix: Optional[str] = Field(
        default=None,
        description="State key prefix for this agent's data"
    )
    output_key: Optional[str] = Field(
        default=None,
        description="State key to store agent output"
    )
    
    # Tool assignments
    tool_names: List[str] = Field(
        default_factory=list,
        description="Names of tools this agent can use"
    )
    
    # Logging and monitoring
    log_level: str = Field(default="INFO", description="Logging level for this agent")
    enable_performance_tracking: bool = Field(
        default=True,
        description="Enable performance tracking"
    )
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Agent tags for organization")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if not v or not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Agent name must be alphanumeric with underscores/hyphens')
        if v in ['user', 'system', 'assistant']:
            raise ValueError('Agent name cannot be a reserved name')
        return v


class LLMAgentConfig(AgentConfig):
    """Configuration for LLM-powered agents."""
    
    agent_type: AgentType = Field(default=AgentType.LLM, description="Agent type")
    
    # Model configuration
    model_provider: ModelProvider = Field(
        default=ModelProvider.GOOGLE_GEMINI,
        description="LLM provider"
    )
    model_name: str = Field(
        default="gemini-2.0-flash",
        description="Model name/identifier"
    )
    
    # LLM behavior
    instruction: str = Field(description="Agent instruction/prompt")
    instruction_provider: Optional[str] = Field(
        default=None,
        description="Function name for dynamic instruction generation"
    )
    global_instruction: Optional[str] = Field(
        default=None,
        description="Global instruction applied to all agents"
    )
    
    # Content handling
    include_contents: str = Field(
        default="default",
        description="Content inclusion strategy: 'default', 'none'"
    )
    
    # Generation parameters
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Model temperature"
    )
    max_output_tokens: Optional[int] = Field(
        default=None,
        description="Maximum output tokens"
    )
    top_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Top-p sampling parameter"
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Top-k sampling parameter"
    )
    
    # Input/Output schemas
    input_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Expected input schema"
    )
    output_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Expected output schema"
    )
    
    # Advanced features
    enable_planning: bool = Field(
        default=False,
        description="Enable multi-step planning"
    )
    enable_code_execution: bool = Field(
        default=False,
        description="Enable code execution capabilities"
    )
    
    # Transfer control
    disallow_transfer_to_parent: bool = Field(
        default=False,
        description="Disallow transfer to parent agent"
    )
    disallow_transfer_to_peers: bool = Field(
        default=False,
        description="Disallow transfer to peer agents"
    )
    
    @field_validator('instruction')
    @classmethod
    def validate_instruction(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError('Instruction must be at least 10 characters')
        return v


class WorkflowAgentConfig(AgentConfig):
    """Configuration for workflow agents (Sequential, Parallel, Loop)."""
    
    # Sub-agent configuration
    sub_agent_names: List[str] = Field(
        description="Names of sub-agents to execute"
    )
    
    # Workflow-specific settings
    max_iterations: Optional[int] = Field(
        default=None,
        description="Maximum iterations for loop agents"
    )
    parallel_execution: bool = Field(
        default=False,
        description="Whether to execute sub-agents in parallel"
    )
    
    # Error handling
    continue_on_error: bool = Field(
        default=False,
        description="Continue execution if sub-agent fails"
    )
    error_escalation_strategy: str = Field(
        default="stop",
        description="Error escalation strategy: 'stop', 'continue', 'retry'"
    )
    
    # State management between sub-agents
    shared_state_keys: List[str] = Field(
        default_factory=list,
        description="State keys shared between sub-agents"
    )
    
    @field_validator('sub_agent_names')
    @classmethod
    def validate_sub_agents(cls, v):
        if not v:
            raise ValueError('Workflow agent must have at least one sub-agent')
        return v
    
    @field_validator('error_escalation_strategy')
    @classmethod
    def validate_error_strategy(cls, v):
        allowed = ['stop', 'continue', 'retry']
        if v not in allowed:
            raise ValueError(f'Error escalation strategy must be one of: {allowed}')
        return v


class CustomAgentConfig(AgentConfig):
    """Configuration for custom agents."""
    
    agent_type: AgentType = Field(default=AgentType.CUSTOM, description="Agent type")
    
    # Implementation details
    implementation_class: str = Field(
        description="Fully qualified class name for custom agent"
    )
    implementation_module: str = Field(
        description="Module containing the custom agent class"
    )
    
    # Custom configuration
    custom_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom configuration passed to agent"
    )
    
    # Initialization parameters
    init_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters passed to agent constructor"
    )
    
    @field_validator('implementation_class')
    @classmethod
    def validate_implementation_class(cls, v):
        if not v or '.' not in v:
            raise ValueError('Implementation class must be fully qualified')
        return v


class AgentRegistry(BaseConfig):
    """Registry of all agents in the system."""
    
    # Agent definitions
    agents: Dict[str, Union[LLMAgentConfig, WorkflowAgentConfig, CustomAgentConfig]] = Field(
        default_factory=dict,
        description="Registry of agent configurations"
    )
    
    # Default agents
    default_root_agent: str = Field(
        default="research_coordinator",
        description="Default root agent name"
    )
    
    # Agent groups
    agent_groups: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Logical groupings of agents"
    )
    
    # Global agent settings
    global_instruction: Optional[str] = Field(
        default=None,
        description="Global instruction applied to all LLM agents"
    )
    global_timeout: int = Field(
        default=300,
        description="Global timeout for all agents"
    )
    
    def add_agent(self, config: AgentConfig) -> None:
        """Add an agent to the registry."""
        self.agents[config.name] = config
    
    def get_agent(self, name: str) -> Optional[AgentConfig]:
        """Get an agent configuration by name."""
        return self.agents.get(name)
    
    def get_agents_by_type(self, agent_type: AgentType) -> List[AgentConfig]:
        """Get all agents of a specific type."""
        return [
            agent for agent in self.agents.values()
            if agent.agent_type == agent_type
        ]
    
    def get_agents_by_tag(self, tag: str) -> List[AgentConfig]:
        """Get all agents with a specific tag."""
        return [
            agent for agent in self.agents.values()
            if tag in agent.tags
        ]
    
    def validate_agent_dependencies(self) -> None:
        """Validate that all agent dependencies are satisfied."""
        all_agent_names = set(self.agents.keys())
        
        for agent in self.agents.values():
            if isinstance(agent, WorkflowAgentConfig):
                # Check sub-agent references
                for sub_agent_name in agent.sub_agent_names:
                    if sub_agent_name not in all_agent_names:
                        raise ValueError(
                            f"Agent '{agent.name}' references unknown sub-agent '{sub_agent_name}'"
                        )
        
        # Check default root agent
        if self.default_root_agent not in all_agent_names:
            raise ValueError(f"Default root agent '{self.default_root_agent}' not found")
    
    def get_agent_execution_order(self) -> List[str]:
        """Get recommended agent execution order based on dependencies."""
        # Simple topological sort for workflow agents
        visited = set()
        order = []
        
        def visit(agent_name: str):
            if agent_name in visited:
                return
            
            visited.add(agent_name)
            agent = self.agents.get(agent_name)
            
            if isinstance(agent, WorkflowAgentConfig):
                # Visit sub-agents first
                for sub_agent in agent.sub_agent_names:
                    visit(sub_agent)
            
            order.append(agent_name)
        
        # Visit all agents
        for agent_name in self.agents.keys():
            visit(agent_name)
        
        return order
    
    def create_default_agents(self) -> None:
        """Create a default set of agents for the platform."""
        # Research Coordinator (Root LLM Agent)
        self.add_agent(LLMAgentConfig(
            name="research_coordinator",
            description="Master orchestrator for research workflows",
            instruction="""You are the Research Coordinator, responsible for managing complex research workflows.
            
Your primary responsibilities:
1. Analyze user research requests and determine optimal approach
2. Coordinate multiple specialized agents for comprehensive research
3. Ensure quality and completeness of research outputs
4. Manage workflow state and handle errors gracefully

Use your available tools and sub-agents effectively to provide thorough, accurate research results.""",
            tool_names=["agent_transfer", "workflow_control"],
            enable_performance_tracking=True,
            tags=["coordinator", "root", "llm"]
        ))
        
        # Tool Selection Agent
        self.add_agent(LLMAgentConfig(
            name="tool_selection_agent",
            description="Intelligent router for tool and agent selection",
            instruction="""You are the Tool Selection Agent, responsible for intelligent routing decisions.
            
Your role is to:
1. Analyze incoming queries to understand requirements
2. Evaluate available tools and agents for optimal match
3. Consider factors like data sensitivity, accuracy needs, and response time
4. Provide confidence scores for routing decisions

Make data-driven routing decisions to optimize research effectiveness.""",
            tool_names=["tool_analysis", "performance_metrics"],
            primary_context_type=ContextType.CALLBACK,
            tags=["router", "intelligence", "llm"]
        ))
        
        # Web Research Agent
        self.add_agent(LLMAgentConfig(
            name="web_research_agent",
            description="Multi-source web research specialist",
            instruction="""You are the Web Research Agent, specialized in comprehensive online research.
            
Your capabilities:
1. Use Google Search for broad web research
2. Access MCP servers (Perplexity, Tavily, Brave) for specialized search
3. Evaluate source credibility and cross-reference information
4. Provide well-cited, accurate research results

Always prioritize accuracy and provide proper citations for your findings.""",
            tool_names=["google_search", "perplexity_search", "tavily_search", "brave_search"],
            primary_context_type=ContextType.TOOL,
            tags=["research", "web", "llm"]
        ))
        
        # Parallel Research Workflow
        self.add_agent(WorkflowAgentConfig(
            name="parallel_research_workflow",
            agent_type=AgentType.WORKFLOW_PARALLEL,
            description="Executes multiple research tasks concurrently",
            sub_agent_names=["web_research_agent", "data_analysis_agent", "knowledge_base_agent"],
            parallel_execution=True,
            continue_on_error=True,
            shared_state_keys=["research_topic", "quality_threshold"],
            tags=["workflow", "parallel", "research"]
        ))
        
        # Quality Assurance Sequential Workflow
        self.add_agent(WorkflowAgentConfig(
            name="quality_assurance_workflow",
            agent_type=AgentType.WORKFLOW_SEQUENTIAL,
            description="Sequential quality assurance pipeline",
            sub_agent_names=["fact_checker_agent", "report_writer_agent", "final_reviewer_agent"],
            error_escalation_strategy="retry",
            tags=["workflow", "sequential", "quality"]
        ))
    
    @field_validator('agents')
    @classmethod
    def validate_agents_dict(cls, v):
        # Ensure all agent names in the dict match their config names
        for name, agent in v.items():
            if agent.name != name:
                raise ValueError(f"Agent dict key '{name}' doesn't match agent name '{agent.name}'")
        return v