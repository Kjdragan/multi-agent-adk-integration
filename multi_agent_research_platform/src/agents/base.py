"""
Base Agent System for Multi-Agent Research Platform

Provides the foundational classes and interfaces for all agent types,
integrating with ADK tools and MCP servers.
"""

import time
import asyncio
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable, Set
from enum import Enum
import uuid

from google.adk.agents import Agent as ADKAgent

from ..platform_logging import RunLogger
from ..services import SessionService, MemoryService, ArtifactService
from ..context import ToolContextPattern, MemoryAccessPattern
from ..mcp import MCPOrchestrator


class AgentType(str, Enum):
    """Types of agents in the multi-agent system."""
    LLM = "llm"                        # Language model-based agents
    WORKFLOW = "workflow"              # Workflow orchestration agents  
    CUSTOM = "custom"                  # Custom specialized agents
    HYBRID = "hybrid"                  # Hybrid agents combining multiple types


class AgentCapability(str, Enum):
    """Capabilities that agents can possess."""
    REASONING = "reasoning"            # Logical reasoning and analysis
    RESEARCH = "research"              # Information gathering and research
    ANALYSIS = "analysis"              # Data and content analysis
    SYNTHESIS = "synthesis"            # Information synthesis and summarization
    PLANNING = "planning"              # Task planning and decomposition
    EXECUTION = "execution"            # Task execution and orchestration
    COMMUNICATION = "communication"    # Agent-to-agent communication
    LEARNING = "learning"              # Learning from interactions
    TOOL_USE = "tool_use"             # Using external tools and APIs
    MEMORY_ACCESS = "memory_access"    # Accessing and storing memories
    CONTEXT_MANAGEMENT = "context_management"  # Managing conversation context
    FACT_CHECKING = "fact_checking"    # Verifying information accuracy
    CONTENT_GENERATION = "content_generation"  # Creating new content
    DECISION_MAKING = "decision_making"  # Making informed decisions


@dataclass
class AgentResult:
    """Result from an agent operation."""
    agent_id: str
    agent_type: AgentType
    task: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    context_used: Dict[str, Any] = field(default_factory=dict)
    tools_used: List[str] = field(default_factory=list)
    memory_accessed: bool = False
    artifacts_created: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "task": self.task,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
            "context_used": self.context_used,
            "tools_used": self.tools_used,
            "memory_accessed": self.memory_accessed,
            "artifacts_created": self.artifacts_created,
        }


class Agent(ABC):
    """
    Base class for all agents in the multi-agent research platform.
    
    Provides common functionality for ADK integration, MCP server access,
    context management, and service integration.
    """
    
    def __init__(self,
                 agent_id: Optional[str] = None,
                 name: str = "",
                 description: str = "",
                 agent_type: AgentType = AgentType.CUSTOM,
                 capabilities: Optional[List[AgentCapability]] = None,
                 logger: Optional[RunLogger] = None,
                 session_service: Optional[SessionService] = None,
                 memory_service: Optional[MemoryService] = None,
                 artifact_service: Optional[ArtifactService] = None,
                 mcp_orchestrator: Optional[MCPOrchestrator] = None):
        
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = name or f"Agent-{self.agent_id[:8]}"
        self.description = description
        self.agent_type = agent_type
        self.capabilities = capabilities or []
        
        # Service integration
        self.logger = logger
        self.session_service = session_service
        self.memory_service = memory_service
        self.artifact_service = artifact_service
        self.mcp_orchestrator = mcp_orchestrator
        
        # Context patterns
        self.tool_pattern = ToolContextPattern(
            logger=logger,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service,
        )
        
        self.memory_pattern = MemoryAccessPattern(
            logger=logger,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service,
        )
        
        # Agent state
        self.is_active = False
        self.total_tasks_completed = 0
        self.last_task_time = None
        
        # ADK agent instance (can be set by subclasses)
        self.adk_agent: Optional[ADKAgent] = None
        
        # Register with agent registry
        AgentRegistry.register(self)
    
    @abstractmethod
    async def execute_task(self, 
                          task: str, 
                          context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Execute a task with the agent.
        
        Args:
            task: Task description
            context: Optional context information
            
        Returns:
            Agent result with task outcome
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[AgentCapability]:
        """Get the capabilities of this agent."""
        pass
    
    def has_capability(self, capability: AgentCapability) -> bool:
        """Check if agent has a specific capability."""
        return capability in self.get_capabilities()
    
    def can_handle_task(self, task: str, required_capabilities: List[AgentCapability]) -> bool:
        """Check if agent can handle a task with required capabilities."""
        agent_capabilities = set(self.get_capabilities())
        required_capabilities_set = set(required_capabilities)
        return required_capabilities_set.issubset(agent_capabilities)
    
    async def activate(self) -> bool:
        """Activate the agent."""
        try:
            if self.logger:
                self.logger.info(f"Activating agent {self.name} ({self.agent_id})")
            
            self.is_active = True
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to activate agent {self.name}: {e}")
            return False
    
    async def deactivate(self) -> bool:
        """Deactivate the agent."""
        try:
            if self.logger:
                self.logger.info(f"Deactivating agent {self.name} ({self.agent_id})")
            
            self.is_active = False
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to deactivate agent {self.name}: {e}")
            return False
    
    async def use_adk_tool(self, 
                          tool_name: str, 
                          parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use an ADK built-in tool.
        
        Args:
            tool_name: Name of the ADK tool
            parameters: Tool parameters
            
        Returns:
            Tool execution result
        """
        if not self.adk_agent:
            raise ValueError("No ADK agent configured for this agent")
        
        try:
            # This would integrate with ADK's tool execution
            # For now, this is a placeholder for the actual ADK integration
            if self.logger:
                self.logger.info(f"Agent {self.name} using ADK tool: {tool_name}")
            
            # Placeholder result
            return {
                "tool": tool_name,
                "parameters": parameters,
                "result": "ADK tool execution placeholder",
                "success": True,
            }
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"ADK tool {tool_name} execution failed: {e}")
            return {
                "tool": tool_name,
                "parameters": parameters,
                "error": str(e),
                "success": False,
            }
    
    async def use_mcp_server(self, 
                           server_name: str, 
                           operation: str,
                           parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use an MCP server through the orchestrator.
        
        Args:
            server_name: Name of the MCP server
            operation: Operation to perform
            parameters: Operation parameters
            
        Returns:
            MCP server operation result
        """
        if not self.mcp_orchestrator:
            raise ValueError("No MCP orchestrator configured for this agent")
        
        try:
            if self.logger:
                self.logger.info(f"Agent {self.name} using MCP server: {server_name}")
            
            # Use the orchestrator to access MCP servers
            if operation == "search":
                from ..mcp.orchestrator import SearchContext, SearchStrategy
                
                search_context = SearchContext(
                    query=parameters.get("query", ""),
                    **parameters.get("context", {})
                )
                
                strategy = SearchStrategy[parameters.get("strategy", "ADAPTIVE")]
                result = await self.mcp_orchestrator.search(search_context, strategy)
                
                return {
                    "server": server_name,
                    "operation": operation,
                    "result": result.to_dict(),
                    "success": True,
                }
            else:
                # Direct server access for other operations
                servers = self.mcp_orchestrator.available_servers
                server = servers.get(server_name)
                
                if not server:
                    raise ValueError(f"MCP server {server_name} not available")
                
                # This would be extended based on specific server operations
                return {
                    "server": server_name,
                    "operation": operation,
                    "result": "MCP server operation placeholder",
                    "success": True,
                }
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"MCP server {server_name} operation failed: {e}")
            return {
                "server": server_name,
                "operation": operation,
                "error": str(e),
                "success": False,
            }
    
    async def store_memory(self, 
                          content: str, 
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store information in agent memory."""
        try:
            if self.memory_service:
                await self.memory_service.store(
                    text=content,
                    metadata={
                        **(metadata or {}),
                        "agent_id": self.agent_id,
                        "agent_name": self.name,
                        "timestamp": time.time(),
                    }
                )
                return True
            return False
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Memory storage failed for agent {self.name}: {e}")
            return False
    
    async def retrieve_memory(self, 
                            query: str, 
                            limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve relevant memories."""
        try:
            if self.memory_service:
                memories = await self.memory_service.search(
                    query=query,
                    filters={"agent_id": self.agent_id},
                    limit=limit
                )
                return memories
            return []
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Memory retrieval failed for agent {self.name}: {e}")
            return []
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "agent_type": self.agent_type.value,
            "capabilities": [cap.value for cap in self.get_capabilities()],
            "is_active": self.is_active,
            "total_tasks_completed": self.total_tasks_completed,
            "last_task_time": self.last_task_time,
            "has_adk_agent": self.adk_agent is not None,
            "has_mcp_orchestrator": self.mcp_orchestrator is not None,
            "services": {
                "session_service": self.session_service is not None,
                "memory_service": self.memory_service is not None,
                "artifact_service": self.artifact_service is not None,
            },
        }


class AgentRegistry:
    """Thread-safe registry for managing agents in the system."""
    
    _agents: Dict[str, Agent] = {}
    _agents_by_type: Dict[AgentType, List[Agent]] = {}
    _agents_by_capability: Dict[AgentCapability, List[Agent]] = {}
    _lock = threading.RLock()  # Use RLock for nested calls
    
    @classmethod
    def register(cls, agent: Agent) -> None:
        """Thread-safe agent registration."""
        with cls._lock:
            # Check for duplicate registration
            if agent.agent_id in cls._agents:
                existing = cls._agents[agent.agent_id]
                if existing is not agent:
                    # Different agent with same ID - unregister old one first
                    cls._unregister_locked(agent.agent_id)
            
            cls._agents[agent.agent_id] = agent
            
            # Index by type (atomic)
            if agent.agent_type not in cls._agents_by_type:
                cls._agents_by_type[agent.agent_type] = []
            cls._agents_by_type[agent.agent_type].append(agent)
            
            # Index by capabilities (atomic)
            for capability in agent.get_capabilities():
                if capability not in cls._agents_by_capability:
                    cls._agents_by_capability[capability] = []
                cls._agents_by_capability[capability].append(agent)
    
    @classmethod
    def unregister(cls, agent_id: str) -> bool:
        """Thread-safe agent unregistration."""
        with cls._lock:
            return cls._unregister_locked(agent_id)
    
    @classmethod
    def _unregister_locked(cls, agent_id: str) -> bool:
        """Internal unregister method (assumes lock is held)."""
        if agent_id in cls._agents:
            agent = cls._agents[agent_id]
            
            # Remove from type index
            if agent.agent_type in cls._agents_by_type:
                try:
                    cls._agents_by_type[agent.agent_type].remove(agent)
                    # Clean up empty lists to prevent memory leaks
                    if not cls._agents_by_type[agent.agent_type]:
                        del cls._agents_by_type[agent.agent_type]
                except ValueError:
                    pass  # Agent wasn't in list (shouldn't happen but be defensive)
            
            # Remove from capability index
            for capability in agent.get_capabilities():
                if capability in cls._agents_by_capability:
                    try:
                        cls._agents_by_capability[capability].remove(agent)
                        # Clean up empty lists to prevent memory leaks
                        if not cls._agents_by_capability[capability]:
                            del cls._agents_by_capability[capability]
                    except ValueError:
                        pass  # Agent wasn't in list (shouldn't happen but be defensive)
            
            del cls._agents[agent_id]
            return True
        
        return False
    
    @classmethod
    def get_agent(cls, agent_id: str) -> Optional[Agent]:
        """Thread-safe agent lookup by ID."""
        with cls._lock:
            return cls._agents.get(agent_id)
    
    @classmethod
    def get_agents_by_type(cls, agent_type: AgentType) -> List[Agent]:
        """Thread-safe agent lookup by type."""
        with cls._lock:
            return cls._agents_by_type.get(agent_type, []).copy()  # Return copy to avoid external modification
    
    @classmethod
    def get_agents_by_capability(cls, capability: AgentCapability) -> List[Agent]:
        """Thread-safe agent lookup by capability."""
        with cls._lock:
            return cls._agents_by_capability.get(capability, []).copy()  # Return copy to avoid external modification
    
    @classmethod
    def find_capable_agents(cls, 
                          required_capabilities: List[AgentCapability],
                          agent_type: Optional[AgentType] = None) -> List[Agent]:
        """Thread-safe search for agents with required capabilities."""
        with cls._lock:
            capable_agents = []
            
            agents_to_check = (
                cls._agents_by_type.get(agent_type, []) if agent_type
                else list(cls._agents.values())
            )
            
            for agent in agents_to_check:
                if agent.can_handle_task("", required_capabilities):
                    capable_agents.append(agent)
            
            return capable_agents
    
    @classmethod
    def get_all_agents(cls) -> List[Agent]:
        """Thread-safe retrieval of all registered agents."""
        with cls._lock:
            return list(cls._agents.values())
    
    @classmethod
    def get_registry_status(cls) -> Dict[str, Any]:
        """Thread-safe registry status retrieval."""
        with cls._lock:
            return {
                "total_agents": len(cls._agents),
                "agents_by_type": {
                    agent_type.value: len(agents)
                    for agent_type, agents in cls._agents_by_type.items()
                },
                "agents_by_capability": {
                    capability.value: len(agents)
                    for capability, agents in cls._agents_by_capability.items()
                },
                "active_agents": sum(1 for agent in cls._agents.values() if agent.is_active),
            }
    
    @classmethod
    def clear(cls) -> None:
        """Thread-safe registry clear operation."""
        with cls._lock:
            cls._agents.clear()
            cls._agents_by_type.clear()
            cls._agents_by_capability.clear()