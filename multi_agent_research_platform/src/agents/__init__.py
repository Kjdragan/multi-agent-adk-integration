"""
Multi-Agent System for Research Platform

Provides a comprehensive multi-agent framework integrating Google ADK
built-in tools with MCP server capabilities for complex research workflows.
"""

from .base import Agent, AgentType, AgentCapability, AgentResult, AgentRegistry
from .llm_agent import LLMAgent, LLMAgentConfig, LLMRole
from .workflow_agent import WorkflowAgent, WorkflowStep, WorkflowConfig
from .custom_agent import CustomAgent, CustomAgentConfig, CustomAgentType
from .orchestrator import AgentOrchestrator, OrchestrationStrategy, TaskAllocation, TaskPriority
from .factory import AgentFactory, AgentSuite, create_agent, create_research_team

__all__ = [
    # Base agent system
    "Agent",
    "AgentType",
    "AgentCapability", 
    "AgentResult",
    "AgentRegistry",
    
    # LLM Agents
    "LLMAgent",
    "LLMAgentConfig",
    "LLMRole",
    
    # Workflow Agents
    "WorkflowAgent", 
    "WorkflowStep",
    "WorkflowConfig",
    
    # Custom Agents
    "CustomAgent",
    "CustomAgentConfig",
    "CustomAgentType",
    
    # Orchestration
    "AgentOrchestrator",
    "OrchestrationStrategy", 
    "TaskAllocation",
    "TaskPriority",
    
    # Factory and utilities
    "AgentFactory",
    "AgentSuite",
    "create_agent",
    "create_research_team",
]