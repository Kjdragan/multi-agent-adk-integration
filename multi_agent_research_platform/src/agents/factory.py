"""
Agent Factory for Easy Creation and Management

Provides factory methods, templates, and utilities for creating and managing
different types of agents and agent teams for various use cases.
"""

import uuid
import asyncio
from typing import Any, Dict, List, Optional, Type, Union, Callable
from enum import Enum

from .base import Agent, AgentType, AgentCapability, AgentRegistry
from .llm_agent import LLMAgent, LLMAgentConfig, LLMRole
from .workflow_agent import WorkflowAgent
from .custom_agent import CustomAgent, CustomAgentConfig, CustomAgentType
from .orchestrator import AgentOrchestrator
from ..platform_logging import RunLogger
from ..services import SessionService, MemoryService, ArtifactService
from ..mcp import MCPOrchestrator
from ..config.gemini_models import (
    GeminiModel, TaskComplexity, ModelSelector, select_model_for_task,
    analyze_task_complexity
)


class AgentSuite(str, Enum):
    """Predefined agent suites for different use cases."""
    RESEARCH_TEAM = "research_team"           # Comprehensive research team
    CONTENT_CREATION = "content_creation"     # Content creation specialists
    DATA_ANALYSIS = "data_analysis"           # Data analysis team
    FACT_CHECKING = "fact_checking"           # Fact-checking and validation
    GENERAL_PURPOSE = "general_purpose"       # General-purpose agents
    DOMAIN_EXPERTS = "domain_experts"         # Domain-specific experts
    WORKFLOW_AUTOMATION = "workflow_automation" # Workflow and automation
    QA_SPECIALISTS = "qa_specialists"         # Question-answering specialists


class AgentFactory:
    """
    Factory for creating and managing agents and agent teams.
    
    Provides templates, configurations, and utilities for easy agent creation
    and management with proper service integration.
    """
    
    def __init__(self,
                 logger: Optional[RunLogger] = None,
                 session_service: Optional[SessionService] = None,
                 memory_service: Optional[MemoryService] = None,
                 artifact_service: Optional[ArtifactService] = None,
                 mcp_orchestrator: Optional[MCPOrchestrator] = None):
        
        self.logger = logger
        self.session_service = session_service
        self.memory_service = memory_service
        self.artifact_service = artifact_service
        self.mcp_orchestrator = mcp_orchestrator
        
        # Created agents tracking
        self.created_agents: Dict[str, Agent] = {}
        self.agent_teams: Dict[str, List[Agent]] = {}
        
        # Default configurations
        self.default_llm_configs = self._get_default_llm_configs()
        self.default_custom_configs = self._get_default_custom_configs()
        
        # Model selector for intelligent model selection
        self.model_selector = ModelSelector()
        
        # Agent templates
        self.agent_templates = self._load_agent_templates()
    
    def create_llm_agent(self,
                        role: LLMRole,
                        model: Optional[GeminiModel] = None,
                        name: Optional[str] = None,
                        tools: Optional[List[Callable]] = None,
                        custom_config: Optional[Dict[str, Any]] = None,
                        auto_optimize_model: bool = True,
                        enable_thinking: bool = True,
                        enable_structured_output: bool = False,
                        priority_speed: bool = False,
                        priority_cost: bool = False) -> LLMAgent:
        """
        Create an LLM agent with specified role and intelligent model selection.
        
        Args:
            role: LLM agent role
            model: Specific model to use (None for auto-selection)
            name: Optional custom name
            tools: Optional custom tools
            custom_config: Optional custom configuration
            auto_optimize_model: Enable automatic model optimization
            enable_thinking: Enable thinking capabilities
            enable_structured_output: Enable structured output
            priority_speed: Prioritize speed over intelligence
            priority_cost: Prioritize cost over performance
            
        Returns:
            Configured LLM agent
        """
        # Get default config for role
        base_config = self.default_llm_configs.get(role, LLMAgentConfig())
        
        # Set Gemini 2.5 specific features
        base_config.model = model  # None allows auto-selection
        base_config.role = role
        base_config.auto_optimize_model = auto_optimize_model
        base_config.enable_thinking = enable_thinking
        base_config.enable_structured_output = enable_structured_output
        base_config.priority_speed = priority_speed
        base_config.priority_cost = priority_cost
        
        # Override with custom config if provided
        if custom_config:
            for key, value in custom_config.items():
                if hasattr(base_config, key):
                    setattr(base_config, key, value)
        
        # Create agent
        agent = LLMAgent(
            config=base_config,
            tools=tools,
            name=name,
            logger=self.logger,
            session_service=self.session_service,
            memory_service=self.memory_service,
            artifact_service=self.artifact_service,
            mcp_orchestrator=self.mcp_orchestrator,
        )
        
        # Track created agent
        self.created_agents[agent.agent_id] = agent
        
        if self.logger:
            model_info = model.value if model else "auto-selected"
            self.logger.info(
                f"Created LLM agent: {agent.name} (role: {role.value}, model: {model_info}, "
                f"thinking: {enable_thinking}, structured: {enable_structured_output})"
            )
        
        return agent
    
    def create_workflow_agent(self,
                             name: Optional[str] = None,
                             description: Optional[str] = None) -> WorkflowAgent:
        """
        Create a workflow orchestration agent.
        
        Args:
            name: Optional custom name
            description: Optional custom description
            
        Returns:
            Configured workflow agent
        """
        agent = WorkflowAgent(
            name=name,
            description=description,
            logger=self.logger,
            session_service=self.session_service,
            memory_service=self.memory_service,
            artifact_service=self.artifact_service,
            mcp_orchestrator=self.mcp_orchestrator,
        )
        
        # Track created agent
        self.created_agents[agent.agent_id] = agent
        
        if self.logger:
            self.logger.info(f"Created workflow agent: {agent.name}")
        
        return agent
    
    def create_custom_agent(self,
                           agent_type: CustomAgentType,
                           domain: str = "",
                           name: Optional[str] = None,
                           custom_tools: Optional[List[Callable]] = None,
                           custom_config: Optional[Dict[str, Any]] = None) -> CustomAgent:
        """
        Create a custom specialized agent.
        
        Args:
            agent_type: Type of custom agent
            domain: Domain of expertise
            name: Optional custom name
            custom_tools: Optional custom tools
            custom_config: Optional custom configuration
            
        Returns:
            Configured custom agent
        """
        # Get default config for type
        base_config = self.default_custom_configs.get(agent_type, CustomAgentConfig(agent_type))
        
        # Set domain
        base_config.domain = domain
        
        # Override with custom config if provided
        if custom_config:
            for key, value in custom_config.items():
                if hasattr(base_config, key):
                    setattr(base_config, key, value)
        
        # Create agent
        agent = CustomAgent(
            config=base_config,
            name=name,
            custom_tools=custom_tools,
            logger=self.logger,
            session_service=self.session_service,
            memory_service=self.memory_service,
            artifact_service=self.artifact_service,
            mcp_orchestrator=self.mcp_orchestrator,
        )
        
        # Track created agent
        self.created_agents[agent.agent_id] = agent
        
        if self.logger:
            self.logger.info(f"Created custom agent: {agent.name} (type: {agent_type.value})")
        
        return agent
    
    def create_agent_suite(self,
                          suite_type: AgentSuite,
                          custom_configs: Optional[Dict[str, Any]] = None) -> List[Agent]:
        """
        Create a predefined suite of agents for specific use cases.
        
        Args:
            suite_type: Type of agent suite to create
            custom_configs: Optional custom configurations
            
        Returns:
            List of configured agents in the suite
        """
        custom_configs = custom_configs or {}
        
        if suite_type == AgentSuite.RESEARCH_TEAM:
            return self._create_research_team(custom_configs)
        elif suite_type == AgentSuite.CONTENT_CREATION:
            return self._create_content_creation_team(custom_configs)
        elif suite_type == AgentSuite.DATA_ANALYSIS:
            return self._create_data_analysis_team(custom_configs)
        elif suite_type == AgentSuite.FACT_CHECKING:
            return self._create_fact_checking_team(custom_configs)
        elif suite_type == AgentSuite.GENERAL_PURPOSE:
            return self._create_general_purpose_team(custom_configs)
        elif suite_type == AgentSuite.DOMAIN_EXPERTS:
            return self._create_domain_experts_team(custom_configs)
        elif suite_type == AgentSuite.WORKFLOW_AUTOMATION:
            return self._create_workflow_automation_team(custom_configs)
        elif suite_type == AgentSuite.QA_SPECIALISTS:
            return self._create_qa_specialists_team(custom_configs)
        else:
            raise ValueError(f"Unknown agent suite type: {suite_type}")
    
    def create_orchestrator(self) -> AgentOrchestrator:
        """
        Create an agent orchestrator.
        
        Returns:
            Configured agent orchestrator
        """
        orchestrator = AgentOrchestrator(
            logger=self.logger,
            session_service=self.session_service,
            memory_service=self.memory_service,
            artifact_service=self.artifact_service,
            mcp_orchestrator=self.mcp_orchestrator,
        )
        
        if self.logger:
            self.logger.info("Created agent orchestrator")
        
        return orchestrator
    
    def activate_all_agents(self) -> int:
        """
        Activate all created agents.
        
        Returns:
            Number of agents activated
        """
        activated_count = 0
        
        for agent in self.created_agents.values():
            try:
                if asyncio.run(agent.activate()):
                    activated_count += 1
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to activate agent {agent.name}: {e}")
        
        if self.logger:
            self.logger.info(f"Activated {activated_count} agents")
        
        return activated_count
    
    def deactivate_all_agents(self) -> int:
        """
        Deactivate all created agents.
        
        Returns:
            Number of agents deactivated
        """
        deactivated_count = 0
        
        for agent in self.created_agents.values():
            try:
                if asyncio.run(agent.deactivate()):
                    deactivated_count += 1
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to deactivate agent {agent.name}: {e}")
        
        if self.logger:
            self.logger.info(f"Deactivated {deactivated_count} agents")
        
        return deactivated_count
    
    def get_created_agents(self) -> List[Agent]:
        """Get all agents created by this factory."""
        return list(self.created_agents.values())
    
    def get_agent_by_id(self, agent_id: str) -> Optional[Agent]:
        """Get agent by ID."""
        return self.created_agents.get(agent_id)
    
    def get_agents_by_type(self, agent_type: AgentType) -> List[Agent]:
        """Get agents by type."""
        return [
            agent for agent in self.created_agents.values()
            if agent.agent_type == agent_type
        ]
    
    def get_agents_by_capability(self, capability: AgentCapability) -> List[Agent]:
        """Get agents by capability."""
        return [
            agent for agent in self.created_agents.values()
            if capability in agent.get_capabilities()
        ]
    
    def _create_research_team(self, custom_configs: Dict[str, Any]) -> List[Agent]:
        """Create a comprehensive research team."""
        team = []
        
        # Research specialist
        researcher = self.create_llm_agent(
            role=LLMRole.RESEARCHER,
            name="Research Specialist",
            custom_config=custom_configs.get("researcher", {})
        )
        team.append(researcher)
        
        # Analysis specialist
        analyst = self.create_llm_agent(
            role=LLMRole.ANALYST,
            name="Analysis Specialist",
            custom_config=custom_configs.get("analyst", {})
        )
        team.append(analyst)
        
        # Synthesis specialist
        synthesizer = self.create_llm_agent(
            role=LLMRole.SYNTHESIZER,
            name="Synthesis Specialist",
            custom_config=custom_configs.get("synthesizer", {})
        )
        team.append(synthesizer)
        
        # Fact checker
        fact_checker = self.create_custom_agent(
            agent_type=CustomAgentType.FACT_CHECKER,
            name="Fact Checker",
            custom_config=custom_configs.get("fact_checker", {})
        )
        team.append(fact_checker)
        
        # Workflow coordinator
        coordinator = self.create_workflow_agent(
            name="Research Coordinator",
        )
        team.append(coordinator)
        
        # Track team
        self.agent_teams["research_team"] = team
        
        return team
    
    def _create_content_creation_team(self, custom_configs: Dict[str, Any]) -> List[Agent]:
        """Create a content creation team."""
        team = []
        
        # Content creator
        creator = self.create_llm_agent(
            role=LLMRole.CREATIVE,
            name="Content Creator",
            custom_config=custom_configs.get("creator", {})
        )
        team.append(creator)
        
        # Content creator specialist
        content_specialist = self.create_custom_agent(
            agent_type=CustomAgentType.CONTENT_CREATOR,
            name="Content Specialist",
            custom_config=custom_configs.get("content_specialist", {})
        )
        team.append(content_specialist)
        
        # Reviewer/Editor
        reviewer = self.create_llm_agent(
            role=LLMRole.CRITIC,
            name="Content Reviewer",
            custom_config=custom_configs.get("reviewer", {})
        )
        team.append(reviewer)
        
        # Track team
        self.agent_teams["content_creation"] = team
        
        return team
    
    def _create_data_analysis_team(self, custom_configs: Dict[str, Any]) -> List[Agent]:
        """Create a data analysis team."""
        team = []
        
        # Data analyst
        analyst = self.create_custom_agent(
            agent_type=CustomAgentType.DATA_ANALYST,
            name="Data Analyst",
            custom_config=custom_configs.get("analyst", {})
        )
        team.append(analyst)
        
        # Statistical analyst
        statistician = self.create_llm_agent(
            role=LLMRole.ANALYST,
            name="Statistical Analyst",
            custom_config=custom_configs.get("statistician", {})
        )
        team.append(statistician)
        
        # Data validator
        validator = self.create_custom_agent(
            agent_type=CustomAgentType.VALIDATOR,
            name="Data Validator",
            custom_config=custom_configs.get("validator", {})
        )
        team.append(validator)
        
        # Track team
        self.agent_teams["data_analysis"] = team
        
        return team
    
    def _create_fact_checking_team(self, custom_configs: Dict[str, Any]) -> List[Agent]:
        """Create a fact-checking team."""
        team = []
        
        # Primary fact checker
        fact_checker = self.create_custom_agent(
            agent_type=CustomAgentType.FACT_CHECKER,
            name="Primary Fact Checker",
            custom_config=custom_configs.get("fact_checker", {})
        )
        team.append(fact_checker)
        
        # Verification specialist
        verifier = self.create_custom_agent(
            agent_type=CustomAgentType.VALIDATOR,
            name="Verification Specialist",
            custom_config=custom_configs.get("verifier", {})
        )
        team.append(verifier)
        
        # Research analyst for fact-checking
        researcher = self.create_llm_agent(
            role=LLMRole.RESEARCHER,
            name="Fact Research Analyst",
            custom_config=custom_configs.get("researcher", {})
        )
        team.append(researcher)
        
        # Track team
        self.agent_teams["fact_checking"] = team
        
        return team
    
    def _create_general_purpose_team(self, custom_configs: Dict[str, Any]) -> List[Agent]:
        """Create a general-purpose team."""
        team = []
        
        # General assistant
        assistant = self.create_llm_agent(
            role=LLMRole.GENERALIST,
            name="General Assistant",
            custom_config=custom_configs.get("assistant", {})
        )
        team.append(assistant)
        
        # Task planner
        planner = self.create_llm_agent(
            role=LLMRole.PLANNER,
            name="Task Planner",
            custom_config=custom_configs.get("planner", {})
        )
        team.append(planner)
        
        # Workflow agent
        workflow = self.create_workflow_agent(
            name="General Workflow Agent",
        )
        team.append(workflow)
        
        # Track team
        self.agent_teams["general_purpose"] = team
        
        return team
    
    def _create_domain_experts_team(self, custom_configs: Dict[str, Any]) -> List[Agent]:
        """Create a team of domain experts."""
        team = []
        
        # Get domains from config or use defaults
        domains = custom_configs.get("domains", ["technology", "science", "business"])
        
        for domain in domains:
            expert = self.create_custom_agent(
                agent_type=CustomAgentType.DOMAIN_EXPERT,
                domain=domain,
                name=f"{domain.title()} Expert",
                custom_config=custom_configs.get(f"{domain}_expert", {})
            )
            team.append(expert)
        
        # Track team
        self.agent_teams["domain_experts"] = team
        
        return team
    
    def _create_workflow_automation_team(self, custom_configs: Dict[str, Any]) -> List[Agent]:
        """Create a workflow automation team."""
        team = []
        
        # Primary workflow agent
        workflow_manager = self.create_workflow_agent(
            name="Workflow Manager",
        )
        team.append(workflow_manager)
        
        # Process optimizer
        optimizer = self.create_custom_agent(
            agent_type=CustomAgentType.OPTIMIZER,
            name="Process Optimizer",
            custom_config=custom_configs.get("optimizer", {})
        )
        team.append(optimizer)
        
        # Integration specialist
        integrator = self.create_custom_agent(
            agent_type=CustomAgentType.INTEGRATOR,
            name="Integration Specialist",
            custom_config=custom_configs.get("integrator", {})
        )
        team.append(integrator)
        
        # Track team
        self.agent_teams["workflow_automation"] = team
        
        return team
    
    def _create_qa_specialists_team(self, custom_configs: Dict[str, Any]) -> List[Agent]:
        """Create a Q&A specialists team."""
        team = []
        
        # Q&A specialist
        qa_specialist = self.create_custom_agent(
            agent_type=CustomAgentType.QA_SPECIALIST,
            name="Q&A Specialist",
            custom_config=custom_configs.get("qa_specialist", {})
        )
        team.append(qa_specialist)
        
        # Research support
        researcher = self.create_llm_agent(
            role=LLMRole.RESEARCHER,
            name="Q&A Researcher",
            custom_config=custom_configs.get("researcher", {})
        )
        team.append(researcher)
        
        # Answer validator
        validator = self.create_custom_agent(
            agent_type=CustomAgentType.VALIDATOR,
            name="Answer Validator",
            custom_config=custom_configs.get("validator", {})
        )
        team.append(validator)
        
        # Track team
        self.agent_teams["qa_specialists"] = team
        
        return team
    
    def create_task_optimized_agent(self,
                                   task_description: str,
                                   role: LLMRole,
                                   context: Optional[Dict[str, Any]] = None,
                                   name: Optional[str] = None,
                                   tools: Optional[List[Callable]] = None) -> LLMAgent:
        """
        Create an LLM agent optimized for a specific task using intelligent model selection.
        
        Args:
            task_description: Description of the task to be performed
            role: Agent role
            context: Additional context for model selection
            name: Optional custom name
            tools: Optional custom tools
            
        Returns:
            Task-optimized LLM agent
        """
        # Analyze task complexity
        task_complexity = analyze_task_complexity(task_description, context)
        
        # Get optimal model configuration for this task
        generation_config = select_model_for_task(
            task=task_description,
            context=context,
            preferences={
                "role": role.value,
                "enable_thinking": True,
                "enable_structured_output": task_complexity in [TaskComplexity.COMPLEX, TaskComplexity.CRITICAL]
            }
        )
        
        # Create custom config based on model recommendation
        custom_config = {
            "model": generation_config.model.model_id,
            "temperature": generation_config.temperature,
            "top_p": generation_config.top_p,
            "top_k": generation_config.top_k,
            "enable_thinking": generation_config.thinking_config.enabled if generation_config.thinking_config else True,
            "thinking_budget": generation_config.thinking_config.budget_tokens if generation_config.thinking_config else None,
            "enable_structured_output": generation_config.structured_output.enabled if generation_config.structured_output else False,
            "output_schema_type": "analysis" if generation_config.structured_output else None,
            "auto_optimize_model": True,
        }
        
        # Create the optimized agent
        agent = self.create_llm_agent(
            role=role,
            name=name or f"Task-Optimized {role.value.title()} Agent",
            tools=tools,
            custom_config=custom_config
        )
        
        if self.logger:
            self.logger.info(
                f"Created task-optimized agent for complexity: {task_complexity.value}, "
                f"using model: {generation_config.model.name}"
            )
        
        return agent
    
    def create_research_team_for_task(self,
                                     task_description: str,
                                     context: Optional[Dict[str, Any]] = None,
                                     team_size: str = "standard") -> List[Agent]:
        """
        Create a research team optimized for a specific task.
        
        Args:
            task_description: Description of the research task
            context: Additional context for optimization
            team_size: Size of team ("minimal", "standard", "comprehensive")
            
        Returns:
            Optimized research team
        """
        team = []
        
        if team_size == "minimal":
            # Just researcher and analyst
            researcher = self.create_task_optimized_agent(
                task_description=f"Research task: {task_description}",
                role=LLMRole.RESEARCHER,
                context=context,
                name="Primary Researcher"
            )
            team.append(researcher)
            
            analyst = self.create_task_optimized_agent(
                task_description=f"Analysis task: {task_description}",
                role=LLMRole.ANALYST,
                context=context,
                name="Primary Analyst"
            )
            team.append(analyst)
            
        elif team_size == "standard":
            # Researcher, analyst, synthesizer
            for role, task_prefix in [
                (LLMRole.RESEARCHER, "Research"),
                (LLMRole.ANALYST, "Analysis"),
                (LLMRole.SYNTHESIZER, "Synthesis")
            ]:
                agent = self.create_task_optimized_agent(
                    task_description=f"{task_prefix} task: {task_description}",
                    role=role,
                    context=context,
                    name=f"{task_prefix} Specialist"
                )
                team.append(agent)
                
        else:  # comprehensive
            # Full research team
            for role, task_prefix in [
                (LLMRole.RESEARCHER, "Research"),
                (LLMRole.ANALYST, "Analysis"),
                (LLMRole.SYNTHESIZER, "Synthesis"),
                (LLMRole.CRITIC, "Critical Review"),
                (LLMRole.SPECIALIST, "Expert Analysis")
            ]:
                agent = self.create_task_optimized_agent(
                    task_description=f"{task_prefix} task: {task_description}",
                    role=role,
                    context=context,
                    name=f"{task_prefix} Specialist"
                )
                team.append(agent)
        
        # Track team
        team_id = f"task_optimized_{len(self.agent_teams)}"
        self.agent_teams[team_id] = team
        
        if self.logger:
            self.logger.info(f"Created {team_size} research team ({len(team)} agents) for task optimization")
        
        return team
    
    def _get_default_llm_configs(self) -> Dict[LLMRole, LLMAgentConfig]:
        """Get default LLM configurations for each role with Gemini 2.5 optimizations."""
        return {
            LLMRole.RESEARCHER: LLMAgentConfig(
                role=LLMRole.RESEARCHER,
                temperature=0.3,
                enable_search=True,
                enable_memory=True,
                enable_thinking=True,  # Enable thinking for research tasks
                enable_structured_output=True,  # Structure research findings
                output_schema_type="research",
                auto_optimize_model=True,
                instruction="Focus on comprehensive information gathering and source verification.",
            ),
            LLMRole.ANALYST: LLMAgentConfig(
                role=LLMRole.ANALYST,
                temperature=0.2,
                enable_search=True,
                enable_thinking=True,  # Enable thinking for complex analysis
                enable_structured_output=True,  # Structure analysis results
                output_schema_type="analysis",
                auto_optimize_model=True,
                instruction="Provide thorough analysis with logical reasoning and evidence-based conclusions.",
            ),
            LLMRole.SYNTHESIZER: LLMAgentConfig(
                role=LLMRole.SYNTHESIZER,
                temperature=0.4,
                enable_memory=True,
                enable_thinking=True,  # Enable thinking for synthesis
                auto_optimize_model=True,
                instruction="Combine information from multiple sources into coherent, well-structured outputs.",
            ),
            LLMRole.CRITIC: LLMAgentConfig(
                role=LLMRole.CRITIC,
                temperature=0.2,
                enable_search=True,
                enable_thinking=True,  # Enable thinking for critical analysis
                enable_structured_output=True,  # Structure critique findings
                output_schema_type="analysis",
                auto_optimize_model=True,
                instruction="Provide critical analysis, identify potential issues, and suggest improvements.",
            ),
            LLMRole.PLANNER: LLMAgentConfig(
                role=LLMRole.PLANNER,
                temperature=0.3,
                enable_thinking=True,  # Enable thinking for planning
                auto_optimize_model=True,
                instruction="Break down complex tasks into actionable steps and create detailed plans.",
            ),
            LLMRole.COMMUNICATOR: LLMAgentConfig(
                role=LLMRole.COMMUNICATOR,
                temperature=0.5,
                priority_speed=True,  # Prioritize speed for communication
                enable_thinking=False,  # Less thinking needed for communication
                auto_optimize_model=True,
                instruction="Focus on clear, effective communication and presentation of information.",
            ),
            LLMRole.CREATIVE: LLMAgentConfig(
                role=LLMRole.CREATIVE,
                temperature=0.7,
                enable_thinking=True,  # Enable thinking for creativity
                auto_optimize_model=True,
                instruction="Generate creative, original content and innovative solutions.",
            ),
            LLMRole.SPECIALIST: LLMAgentConfig(
                role=LLMRole.SPECIALIST,
                temperature=0.2,
                enable_search=True,
                enable_thinking=True,  # Enable thinking for specialist knowledge
                enable_structured_output=True,  # Structure specialist analysis
                output_schema_type="analysis",
                auto_optimize_model=True,
                instruction="Provide deep, expert-level knowledge and analysis in specific domains.",
            ),
            LLMRole.GENERALIST: LLMAgentConfig(
                role=LLMRole.GENERALIST,
                temperature=0.4,
                enable_search=True,
                enable_memory=True,
                enable_thinking=True,  # Enable thinking for general tasks
                auto_optimize_model=True,
                priority_cost=True,  # Balance cost for general use
                instruction="Provide balanced, helpful assistance across a wide range of topics and tasks.",
            ),
        }
    
    def _get_default_custom_configs(self) -> Dict[CustomAgentType, CustomAgentConfig]:
        """Get default custom agent configurations."""
        return {
            CustomAgentType.FACT_CHECKER: CustomAgentConfig(
                agent_type=CustomAgentType.FACT_CHECKER,
                behavior_rules={"verification_threshold": 0.8, "require_sources": True},
                quality_thresholds={"minimum": 0.7},
            ),
            CustomAgentType.DATA_ANALYST: CustomAgentConfig(
                agent_type=CustomAgentType.DATA_ANALYST,
                specialized_tools=["statistical_analysis", "data_visualization"],
                quality_thresholds={"minimum": 0.6},
            ),
            CustomAgentType.CONTENT_CREATOR: CustomAgentConfig(
                agent_type=CustomAgentType.CONTENT_CREATOR,
                output_format="markdown",
                behavior_rules={"creativity_level": "high", "originality_check": True},
                quality_thresholds={"minimum": 0.7},
            ),
            CustomAgentType.VALIDATOR: CustomAgentConfig(
                agent_type=CustomAgentType.VALIDATOR,
                behavior_rules={"strict_validation": True, "detailed_feedback": True},
                quality_thresholds={"minimum": 0.8},
            ),
            CustomAgentType.QA_SPECIALIST: CustomAgentConfig(
                agent_type=CustomAgentType.QA_SPECIALIST,
                behavior_rules={"comprehensive_answers": True, "cite_sources": True},
                quality_thresholds={"minimum": 0.7},
            ),
            CustomAgentType.DOMAIN_EXPERT: CustomAgentConfig(
                agent_type=CustomAgentType.DOMAIN_EXPERT,
                behavior_rules={"deep_expertise": True, "domain_focus": True},
                quality_thresholds={"minimum": 0.8},
            ),
        }
    
    def _load_agent_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load agent templates for common use cases."""
        return {
            "academic_researcher": {
                "type": "llm",
                "role": LLMRole.RESEARCHER,
                "config": {
                    "temperature": 0.2,
                    "instruction": "Focus on academic research with peer-reviewed sources and rigorous methodology.",
                    "enable_search": True,
                }
            },
            "business_analyst": {
                "type": "custom",
                "agent_type": CustomAgentType.DATA_ANALYST,
                "domain": "business",
                "config": {
                    "specialized_tools": ["financial_analysis", "market_research"],
                }
            },
            "technical_writer": {
                "type": "custom",
                "agent_type": CustomAgentType.CONTENT_CREATOR,
                "domain": "technology",
                "config": {
                    "output_format": "markdown",
                    "behavior_rules": {"technical_accuracy": True, "clear_explanations": True},
                }
            },
        }
    
    def get_factory_status(self) -> Dict[str, Any]:
        """Get factory status and statistics."""
        return {
            "total_agents_created": len(self.created_agents),
            "active_agents": sum(1 for agent in self.created_agents.values() if agent.is_active),
            "agents_by_type": {
                agent_type.value: len(self.get_agents_by_type(agent_type))
                for agent_type in AgentType
            },
            "teams_created": len(self.agent_teams),
            "team_names": list(self.agent_teams.keys()),
            "available_templates": list(self.agent_templates.keys()),
            "available_suites": [suite.value for suite in AgentSuite],
        }


# Convenience functions for easy agent creation

def create_agent(agent_type: str,
                name: Optional[str] = None,
                config: Optional[Dict[str, Any]] = None,
                logger: Optional[RunLogger] = None,
                **kwargs) -> Agent:
    """
    Convenience function to create an agent quickly.
    
    Args:
        agent_type: Type of agent ("llm", "workflow", "custom")
        name: Optional agent name
        config: Optional configuration
        logger: Optional logger
        **kwargs: Additional arguments
        
    Returns:
        Created agent
    """
    factory = AgentFactory(logger=logger, **kwargs)
    
    if agent_type.lower() == "llm":
        role = LLMRole(config.get("role", "generalist")) if config else LLMRole.GENERALIST
        return factory.create_llm_agent(role=role, name=name, custom_config=config)
    elif agent_type.lower() == "workflow":
        return factory.create_workflow_agent(name=name)
    elif agent_type.lower() == "custom":
        custom_type = CustomAgentType(config.get("agent_type", "domain_expert")) if config else CustomAgentType.DOMAIN_EXPERT
        domain = config.get("domain", "") if config else ""
        return factory.create_custom_agent(agent_type=custom_type, domain=domain, name=name, custom_config=config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def create_research_team(logger: Optional[RunLogger] = None,
                        **kwargs) -> List[Agent]:
    """
    Convenience function to create a research team quickly.
    
    Args:
        logger: Optional logger
        **kwargs: Additional arguments for services
        
    Returns:
        List of research team agents
    """
    factory = AgentFactory(logger=logger, **kwargs)
    return factory.create_agent_suite(AgentSuite.RESEARCH_TEAM)


def create_content_team(logger: Optional[RunLogger] = None,
                       **kwargs) -> List[Agent]:
    """
    Convenience function to create a content creation team quickly.
    
    Args:
        logger: Optional logger
        **kwargs: Additional arguments for services
        
    Returns:
        List of content creation team agents
    """
    factory = AgentFactory(logger=logger, **kwargs)
    return factory.create_agent_suite(AgentSuite.CONTENT_CREATION)