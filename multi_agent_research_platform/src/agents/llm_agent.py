"""
LLM Agent Implementation

Provides language model-based agents that can perform reasoning, analysis,
and content generation using Google ADK's latest Gemini 2.5 models with
thinking budgets and structured output capabilities.
"""

import time
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from google.adk.agents import Agent as ADKAgent
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig, HttpOptions

from .base import Agent, AgentType, AgentCapability, AgentResult
from ..platform_logging import RunLogger
from ..services import SessionService, MemoryService, ArtifactService
from ..mcp import MCPOrchestrator
from ..config.gemini_models import (
    GeminiModel, TaskComplexity, ModelSelector, GeminiGenerationConfig,
    ThinkingBudgetConfig, StructuredOutputConfig, select_model_for_task,
    analyze_task_complexity
)


class LLMRole(str, Enum):
    """Specialized roles for LLM agents."""
    RESEARCHER = "researcher"           # Research and information gathering
    ANALYST = "analyst"                # Data analysis and interpretation
    SYNTHESIZER = "synthesizer"        # Information synthesis and summarization
    CRITIC = "critic"                  # Critical analysis and fact-checking
    PLANNER = "planner"                # Task planning and decomposition
    COMMUNICATOR = "communicator"      # Communication and presentation
    CREATIVE = "creative"              # Creative content generation
    SPECIALIST = "specialist"          # Domain-specific expertise
    GENERALIST = "generalist"          # General-purpose assistance


@dataclass
class LLMAgentConfig:
    """Enhanced configuration for LLM agents with Gemini 2.5 models."""
    # Model selection - can be specific model or auto-select based on task
    model: Optional[GeminiModel] = None  # None for auto-selection
    role: LLMRole = LLMRole.GENERALIST
    
    # Generation parameters  
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 0.95
    top_k: int = 64
    
    # Instructions and prompts
    system_prompt: str = ""
    instruction: str = ""
    
    # Advanced Gemini 2.5 features
    enable_thinking: bool = True
    thinking_budget: Optional[int] = None  # None for auto, -1 for model control, >0 for manual
    enable_structured_output: bool = False
    output_schema_type: str = "analysis"  # "analysis", "research", "custom"
    custom_output_schema: Optional[Dict[str, Any]] = None
    
    # Model selection preferences
    priority_speed: bool = False
    priority_cost: bool = False
    task_complexity: Optional[TaskComplexity] = None  # None for auto-detection
    
    # Traditional settings
    enable_function_calling: bool = True
    enable_search: bool = True
    enable_memory: bool = True
    safety_settings: Dict[str, str] = field(default_factory=dict)
    
    # Auto-optimization settings
    auto_optimize_model: bool = True  # Automatically select best model for each task
    enable_cost_optimization: bool = False  # Try to use thinking budget to reduce model tier
    
    def get_model_preferences(self) -> Dict[str, Any]:
        """Get model selection preferences."""
        return {
            "priority_speed": self.priority_speed,
            "priority_cost": self.priority_cost,
            "enable_thinking": self.enable_thinking,
            "thinking_budget": self.thinking_budget,
            "enable_structured_output": self.enable_structured_output,
            "output_schema_type": self.output_schema_type,
            "custom_schema": self.custom_output_schema,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "system_instruction": self.get_full_system_instruction()
        }
    
    def get_full_system_instruction(self) -> str:
        """Get complete system instruction combining all prompts."""
        parts = []
        if self.system_prompt:
            parts.append(self.system_prompt)
        if self.instruction:
            parts.append(self.instruction)
        return "\n\n".join(parts) if parts else ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "model": self.model.value if self.model else None,
            "role": self.role.value,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "system_prompt": self.system_prompt,
            "instruction": self.instruction,
            "enable_thinking": self.enable_thinking,
            "thinking_budget": self.thinking_budget,
            "enable_structured_output": self.enable_structured_output,
            "output_schema_type": self.output_schema_type,
            "priority_speed": self.priority_speed,
            "priority_cost": self.priority_cost,
            "auto_optimize_model": self.auto_optimize_model,
            "enable_cost_optimization": self.enable_cost_optimization,
            "enable_function_calling": self.enable_function_calling,
            "enable_search": self.enable_search,
            "enable_memory": self.enable_memory,
            "safety_settings": self.safety_settings,
        }


class LLMAgent(Agent):
    """
    Language model-based agent using Google ADK and Gemini models.
    
    Provides advanced reasoning, analysis, and content generation capabilities
    with integration to both ADK tools and MCP servers.
    """
    
    def __init__(self,
                 config: LLMAgentConfig,
                 tools: Optional[List[callable]] = None,
                 agent_id: Optional[str] = None,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 logger: Optional[RunLogger] = None,
                 session_service: Optional[SessionService] = None,
                 memory_service: Optional[MemoryService] = None,
                 artifact_service: Optional[ArtifactService] = None,
                 mcp_orchestrator: Optional[MCPOrchestrator] = None):
        
        # Generate name and description based on role if not provided
        role_names = {
            LLMRole.RESEARCHER: "Research Agent",
            LLMRole.ANALYST: "Analysis Agent", 
            LLMRole.SYNTHESIZER: "Synthesis Agent",
            LLMRole.CRITIC: "Critical Analysis Agent",
            LLMRole.PLANNER: "Planning Agent",
            LLMRole.COMMUNICATOR: "Communication Agent",
            LLMRole.CREATIVE: "Creative Agent",
            LLMRole.SPECIALIST: "Specialist Agent",
            LLMRole.GENERALIST: "General Assistant Agent",
        }
        
        role_descriptions = {
            LLMRole.RESEARCHER: "Specializes in research, information gathering, and fact-finding using multiple sources",
            LLMRole.ANALYST: "Focuses on data analysis, interpretation, and drawing insights from information",
            LLMRole.SYNTHESIZER: "Combines information from multiple sources into coherent summaries and reports",
            LLMRole.CRITIC: "Provides critical analysis, fact-checking, and quality assessment",
            LLMRole.PLANNER: "Breaks down complex tasks into manageable steps and creates execution plans",
            LLMRole.COMMUNICATOR: "Specializes in clear communication, presentations, and stakeholder engagement",
            LLMRole.CREATIVE: "Generates creative content, ideas, and innovative solutions",
            LLMRole.SPECIALIST: "Provides deep expertise in specific domains or technical areas",
            LLMRole.GENERALIST: "Handles a wide variety of tasks with balanced capabilities",
        }
        
        agent_name = name or role_names.get(config.role, "LLM Agent")
        agent_description = description or role_descriptions.get(config.role, "Language model-based agent")
        
        super().__init__(
            agent_id=agent_id,
            name=agent_name,
            description=agent_description,
            agent_type=AgentType.LLM,
            capabilities=self._determine_capabilities(config.role),
            logger=logger,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service,
            mcp_orchestrator=mcp_orchestrator,
        )
        
        self.config = config
        self.tools = tools or []
        
        # Create ADK agent instance
        self._create_adk_agent()
        
        # Conversation history for context
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history_length = 20
        
        # Performance metrics
        self.total_tokens_used = 0
        self.average_response_time_ms = 0.0
        self.successful_tasks = 0
        self.failed_tasks = 0
    
    def _determine_capabilities(self, role: LLMRole) -> List[AgentCapability]:
        """Determine capabilities based on agent role."""
        # Base capabilities all LLM agents have
        base_capabilities = [
            AgentCapability.REASONING,
            AgentCapability.COMMUNICATION,
            AgentCapability.TOOL_USE,
            AgentCapability.CONTEXT_MANAGEMENT,
        ]
        
        # Role-specific capabilities
        role_capabilities = {
            LLMRole.RESEARCHER: [
                AgentCapability.RESEARCH,
                AgentCapability.ANALYSIS,
                AgentCapability.MEMORY_ACCESS,
            ],
            LLMRole.ANALYST: [
                AgentCapability.ANALYSIS,
                AgentCapability.DECISION_MAKING,
                AgentCapability.FACT_CHECKING,
            ],
            LLMRole.SYNTHESIZER: [
                AgentCapability.SYNTHESIS,
                AgentCapability.CONTENT_GENERATION,
                AgentCapability.ANALYSIS,
            ],
            LLMRole.CRITIC: [
                AgentCapability.ANALYSIS,
                AgentCapability.FACT_CHECKING,
                AgentCapability.DECISION_MAKING,
            ],
            LLMRole.PLANNER: [
                AgentCapability.PLANNING,
                AgentCapability.DECISION_MAKING,
                AgentCapability.EXECUTION,
            ],
            LLMRole.COMMUNICATOR: [
                AgentCapability.CONTENT_GENERATION,
                AgentCapability.SYNTHESIS,
                AgentCapability.COMMUNICATION,
            ],
            LLMRole.CREATIVE: [
                AgentCapability.CONTENT_GENERATION,
                AgentCapability.REASONING,
                AgentCapability.SYNTHESIS,
            ],
            LLMRole.SPECIALIST: [
                AgentCapability.ANALYSIS,
                AgentCapability.DECISION_MAKING,
                AgentCapability.RESEARCH,
            ],
            LLMRole.GENERALIST: [
                AgentCapability.ANALYSIS,
                AgentCapability.RESEARCH,
                AgentCapability.SYNTHESIS,
                AgentCapability.CONTENT_GENERATION,
                AgentCapability.PLANNING,
            ],
        }
        
        return base_capabilities + role_capabilities.get(role, [])
    
    def _create_adk_agent(self) -> None:
        """Create the underlying ADK agent."""
        try:
            # Construct system instruction
            role_instructions = {
                LLMRole.RESEARCHER: "You are a research specialist agent. Focus on gathering comprehensive information from multiple sources, fact-checking, and providing well-researched responses.",
                LLMRole.ANALYST: "You are an analysis specialist agent. Focus on interpreting data, identifying patterns, and providing insightful analysis of information.",
                LLMRole.SYNTHESIZER: "You are a synthesis specialist agent. Focus on combining information from multiple sources into coherent, well-structured summaries.",
                LLMRole.CRITIC: "You are a critical analysis agent. Focus on evaluating information quality, fact-checking claims, and identifying potential issues or biases.",
                LLMRole.PLANNER: "You are a planning specialist agent. Focus on breaking down complex tasks into actionable steps and creating detailed execution plans.",
                LLMRole.COMMUNICATOR: "You are a communication specialist agent. Focus on clear, effective communication and presenting information in accessible formats.",
                LLMRole.CREATIVE: "You are a creative agent. Focus on generating original ideas, creative solutions, and innovative approaches to problems.",
                LLMRole.SPECIALIST: "You are a domain specialist agent. Focus on providing deep, expert-level knowledge and analysis in specific subject areas.",
                LLMRole.GENERALIST: "You are a general-purpose assistant agent. Provide balanced, helpful responses across a wide range of topics and tasks.",
            }
            
            base_instruction = role_instructions.get(self.config.role, "You are a helpful AI assistant.")
            
            full_instruction = f"""{base_instruction}

{self.config.instruction}

{self.config.system_prompt}

You have access to various tools and can search for information when needed. Always strive for accuracy and cite your sources when possible.""".strip()
            
            # Determine model for ADK agent
            if self.config.auto_optimize_model:
                # For ADK agent creation, use a default task to determine model
                sample_task = f"General {self.config.role.value} task"
                generation_config = select_model_for_task(
                    task=sample_task,
                    preferences=self.config.get_model_preferences()
                )
                model_id = generation_config.model.model_id.value
            else:
                model_id = self.config.model.value if self.config.model else GeminiModel.FLASH.value
            
            # Create ADK agent
            self.adk_agent = ADKAgent(
                name=self.name.lower().replace(" ", "_"),
                model=model_id,
                description=self.description,
                instruction=full_instruction,
                tools=self.tools,
            )
            
            if self.logger:
                self.logger.info(f"Created ADK agent for {self.name} with model {self.config.model}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to create ADK agent for {self.name}: {e}")
            self.adk_agent = None
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Get the capabilities of this LLM agent."""
        return self.capabilities
    
    async def execute_task(self, 
                          task: str, 
                          context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Execute a task using the LLM agent.
        
        Args:
            task: Task description or question
            context: Optional context information
            
        Returns:
            Agent result with LLM response
        """
        start_time = time.time()
        
        if self.logger:
            self.logger.info(f"LLM Agent {self.name} executing task: {task[:100]}...")
        
        if not self.is_active:
            return AgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                task=task,
                success=False,
                error="Agent is not active",
            )
        
        tools_used = []
        memory_accessed = False
        artifacts_created = []
        
        try:
            # Prepare context
            context_info = await self._prepare_context(task, context)
            
            # Enhance task with context if needed
            enhanced_task = await self._enhance_task_with_context(task, context_info)
            
            # Execute with ADK agent
            if self.adk_agent:
                # This would be the actual ADK execution
                # For now, we'll simulate the response
                response = await self._simulate_llm_response(enhanced_task, context_info)
                tools_used.append("adk_gemini")
            else:
                # Fallback to direct LLM call
                response = await self._direct_llm_call(enhanced_task, context_info)
                tools_used.append("direct_llm")
            
            # Store result in memory if enabled
            if self.config.enable_memory and self.memory_service:
                await self.store_memory(
                    f"Task: {task}\nResponse: {response}",
                    metadata={
                        "task_type": "llm_execution",
                        "role": self.config.role.value,
                        "success": True,
                    }
                )
                memory_accessed = True
            
            # Update conversation history
            self._update_conversation_history(task, response)
            
            # Update metrics
            execution_time = (time.time() - start_time) * 1000
            self.successful_tasks += 1
            self.total_tasks_completed += 1
            self.last_task_time = time.time()
            self._update_average_response_time(execution_time)
            
            return AgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                task=task,
                success=True,
                result=response,
                execution_time_ms=execution_time,
                metadata={
                    "model": self.config.model.value if self.config.model else "auto-selected",
                    "role": self.config.role.value,
                    "temperature": self.config.temperature,
                    "context_used": bool(context_info),
                    "thinking_enabled": self.config.enable_thinking,
                    "structured_output": self.config.enable_structured_output,
                    "auto_optimize": self.config.auto_optimize_model,
                },
                context_used=context_info,
                tools_used=tools_used,
                memory_accessed=memory_accessed,
                artifacts_created=artifacts_created,
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.failed_tasks += 1
            
            if self.logger:
                self.logger.error(f"LLM Agent {self.name} task execution failed: {e}")
            
            return AgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                task=task,
                success=False,
                error=str(e),
                execution_time_ms=execution_time,
            )
    
    async def chat(self, 
                  message: str, 
                  maintain_context: bool = True) -> str:
        """
        Have a conversational interaction with the agent.
        
        Args:
            message: User message
            maintain_context: Whether to maintain conversation context
            
        Returns:
            Agent's response
        """
        if maintain_context:
            # Include conversation history in context
            context = {
                "conversation_history": self.conversation_history[-self.max_history_length:],
                "maintain_context": True,
            }
        else:
            context = {"maintain_context": False}
        
        result = await self.execute_task(message, context)
        
        if result.success:
            return result.result
        else:
            return f"I encountered an error: {result.error}"
    
    async def analyze_with_sources(self, 
                                 content: str, 
                                 analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Analyze content with source verification using MCP servers.
        
        Args:
            content: Content to analyze
            analysis_type: Type of analysis to perform
            
        Returns:
            Analysis results with source verification
        """
        if not self.mcp_orchestrator:
            # Fallback to basic analysis
            task = f"Analyze the following content ({analysis_type} analysis):\n\n{content}"
            result = await self.execute_task(task)
            return {"analysis": result.result, "sources_verified": False}
        
        # Use MCP servers for enhanced analysis
        try:
            # First, perform basic analysis
            analysis_task = f"Provide a {analysis_type} analysis of the following content:\n\n{content}"
            analysis_result = await self.execute_task(analysis_task)
            
            # Then verify with external sources using MCP
            verification_task = f"Verify the key claims in this content using external sources:\n\n{content}"
            verification_result = await self.use_mcp_server(
                "perplexity",
                "search",
                {
                    "query": f"fact check verify: {content[:200]}",
                    "strategy": "HYBRID_VALIDATION",
                }
            )
            
            return {
                "analysis": analysis_result.result,
                "verification": verification_result,
                "sources_verified": verification_result.get("success", False),
                "confidence": "high" if verification_result.get("success") else "medium",
            }
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Analysis with sources failed: {e}")
            
            # Fallback to basic analysis
            task = f"Analyze the following content ({analysis_type} analysis):\n\n{content}"
            result = await self.execute_task(task)
            return {
                "analysis": result.result, 
                "sources_verified": False,
                "error": str(e),
            }
    
    async def _prepare_context(self, 
                             task: str, 
                             context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare context information for task execution."""
        context_info = context or {}
        
        # Add memory context if enabled
        if self.config.enable_memory and self.memory_service:
            try:
                relevant_memories = await self.retrieve_memory(task, limit=5)
                if relevant_memories:
                    context_info["relevant_memories"] = relevant_memories
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to retrieve memories: {e}")
        
        # Add conversation context
        if self.conversation_history:
            context_info["recent_conversation"] = self.conversation_history[-5:]
        
        # Add agent role context
        context_info["agent_role"] = self.config.role.value
        context_info["agent_capabilities"] = [cap.value for cap in self.get_capabilities()]
        
        return context_info
    
    async def _enhance_task_with_context(self, 
                                       task: str, 
                                       context: Dict[str, Any]) -> str:
        """Enhance task description with relevant context."""
        if not context:
            return task
        
        enhanced_task = task
        
        # Add role context
        if "agent_role" in context:
            role_prefix = f"As a {context['agent_role']} agent, "
            enhanced_task = role_prefix + enhanced_task
        
        # Add memory context
        if "relevant_memories" in context and context["relevant_memories"]:
            memory_context = "\n\nRelevant context from previous interactions:\n"
            for memory in context["relevant_memories"][:3]:
                memory_context += f"- {memory.get('text', '')[:200]}...\n"
            enhanced_task += memory_context
        
        return enhanced_task
    
    async def _simulate_llm_response(self, 
                                   task: str, 
                                   context: Dict[str, Any]) -> str:
        """
        Generate LLM response using Google GenAI SDK with Gemini 2.5 models.
        
        Uses intelligent model selection, thinking budgets, and structured output
        based on task complexity and agent configuration.
        """
        try:
            # Get model preferences from config
            preferences = self.config.get_model_preferences()
            
            # Analyze task and select optimal model configuration
            if self.config.auto_optimize_model:
                generation_config = select_model_for_task(
                    task=task,
                    context=context,
                    preferences=preferences
                )
            else:
                # Use specified model or default
                model_id = self.config.model or GeminiModel.FLASH
                model_config = get_model_config(model_id)
                
                # Create thinking config
                thinking_config = None
                if (self.config.enable_thinking and 
                    model_config.supports_thinking):
                    
                    thinking_config = ThinkingBudgetConfig(
                        enabled=True,
                        budget_tokens=self.config.thinking_budget or -1  # Auto mode
                    )
                
                # Create structured output config
                structured_config = None
                if (self.config.enable_structured_output and 
                    ModelCapability.STRUCTURED_OUTPUT in model_config.capabilities):
                    
                    if self.config.output_schema_type == "analysis":
                        schema = StructuredOutputConfig.get_analysis_schema()
                    elif self.config.output_schema_type == "research":
                        schema = StructuredOutputConfig.get_research_schema()
                    else:
                        schema = self.config.custom_output_schema
                    
                    if schema:
                        structured_config = StructuredOutputConfig(
                            enabled=True,
                            response_schema=schema
                        )
                
                generation_config = GeminiGenerationConfig(
                    model=model_config,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    thinking_config=thinking_config,
                    structured_output=structured_config,
                    system_instruction=self.config.get_full_system_instruction()
                )
            
            # Create Google GenAI client
            client = genai.Client(
                http_options=HttpOptions(api_version="v1")
            )
            
            # Prepare generation config for API
            api_config = generation_config.to_generate_content_config()
            
            # Add system instruction if provided
            if generation_config.system_instruction:
                api_config["system_instruction"] = generation_config.system_instruction
            
            # Log model selection if logger available
            if self.logger:
                self.logger.info(
                    f"Using {generation_config.model.name} for task complexity analysis. "
                    f"Thinking: {generation_config.thinking_config.enabled if generation_config.thinking_config else False}, "
                    f"Structured: {generation_config.structured_output.enabled if generation_config.structured_output else False}"
                )
            
            # Generate response
            response = await client.models.generate_content(
                model=generation_config.model.model_id.value,
                contents=task,
                config=GenerateContentConfig(**api_config)
            )
            
            # Extract response text
            response_text = response.text
            
            # Update token usage metrics if available
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                self.total_tokens_used += (
                    response.usage_metadata.prompt_token_count + 
                    response.usage_metadata.candidates_token_count
                )
            
            # Log thinking process if enabled and available
            if (generation_config.thinking_config and 
                generation_config.thinking_config.enabled and 
                self.logger):
                
                # Check for thinking content in response
                for candidate in response.candidates:
                    for part in candidate.content.parts:
                        if hasattr(part, 'thought') and part.thought:
                            self.logger.debug(f"Model thinking process: {part.text[:200]}...")
            
            return response_text
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"LLM response generation failed: {e}")
            
            # Fallback to direct LLM call
            return await self._direct_llm_call(task, context)
    
    async def _direct_llm_call(self, 
                             task: str, 
                             context: Dict[str, Any]) -> str:
        """
        Direct LLM call fallback using basic Gemini configuration.
        
        Used when ADK agent creation fails or as a fallback method.
        """
        try:
            # Use Flash model as reliable fallback
            model_config = get_model_config(GeminiModel.FLASH)
            
            # Create basic generation config
            generation_config = GeminiGenerationConfig(
                model=model_config,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                system_instruction=self.config.get_full_system_instruction()
            )
            
            # Create client
            client = genai.Client(
                http_options=HttpOptions(api_version="v1")
            )
            
            # Prepare task with role context
            enhanced_task = f"As a {self.config.role.value} agent, {task}"
            
            if self.logger:
                self.logger.info(f"Direct LLM fallback using {model_config.name}")
            
            # Generate response with basic configuration
            response = await client.models.generate_content(
                model=generation_config.model.model_id.value,
                contents=enhanced_task,
                config=GenerateContentConfig(
                    temperature=generation_config.temperature,
                    top_p=generation_config.top_p,
                    top_k=generation_config.top_k,
                    system_instruction=generation_config.system_instruction
                )
            )
            
            # Update token usage
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                self.total_tokens_used += (
                    response.usage_metadata.prompt_token_count + 
                    response.usage_metadata.candidates_token_count
                )
            
            return response.text
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Direct LLM call failed: {e}")
            
            # Ultimate fallback - return error message
            role = self.config.role.value
            return f"I apologize, but I encountered an error while processing your request as a {role} agent. The error was: {str(e)[:100]}. Please try again or contact support if this issue persists."
    
    def _update_conversation_history(self, task: str, response: str) -> None:
        """Update conversation history."""
        self.conversation_history.append({
            "task": task,
            "response": response,
            "timestamp": time.time(),
        })
        
        # Trim history to max length
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def _update_average_response_time(self, execution_time_ms: float) -> None:
        """Update average response time metric."""
        if self.successful_tasks == 1:
            self.average_response_time_ms = execution_time_ms
        else:
            # Running average
            self.average_response_time_ms = (
                (self.average_response_time_ms * (self.successful_tasks - 1) + execution_time_ms) / 
                self.successful_tasks
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this agent."""
        success_rate = (
            (self.successful_tasks / (self.successful_tasks + self.failed_tasks) * 100)
            if (self.successful_tasks + self.failed_tasks) > 0
            else 0
        )
        
        return {
            "total_tasks": self.total_tasks_completed,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate_percent": success_rate,
            "average_response_time_ms": self.average_response_time_ms,
            "total_tokens_used": self.total_tokens_used,
            "conversation_length": len(self.conversation_history),
        }
    
    def clear_conversation_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        if self.logger:
            self.logger.info(f"Cleared conversation history for agent {self.name}")
    
    async def update_config(self, new_config: LLMAgentConfig) -> bool:
        """
        Update agent configuration and recreate ADK agent if needed.
        
        Args:
            new_config: New configuration
            
        Returns:
            Success status
        """
        try:
            old_model = self.config.model
            self.config = new_config
            
            # Recreate ADK agent if model changed
            if old_model != new_config.model:
                self._create_adk_agent()
            
            if self.logger:
                self.logger.info(f"Updated configuration for agent {self.name}")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to update configuration for agent {self.name}: {e}")
            return False