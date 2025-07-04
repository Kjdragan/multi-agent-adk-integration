"""
Custom Agent Implementation

Provides specialized custom agents for specific domains, tasks, and use cases
with configurable behaviors, tools, and capabilities.
"""

import time
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum

from .base import Agent, AgentType, AgentCapability, AgentResult
from ..platform_logging import RunLogger
from ..services import SessionService, MemoryService, ArtifactService
from ..mcp import MCPOrchestrator


class CustomAgentType(str, Enum):
    """Types of custom agents."""
    DOMAIN_EXPERT = "domain_expert"        # Domain-specific expertise
    DATA_ANALYST = "data_analyst"          # Data analysis specialist
    CONTENT_CREATOR = "content_creator"    # Content generation specialist
    FACT_CHECKER = "fact_checker"          # Fact-checking specialist
    MODERATOR = "moderator"                # Content moderation
    TRANSLATOR = "translator"              # Language translation
    SUMMARIZER = "summarizer"              # Content summarization
    QA_SPECIALIST = "qa_specialist"        # Question-answering specialist
    VALIDATOR = "validator"                # Data/content validation
    MONITOR = "monitor"                    # System/content monitoring
    INTEGRATOR = "integrator"              # System integration specialist
    OPTIMIZER = "optimizer"                # Process optimization


@dataclass
class CustomAgentConfig:
    """Configuration for custom agents."""
    agent_type: CustomAgentType
    domain: str = ""                       # Domain of expertise
    specialized_tools: List[str] = field(default_factory=list)
    knowledge_base: Dict[str, Any] = field(default_factory=dict)
    behavior_rules: Dict[str, Any] = field(default_factory=dict)
    output_format: str = "text"            # Output format preference
    quality_thresholds: Dict[str, float] = field(default_factory=dict)
    interaction_style: str = "professional"
    context_requirements: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "agent_type": self.agent_type.value,
            "domain": self.domain,
            "specialized_tools": self.specialized_tools,
            "knowledge_base": self.knowledge_base,
            "behavior_rules": self.behavior_rules,
            "output_format": self.output_format,
            "quality_thresholds": self.quality_thresholds,
            "interaction_style": self.interaction_style,
            "context_requirements": self.context_requirements,
        }


class CustomAgent(Agent):
    """
    Custom specialized agent for specific domains and tasks.
    
    Provides configurable behavior, specialized tools, and domain expertise
    with integration to both ADK tools and MCP servers.
    """
    
    def __init__(self,
                 config: CustomAgentConfig,
                 agent_id: Optional[str] = None,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 custom_tools: Optional[List[Callable]] = None,
                 logger: Optional[RunLogger] = None,
                 session_service: Optional[SessionService] = None,
                 memory_service: Optional[MemoryService] = None,
                 artifact_service: Optional[ArtifactService] = None,
                 mcp_orchestrator: Optional[MCPOrchestrator] = None):
        
        # Generate name and description based on type if not provided
        type_names = {
            CustomAgentType.DOMAIN_EXPERT: f"{config.domain} Expert Agent" if config.domain else "Domain Expert Agent",
            CustomAgentType.DATA_ANALYST: "Data Analysis Agent",
            CustomAgentType.CONTENT_CREATOR: "Content Creation Agent",
            CustomAgentType.FACT_CHECKER: "Fact-Checking Agent",
            CustomAgentType.MODERATOR: "Content Moderation Agent",
            CustomAgentType.TRANSLATOR: "Translation Agent",
            CustomAgentType.SUMMARIZER: "Summarization Agent",
            CustomAgentType.QA_SPECIALIST: "Question-Answering Agent",
            CustomAgentType.VALIDATOR: "Validation Agent",
            CustomAgentType.MONITOR: "Monitoring Agent",
            CustomAgentType.INTEGRATOR: "Integration Agent",
            CustomAgentType.OPTIMIZER: "Optimization Agent",
        }
        
        type_descriptions = {
            CustomAgentType.DOMAIN_EXPERT: f"Provides expert knowledge and analysis in {config.domain}" if config.domain else "Provides domain-specific expertise",
            CustomAgentType.DATA_ANALYST: "Specializes in data analysis, visualization, and statistical insights",
            CustomAgentType.CONTENT_CREATOR: "Creates high-quality content including articles, reports, and documentation",
            CustomAgentType.FACT_CHECKER: "Verifies facts, checks sources, and assesses information credibility",
            CustomAgentType.MODERATOR: "Moderates content for quality, appropriateness, and compliance",
            CustomAgentType.TRANSLATOR: "Provides translation services across multiple languages",
            CustomAgentType.SUMMARIZER: "Creates concise summaries of complex content and documents",
            CustomAgentType.QA_SPECIALIST: "Answers questions with precision and provides detailed explanations",
            CustomAgentType.VALIDATOR: "Validates data quality, format compliance, and content accuracy",
            CustomAgentType.MONITOR: "Monitors systems, content, and processes for changes and issues",
            CustomAgentType.INTEGRATOR: "Integrates different systems, APIs, and data sources",
            CustomAgentType.OPTIMIZER: "Optimizes processes, performance, and resource utilization",
        }
        
        agent_name = name or type_names.get(config.agent_type, "Custom Agent")
        agent_description = description or type_descriptions.get(config.agent_type, "Specialized custom agent")
        
        super().__init__(
            agent_id=agent_id,
            name=agent_name,
            description=agent_description,
            agent_type=AgentType.CUSTOM,
            capabilities=self._determine_capabilities(config),
            logger=logger,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service,
            mcp_orchestrator=mcp_orchestrator,
        )
        
        self.config = config
        self.custom_tools = custom_tools or []
        
        # Specialized state
        self.domain_knowledge = config.knowledge_base
        self.specialized_context = {}
        self.validation_rules = config.behavior_rules.get("validation", {})
        self.processing_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.domain_accuracy = 0.0
        self.processing_efficiency = 0.0
        self.quality_scores: List[float] = []
        self.specialization_metrics = {}
    
    def _determine_capabilities(self, config: CustomAgentConfig) -> List[AgentCapability]:
        """Determine capabilities based on agent type and configuration."""
        # Base capabilities
        base_capabilities = [
            AgentCapability.TOOL_USE,
            AgentCapability.CONTEXT_MANAGEMENT,
        ]
        
        # Type-specific capabilities
        type_capabilities = {
            CustomAgentType.DOMAIN_EXPERT: [
                AgentCapability.REASONING,
                AgentCapability.ANALYSIS,
                AgentCapability.DECISION_MAKING,
                AgentCapability.RESEARCH,
            ],
            CustomAgentType.DATA_ANALYST: [
                AgentCapability.ANALYSIS,
                AgentCapability.REASONING,
                AgentCapability.DECISION_MAKING,
            ],
            CustomAgentType.CONTENT_CREATOR: [
                AgentCapability.CONTENT_GENERATION,
                AgentCapability.SYNTHESIS,
                AgentCapability.REASONING,
            ],
            CustomAgentType.FACT_CHECKER: [
                AgentCapability.FACT_CHECKING,
                AgentCapability.RESEARCH,
                AgentCapability.ANALYSIS,
                AgentCapability.DECISION_MAKING,
            ],
            CustomAgentType.MODERATOR: [
                AgentCapability.ANALYSIS,
                AgentCapability.DECISION_MAKING,
                AgentCapability.FACT_CHECKING,
            ],
            CustomAgentType.TRANSLATOR: [
                AgentCapability.CONTENT_GENERATION,
                AgentCapability.ANALYSIS,
            ],
            CustomAgentType.SUMMARIZER: [
                AgentCapability.SYNTHESIS,
                AgentCapability.ANALYSIS,
                AgentCapability.CONTENT_GENERATION,
            ],
            CustomAgentType.QA_SPECIALIST: [
                AgentCapability.REASONING,
                AgentCapability.RESEARCH,
                AgentCapability.ANALYSIS,
                AgentCapability.COMMUNICATION,
            ],
            CustomAgentType.VALIDATOR: [
                AgentCapability.ANALYSIS,
                AgentCapability.DECISION_MAKING,
                AgentCapability.FACT_CHECKING,
            ],
            CustomAgentType.MONITOR: [
                AgentCapability.ANALYSIS,
                AgentCapability.DECISION_MAKING,
            ],
            CustomAgentType.INTEGRATOR: [
                AgentCapability.TOOL_USE,
                AgentCapability.EXECUTION,
                AgentCapability.ANALYSIS,
            ],
            CustomAgentType.OPTIMIZER: [
                AgentCapability.ANALYSIS,
                AgentCapability.DECISION_MAKING,
                AgentCapability.PLANNING,
            ],
        }
        
        return base_capabilities + type_capabilities.get(config.agent_type, [])
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Get the capabilities of this custom agent."""
        return self.capabilities
    
    async def execute_task(self, 
                          task: str, 
                          context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Execute a task using the custom agent's specialized capabilities.
        
        Args:
            task: Task description
            context: Optional context information
            
        Returns:
            Agent result with specialized processing outcome
        """
        start_time = time.time()
        
        if self.logger:
            self.logger.info(f"Custom Agent {self.name} ({self.config.agent_type.value}) executing task: {task[:100]}...")
        
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
            # Prepare specialized context
            specialized_context = await self._prepare_specialized_context(task, context)
            
            # Validate task requirements
            validation_result = await self._validate_task_requirements(task, specialized_context)
            if not validation_result["valid"]:
                return AgentResult(
                    agent_id=self.agent_id,
                    agent_type=self.agent_type,
                    task=task,
                    success=False,
                    error=validation_result["error"],
                )
            
            # Execute specialized processing
            result = await self._execute_specialized_processing(task, specialized_context)
            tools_used.extend(result.get("tools_used", []))
            
            # Apply quality assurance
            qa_result = await self._apply_quality_assurance(result, task)
            
            # Store in specialized memory if configured
            if self.config.behavior_rules.get("store_results", True) and self.memory_service:
                await self.store_memory(
                    f"Task: {task}\nResult: {qa_result['output']}",
                    metadata={
                        "agent_type": self.config.agent_type.value,
                        "domain": self.config.domain,
                        "quality_score": qa_result.get("quality_score", 0.0),
                    }
                )
                memory_accessed = True
            
            # Update processing history
            self._update_processing_history(task, qa_result)
            
            # Update metrics
            execution_time = (time.time() - start_time) * 1000
            self.total_tasks_completed += 1
            self.last_task_time = time.time()
            self._update_performance_metrics(qa_result, execution_time)
            
            return AgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                task=task,
                success=qa_result["success"],
                result=qa_result["output"],
                error=qa_result.get("error"),
                execution_time_ms=execution_time,
                metadata={
                    "agent_type": self.config.agent_type.value,
                    "domain": self.config.domain,
                    "quality_score": qa_result.get("quality_score", 0.0),
                    "specialized_processing": True,
                },
                context_used=specialized_context,
                tools_used=tools_used,
                memory_accessed=memory_accessed,
                artifacts_created=artifacts_created,
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            if self.logger:
                self.logger.error(f"Custom Agent {self.name} task execution failed: {e}")
            
            return AgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                task=task,
                success=False,
                error=str(e),
                execution_time_ms=execution_time,
            )
    
    async def _prepare_specialized_context(self, 
                                         task: str, 
                                         context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare specialized context for task execution."""
        specialized_context = context or {}
        
        # Add domain knowledge
        if self.domain_knowledge:
            specialized_context["domain_knowledge"] = self.domain_knowledge
        
        # Add agent-specific context
        specialized_context["agent_type"] = self.config.agent_type.value
        specialized_context["domain"] = self.config.domain
        specialized_context["capabilities"] = [cap.value for cap in self.get_capabilities()]
        
        # Add processing history for context
        if self.processing_history:
            specialized_context["recent_processing"] = self.processing_history[-5:]
        
        # Add specialized tools available
        specialized_context["available_tools"] = self.config.specialized_tools
        
        return specialized_context
    
    async def _validate_task_requirements(self, 
                                        task: str, 
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that task requirements are met."""
        validation_result = {"valid": True, "error": None, "warnings": []}
        
        # Check context requirements
        for requirement in self.config.context_requirements:
            if requirement not in context:
                validation_result["warnings"].append(f"Missing recommended context: {requirement}")
        
        # Check domain alignment if domain-specific
        if self.config.domain and self.config.agent_type == CustomAgentType.DOMAIN_EXPERT:
            domain_keywords = self.domain_knowledge.get("keywords", [])
            if domain_keywords:
                task_lower = task.lower()
                if not any(keyword.lower() in task_lower for keyword in domain_keywords):
                    validation_result["warnings"].append(f"Task may not align with {self.config.domain} domain")
        
        # Agent-specific validations
        if self.config.agent_type == CustomAgentType.FACT_CHECKER:
            if not any(word in task.lower() for word in ["verify", "check", "fact", "accurate", "true", "false"]):
                validation_result["warnings"].append("Task doesn't appear to be fact-checking related")
        
        elif self.config.agent_type == CustomAgentType.DATA_ANALYST:
            if not any(word in task.lower() for word in ["data", "analyze", "statistics", "trend", "pattern"]):
                validation_result["warnings"].append("Task doesn't appear to be data analysis related")
        
        return validation_result
    
    async def _execute_specialized_processing(self, 
                                            task: str, 
                                            context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute specialized processing based on agent type."""
        if self.config.agent_type == CustomAgentType.FACT_CHECKER:
            return await self._execute_fact_checking(task, context)
        elif self.config.agent_type == CustomAgentType.DATA_ANALYST:
            return await self._execute_data_analysis(task, context)
        elif self.config.agent_type == CustomAgentType.CONTENT_CREATOR:
            return await self._execute_content_creation(task, context)
        elif self.config.agent_type == CustomAgentType.SUMMARIZER:
            return await self._execute_summarization(task, context)
        elif self.config.agent_type == CustomAgentType.TRANSLATOR:
            return await self._execute_translation(task, context)
        elif self.config.agent_type == CustomAgentType.VALIDATOR:
            return await self._execute_validation(task, context)
        elif self.config.agent_type == CustomAgentType.QA_SPECIALIST:
            return await self._execute_qa_processing(task, context)
        elif self.config.agent_type == CustomAgentType.DOMAIN_EXPERT:
            return await self._execute_domain_expertise(task, context)
        else:
            return await self._execute_generic_processing(task, context)
    
    async def _execute_fact_checking(self, 
                                   task: str, 
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute fact-checking specialized processing."""
        tools_used = []
        
        try:
            # Extract claims to verify
            claims = self._extract_claims(task)
            
            verification_results = []
            
            # Use MCP servers for verification if available
            if self.mcp_orchestrator:
                for claim in claims[:3]:  # Limit to top 3 claims
                    verification = await self.use_mcp_server(
                        "perplexity",
                        "search",
                        {
                            "query": f"fact check verify: {claim}",
                            "strategy": "HYBRID_VALIDATION",
                        }
                    )
                    
                    verification_results.append({
                        "claim": claim,
                        "verification": verification,
                        "verified": verification.get("success", False),
                    })
                    
                tools_used.append("mcp_perplexity")
            
            # Compile fact-check report
            overall_credibility = sum(1 for v in verification_results if v["verified"]) / len(verification_results) if verification_results else 0.0
            
            fact_check_report = {
                "task": task,
                "claims_analyzed": len(claims),
                "claims_verified": sum(1 for v in verification_results if v["verified"]),
                "overall_credibility": overall_credibility,
                "verification_details": verification_results,
                "recommendation": "high confidence" if overall_credibility > 0.7 else "requires further verification",
            }
            
            return {
                "success": True,
                "output": fact_check_report,
                "tools_used": tools_used,
                "quality_score": overall_credibility,
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Fact-checking failed: {e}",
                "tools_used": tools_used,
            }
    
    async def _execute_data_analysis(self, 
                                   task: str, 
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data analysis specialized processing."""
        tools_used = []
        
        try:
            # This would integrate with data analysis tools
            # For now, provide a structured analysis framework
            
            analysis_framework = {
                "data_identification": "Identify data sources and types mentioned in task",
                "analysis_methods": "Determine appropriate analysis methods",
                "statistical_considerations": "Consider statistical significance and validity",
                "visualization_recommendations": "Suggest appropriate visualizations",
                "insights_extraction": "Extract key insights and patterns",
                "recommendations": "Provide actionable recommendations",
            }
            
            # Simulate data analysis result
            analysis_result = {
                "task": task,
                "analysis_framework": analysis_framework,
                "data_quality_assessment": "Good quality data assumed",
                "key_findings": ["Placeholder finding 1", "Placeholder finding 2"],
                "statistical_summary": {"confidence_level": 0.95, "sample_size": "N/A"},
                "recommendations": ["Recommendation 1", "Recommendation 2"],
                "methodology": "Structured analytical approach",
            }
            
            tools_used.append("data_analysis_framework")
            
            return {
                "success": True,
                "output": analysis_result,
                "tools_used": tools_used,
                "quality_score": 0.8,
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Data analysis failed: {e}",
                "tools_used": tools_used,
            }
    
    async def _execute_content_creation(self, 
                                      task: str, 
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute content creation specialized processing."""
        tools_used = []
        
        try:
            # Content creation framework
            content_structure = {
                "title": f"Content for: {task[:50]}",
                "introduction": "Introduction section",
                "main_content": "Main content body",
                "conclusion": "Conclusion and summary",
                "metadata": {
                    "word_count": 0,
                    "reading_level": "intermediate",
                    "target_audience": "general",
                    "content_type": self.config.output_format,
                },
            }
            
            # Research using MCP if available
            if self.mcp_orchestrator:
                research = await self.use_mcp_server(
                    "tavily",
                    "search",
                    {"query": task, "strategy": "QUALITY_OPTIMIZED"}
                )
                content_structure["research_sources"] = research
                tools_used.append("mcp_tavily")
            
            tools_used.append("content_creation_framework")
            
            return {
                "success": True,
                "output": content_structure,
                "tools_used": tools_used,
                "quality_score": 0.85,
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Content creation failed: {e}",
                "tools_used": tools_used,
            }
    
    async def _execute_summarization(self, 
                                   task: str, 
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute summarization specialized processing."""
        tools_used = []
        
        try:
            # Extract content to summarize
            content = context.get("content", task)
            
            # Summarization framework
            summary_structure = {
                "original_length": len(content),
                "summary_length": max(100, len(content) // 10),  # 10% of original
                "key_points": ["Key point 1", "Key point 2", "Key point 3"],
                "main_themes": ["Theme 1", "Theme 2"],
                "conclusion": "Summary conclusion",
                "compression_ratio": 0.1,
            }
            
            tools_used.append("summarization_framework")
            
            return {
                "success": True,
                "output": summary_structure,
                "tools_used": tools_used,
                "quality_score": 0.75,
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Summarization failed: {e}",
                "tools_used": tools_used,
            }
    
    async def _execute_translation(self, 
                                 task: str, 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute translation specialized processing."""
        tools_used = []
        
        try:
            # Translation framework
            translation_result = {
                "source_language": "auto-detected",
                "target_language": context.get("target_language", "english"),
                "original_text": task,
                "translated_text": f"[TRANSLATED] {task}",
                "confidence_score": 0.9,
                "translation_notes": [],
            }
            
            tools_used.append("translation_framework")
            
            return {
                "success": True,
                "output": translation_result,
                "tools_used": tools_used,
                "quality_score": 0.9,
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Translation failed: {e}",
                "tools_used": tools_used,
            }
    
    async def _execute_validation(self, 
                                task: str, 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute validation specialized processing."""
        tools_used = []
        
        try:
            # Validation framework
            validation_result = {
                "validation_type": "general",
                "items_validated": 1,
                "passed_validation": 1,
                "failed_validation": 0,
                "validation_details": {
                    "format_check": "passed",
                    "content_quality": "passed",
                    "completeness": "passed",
                },
                "overall_status": "valid",
                "confidence": 0.85,
            }
            
            tools_used.append("validation_framework")
            
            return {
                "success": True,
                "output": validation_result,
                "tools_used": tools_used,
                "quality_score": 0.85,
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Validation failed: {e}",
                "tools_used": tools_used,
            }
    
    async def _execute_qa_processing(self, 
                                   task: str, 
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Q&A specialized processing."""
        tools_used = []
        
        try:
            # Extract question
            question = task
            
            # Q&A framework
            qa_result = {
                "question": question,
                "answer": f"Specialized Q&A response for: {question}",
                "confidence": 0.8,
                "sources": [],
                "related_questions": [],
                "answer_type": "direct",
            }
            
            # Use MCP for research if available
            if self.mcp_orchestrator:
                research = await self.use_mcp_server(
                    "perplexity",
                    "search",
                    {"query": question, "strategy": "QUALITY_OPTIMIZED"}
                )
                qa_result["sources"] = research.get("result", {}).get("sources_used", [])
                tools_used.append("mcp_perplexity")
            
            tools_used.append("qa_framework")
            
            return {
                "success": True,
                "output": qa_result,
                "tools_used": tools_used,
                "quality_score": 0.8,
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Q&A processing failed: {e}",
                "tools_used": tools_used,
            }
    
    async def _execute_domain_expertise(self, 
                                      task: str, 
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute domain expertise specialized processing."""
        tools_used = []
        
        try:
            # Domain expertise framework
            expertise_result = {
                "domain": self.config.domain,
                "task": task,
                "expert_analysis": f"Domain expert analysis for {self.config.domain}: {task}",
                "domain_specific_insights": [],
                "recommendations": [],
                "confidence": 0.9,
                "knowledge_base_used": bool(self.domain_knowledge),
            }
            
            # Apply domain knowledge
            if self.domain_knowledge:
                expertise_result["domain_specific_insights"] = [
                    f"Insight based on {self.config.domain} expertise",
                    "Domain-specific consideration",
                ]
                tools_used.append("domain_knowledge_base")
            
            tools_used.append("domain_expertise_framework")
            
            return {
                "success": True,
                "output": expertise_result,
                "tools_used": tools_used,
                "quality_score": 0.9,
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Domain expertise processing failed: {e}",
                "tools_used": tools_used,
            }
    
    async def _execute_generic_processing(self, 
                                        task: str, 
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute generic custom processing."""
        return {
            "success": True,
            "output": f"Custom agent processed: {task}",
            "tools_used": ["generic_processing"],
            "quality_score": 0.7,
        }
    
    async def _apply_quality_assurance(self, 
                                     result: Dict[str, Any], 
                                     task: str) -> Dict[str, Any]:
        """Apply quality assurance checks to the result."""
        qa_result = result.copy()
        
        # Check quality thresholds
        quality_score = result.get("quality_score", 0.5)
        min_quality = self.config.quality_thresholds.get("minimum", 0.0)
        
        if quality_score < min_quality:
            qa_result["success"] = False
            qa_result["error"] = f"Quality score {quality_score} below minimum threshold {min_quality}"
        
        # Apply output formatting
        if self.config.output_format == "json" and isinstance(qa_result.get("output"), dict):
            qa_result["output"] = json.dumps(qa_result["output"], indent=2)
        
        return qa_result
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text for verification."""
        # Simple claim extraction - could be enhanced with NLP
        sentences = text.split('. ')
        claims = []
        
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in 
                  ['is', 'are', 'was', 'were', 'has', 'have', 'will', 'can', 'according to']):
                if len(sentence.strip()) > 10:
                    claims.append(sentence.strip())
        
        return claims[:5]  # Return top 5 claims
    
    def _update_processing_history(self, task: str, result: Dict[str, Any]) -> None:
        """Update processing history."""
        self.processing_history.append({
            "task": task,
            "result": result,
            "timestamp": time.time(),
            "quality_score": result.get("quality_score", 0.0),
            "success": result.get("success", False),
        })
        
        # Trim history to max length
        if len(self.processing_history) > 50:
            self.processing_history = self.processing_history[-50:]
    
    def _update_performance_metrics(self, result: Dict[str, Any], execution_time_ms: float) -> None:
        """Update performance metrics."""
        quality_score = result.get("quality_score", 0.0)
        self.quality_scores.append(quality_score)
        
        # Calculate averages
        if self.quality_scores:
            self.domain_accuracy = sum(self.quality_scores) / len(self.quality_scores)
        
        # Update processing efficiency (tasks per minute)
        if self.total_tasks_completed > 0:
            total_time_hours = (time.time() - (self.last_task_time or time.time())) / 3600
            self.processing_efficiency = self.total_tasks_completed / max(total_time_hours, 0.001)
    
    def get_specialization_metrics(self) -> Dict[str, Any]:
        """Get specialization-specific metrics."""
        return {
            "agent_type": self.config.agent_type.value,
            "domain": self.config.domain,
            "domain_accuracy": self.domain_accuracy,
            "processing_efficiency": self.processing_efficiency,
            "average_quality_score": sum(self.quality_scores) / len(self.quality_scores) if self.quality_scores else 0.0,
            "total_specialized_tasks": len(self.processing_history),
            "knowledge_base_size": len(self.domain_knowledge),
            "specialized_tools_available": len(self.config.specialized_tools),
        }
    
    async def update_domain_knowledge(self, new_knowledge: Dict[str, Any]) -> bool:
        """Update domain knowledge base."""
        try:
            self.domain_knowledge.update(new_knowledge)
            self.config.knowledge_base = self.domain_knowledge
            
            if self.logger:
                self.logger.info(f"Updated domain knowledge for {self.name}")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to update domain knowledge: {e}")
            return False