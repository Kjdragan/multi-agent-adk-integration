"""
Agent Orchestrator for Multi-Agent Coordination

Provides sophisticated orchestration across multiple agents including task allocation,
agent coordination, workflow management, and performance optimization.
"""

import time
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum

from .base import Agent, AgentType, AgentCapability, AgentResult, AgentRegistry
from ..platform_logging import RunLogger
from ..services import SessionService, MemoryService, ArtifactService
from ..mcp import MCPOrchestrator


class OrchestrationStrategy(str, Enum):
    """Strategies for agent orchestration."""
    SINGLE_BEST = "single_best"           # Use single best agent for task
    PARALLEL_ALL = "parallel_all"         # Execute with all capable agents in parallel
    SEQUENTIAL = "sequential"             # Execute with agents sequentially
    HIERARCHICAL = "hierarchical"         # Hierarchical execution with lead agent
    CONSENSUS = "consensus"               # Execute with multiple agents and find consensus
    PIPELINE = "pipeline"                 # Pipeline execution through multiple agents
    COMPETITIVE = "competitive"           # Competitive execution, best result wins
    COLLABORATIVE = "collaborative"       # Collaborative execution with coordination
    ADAPTIVE = "adaptive"                 # Adaptive strategy based on task complexity


class TaskPriority(str, Enum):
    """Priority levels for tasks."""
    CRITICAL = "critical"                 # Highest priority
    HIGH = "high"                        # High priority
    MEDIUM = "medium"                    # Medium priority
    LOW = "low"                          # Low priority
    BACKGROUND = "background"            # Background processing


@dataclass
class TaskAllocation:
    """Task allocation to agents."""
    task_id: str
    task: str
    assigned_agents: List[str] = field(default_factory=list)
    strategy: OrchestrationStrategy = OrchestrationStrategy.SINGLE_BEST
    priority: TaskPriority = TaskPriority.MEDIUM
    requirements: List[AgentCapability] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Execution state
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    results: Dict[str, AgentResult] = field(default_factory=dict)
    final_result: Optional[Any] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "task_id": self.task_id,
            "task": self.task,
            "assigned_agents": self.assigned_agents,
            "strategy": self.strategy.value,
            "priority": self.priority.value,
            "requirements": [req.value for req in self.requirements],
            "context": self.context,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "results": {agent_id: result.to_dict() for agent_id, result in self.results.items()},
            "final_result": self.final_result,
            "error": self.error,
        }


@dataclass
class OrchestrationResult:
    """Result from agent orchestration."""
    task_id: str
    success: bool
    strategy_used: OrchestrationStrategy
    agents_used: List[str]
    primary_result: Any
    all_results: Dict[str, Any] = field(default_factory=dict)
    consensus_score: Optional[float] = None
    execution_time_ms: float = 0.0
    coordination_overhead_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "task_id": self.task_id,
            "success": self.success,
            "strategy_used": self.strategy_used.value,
            "agents_used": self.agents_used,
            "primary_result": self.primary_result,
            "all_results": self.all_results,
            "consensus_score": self.consensus_score,
            "execution_time_ms": self.execution_time_ms,
            "coordination_overhead_ms": self.coordination_overhead_ms,
            "error": self.error,
            "metadata": self.metadata,
        }


class AgentOrchestrator:
    """
    Agent orchestrator for sophisticated multi-agent coordination.
    
    Manages task allocation, agent coordination, workflow optimization,
    and performance monitoring across the multi-agent system.
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
        
        # Task management
        self.active_tasks: Dict[str, TaskAllocation] = {}
        self.task_queue: List[TaskAllocation] = []
        self.completed_tasks: List[TaskAllocation] = []
        
        # Resource management settings
        self.max_completed_tasks_history = 1000  # Limit completed task history
        self.metrics_cleanup_threshold = 10000   # Cleanup metrics after this many orchestrations
        self.last_cleanup_time = time.time()
        
        # Orchestration settings
        self.max_concurrent_tasks = 10
        self.default_timeout_seconds = 300  # 5 minutes
        self.enable_adaptive_strategies = True
        self.enable_performance_optimization = True
        
        # Performance tracking
        self.total_tasks_orchestrated = 0
        self.successful_orchestrations = 0
        self.failed_orchestrations = 0
        self.average_coordination_time_ms = 0.0
        self.agent_performance_scores: Dict[str, float] = {}
        self.strategy_success_rates: Dict[OrchestrationStrategy, float] = {}
        
        # Agent coordination state
        self.agent_workloads: Dict[str, int] = {}
        self.agent_availability: Dict[str, bool] = {}
        self.agent_specializations: Dict[str, List[str]] = {}
        
        # Strategy configurations
        self.strategy_configs = {
            OrchestrationStrategy.SINGLE_BEST: {"selection_criteria": "performance"},
            OrchestrationStrategy.PARALLEL_ALL: {"max_agents": 5},
            OrchestrationStrategy.CONSENSUS: {"min_agents": 3, "consensus_threshold": 0.7},
            OrchestrationStrategy.PIPELINE: {"preserve_order": True},
            OrchestrationStrategy.COMPETITIVE: {"evaluation_criteria": "quality"},
        }
    
    async def orchestrate_task(self, 
                              task: str,
                              strategy: OrchestrationStrategy = OrchestrationStrategy.ADAPTIVE,
                              requirements: Optional[List[AgentCapability]] = None,
                              context: Optional[Dict[str, Any]] = None,
                              priority: TaskPriority = TaskPriority.MEDIUM,
                              timeout_seconds: Optional[int] = None) -> OrchestrationResult:
        """
        Orchestrate a task across multiple agents.
        
        Args:
            task: Task description
            strategy: Orchestration strategy to use
            requirements: Required agent capabilities
            context: Task context
            priority: Task priority
            timeout_seconds: Optional timeout for task execution
            
        Returns:
            Orchestration result with outcomes from agents
        """
        start_time = time.time()
        
        if self.logger:
            self.logger.info(f"Orchestrating task with {strategy.value} strategy: {task[:100]}...")
        
        # Create task allocation
        task_id = f"task_{int(time.time() * 1000)}"
        task_allocation = TaskAllocation(
            task_id=task_id,
            task=task,
            strategy=strategy,
            priority=priority,
            requirements=requirements or [],
            context=context or {},
            timeout_seconds=timeout_seconds or self.default_timeout_seconds,
        )
        
        try:
            # Determine optimal strategy if adaptive
            if strategy == OrchestrationStrategy.ADAPTIVE:
                strategy = await self._determine_optimal_strategy(task, requirements)
                task_allocation.strategy = strategy
            
            # Select appropriate agents
            selected_agents = await self._select_agents(task_allocation)
            task_allocation.assigned_agents = [agent.agent_id for agent in selected_agents]
            
            if not selected_agents:
                return OrchestrationResult(
                    task_id=task_id,
                    success=False,
                    strategy_used=strategy,
                    agents_used=[],
                    primary_result=None,
                    error="No suitable agents found for task",
                )
            
            # Execute orchestration strategy with timeout protection
            self.active_tasks[task_id] = task_allocation
            task_allocation.status = "running"
            task_allocation.start_time = time.time()
            
            try:
                # Apply timeout protection to orchestration execution
                timeout = task_allocation.timeout_seconds
                if timeout and timeout > 0:
                    orchestration_result = await asyncio.wait_for(
                        self._execute_orchestration_strategy(task_allocation, selected_agents),
                        timeout=timeout
                    )
                else:
                    orchestration_result = await self._execute_orchestration_strategy(
                        task_allocation, selected_agents
                    )
            except asyncio.TimeoutError:
                # Handle timeout with proper cleanup
                orchestration_result = OrchestrationResult(
                    task_id=task_id,
                    success=False,
                    strategy_used=strategy,
                    agents_used=[agent.agent_id for agent in selected_agents],
                    primary_result=None,
                    error=f"Task execution timed out after {timeout} seconds"
                )
                
                if self.logger:
                    self.logger.warning(f"Task {task_id} timed out after {timeout} seconds")
                
                # Update agent workloads (reduce since task was cancelled)
                for agent in selected_agents:
                    current_workload = self.agent_workloads.get(agent.agent_id, 0)
                    self.agent_workloads[agent.agent_id] = max(0, current_workload - 1)
            
            # Update performance metrics
            execution_time = (time.time() - start_time) * 1000
            orchestration_result.execution_time_ms = execution_time
            
            self.total_tasks_orchestrated += 1
            if orchestration_result.success:
                self.successful_orchestrations += 1
            else:
                self.failed_orchestrations += 1
            
            self._update_performance_metrics(orchestration_result, selected_agents)
            
            # Store in memory if successful
            if orchestration_result.success and self.memory_service:
                await self.memory_service.store(
                    text=f"Orchestrated task: {task}\nStrategy: {strategy.value}\nResult: {orchestration_result.primary_result}",
                    metadata={
                        "task_id": task_id,
                        "strategy": strategy.value,
                        "agents_used": orchestration_result.agents_used,
                        "success": orchestration_result.success,
                    }
                )
            
            if self.logger:
                self.logger.info(f"Completed task orchestration {task_id}: success={orchestration_result.success}")
            
            return orchestration_result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            if self.logger:
                self.logger.error(f"Task orchestration {task_id} failed: {e}")
            
            return OrchestrationResult(
                task_id=task_id,
                success=False,
                strategy_used=strategy,
                agents_used=[],
                primary_result=None,
                execution_time_ms=execution_time,
                error=str(e),
            )
        
        finally:
            # Clean up
            if task_id in self.active_tasks:
                completed_task = self.active_tasks[task_id]
                completed_task.status = "completed"
                completed_task.end_time = time.time()
                self.completed_tasks.append(completed_task)
                del self.active_tasks[task_id]
            
            # Periodic resource cleanup
            self._cleanup_resources_if_needed()
    
    def _cleanup_resources_if_needed(self) -> None:
        """Perform periodic cleanup to prevent memory leaks."""
        current_time = time.time()
        
        # Cleanup every 5 minutes or when metrics reach threshold
        should_cleanup = (
            current_time - self.last_cleanup_time > 300 or  # 5 minutes
            self.total_tasks_orchestrated > self.metrics_cleanup_threshold
        )
        
        if not should_cleanup:
            return
        
        # Limit completed tasks history
        if len(self.completed_tasks) > self.max_completed_tasks_history:
            # Keep only the most recent tasks
            excess_count = len(self.completed_tasks) - self.max_completed_tasks_history
            self.completed_tasks = self.completed_tasks[excess_count:]
            
            if self.logger:
                self.logger.info(f"Cleaned up {excess_count} old completed tasks")
        
        # Reset performance metrics if they grow too large
        if len(self.agent_performance_scores) > 1000:
            # Keep only scores above threshold to maintain performance data for active agents
            filtered_scores = {
                agent_id: score for agent_id, score in self.agent_performance_scores.items()
                if score > 0.3  # Keep only agents with decent performance
            }
            cleared_count = len(self.agent_performance_scores) - len(filtered_scores)
            self.agent_performance_scores = filtered_scores
            
            if self.logger:
                self.logger.info(f"Cleaned up {cleared_count} low-performing agent metrics")
        
        # Clean up agent workload tracking for inactive agents
        active_agent_ids = {agent.agent_id for agent in AgentRegistry.get_all_agents() if agent.is_active}
        workload_keys_to_remove = [
            agent_id for agent_id in self.agent_workloads.keys()
            if agent_id not in active_agent_ids
        ]
        for agent_id in workload_keys_to_remove:
            del self.agent_workloads[agent_id]
            self.agent_availability.pop(agent_id, None)
        
        if workload_keys_to_remove and self.logger:
            self.logger.info(f"Cleaned up workload tracking for {len(workload_keys_to_remove)} inactive agents")
        
        self.last_cleanup_time = current_time
    
    async def _determine_optimal_strategy(self, 
                                        task: str,
                                        requirements: List[AgentCapability]) -> OrchestrationStrategy:
        """Determine optimal orchestration strategy for a task."""
        # Analyze task complexity
        task_complexity = self._assess_task_complexity(task, requirements)
        
        # Get available agents
        available_agents = AgentRegistry.get_all_agents()
        capable_agents = [
            agent for agent in available_agents 
            if agent.can_handle_task(task, requirements) and agent.is_active
        ]
        
        # Strategy selection logic
        if len(capable_agents) == 0:
            return OrchestrationStrategy.SINGLE_BEST
        elif len(capable_agents) == 1:
            return OrchestrationStrategy.SINGLE_BEST
        elif task_complexity >= 0.8:  # High complexity
            if len(capable_agents) >= 3:
                return OrchestrationStrategy.CONSENSUS
            else:
                return OrchestrationStrategy.COLLABORATIVE
        elif task_complexity >= 0.6:  # Medium complexity
            return OrchestrationStrategy.PIPELINE
        elif len(capable_agents) >= 3 and any(req in [AgentCapability.FACT_CHECKING, AgentCapability.ANALYSIS] for req in requirements):
            return OrchestrationStrategy.COMPETITIVE
        else:
            return OrchestrationStrategy.SINGLE_BEST
    
    def _assess_task_complexity(self, 
                              task: str, 
                              requirements: List[AgentCapability]) -> float:
        """Assess task complexity on a scale of 0-1."""
        complexity_score = 0.0
        
        # Length-based complexity
        complexity_score += min(len(task) / 1000, 0.3)
        
        # Requirements-based complexity
        complexity_score += len(requirements) * 0.1
        
        # Keyword-based complexity
        complex_keywords = [
            "analyze", "research", "comprehensive", "detailed", "complex",
            "multi-step", "workflow", "orchestrate", "coordinate", "integrate"
        ]
        
        keyword_matches = sum(1 for keyword in complex_keywords if keyword in task.lower())
        complexity_score += keyword_matches * 0.1
        
        return min(complexity_score, 1.0)
    
    async def _select_agents(self, task_allocation: TaskAllocation) -> List[Agent]:
        """Select appropriate agents for task execution."""
        # Get capable agents
        capable_agents = AgentRegistry.find_capable_agents(
            task_allocation.requirements,
            None  # Don't filter by type
        )
        
        # Filter active agents
        active_agents = [agent for agent in capable_agents if agent.is_active]
        
        if not active_agents:
            return []
        
        # Strategy-specific selection
        strategy = task_allocation.strategy
        
        if strategy == OrchestrationStrategy.SINGLE_BEST:
            return [self._select_best_agent(active_agents, task_allocation)]
        
        elif strategy == OrchestrationStrategy.PARALLEL_ALL:
            max_agents = self.strategy_configs[strategy]["max_agents"]
            return self._select_top_agents(active_agents, task_allocation, max_agents)
        
        elif strategy == OrchestrationStrategy.CONSENSUS:
            min_agents = self.strategy_configs[strategy]["min_agents"]
            return self._select_diverse_agents(active_agents, task_allocation, min_agents)
        
        elif strategy == OrchestrationStrategy.PIPELINE:
            return self._select_pipeline_agents(active_agents, task_allocation)
        
        elif strategy == OrchestrationStrategy.COMPETITIVE:
            return self._select_competitive_agents(active_agents, task_allocation)
        
        else:
            # Default to best agent
            return [self._select_best_agent(active_agents, task_allocation)]
    
    def _select_best_agent(self, agents: List[Agent], task_allocation: TaskAllocation) -> Agent:
        """Select the single best agent for a task."""
        # Score agents based on performance and capability match
        agent_scores = {}
        
        for agent in agents:
            score = 0.0
            
            # Performance score
            performance_score = self.agent_performance_scores.get(agent.agent_id, 0.5)
            score += performance_score * 0.4
            
            # Capability match score
            agent_capabilities = set(agent.get_capabilities())
            required_capabilities = set(task_allocation.requirements)
            
            if required_capabilities:
                capability_match = len(required_capabilities & agent_capabilities) / len(required_capabilities)
                score += capability_match * 0.4
            else:
                score += 0.4  # No specific requirements
            
            # Workload score (prefer less busy agents)
            workload = self.agent_workloads.get(agent.agent_id, 0)
            workload_score = max(0, 1.0 - workload / 10)  # Normalize to 0-1
            score += workload_score * 0.2
            
            agent_scores[agent.agent_id] = score
        
        # Return agent with highest score
        best_agent_id = max(agent_scores, key=agent_scores.get)
        return next(agent for agent in agents if agent.agent_id == best_agent_id)
    
    def _select_top_agents(self, 
                          agents: List[Agent], 
                          task_allocation: TaskAllocation, 
                          max_count: int) -> List[Agent]:
        """Select top N agents for parallel execution."""
        # Score all agents
        agent_scores = []
        
        for agent in agents:
            performance_score = self.agent_performance_scores.get(agent.agent_id, 0.5)
            workload = self.agent_workloads.get(agent.agent_id, 0)
            workload_score = max(0, 1.0 - workload / 10)
            
            total_score = performance_score * 0.7 + workload_score * 0.3
            agent_scores.append((agent, total_score))
        
        # Sort by score and take top N
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        return [agent for agent, score in agent_scores[:max_count]]
    
    def _select_diverse_agents(self, 
                             agents: List[Agent], 
                             task_allocation: TaskAllocation, 
                             min_count: int) -> List[Agent]:
        """Select diverse agents for consensus building."""
        selected_agents = []
        agent_types_used = set()
        
        # Prefer diversity in agent types
        for agent in agents:
            if len(selected_agents) >= min_count and agent.agent_type in agent_types_used:
                continue
            
            selected_agents.append(agent)
            agent_types_used.add(agent.agent_type)
            
            if len(selected_agents) >= 5:  # Max 5 agents for consensus
                break
        
        # Fill remaining slots with best performers if needed
        if len(selected_agents) < min_count:
            remaining_agents = [a for a in agents if a not in selected_agents]
            remaining_agents.sort(
                key=lambda a: self.agent_performance_scores.get(a.agent_id, 0.5),
                reverse=True
            )
            
            needed = min_count - len(selected_agents)
            selected_agents.extend(remaining_agents[:needed])
        
        return selected_agents
    
    def _select_pipeline_agents(self, 
                              agents: List[Agent], 
                              task_allocation: TaskAllocation) -> List[Agent]:
        """Select agents for pipeline execution."""
        # Order agents by complementary capabilities
        pipeline_order = [
            AgentCapability.RESEARCH,
            AgentCapability.ANALYSIS, 
            AgentCapability.SYNTHESIS,
            AgentCapability.FACT_CHECKING,
        ]
        
        selected_agents = []
        
        for capability in pipeline_order:
            suitable_agents = [
                agent for agent in agents 
                if capability in agent.get_capabilities() and agent not in selected_agents
            ]
            
            if suitable_agents:
                # Select best agent for this capability
                best_agent = max(
                    suitable_agents,
                    key=lambda a: self.agent_performance_scores.get(a.agent_id, 0.5)
                )
                selected_agents.append(best_agent)
        
        # Ensure at least one agent
        if not selected_agents and agents:
            selected_agents.append(self._select_best_agent(agents, task_allocation))
        
        return selected_agents
    
    def _select_competitive_agents(self, 
                                 agents: List[Agent], 
                                 task_allocation: TaskAllocation) -> List[Agent]:
        """Select agents for competitive execution."""
        # Select 2-3 top performers for competition
        top_agents = self._select_top_agents(agents, task_allocation, 3)
        return top_agents
    
    async def _execute_orchestration_strategy(self, 
                                            task_allocation: TaskAllocation,
                                            agents: List[Agent]) -> OrchestrationResult:
        """Execute the specific orchestration strategy."""
        strategy = task_allocation.strategy
        
        if strategy == OrchestrationStrategy.SINGLE_BEST:
            return await self._execute_single_best(task_allocation, agents)
        elif strategy == OrchestrationStrategy.PARALLEL_ALL:
            return await self._execute_parallel_all(task_allocation, agents)
        elif strategy == OrchestrationStrategy.CONSENSUS:
            return await self._execute_consensus(task_allocation, agents)
        elif strategy == OrchestrationStrategy.PIPELINE:
            return await self._execute_pipeline(task_allocation, agents)
        elif strategy == OrchestrationStrategy.COMPETITIVE:
            return await self._execute_competitive(task_allocation, agents)
        elif strategy == OrchestrationStrategy.COLLABORATIVE:
            return await self._execute_collaborative(task_allocation, agents)
        else:
            # Default to single best
            return await self._execute_single_best(task_allocation, agents)
    
    async def _execute_single_best(self, 
                                 task_allocation: TaskAllocation,
                                 agents: List[Agent]) -> OrchestrationResult:
        """Execute task with single best agent."""
        agent = agents[0]
        
        if self.logger:
            self.logger.info(f"🎯 EXECUTING SINGLE BEST STRATEGY")
            self.logger.info(f"Selected agent: {agent.name} ({agent.agent_id})")
            self.logger.info(f"Agent type: {agent.agent_type.value}")
            self.logger.info(f"Task: {task_allocation.task}")
            if hasattr(agent, 'adk_agent'):
                self.logger.info(f"ADK agent available: {agent.adk_agent is not None}")
            if hasattr(agent, 'runner'):
                self.logger.info(f"ADK runner available: {agent.runner is not None}")
        
        try:
            if self.logger:
                self.logger.info(f"🚀 Calling agent.execute_task...")
            
            import time
            start_time = time.time()
            result = await agent.execute_task(task_allocation.task, task_allocation.context)
            execution_time = time.time() - start_time
            
            if self.logger:
                self.logger.info(f"✅ Agent execution completed in {execution_time:.2f}s")
                self.logger.info(f"Result success: {result.success}")
                if result.success:
                    result_preview = str(result.result)[:300] + "..." if len(str(result.result)) > 300 else str(result.result)
                    self.logger.info(f"Result preview: {result_preview}")
                    if hasattr(result, 'tools_used') and result.tools_used:
                        self.logger.info(f"Tools used: {result.tools_used}")
                    if hasattr(result, 'execution_time_ms'):
                        self.logger.info(f"Agent reported execution time: {result.execution_time_ms}ms")
                else:
                    if hasattr(result, 'error'):
                        self.logger.error(f"Agent execution error: {result.error}")
            
            return OrchestrationResult(
                task_id=task_allocation.task_id,
                success=result.success,
                strategy_used=task_allocation.strategy,
                agents_used=[agent.agent_id],
                primary_result=result.result,
                all_results={agent.agent_id: result.to_dict()},
                metadata={
                    "single_agent_execution": True,
                    "execution_time_seconds": execution_time,
                    "agent_name": agent.name,
                    "agent_type": agent.agent_type.value
                },
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ AGENT EXECUTION EXCEPTION")
                self.logger.error(f"Exception type: {type(e).__name__}")
                self.logger.error(f"Exception message: {str(e)}")
                
                import traceback
                full_traceback = traceback.format_exc()
                self.logger.error(f"Full traceback:\n{full_traceback}")
            
            return OrchestrationResult(
                task_id=task_allocation.task_id,
                success=False,
                strategy_used=task_allocation.strategy,
                agents_used=[agent.agent_id],
                primary_result=None,
                error=str(e),
                metadata={
                    "single_agent_execution": True,
                    "exception_type": type(e).__name__,
                    "agent_name": agent.name,
                    "agent_type": agent.agent_type.value
                }
            )
    
    async def _execute_parallel_all(self, 
                                  task_allocation: TaskAllocation,
                                  agents: List[Agent]) -> OrchestrationResult:
        """Execute task with all agents in parallel."""
        try:
            # Execute with all agents simultaneously
            tasks = [
                agent.execute_task(task_allocation.task, task_allocation.context)
                for agent in agents
            ]
            
            # Apply timeout protection to parallel execution
            timeout = task_allocation.timeout_seconds
            if timeout and timeout > 0:
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    # Cancel remaining tasks
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    
                    # Wait briefly for cancellation to complete
                    try:
                        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=1.0)
                    except asyncio.TimeoutError:
                        pass  # Some tasks may not respond to cancellation
                    
                    return OrchestrationResult(
                        task_id=task_allocation.task_id,
                        success=False,
                        strategy_used=task_allocation.strategy,
                        agents_used=[agent.agent_id for agent in agents],
                        primary_result=None,
                        error=f"Parallel execution timed out after {timeout} seconds"
                    )
            else:
                results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            all_results = {}
            successful_results = []
            
            for agent, result in zip(agents, results):
                if isinstance(result, Exception):
                    all_results[agent.agent_id] = {"success": False, "error": str(result)}
                else:
                    all_results[agent.agent_id] = result.to_dict()
                    if result.success:
                        successful_results.append(result)
            
            # Select primary result (best quality or first successful)
            primary_result = None
            if successful_results:
                # Use first successful result as primary
                primary_result = successful_results[0].result
            
            return OrchestrationResult(
                task_id=task_allocation.task_id,
                success=bool(successful_results),
                strategy_used=task_allocation.strategy,
                agents_used=[agent.agent_id for agent in agents],
                primary_result=primary_result,
                all_results=all_results,
                metadata={
                    "parallel_execution": True,
                    "successful_agents": len(successful_results),
                    "total_agents": len(agents),
                },
            )
            
        except Exception as e:
            return OrchestrationResult(
                task_id=task_allocation.task_id,
                success=False,
                strategy_used=task_allocation.strategy,
                agents_used=[agent.agent_id for agent in agents],
                primary_result=None,
                error=str(e),
            )
    
    async def _execute_consensus(self, 
                               task_allocation: TaskAllocation,
                               agents: List[Agent]) -> OrchestrationResult:
        """Execute task with consensus building."""
        try:
            # Execute with all agents
            tasks = [
                agent.execute_task(task_allocation.task, task_allocation.context)
                for agent in agents
            ]
            
            # Apply timeout protection to consensus execution
            timeout = task_allocation.timeout_seconds
            if timeout and timeout > 0:
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    # Cancel remaining tasks
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    
                    # Wait briefly for cancellation to complete
                    try:
                        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=1.0)
                    except asyncio.TimeoutError:
                        pass  # Some tasks may not respond to cancellation
                    
                    return OrchestrationResult(
                        task_id=task_allocation.task_id,
                        success=False,
                        strategy_used=task_allocation.strategy,
                        agents_used=[agent.agent_id for agent in agents],
                        primary_result=None,
                        error=f"Consensus execution timed out after {timeout} seconds"
                    )
            else:
                results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results for consensus
            successful_results = []
            all_results = {}
            
            for agent, result in zip(agents, results):
                if isinstance(result, Exception):
                    all_results[agent.agent_id] = {"success": False, "error": str(result)}
                else:
                    all_results[agent.agent_id] = result.to_dict()
                    if result.success:
                        successful_results.append(result)
            
            # Build consensus
            consensus_result, consensus_score = self._build_consensus(successful_results)
            
            return OrchestrationResult(
                task_id=task_allocation.task_id,
                success=consensus_score >= self.strategy_configs[OrchestrationStrategy.CONSENSUS]["consensus_threshold"],
                strategy_used=task_allocation.strategy,
                agents_used=[agent.agent_id for agent in agents],
                primary_result=consensus_result,
                all_results=all_results,
                consensus_score=consensus_score,
                metadata={
                    "consensus_building": True,
                    "consensus_threshold": self.strategy_configs[OrchestrationStrategy.CONSENSUS]["consensus_threshold"],
                },
            )
            
        except Exception as e:
            return OrchestrationResult(
                task_id=task_allocation.task_id,
                success=False,
                strategy_used=task_allocation.strategy,
                agents_used=[agent.agent_id for agent in agents],
                primary_result=None,
                error=str(e),
            )
    
    async def _execute_pipeline(self, 
                              task_allocation: TaskAllocation,
                              agents: List[Agent]) -> OrchestrationResult:
        """Execute task as pipeline through multiple agents."""
        try:
            current_input = task_allocation.task
            current_context = task_allocation.context.copy()
            all_results = {}
            
            # Execute through pipeline
            for i, agent in enumerate(agents):
                result = await agent.execute_task(current_input, current_context)
                all_results[agent.agent_id] = result.to_dict()
                
                if not result.success:
                    # Pipeline broken
                    return OrchestrationResult(
                        task_id=task_allocation.task_id,
                        success=False,
                        strategy_used=task_allocation.strategy,
                        agents_used=[agent.agent_id for agent in agents[:i+1]],
                        primary_result=None,
                        all_results=all_results,
                        error=f"Pipeline broken at agent {agent.name}: {result.error}",
                    )
                
                # Use result as input for next agent
                if result.result:
                    current_input = str(result.result)
                    current_context["pipeline_stage"] = i + 1
                    current_context["previous_result"] = result.result
            
            # Return final result
            final_result = list(all_results.values())[-1]["result"] if all_results else None
            
            return OrchestrationResult(
                task_id=task_allocation.task_id,
                success=True,
                strategy_used=task_allocation.strategy,
                agents_used=[agent.agent_id for agent in agents],
                primary_result=final_result,
                all_results=all_results,
                metadata={
                    "pipeline_execution": True,
                    "pipeline_stages": len(agents),
                },
            )
            
        except Exception as e:
            return OrchestrationResult(
                task_id=task_allocation.task_id,
                success=False,
                strategy_used=task_allocation.strategy,
                agents_used=[agent.agent_id for agent in agents],
                primary_result=None,
                error=str(e),
            )
    
    async def _execute_competitive(self, 
                                 task_allocation: TaskAllocation,
                                 agents: List[Agent]) -> OrchestrationResult:
        """Execute task competitively and select best result."""
        try:
            # Execute with all agents
            tasks = [
                agent.execute_task(task_allocation.task, task_allocation.context)
                for agent in agents
            ]
            
            # Apply timeout protection to competitive execution
            timeout = task_allocation.timeout_seconds
            if timeout and timeout > 0:
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    # Cancel remaining tasks
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    
                    # Wait briefly for cancellation to complete
                    try:
                        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=1.0)
                    except asyncio.TimeoutError:
                        pass  # Some tasks may not respond to cancellation
                    
                    return OrchestrationResult(
                        task_id=task_allocation.task_id,
                        success=False,
                        strategy_used=task_allocation.strategy,
                        agents_used=[agent.agent_id for agent in agents],
                        primary_result=None,
                        error=f"Competitive execution timed out after {timeout} seconds"
                    )
            else:
                results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process and evaluate results
            successful_results = []
            all_results = {}
            
            for agent, result in zip(agents, results):
                if isinstance(result, Exception):
                    all_results[agent.agent_id] = {"success": False, "error": str(result)}
                else:
                    all_results[agent.agent_id] = result.to_dict()
                    if result.success:
                        successful_results.append((agent.agent_id, result))
            
            # Select best result based on quality/performance
            best_result = None
            best_agent_id = None
            
            if successful_results:
                # Simple selection: use execution time and success as criteria
                best_agent_id, best_result = min(
                    successful_results,
                    key=lambda x: x[1].execution_time_ms
                )
            
            return OrchestrationResult(
                task_id=task_allocation.task_id,
                success=best_result is not None,
                strategy_used=task_allocation.strategy,
                agents_used=[agent.agent_id for agent in agents],
                primary_result=best_result.result if best_result else None,
                all_results=all_results,
                metadata={
                    "competitive_execution": True,
                    "winning_agent": best_agent_id,
                    "contestants": len(agents),
                },
            )
            
        except Exception as e:
            return OrchestrationResult(
                task_id=task_allocation.task_id,
                success=False,
                strategy_used=task_allocation.strategy,
                agents_used=[agent.agent_id for agent in agents],
                primary_result=None,
                error=str(e),
            )
    
    async def _execute_collaborative(self, 
                                   task_allocation: TaskAllocation,
                                   agents: List[Agent]) -> OrchestrationResult:
        """Execute task collaboratively with agent coordination."""
        try:
            # For now, implement as sequential with shared context
            shared_context = task_allocation.context.copy()
            shared_context["collaborative_mode"] = True
            all_results = {}
            
            for i, agent in enumerate(agents):
                # Add previous agents' results to context
                if i > 0:
                    shared_context["previous_agents_results"] = [
                        all_results[agents[j].agent_id] for j in range(i)
                    ]
                
                result = await agent.execute_task(task_allocation.task, shared_context)
                all_results[agent.agent_id] = result.to_dict()
                
                # Add this result to shared context for next agent
                shared_context[f"agent_{agent.agent_id}_result"] = result.result
            
            # Combine results
            final_result = self._combine_collaborative_results(all_results)
            
            return OrchestrationResult(
                task_id=task_allocation.task_id,
                success=bool(final_result),
                strategy_used=task_allocation.strategy,
                agents_used=[agent.agent_id for agent in agents],
                primary_result=final_result,
                all_results=all_results,
                metadata={
                    "collaborative_execution": True,
                    "collaborating_agents": len(agents),
                },
            )
            
        except Exception as e:
            return OrchestrationResult(
                task_id=task_allocation.task_id,
                success=False,
                strategy_used=task_allocation.strategy,
                agents_used=[agent.agent_id for agent in agents],
                primary_result=None,
                error=str(e),
            )
    
    def _build_consensus(self, results: List[AgentResult]) -> Tuple[Any, float]:
        """Build consensus from multiple agent results."""
        if not results:
            return None, 0.0
        
        if len(results) == 1:
            return results[0].result, 1.0
        
        # Simple consensus: majority rule for similar results
        result_strings = [str(result.result) for result in results]
        
        # Find most common result
        from collections import Counter
        result_counts = Counter(result_strings)
        most_common_result, count = result_counts.most_common(1)[0]
        
        # Calculate consensus score
        consensus_score = count / len(results)
        
        # Return the actual result object that matches the consensus
        for result in results:
            if str(result.result) == most_common_result:
                return result.result, consensus_score
        
        # Fallback
        return results[0].result, consensus_score
    
    def _combine_collaborative_results(self, all_results: Dict[str, Any]) -> Any:
        """Combine results from collaborative execution."""
        successful_results = [
            result for result in all_results.values() 
            if result.get("success", False)
        ]
        
        if not successful_results:
            return None
        
        # Simple combination: concatenate results
        combined_result = {
            "collaborative_synthesis": True,
            "contributing_agents": list(all_results.keys()),
            "individual_results": {
                agent_id: result.get("result") 
                for agent_id, result in all_results.items()
                if result.get("success", False)
            },
            "summary": "Combined results from collaborative agent execution",
        }
        
        return combined_result
    
    def _update_performance_metrics(self, 
                                  orchestration_result: OrchestrationResult,
                                  agents: List[Agent]) -> None:
        """Update performance metrics for agents and strategies."""
        # Update agent performance scores
        for agent in agents:
            agent_result = orchestration_result.all_results.get(agent.agent_id, {})
            if agent_result.get("success", False):
                # Positive feedback
                current_score = self.agent_performance_scores.get(agent.agent_id, 0.5)
                self.agent_performance_scores[agent.agent_id] = min(1.0, current_score + 0.05)
            else:
                # Negative feedback
                current_score = self.agent_performance_scores.get(agent.agent_id, 0.5)
                self.agent_performance_scores[agent.agent_id] = max(0.0, current_score - 0.02)
        
        # Update strategy success rates
        strategy = orchestration_result.strategy_used
        if strategy not in self.strategy_success_rates:
            self.strategy_success_rates[strategy] = 0.5
        
        current_rate = self.strategy_success_rates[strategy]
        if orchestration_result.success:
            self.strategy_success_rates[strategy] = min(1.0, current_rate + 0.02)
        else:
            self.strategy_success_rates[strategy] = max(0.0, current_rate - 0.01)
    
    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get current orchestration status."""
        return {
            "active_tasks": len(self.active_tasks),
            "queued_tasks": len(self.task_queue),
            "completed_tasks": len(self.completed_tasks),
            "total_orchestrated": self.total_tasks_orchestrated,
            "success_rate": (
                self.successful_orchestrations / self.total_tasks_orchestrated * 100
                if self.total_tasks_orchestrated > 0 else 0
            ),
            "agent_performance_scores": self.agent_performance_scores,
            "strategy_success_rates": {
                strategy.value: rate for strategy, rate in self.strategy_success_rates.items()
            },
            "available_agents": len(AgentRegistry.get_all_agents()),
            "active_agents": len([a for a in AgentRegistry.get_all_agents() if a.is_active]),
        }
    
    def get_agent_workload_status(self) -> Dict[str, Any]:
        """Get current agent workload status."""
        all_agents = AgentRegistry.get_all_agents()
        
        return {
            "agent_workloads": self.agent_workloads,
            "agent_availability": self.agent_availability,
            "total_agents": len(all_agents),
            "active_agents": sum(1 for agent in all_agents if agent.is_active),
            "busy_agents": sum(1 for workload in self.agent_workloads.values() if workload > 0),
        }