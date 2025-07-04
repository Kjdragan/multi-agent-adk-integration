"""
Workflow Agent Implementation

Provides workflow orchestration agents that can manage complex multi-step
processes, coordinate between different tools and agents, and handle
conditional logic and error recovery.
"""

import time
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum

from .base import Agent, AgentType, AgentCapability, AgentResult, AgentRegistry
from ..platform_logging import RunLogger
from ..services import SessionService, MemoryService, ArtifactService
from ..mcp import MCPOrchestrator


class WorkflowStepType(str, Enum):
    """Types of workflow steps."""
    TASK = "task"                      # Execute a specific task
    CONDITION = "condition"            # Conditional branching
    LOOP = "loop"                      # Iterative processing
    PARALLEL = "parallel"              # Parallel execution
    AGENT_CALL = "agent_call"          # Call another agent
    TOOL_CALL = "tool_call"           # Call a tool or API
    MCP_CALL = "mcp_call"             # Call MCP server
    MEMORY_STORE = "memory_store"      # Store information in memory
    MEMORY_RETRIEVE = "memory_retrieve" # Retrieve information from memory
    ARTIFACT_CREATE = "artifact_create" # Create artifact
    WAIT = "wait"                      # Wait/delay step
    MERGE = "merge"                    # Merge results from multiple steps


class WorkflowStepStatus(str, Enum):
    """Status of workflow steps."""
    PENDING = "pending"                # Not yet executed
    RUNNING = "running"                # Currently executing
    COMPLETED = "completed"            # Successfully completed
    FAILED = "failed"                  # Failed with error
    SKIPPED = "skipped"                # Skipped due to conditions
    RETRYING = "retrying"              # Retrying after failure


@dataclass
class WorkflowStep:
    """Individual step in a workflow."""
    step_id: str
    step_type: WorkflowStepType
    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # Step IDs this depends on
    conditions: Dict[str, Any] = field(default_factory=dict)  # Execution conditions
    error_handling: Dict[str, Any] = field(default_factory=dict)  # Error handling config
    timeout_seconds: Optional[int] = None
    retry_config: Dict[str, Any] = field(default_factory=dict)  # Retry configuration
    
    # Execution state
    status: WorkflowStepStatus = WorkflowStepStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    
    @property
    def execution_time_ms(self) -> float:
        """Calculate execution time in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "step_id": self.step_id,
            "step_type": self.step_type.value,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "conditions": self.conditions,
            "error_handling": self.error_handling,
            "timeout_seconds": self.timeout_seconds,
            "retry_config": self.retry_config,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "result": self.result,
            "error": self.error,
            "retry_count": self.retry_count,
            "execution_time_ms": self.execution_time_ms,
        }


@dataclass
class WorkflowConfig:
    """Configuration for workflow execution."""
    workflow_id: str
    name: str
    description: str = ""
    steps: List[WorkflowStep] = field(default_factory=list)
    global_timeout_seconds: Optional[int] = None
    parallel_execution: bool = False
    stop_on_first_failure: bool = False
    enable_checkpoints: bool = True
    enable_recovery: bool = True
    context_sharing: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "steps": [step.to_dict() for step in self.steps],
            "global_timeout_seconds": self.global_timeout_seconds,
            "parallel_execution": self.parallel_execution,
            "stop_on_first_failure": self.stop_on_first_failure,
            "enable_checkpoints": self.enable_checkpoints,
            "enable_recovery": self.enable_recovery,
            "context_sharing": self.context_sharing,
        }


@dataclass
class WorkflowResult:
    """Result from workflow execution."""
    workflow_id: str
    success: bool
    steps_executed: int
    steps_successful: int
    steps_failed: int
    steps_skipped: int
    total_execution_time_ms: float
    step_results: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    checkpoint_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "workflow_id": self.workflow_id,
            "success": self.success,
            "steps_executed": self.steps_executed,
            "steps_successful": self.steps_successful,
            "steps_failed": self.steps_failed,
            "steps_skipped": self.steps_skipped,
            "total_execution_time_ms": self.total_execution_time_ms,
            "step_results": self.step_results,
            "error": self.error,
            "checkpoint_data": self.checkpoint_data,
        }


class WorkflowAgent(Agent):
    """
    Workflow orchestration agent for managing complex multi-step processes.
    
    Provides workflow definition, execution, monitoring, and error recovery
    with integration to both ADK tools and MCP servers.
    """
    
    def __init__(self,
                 agent_id: Optional[str] = None,
                 name: str = "Workflow Agent",
                 description: str = "Orchestrates complex multi-step workflows and processes",
                 logger: Optional[RunLogger] = None,
                 session_service: Optional[SessionService] = None,
                 memory_service: Optional[MemoryService] = None,
                 artifact_service: Optional[ArtifactService] = None,
                 mcp_orchestrator: Optional[MCPOrchestrator] = None):
        
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            agent_type=AgentType.WORKFLOW,
            capabilities=[
                AgentCapability.PLANNING,
                AgentCapability.EXECUTION,
                AgentCapability.TOOL_USE,
                AgentCapability.MEMORY_ACCESS,
                AgentCapability.CONTEXT_MANAGEMENT,
                AgentCapability.DECISION_MAKING,
            ],
            logger=logger,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service,
            mcp_orchestrator=mcp_orchestrator,
        )
        
        # Workflow management
        self.active_workflows: Dict[str, WorkflowConfig] = {}
        self.workflow_history: List[WorkflowResult] = []
        self.workflow_templates: Dict[str, WorkflowConfig] = {}
        
        # Execution context
        self.shared_context: Dict[str, Any] = {}
        self.checkpoint_storage: Dict[str, Dict[str, Any]] = {}
        
        # Performance metrics
        self.total_workflows_executed = 0
        self.successful_workflows = 0
        self.failed_workflows = 0
        self.average_workflow_time_ms = 0.0
        
        # Load default workflow templates
        self._load_default_templates()
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Get the capabilities of this workflow agent."""
        return self.capabilities
    
    async def execute_task(self, 
                          task: str, 
                          context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Execute a task by creating and running a workflow.
        
        Args:
            task: Task description
            context: Optional context information
            
        Returns:
            Agent result with workflow execution outcome
        """
        start_time = time.time()
        
        if self.logger:
            self.logger.info(f"Workflow Agent {self.name} executing task: {task[:100]}...")
        
        try:
            # Analyze task and create workflow
            workflow_config = await self._create_workflow_from_task(task, context)
            
            # Execute workflow
            workflow_result = await self.execute_workflow(workflow_config)
            
            execution_time = (time.time() - start_time) * 1000
            
            return AgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                task=task,
                success=workflow_result.success,
                result=workflow_result.to_dict(),
                error=workflow_result.error,
                execution_time_ms=execution_time,
                metadata={
                    "workflow_id": workflow_result.workflow_id,
                    "steps_executed": workflow_result.steps_executed,
                    "steps_successful": workflow_result.steps_successful,
                    "steps_failed": workflow_result.steps_failed,
                },
                tools_used=["workflow_engine"],
                memory_accessed=True,
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            if self.logger:
                self.logger.error(f"Workflow Agent {self.name} task execution failed: {e}")
            
            return AgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                task=task,
                success=False,
                error=str(e),
                execution_time_ms=execution_time,
            )
    
    async def execute_workflow(self, workflow_config: WorkflowConfig) -> WorkflowResult:
        """
        Execute a complete workflow.
        
        Args:
            workflow_config: Workflow configuration
            
        Returns:
            Workflow execution result
        """
        start_time = time.time()
        
        if self.logger:
            self.logger.info(f"Starting workflow execution: {workflow_config.name}")
        
        self.active_workflows[workflow_config.workflow_id] = workflow_config
        
        # Initialize workflow execution context
        workflow_context = {
            "workflow_id": workflow_config.workflow_id,
            "shared_data": {},
            "step_outputs": {},
        }
        
        steps_executed = 0
        steps_successful = 0
        steps_failed = 0
        steps_skipped = 0
        step_results = {}
        
        try:
            # Execute workflow steps
            if workflow_config.parallel_execution:
                # Execute steps in parallel where possible
                step_results = await self._execute_steps_parallel(
                    workflow_config.steps, workflow_context
                )
            else:
                # Execute steps sequentially
                step_results = await self._execute_steps_sequential(
                    workflow_config.steps, workflow_context
                )
            
            # Count step outcomes
            for step_id, result in step_results.items():
                steps_executed += 1
                if result.get("success", False):
                    steps_successful += 1
                elif result.get("skipped", False):
                    steps_skipped += 1
                else:
                    steps_failed += 1
            
            # Determine overall success
            workflow_success = (
                steps_failed == 0 or not workflow_config.stop_on_first_failure
            )
            
            total_execution_time = (time.time() - start_time) * 1000
            
            # Create workflow result
            workflow_result = WorkflowResult(
                workflow_id=workflow_config.workflow_id,
                success=workflow_success,
                steps_executed=steps_executed,
                steps_successful=steps_successful,
                steps_failed=steps_failed,
                steps_skipped=steps_skipped,
                total_execution_time_ms=total_execution_time,
                step_results=step_results,
            )
            
            # Update metrics
            self.total_workflows_executed += 1
            if workflow_success:
                self.successful_workflows += 1
            else:
                self.failed_workflows += 1
            
            self._update_average_workflow_time(total_execution_time)
            
            # Store in history
            self.workflow_history.append(workflow_result)
            
            # Store in memory
            if self.memory_service:
                await self.store_memory(
                    f"Workflow executed: {workflow_config.name}",
                    metadata={
                        "workflow_id": workflow_config.workflow_id,
                        "success": workflow_success,
                        "execution_time_ms": total_execution_time,
                    }
                )
            
            if self.logger:
                self.logger.info(f"Completed workflow {workflow_config.name}: success={workflow_success}")
            
            return workflow_result
            
        except Exception as e:
            total_execution_time = (time.time() - start_time) * 1000
            
            if self.logger:
                self.logger.error(f"Workflow {workflow_config.name} execution failed: {e}")
            
            return WorkflowResult(
                workflow_id=workflow_config.workflow_id,
                success=False,
                steps_executed=steps_executed,
                steps_successful=steps_successful,
                steps_failed=steps_failed + 1,  # Add the failure
                steps_skipped=steps_skipped,
                total_execution_time_ms=total_execution_time,
                step_results=step_results,
                error=str(e),
            )
        
        finally:
            # Clean up
            if workflow_config.workflow_id in self.active_workflows:
                del self.active_workflows[workflow_config.workflow_id]
    
    async def _execute_steps_sequential(self, 
                                      steps: List[WorkflowStep],
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow steps sequentially."""
        step_results = {}
        
        for step in steps:
            # Check dependencies
            if not self._check_step_dependencies(step, step_results):
                step.status = WorkflowStepStatus.SKIPPED
                step_results[step.step_id] = {"skipped": True, "reason": "Dependencies not met"}
                continue
            
            # Check conditions
            if not self._check_step_conditions(step, context):
                step.status = WorkflowStepStatus.SKIPPED
                step_results[step.step_id] = {"skipped": True, "reason": "Conditions not met"}
                continue
            
            # Execute step
            result = await self._execute_single_step(step, context)
            step_results[step.step_id] = result
            
            # Update context with step output
            if result.get("success") and "output" in result:
                context["step_outputs"][step.step_id] = result["output"]
        
        return step_results
    
    async def _execute_steps_parallel(self, 
                                    steps: List[WorkflowStep],
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow steps in parallel where dependencies allow."""
        step_results = {}
        pending_steps = steps.copy()
        
        while pending_steps:
            # Find steps that can execute now (dependencies met)
            ready_steps = []
            remaining_steps = []
            
            for step in pending_steps:
                if self._check_step_dependencies(step, step_results):
                    if self._check_step_conditions(step, context):
                        ready_steps.append(step)
                    else:
                        step.status = WorkflowStepStatus.SKIPPED
                        step_results[step.step_id] = {"skipped": True, "reason": "Conditions not met"}
                else:
                    remaining_steps.append(step)
            
            if not ready_steps:
                # No more steps can execute
                for step in remaining_steps:
                    step.status = WorkflowStepStatus.SKIPPED
                    step_results[step.step_id] = {"skipped": True, "reason": "Dependencies not met"}
                break
            
            # Execute ready steps in parallel
            tasks = [
                self._execute_single_step(step, context)
                for step in ready_steps
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for step, result in zip(ready_steps, results):
                if isinstance(result, Exception):
                    step_results[step.step_id] = {
                        "success": False,
                        "error": str(result),
                    }
                else:
                    step_results[step.step_id] = result
                    
                    # Update context with step output
                    if result.get("success") and "output" in result:
                        context["step_outputs"][step.step_id] = result["output"]
            
            pending_steps = remaining_steps
        
        return step_results
    
    async def _execute_single_step(self, 
                                 step: WorkflowStep,
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step."""
        step.status = WorkflowStepStatus.RUNNING
        step.start_time = time.time()
        
        try:
            if self.logger:
                self.logger.info(f"Executing workflow step: {step.name}")
            
            # Execute based on step type
            if step.step_type == WorkflowStepType.TASK:
                result = await self._execute_task_step(step, context)
            elif step.step_type == WorkflowStepType.AGENT_CALL:
                result = await self._execute_agent_call_step(step, context)
            elif step.step_type == WorkflowStepType.TOOL_CALL:
                result = await self._execute_tool_call_step(step, context)
            elif step.step_type == WorkflowStepType.MCP_CALL:
                result = await self._execute_mcp_call_step(step, context)
            elif step.step_type == WorkflowStepType.MEMORY_STORE:
                result = await self._execute_memory_store_step(step, context)
            elif step.step_type == WorkflowStepType.MEMORY_RETRIEVE:
                result = await self._execute_memory_retrieve_step(step, context)
            elif step.step_type == WorkflowStepType.CONDITION:
                result = await self._execute_condition_step(step, context)
            elif step.step_type == WorkflowStepType.WAIT:
                result = await self._execute_wait_step(step, context)
            else:
                result = {"success": False, "error": f"Unsupported step type: {step.step_type}"}
            
            step.status = WorkflowStepStatus.COMPLETED if result.get("success") else WorkflowStepStatus.FAILED
            step.result = result.get("output")
            step.error = result.get("error")
            
            return result
            
        except Exception as e:
            step.status = WorkflowStepStatus.FAILED
            step.error = str(e)
            
            if self.logger:
                self.logger.error(f"Workflow step {step.name} failed: {e}")
            
            return {"success": False, "error": str(e)}
        
        finally:
            step.end_time = time.time()
    
    async def _execute_task_step(self, 
                               step: WorkflowStep,
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a general task step."""
        task = step.parameters.get("task", "")
        
        if not task:
            return {"success": False, "error": "No task specified"}
        
        # Create an LLM agent to handle the task
        llm_agents = AgentRegistry.get_agents_by_type(AgentType.LLM)
        if llm_agents:
            agent = llm_agents[0]  # Use first available LLM agent
            result = await agent.execute_task(task, context)
            
            return {
                "success": result.success,
                "output": result.result,
                "error": result.error,
                "execution_time_ms": result.execution_time_ms,
            }
        
        return {"success": False, "error": "No LLM agent available for task execution"}
    
    async def _execute_agent_call_step(self, 
                                     step: WorkflowStep,
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an agent call step."""
        agent_id = step.parameters.get("agent_id")
        task = step.parameters.get("task", "")
        
        if not agent_id or not task:
            return {"success": False, "error": "Agent ID and task are required"}
        
        agent = AgentRegistry.get_agent(agent_id)
        if not agent:
            return {"success": False, "error": f"Agent {agent_id} not found"}
        
        result = await agent.execute_task(task, context)
        
        return {
            "success": result.success,
            "output": result.result,
            "error": result.error,
            "execution_time_ms": result.execution_time_ms,
            "agent_used": agent_id,
        }
    
    async def _execute_tool_call_step(self, 
                                    step: WorkflowStep,
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call step."""
        tool_name = step.parameters.get("tool_name")
        tool_params = step.parameters.get("parameters", {})
        
        if not tool_name:
            return {"success": False, "error": "Tool name is required"}
        
        # Use ADK tool
        result = await self.use_adk_tool(tool_name, tool_params)
        
        return {
            "success": result.get("success", False),
            "output": result.get("result"),
            "error": result.get("error"),
            "tool_used": tool_name,
        }
    
    async def _execute_mcp_call_step(self, 
                                   step: WorkflowStep,
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an MCP server call step."""
        server_name = step.parameters.get("server_name")
        operation = step.parameters.get("operation")
        operation_params = step.parameters.get("parameters", {})
        
        if not server_name or not operation:
            return {"success": False, "error": "Server name and operation are required"}
        
        result = await self.use_mcp_server(server_name, operation, operation_params)
        
        return {
            "success": result.get("success", False),
            "output": result.get("result"),
            "error": result.get("error"),
            "server_used": server_name,
        }
    
    async def _execute_memory_store_step(self, 
                                       step: WorkflowStep,
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a memory store step."""
        content = step.parameters.get("content", "")
        metadata = step.parameters.get("metadata", {})
        
        success = await self.store_memory(content, metadata)
        
        return {
            "success": success,
            "output": "Memory stored successfully" if success else "Memory storage failed",
        }
    
    async def _execute_memory_retrieve_step(self, 
                                          step: WorkflowStep,
                                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a memory retrieve step."""
        query = step.parameters.get("query", "")
        limit = step.parameters.get("limit", 10)
        
        memories = await self.retrieve_memory(query, limit)
        
        return {
            "success": True,
            "output": memories,
            "count": len(memories),
        }
    
    async def _execute_condition_step(self, 
                                    step: WorkflowStep,
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a condition step."""
        condition = step.parameters.get("condition", "")
        
        # Simple condition evaluation (could be enhanced with a proper expression evaluator)
        try:
            # This is a simplified implementation
            result = eval(condition, {"context": context})
            
            return {
                "success": True,
                "output": result,
                "condition_met": bool(result),
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Condition evaluation failed: {e}",
            }
    
    async def _execute_wait_step(self, 
                               step: WorkflowStep,
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a wait step."""
        duration = step.parameters.get("duration_seconds", 1)
        
        await asyncio.sleep(duration)
        
        return {
            "success": True,
            "output": f"Waited for {duration} seconds",
        }
    
    def _check_step_dependencies(self, 
                               step: WorkflowStep,
                               step_results: Dict[str, Any]) -> bool:
        """Check if step dependencies are satisfied."""
        for dep_id in step.dependencies:
            if dep_id not in step_results:
                return False
            if not step_results[dep_id].get("success", False):
                return False
        return True
    
    def _check_step_conditions(self, 
                             step: WorkflowStep,
                             context: Dict[str, Any]) -> bool:
        """Check if step conditions are met."""
        if not step.conditions:
            return True
        
        # Simple condition checking (could be enhanced)
        for condition_key, condition_value in step.conditions.items():
            if condition_key == "context_has":
                if condition_value not in context:
                    return False
            elif condition_key == "step_output_equals":
                step_id, expected_value = condition_value.split("=", 1)
                if step_id not in context.get("step_outputs", {}):
                    return False
                if context["step_outputs"][step_id] != expected_value:
                    return False
        
        return True
    
    async def _create_workflow_from_task(self, 
                                       task: str,
                                       context: Optional[Dict[str, Any]]) -> WorkflowConfig:
        """Create a workflow configuration from a task description."""
        # This is a simplified implementation - in practice, this would use
        # sophisticated task analysis to create appropriate workflows
        
        workflow_id = f"task_workflow_{int(time.time())}"
        
        # Check if it's a research task
        if any(keyword in task.lower() for keyword in ["research", "analyze", "investigate", "study"]):
            return self._create_research_workflow(workflow_id, task, context)
        
        # Check if it's a planning task
        elif any(keyword in task.lower() for keyword in ["plan", "organize", "schedule", "prepare"]):
            return self._create_planning_workflow(workflow_id, task, context)
        
        # Default: simple task workflow
        else:
            return self._create_simple_task_workflow(workflow_id, task, context)
    
    def _create_research_workflow(self, 
                                workflow_id: str,
                                task: str,
                                context: Optional[Dict[str, Any]]) -> WorkflowConfig:
        """Create a research workflow."""
        steps = [
            WorkflowStep(
                step_id="research_planning",
                step_type=WorkflowStepType.TASK,
                name="Research Planning",
                description="Plan the research approach",
                parameters={"task": f"Create a research plan for: {task}"},
            ),
            WorkflowStep(
                step_id="information_gathering",
                step_type=WorkflowStepType.MCP_CALL,
                name="Information Gathering",
                description="Gather information using MCP servers",
                parameters={
                    "server_name": "perplexity",
                    "operation": "search",
                    "parameters": {"query": task, "strategy": "QUALITY_OPTIMIZED"},
                },
                dependencies=["research_planning"],
            ),
            WorkflowStep(
                step_id="analysis",
                step_type=WorkflowStepType.TASK,
                name="Analysis",
                description="Analyze gathered information",
                parameters={"task": f"Analyze the research findings for: {task}"},
                dependencies=["information_gathering"],
            ),
            WorkflowStep(
                step_id="synthesis",
                step_type=WorkflowStepType.TASK,
                name="Synthesis",
                description="Synthesize findings into report",
                parameters={"task": f"Create a comprehensive report synthesizing findings for: {task}"},
                dependencies=["analysis"],
            ),
        ]
        
        return WorkflowConfig(
            workflow_id=workflow_id,
            name=f"Research Workflow: {task[:50]}",
            description=f"Comprehensive research workflow for: {task}",
            steps=steps,
        )
    
    def _create_planning_workflow(self, 
                                workflow_id: str,
                                task: str,
                                context: Optional[Dict[str, Any]]) -> WorkflowConfig:
        """Create a planning workflow."""
        steps = [
            WorkflowStep(
                step_id="goal_analysis",
                step_type=WorkflowStepType.TASK,
                name="Goal Analysis",
                description="Analyze the planning goal",
                parameters={"task": f"Analyze the planning requirements for: {task}"},
            ),
            WorkflowStep(
                step_id="resource_assessment",
                step_type=WorkflowStepType.TASK,
                name="Resource Assessment",
                description="Assess available resources",
                parameters={"task": f"Assess resources needed for: {task}"},
                dependencies=["goal_analysis"],
            ),
            WorkflowStep(
                step_id="plan_creation",
                step_type=WorkflowStepType.TASK,
                name="Plan Creation",
                description="Create detailed plan",
                parameters={"task": f"Create a detailed execution plan for: {task}"},
                dependencies=["resource_assessment"],
            ),
            WorkflowStep(
                step_id="plan_review",
                step_type=WorkflowStepType.TASK,
                name="Plan Review",
                description="Review and validate plan",
                parameters={"task": f"Review and validate the execution plan for: {task}"},
                dependencies=["plan_creation"],
            ),
        ]
        
        return WorkflowConfig(
            workflow_id=workflow_id,
            name=f"Planning Workflow: {task[:50]}",
            description=f"Comprehensive planning workflow for: {task}",
            steps=steps,
        )
    
    def _create_simple_task_workflow(self, 
                                   workflow_id: str,
                                   task: str,
                                   context: Optional[Dict[str, Any]]) -> WorkflowConfig:
        """Create a simple task workflow."""
        steps = [
            WorkflowStep(
                step_id="task_execution",
                step_type=WorkflowStepType.TASK,
                name="Task Execution",
                description="Execute the main task",
                parameters={"task": task},
            ),
        ]
        
        return WorkflowConfig(
            workflow_id=workflow_id,
            name=f"Simple Task: {task[:50]}",
            description=f"Simple task execution for: {task}",
            steps=steps,
        )
    
    def _load_default_templates(self) -> None:
        """Load default workflow templates."""
        # This would load workflow templates from configuration
        # For now, we'll skip this implementation
        pass
    
    def _update_average_workflow_time(self, execution_time_ms: float) -> None:
        """Update average workflow execution time."""
        if self.total_workflows_executed == 1:
            self.average_workflow_time_ms = execution_time_ms
        else:
            self.average_workflow_time_ms = (
                (self.average_workflow_time_ms * (self.total_workflows_executed - 1) + execution_time_ms) / 
                self.total_workflows_executed
            )
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an active workflow."""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            return {
                "workflow_id": workflow_id,
                "name": workflow.name,
                "status": "active",
                "steps": [step.to_dict() for step in workflow.steps],
            }
        return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the workflow agent."""
        success_rate = (
            (self.successful_workflows / self.total_workflows_executed * 100)
            if self.total_workflows_executed > 0
            else 0
        )
        
        return {
            "total_workflows": self.total_workflows_executed,
            "successful_workflows": self.successful_workflows,
            "failed_workflows": self.failed_workflows,
            "success_rate_percent": success_rate,
            "average_workflow_time_ms": self.average_workflow_time_ms,
            "active_workflows": len(self.active_workflows),
            "workflow_history_length": len(self.workflow_history),
        }