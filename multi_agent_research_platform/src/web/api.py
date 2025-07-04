"""
Web API Endpoints

REST API implementations for agent management, orchestration,
and debugging functionality.
"""

import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from ..agents import (
    AgentRegistry, AgentOrchestrator, AgentFactory, AgentSuite,
    OrchestrationStrategy, TaskPriority, AgentType, AgentCapability
)
from ..agents.llm_agent import LLMRole
from ..agents.custom_agent import CustomAgentType
from ..platform_logging import RunLogger
from ..services import SessionService


# Pydantic models for API requests/responses

class AgentCreateRequest(BaseModel):
    """Request model for creating an agent."""
    agent_type: str = Field(..., description="Type of agent (llm, workflow, custom)")
    name: Optional[str] = Field(None, description="Agent name")
    config: Optional[Dict[str, Any]] = Field(None, description="Agent configuration")


class AgentResponse(BaseModel):
    """Response model for agent information."""
    agent_id: str
    name: str
    agent_type: str
    capabilities: List[str]
    is_active: bool
    total_tasks_completed: int
    status: Dict[str, Any]


class TaskRequest(BaseModel):
    """Request model for task execution."""
    task: str = Field(..., description="Task description")
    context: Optional[Dict[str, Any]] = Field(None, description="Task context")


class OrchestrationRequest(BaseModel):
    """Request model for orchestrated task execution."""
    task: str = Field(..., description="Task description")
    strategy: str = Field("adaptive", description="Orchestration strategy")
    priority: str = Field("medium", description="Task priority")
    requirements: Optional[List[str]] = Field(None, description="Required capabilities")
    context: Optional[Dict[str, Any]] = Field(None, description="Task context")


class TeamCreateRequest(BaseModel):
    """Request model for creating agent teams."""
    suite_type: str = Field(..., description="Type of agent suite")
    custom_configs: Optional[Dict[str, Any]] = Field(None, description="Custom configurations")


class DebugInspectionRequest(BaseModel):
    """Request model for agent inspection."""
    agent_id: str = Field(..., description="Agent ID to inspect")
    include_performance: bool = Field(True, description="Include performance metrics")
    include_memory: bool = Field(False, description="Include memory contents")


# API Router classes

class AgentAPI:
    """API endpoints for agent management."""
    
    def __init__(self,
                 logger: Optional[RunLogger] = None,
                 session_service: Optional[SessionService] = None):
        
        self.logger = logger
        self.session_service = session_service
        self.router = APIRouter()
        self.factory = AgentFactory(
            logger=logger,
            session_service=session_service
        )
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register API routes."""
        
        @self.router.get("/", response_model=List[AgentResponse])
        async def list_agents():
            """List all registered agents."""
            agents = AgentRegistry.get_all_agents()
            return [
                AgentResponse(
                    agent_id=agent.agent_id,
                    name=agent.name,
                    agent_type=agent.agent_type.value,
                    capabilities=[cap.value for cap in agent.get_capabilities()],
                    is_active=agent.is_active,
                    total_tasks_completed=agent.total_tasks_completed,
                    status=agent.get_status()
                )
                for agent in agents
            ]
        
        @self.router.get("/{agent_id}", response_model=AgentResponse)
        async def get_agent(agent_id: str):
            """Get specific agent by ID."""
            agent = AgentRegistry.get_agent(agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
            
            return AgentResponse(
                agent_id=agent.agent_id,
                name=agent.name,
                agent_type=agent.agent_type.value,
                capabilities=[cap.value for cap in agent.get_capabilities()],
                is_active=agent.is_active,
                total_tasks_completed=agent.total_tasks_completed,
                status=agent.get_status()
            )
        
        @self.router.post("/", response_model=AgentResponse)
        async def create_agent(request: AgentCreateRequest):
            """Create a new agent."""
            try:
                if request.agent_type.lower() == "llm":
                    role = LLMRole(request.config.get("role", "generalist")) if request.config else LLMRole.GENERALIST
                    agent = self.factory.create_llm_agent(
                        role=role,
                        name=request.name,
                        custom_config=request.config
                    )
                elif request.agent_type.lower() == "workflow":
                    agent = self.factory.create_workflow_agent(
                        name=request.name
                    )
                elif request.agent_type.lower() == "custom":
                    agent_type = CustomAgentType(request.config.get("agent_type", "domain_expert")) if request.config else CustomAgentType.DOMAIN_EXPERT
                    domain = request.config.get("domain", "") if request.config else ""
                    agent = self.factory.create_custom_agent(
                        agent_type=agent_type,
                        domain=domain,
                        name=request.name,
                        custom_config=request.config
                    )
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown agent type: {request.agent_type}")
                
                # Activate the agent
                await agent.activate()
                
                return AgentResponse(
                    agent_id=agent.agent_id,
                    name=agent.name,
                    agent_type=agent.agent_type.value,
                    capabilities=[cap.value for cap in agent.get_capabilities()],
                    is_active=agent.is_active,
                    total_tasks_completed=agent.total_tasks_completed,
                    status=agent.get_status()
                )
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to create agent: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.post("/teams", response_model=List[AgentResponse])
        async def create_team(request: TeamCreateRequest):
            """Create a team of agents."""
            try:
                suite_type = AgentSuite(request.suite_type)
                agents = self.factory.create_agent_suite(
                    suite_type=suite_type,
                    custom_configs=request.custom_configs
                )
                
                # Activate all agents
                for agent in agents:
                    await agent.activate()
                
                return [
                    AgentResponse(
                        agent_id=agent.agent_id,
                        name=agent.name,
                        agent_type=agent.agent_type.value,
                        capabilities=[cap.value for cap in agent.get_capabilities()],
                        is_active=agent.is_active,
                        total_tasks_completed=agent.total_tasks_completed,
                        status=agent.get_status()
                    )
                    for agent in agents
                ]
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to create team: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.post("/{agent_id}/activate")
        async def activate_agent(agent_id: str):
            """Activate an agent."""
            agent = AgentRegistry.get_agent(agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
            
            success = await agent.activate()
            return {"success": success, "agent_id": agent_id}
        
        @self.router.post("/{agent_id}/deactivate")
        async def deactivate_agent(agent_id: str):
            """Deactivate an agent."""
            agent = AgentRegistry.get_agent(agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
            
            success = await agent.deactivate()
            return {"success": success, "agent_id": agent_id}
        
        @self.router.post("/{agent_id}/task")
        async def execute_task(agent_id: str, request: TaskRequest):
            """Execute a task with a specific agent."""
            agent = AgentRegistry.get_agent(agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
            
            if not agent.is_active:
                raise HTTPException(status_code=400, detail="Agent is not active")
            
            try:
                result = await agent.execute_task(request.task, request.context)
                return {
                    "success": result.success,
                    "result": result.result,
                    "error": result.error,
                    "execution_time_ms": result.execution_time_ms,
                    "metadata": result.metadata
                }
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Task execution failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.get("/registry/status")
        async def get_registry_status():
            """Get agent registry status."""
            return AgentRegistry.get_registry_status()


class OrchestrationAPI:
    """API endpoints for agent orchestration."""
    
    def __init__(self,
                 logger: Optional[RunLogger] = None,
                 session_service: Optional[SessionService] = None):
        
        self.logger = logger
        self.session_service = session_service
        self.router = APIRouter()
        self.orchestrator = AgentOrchestrator(
            logger=logger,
            session_service=session_service
        )
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register orchestration API routes."""
        
        @self.router.post("/task")
        async def orchestrate_task(request: OrchestrationRequest):
            """Execute a task using agent orchestration."""
            try:
                # Convert string enums
                strategy = OrchestrationStrategy(request.strategy.lower())
                priority = TaskPriority(request.priority.lower())
                
                # Convert capability requirements
                requirements = []
                if request.requirements:
                    for req in request.requirements:
                        try:
                            requirements.append(AgentCapability(req.lower()))
                        except ValueError:
                            # Skip unknown capabilities
                            pass
                
                result = await self.orchestrator.orchestrate_task(
                    task=request.task,
                    strategy=strategy,
                    requirements=requirements,
                    context=request.context,
                    priority=priority
                )
                
                return {
                    "success": result.success,
                    "task_id": result.task_id,
                    "strategy_used": result.strategy_used.value,
                    "agents_used": result.agents_used,
                    "primary_result": result.primary_result,
                    "consensus_score": result.consensus_score,
                    "execution_time_ms": result.execution_time_ms,
                    "error": result.error,
                    "metadata": result.metadata
                }
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Orchestration failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.get("/status")
        async def get_orchestration_status():
            """Get orchestration status."""
            return self.orchestrator.get_orchestration_status()
        
        @self.router.get("/strategies")
        async def list_strategies():
            """List available orchestration strategies."""
            return {
                "strategies": [strategy.value for strategy in OrchestrationStrategy],
                "priorities": [priority.value for priority in TaskPriority],
                "capabilities": [capability.value for capability in AgentCapability]
            }
        
        @self.router.get("/workload")
        async def get_workload_status():
            """Get agent workload status."""
            return self.orchestrator.get_agent_workload_status()


class DebugAPI:
    """API endpoints for debugging functionality."""
    
    def __init__(self,
                 debug_interface,
                 logger: Optional[RunLogger] = None):
        
        self.debug_interface = debug_interface
        self.logger = logger
        self.router = APIRouter()
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register debug API routes."""
        
        @self.router.get("/status")
        async def get_debug_status():
            """Get debug interface status."""
            return self.debug_interface.get_status()
        
        @self.router.post("/inspect")
        async def inspect_agent(request: DebugInspectionRequest):
            """Inspect an agent's state and configuration."""
            try:
                inspection_data = await self.debug_interface.inspect_agent(request.agent_id)
                return inspection_data
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Agent inspection failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.get("/logs")
        async def get_captured_logs(
            limit: int = 100,
            level: Optional[str] = None,
            agent_id: Optional[str] = None
        ):
            """Get captured log entries."""
            logs = self.debug_interface.captured_logs.copy()
            
            # Apply filters
            if level:
                logs = [log for log in logs if log.get("level", "").upper() == level.upper()]
            
            if agent_id:
                logs = [log for log in logs if log.get("agent_id") == agent_id]
            
            # Apply limit
            logs = logs[-limit:] if limit > 0 else logs
            
            return {
                "logs": logs,
                "total_count": len(self.debug_interface.captured_logs),
                "filtered_count": len(logs)
            }
        
        @self.router.post("/step-mode/{enabled}")
        async def set_step_mode(enabled: bool):
            """Enable or disable step-by-step debugging mode."""
            self.debug_interface.step_mode_enabled = enabled
            return {"step_mode_enabled": enabled}
        
        @self.router.get("/performance")
        async def get_performance_data():
            """Get performance profiling data."""
            return {
                "performance_data": self.debug_interface.performance_data,
                "execution_traces": self.debug_interface.execution_traces[-100:],  # Last 100 traces
                "total_traces": len(self.debug_interface.execution_traces)
            }
        
        @self.router.post("/breakpoint")
        async def set_breakpoint(
            agent_id: str,
            condition: Optional[str] = None
        ):
            """Set a debugging breakpoint for an agent."""
            breakpoint_id = f"{agent_id}_{int(time.time())}"
            
            self.debug_interface.breakpoints[breakpoint_id] = {
                "agent_id": agent_id,
                "condition": condition,
                "created_at": time.time(),
                "hit_count": 0
            }
            
            return {
                "breakpoint_id": breakpoint_id,
                "agent_id": agent_id,
                "condition": condition
            }
        
        @self.router.delete("/breakpoint/{breakpoint_id}")
        async def remove_breakpoint(breakpoint_id: str):
            """Remove a debugging breakpoint."""
            if breakpoint_id in self.debug_interface.breakpoints:
                del self.debug_interface.breakpoints[breakpoint_id]
                return {"success": True, "breakpoint_id": breakpoint_id}
            else:
                raise HTTPException(status_code=404, detail="Breakpoint not found")
        
        @self.router.get("/breakpoints")
        async def list_breakpoints():
            """List all active breakpoints."""
            return {
                "breakpoints": self.debug_interface.breakpoints,
                "total_count": len(self.debug_interface.breakpoints)
            }


class MonitoringAPI:
    """API endpoints for monitoring functionality."""
    
    def __init__(self,
                 monitoring_interface,
                 logger: Optional[RunLogger] = None):
        
        self.monitoring_interface = monitoring_interface
        self.logger = logger
        self.router = APIRouter()
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register monitoring API routes."""
        
        @self.router.get("/status")
        async def get_monitoring_status():
            """Get monitoring interface status."""
            return self.monitoring_interface.get_status()
        
        @self.router.get("/metrics/current")
        async def get_current_metrics():
            """Get current system metrics."""
            return self.monitoring_interface.current_metrics
        
        @self.router.get("/metrics/performance")
        async def get_performance_metrics(
            limit: int = 100,
            start_time: Optional[float] = None,
            end_time: Optional[float] = None
        ):
            """Get performance metrics within time range."""
            metrics = self.monitoring_interface.performance_metrics.copy()
            
            # Apply time filters
            if start_time:
                metrics = [m for m in metrics if m["timestamp"] >= start_time]
            
            if end_time:
                metrics = [m for m in metrics if m["timestamp"] <= end_time]
            
            # Apply limit
            metrics = metrics[-limit:] if limit > 0 else metrics
            
            return {
                "metrics": metrics,
                "total_count": len(self.monitoring_interface.performance_metrics),
                "filtered_count": len(metrics)
            }
        
        @self.router.get("/alerts")
        async def get_active_alerts():
            """Get currently active alerts."""
            return {
                "alerts": self.monitoring_interface.alert_states,
                "alert_count": len(self.monitoring_interface.alert_states),
                "alert_thresholds": self.monitoring_interface.config.alert_thresholds
            }
        
        @self.router.post("/alerts/acknowledge/{metric_name}")
        async def acknowledge_alert(metric_name: str):
            """Acknowledge an active alert."""
            if metric_name in self.monitoring_interface.alert_states:
                self.monitoring_interface.alert_states[metric_name]["acknowledged"] = True
                self.monitoring_interface.alert_states[metric_name]["acknowledged_at"] = time.time()
                return {"success": True, "metric_name": metric_name}
            else:
                raise HTTPException(status_code=404, detail="Alert not found")
        
        @self.router.get("/export")
        async def export_metrics(
            format: str = "json",
            metric_type: str = "all",
            start_time: Optional[float] = None,
            end_time: Optional[float] = None
        ):
            """Export metrics in specified format."""
            if format not in self.monitoring_interface.config.export_formats:
                raise HTTPException(status_code=400, detail=f"Format {format} not supported")
            
            # Collect metrics based on type
            if metric_type == "performance":
                data = self.monitoring_interface.performance_metrics
            elif metric_type == "usage":
                data = self.monitoring_interface.usage_metrics
            elif metric_type == "error":
                data = self.monitoring_interface.error_metrics
            else:
                data = {
                    "performance": self.monitoring_interface.performance_metrics,
                    "usage": self.monitoring_interface.usage_metrics,
                    "error": self.monitoring_interface.error_metrics,
                }
            
            # Apply time filters if specified
            if start_time or end_time:
                # Implementation would filter by timestamp
                pass
            
            return {
                "data": data,
                "format": format,
                "exported_at": time.time(),
                "record_count": len(data) if isinstance(data, list) else sum(len(v) for v in data.values())
            }