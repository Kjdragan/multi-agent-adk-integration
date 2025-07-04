"""
Web Dashboard Components

Dashboard implementations for monitoring agents, tasks, and performance
with real-time updates and interactive debugging capabilities.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..agents import AgentRegistry, AgentType, AgentCapability
from ..agents.orchestrator import OrchestrationStrategy, TaskPriority
from ..platform_logging import RunLogger


class DashboardType(str, Enum):
    """Types of dashboards available."""
    OVERVIEW = "overview"
    AGENTS = "agents"
    TASKS = "tasks"
    PERFORMANCE = "performance"
    DEBUG = "debug"
    MONITORING = "monitoring"
    ORCHESTRATION = "orchestration"


@dataclass
class DashboardWidget:
    """Individual dashboard widget."""
    widget_id: str
    widget_type: str
    title: str
    data: Any = None
    config: Dict[str, Any] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)
    refresh_interval: int = 5  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "widget_id": self.widget_id,
            "widget_type": self.widget_type,
            "title": self.title,
            "data": self.data,
            "config": self.config,
            "last_updated": self.last_updated,
            "refresh_interval": self.refresh_interval,
        }


class BaseDashboard:
    """Base class for all dashboards."""
    
    def __init__(self,
                 dashboard_id: str,
                 dashboard_type: DashboardType,
                 title: str,
                 logger: Optional[RunLogger] = None):
        
        self.dashboard_id = dashboard_id
        self.dashboard_type = dashboard_type
        self.title = title
        self.logger = logger
        
        # Dashboard state
        self.widgets: Dict[str, DashboardWidget] = {}
        self.layout_config: Dict[str, Any] = {}
        self.is_active = False
        self.last_refresh = time.time()
        self.auto_refresh_enabled = True
        self.refresh_interval = 5  # seconds
        
        # Initialize widgets
        self._initialize_widgets()
    
    def _initialize_widgets(self) -> None:
        """Initialize dashboard widgets - to be implemented by subclasses."""
        pass
    
    async def refresh(self) -> None:
        """Refresh all dashboard data."""
        try:
            for widget in self.widgets.values():
                await self._refresh_widget(widget)
            
            self.last_refresh = time.time()
            
            if self.logger:
                self.logger.debug(f"Dashboard {self.dashboard_id} refreshed")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to refresh dashboard {self.dashboard_id}: {e}")
    
    async def _refresh_widget(self, widget: DashboardWidget) -> None:
        """Refresh individual widget data - to be implemented by subclasses."""
        widget.last_updated = time.time()
    
    def add_widget(self, widget: DashboardWidget) -> None:
        """Add widget to dashboard."""
        self.widgets[widget.widget_id] = widget
    
    def remove_widget(self, widget_id: str) -> bool:
        """Remove widget from dashboard."""
        if widget_id in self.widgets:
            del self.widgets[widget_id]
            return True
        return False
    
    def get_widget_data(self, widget_id: str) -> Optional[Dict[str, Any]]:
        """Get data for specific widget."""
        widget = self.widgets.get(widget_id)
        return widget.to_dict() if widget else None
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data."""
        return {
            "dashboard_id": self.dashboard_id,
            "dashboard_type": self.dashboard_type.value,
            "title": self.title,
            "is_active": self.is_active,
            "last_refresh": self.last_refresh,
            "auto_refresh_enabled": self.auto_refresh_enabled,
            "refresh_interval": self.refresh_interval,
            "widgets": {wid: widget.to_dict() for wid, widget in self.widgets.items()},
            "layout_config": self.layout_config,
        }
    
    def set_layout_config(self, layout_config: Dict[str, Any]) -> None:
        """Set dashboard layout configuration."""
        self.layout_config = layout_config


class AgentDashboard(BaseDashboard):
    """Dashboard for monitoring agents."""
    
    def __init__(self, logger: Optional[RunLogger] = None):
        super().__init__(
            dashboard_id="agents",
            dashboard_type=DashboardType.AGENTS,
            title="Agent Monitoring Dashboard",
            logger=logger
        )
    
    def _initialize_widgets(self) -> None:
        """Initialize agent monitoring widgets."""
        # Agent overview widget
        self.add_widget(DashboardWidget(
            widget_id="agent_overview",
            widget_type="stats_card",
            title="Agent Overview",
            config={"chart_type": "donut"}
        ))
        
        # Active agents list widget
        self.add_widget(DashboardWidget(
            widget_id="active_agents_list",
            widget_type="list",
            title="Active Agents",
            config={"sortable": True, "filterable": True}
        ))
        
        # Agent types distribution widget
        self.add_widget(DashboardWidget(
            widget_id="agent_types_chart",
            widget_type="bar_chart",
            title="Agent Types Distribution",
            config={"horizontal": True}
        ))
        
        # Agent capabilities matrix widget
        self.add_widget(DashboardWidget(
            widget_id="capabilities_matrix",
            widget_type="heatmap",
            title="Agent Capabilities Matrix",
            config={"color_scheme": "viridis"}
        ))
        
        # Agent performance widget
        self.add_widget(DashboardWidget(
            widget_id="agent_performance",
            widget_type="line_chart",
            title="Agent Performance Trends",
            config={"time_range": "1h"}
        ))
        
        # Recent agent activities widget
        self.add_widget(DashboardWidget(
            widget_id="recent_activities",
            widget_type="timeline",
            title="Recent Agent Activities",
            config={"limit": 20}
        ))
    
    async def _refresh_widget(self, widget: DashboardWidget) -> None:
        """Refresh agent dashboard widget data."""
        try:
            if widget.widget_id == "agent_overview":
                await self._refresh_agent_overview(widget)
            elif widget.widget_id == "active_agents_list":
                await self._refresh_active_agents_list(widget)
            elif widget.widget_id == "agent_types_chart":
                await self._refresh_agent_types_chart(widget)
            elif widget.widget_id == "capabilities_matrix":
                await self._refresh_capabilities_matrix(widget)
            elif widget.widget_id == "agent_performance":
                await self._refresh_agent_performance(widget)
            elif widget.widget_id == "recent_activities":
                await self._refresh_recent_activities(widget)
            
            widget.last_updated = time.time()
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to refresh widget {widget.widget_id}: {e}")
    
    async def _refresh_agent_overview(self, widget: DashboardWidget) -> None:
        """Refresh agent overview statistics."""
        registry_status = AgentRegistry.get_registry_status()
        all_agents = AgentRegistry.get_all_agents()
        
        # Calculate performance metrics
        total_tasks = sum(agent.total_tasks_completed for agent in all_agents)
        avg_performance = 0.0
        
        if all_agents:
            performing_agents = [
                agent for agent in all_agents 
                if hasattr(agent, 'get_performance_metrics')
            ]
            
            if performing_agents:
                success_rates = []
                for agent in performing_agents:
                    metrics = agent.get_performance_metrics()
                    success_rates.append(metrics.get('success_rate_percent', 0))
                
                avg_performance = sum(success_rates) / len(success_rates)
        
        widget.data = {
            "total_agents": registry_status["total_agents"],
            "active_agents": registry_status["active_agents"],
            "total_tasks_completed": total_tasks,
            "average_performance": avg_performance,
            "agents_by_type": registry_status["agents_by_type"],
            "agents_by_capability": registry_status["agents_by_capability"],
        }
    
    async def _refresh_active_agents_list(self, widget: DashboardWidget) -> None:
        """Refresh active agents list."""
        all_agents = AgentRegistry.get_all_agents()
        active_agents = [agent for agent in all_agents if agent.is_active]
        
        agent_list = []
        for agent in active_agents:
            agent_data = {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "type": agent.agent_type.value,
                "capabilities": [cap.value for cap in agent.get_capabilities()],
                "tasks_completed": agent.total_tasks_completed,
                "last_task_time": agent.last_task_time,
                "status": "active"
            }
            
            # Add performance metrics if available
            if hasattr(agent, 'get_performance_metrics'):
                metrics = agent.get_performance_metrics()
                agent_data.update({
                    "success_rate": metrics.get('success_rate_percent', 0),
                    "avg_response_time": metrics.get('average_response_time_ms', 0),
                })
            
            agent_list.append(agent_data)
        
        # Sort by performance or tasks completed
        agent_list.sort(key=lambda x: x.get('success_rate', x['tasks_completed']), reverse=True)
        
        widget.data = {
            "agents": agent_list,
            "total_count": len(agent_list),
            "last_updated": time.time(),
        }
    
    async def _refresh_agent_types_chart(self, widget: DashboardWidget) -> None:
        """Refresh agent types distribution chart."""
        registry_status = AgentRegistry.get_registry_status()
        
        # Prepare chart data
        chart_data = []
        for agent_type, count in registry_status["agents_by_type"].items():
            chart_data.append({
                "label": agent_type.replace("_", " ").title(),
                "value": count,
                "percentage": (count / registry_status["total_agents"] * 100) if registry_status["total_agents"] > 0 else 0
            })
        
        widget.data = {
            "chart_data": chart_data,
            "total_agents": registry_status["total_agents"],
        }
    
    async def _refresh_capabilities_matrix(self, widget: DashboardWidget) -> None:
        """Refresh agent capabilities matrix."""
        all_agents = AgentRegistry.get_all_agents()
        
        # Create capability matrix
        capability_matrix = {}
        agent_names = []
        all_capabilities = set()
        
        for agent in all_agents:
            agent_names.append(agent.name)
            agent_capabilities = agent.get_capabilities()
            all_capabilities.update(cap.value for cap in agent_capabilities)
            
            capability_matrix[agent.name] = {
                cap.value: 1 for cap in agent_capabilities
            }
        
        # Fill missing capabilities with 0
        for agent_name in agent_names:
            for capability in all_capabilities:
                if capability not in capability_matrix[agent_name]:
                    capability_matrix[agent_name][capability] = 0
        
        widget.data = {
            "matrix": capability_matrix,
            "agents": agent_names,
            "capabilities": sorted(list(all_capabilities)),
        }
    
    async def _refresh_agent_performance(self, widget: DashboardWidget) -> None:
        """Refresh agent performance trends."""
        all_agents = AgentRegistry.get_all_agents()
        
        performance_data = []
        timestamps = []
        current_time = time.time()
        
        # Generate time series data (last hour with 5-minute intervals)
        for i in range(12):
            timestamp = current_time - (i * 300)  # 5 minutes
            timestamps.append(timestamp)
        
        timestamps.reverse()
        
        # Get performance data for each agent
        for agent in all_agents[:10]:  # Limit to top 10 agents
            if hasattr(agent, 'get_performance_metrics'):
                metrics = agent.get_performance_metrics()
                success_rate = metrics.get('success_rate_percent', 0)
                
                # Simulate time series data (in real implementation, this would come from stored metrics)
                agent_performance = {
                    "agent_name": agent.name,
                    "data": [success_rate + (i % 3) for i in range(len(timestamps))],  # Simulated variation
                }
                performance_data.append(agent_performance)
        
        widget.data = {
            "timestamps": timestamps,
            "performance_data": performance_data,
            "time_range": widget.config.get("time_range", "1h"),
        }
    
    async def _refresh_recent_activities(self, widget: DashboardWidget) -> None:
        """Refresh recent agent activities."""
        all_agents = AgentRegistry.get_all_agents()
        
        activities = []
        for agent in all_agents:
            if agent.last_task_time:
                activities.append({
                    "timestamp": agent.last_task_time,
                    "agent_name": agent.name,
                    "agent_id": agent.agent_id,
                    "activity_type": "task_completed",
                    "description": f"Completed task (Total: {agent.total_tasks_completed})",
                })
        
        # Sort by timestamp (most recent first)
        activities.sort(key=lambda x: x["timestamp"] or 0, reverse=True)
        
        # Limit to configured number
        limit = widget.config.get("limit", 20)
        activities = activities[:limit]
        
        widget.data = {
            "activities": activities,
            "total_count": len(activities),
        }


class TaskDashboard(BaseDashboard):
    """Dashboard for monitoring tasks and orchestration."""
    
    def __init__(self, 
                 orchestrator=None,
                 logger: Optional[RunLogger] = None):
        
        self.orchestrator = orchestrator
        super().__init__(
            dashboard_id="tasks",
            dashboard_type=DashboardType.TASKS,
            title="Task & Orchestration Dashboard",
            logger=logger
        )
    
    def _initialize_widgets(self) -> None:
        """Initialize task monitoring widgets."""
        # Task overview widget
        self.add_widget(DashboardWidget(
            widget_id="task_overview",
            widget_type="stats_card",
            title="Task Overview",
        ))
        
        # Active tasks widget
        self.add_widget(DashboardWidget(
            widget_id="active_tasks",
            widget_type="list",
            title="Active Tasks",
            config={"real_time": True}
        ))
        
        # Orchestration strategies widget
        self.add_widget(DashboardWidget(
            widget_id="orchestration_strategies",
            widget_type="pie_chart",
            title="Strategy Usage Distribution",
        ))
        
        # Task completion rates widget
        self.add_widget(DashboardWidget(
            widget_id="completion_rates",
            widget_type="line_chart",
            title="Task Completion Rates",
            config={"time_range": "24h"}
        ))
        
        # Task duration analysis widget
        self.add_widget(DashboardWidget(
            widget_id="duration_analysis",
            widget_type="histogram",
            title="Task Duration Analysis",
        ))
        
        # Orchestration flow widget
        self.add_widget(DashboardWidget(
            widget_id="orchestration_flow",
            widget_type="flowchart",
            title="Current Orchestration Flow",
            config={"interactive": True}
        ))
    
    async def _refresh_widget(self, widget: DashboardWidget) -> None:
        """Refresh task dashboard widget data."""
        try:
            if widget.widget_id == "task_overview":
                await self._refresh_task_overview(widget)
            elif widget.widget_id == "active_tasks":
                await self._refresh_active_tasks(widget)
            elif widget.widget_id == "orchestration_strategies":
                await self._refresh_orchestration_strategies(widget)
            elif widget.widget_id == "completion_rates":
                await self._refresh_completion_rates(widget)
            elif widget.widget_id == "duration_analysis":
                await self._refresh_duration_analysis(widget)
            elif widget.widget_id == "orchestration_flow":
                await self._refresh_orchestration_flow(widget)
            
            widget.last_updated = time.time()
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to refresh widget {widget.widget_id}: {e}")
    
    async def _refresh_task_overview(self, widget: DashboardWidget) -> None:
        """Refresh task overview statistics."""
        if not self.orchestrator:
            widget.data = {"error": "No orchestrator available"}
            return
        
        status = self.orchestrator.get_orchestration_status()
        
        widget.data = {
            "active_tasks": status.get("active_tasks", 0),
            "queued_tasks": status.get("queued_tasks", 0),
            "completed_tasks": status.get("completed_tasks", 0),
            "total_orchestrated": status.get("total_orchestrated", 0),
            "success_rate": status.get("success_rate", 0),
            "available_agents": status.get("available_agents", 0),
            "active_agents": status.get("active_agents", 0),
        }
    
    async def _refresh_active_tasks(self, widget: DashboardWidget) -> None:
        """Refresh active tasks list."""
        if not self.orchestrator:
            widget.data = {"tasks": [], "total_count": 0}
            return
        
        # Get active tasks from orchestrator
        active_tasks = []
        for task_id, task_allocation in self.orchestrator.active_tasks.items():
            task_data = {
                "task_id": task_id,
                "task": task_allocation.task[:100] + "..." if len(task_allocation.task) > 100 else task_allocation.task,
                "strategy": task_allocation.strategy.value,
                "priority": task_allocation.priority.value,
                "assigned_agents": task_allocation.assigned_agents,
                "status": task_allocation.status,
                "start_time": task_allocation.start_time,
                "progress": self._calculate_task_progress(task_allocation),
            }
            active_tasks.append(task_data)
        
        widget.data = {
            "tasks": active_tasks,
            "total_count": len(active_tasks),
        }
    
    def _calculate_task_progress(self, task_allocation) -> float:
        """Calculate task progress percentage."""
        # Simple progress calculation based on task status
        if task_allocation.status == "completed":
            return 100.0
        elif task_allocation.status == "running":
            # Estimate progress based on time elapsed
            if task_allocation.start_time:
                elapsed = time.time() - task_allocation.start_time
                estimated_duration = task_allocation.timeout_seconds or 300
                return min(90.0, (elapsed / estimated_duration) * 100)
            return 10.0
        else:
            return 0.0
    
    async def _refresh_orchestration_strategies(self, widget: DashboardWidget) -> None:
        """Refresh orchestration strategies usage."""
        if not self.orchestrator:
            widget.data = {"strategies": []}
            return
        
        strategy_success_rates = self.orchestrator.strategy_success_rates
        
        chart_data = []
        for strategy, success_rate in strategy_success_rates.items():
            chart_data.append({
                "label": strategy.replace("_", " ").title(),
                "value": success_rate,
                "usage_count": 1,  # Would track actual usage counts
            })
        
        widget.data = {
            "strategies": chart_data,
            "total_strategies": len(chart_data),
        }
    
    async def _refresh_completion_rates(self, widget: DashboardWidget) -> None:
        """Refresh task completion rates over time."""
        # Simulate completion rate data over time
        current_time = time.time()
        timestamps = []
        completion_rates = []
        
        # Generate data points for the last 24 hours
        for i in range(24):
            timestamp = current_time - (i * 3600)  # 1 hour intervals
            timestamps.append(timestamp)
            # Simulate completion rate (would use actual data)
            completion_rates.append(85 + (i % 10))  # Simulated data
        
        timestamps.reverse()
        completion_rates.reverse()
        
        widget.data = {
            "timestamps": timestamps,
            "completion_rates": completion_rates,
            "average_rate": sum(completion_rates) / len(completion_rates),
            "time_range": widget.config.get("time_range", "24h"),
        }
    
    async def _refresh_duration_analysis(self, widget: DashboardWidget) -> None:
        """Refresh task duration analysis."""
        if not self.orchestrator:
            widget.data = {"duration_buckets": []}
            return
        
        # Analyze completed tasks duration
        completed_tasks = self.orchestrator.completed_tasks
        durations = []
        
        for task in completed_tasks:
            if task.start_time and task.end_time:
                duration = (task.end_time - task.start_time) * 1000  # Convert to ms
                durations.append(duration)
        
        # Create histogram buckets
        if durations:
            min_duration = min(durations)
            max_duration = max(durations)
            bucket_size = (max_duration - min_duration) / 10
            
            buckets = []
            for i in range(10):
                bucket_min = min_duration + (i * bucket_size)
                bucket_max = bucket_min + bucket_size
                count = sum(1 for d in durations if bucket_min <= d < bucket_max)
                
                buckets.append({
                    "range": f"{bucket_min:.0f}-{bucket_max:.0f}ms",
                    "count": count,
                    "percentage": (count / len(durations)) * 100,
                })
            
            widget.data = {
                "duration_buckets": buckets,
                "total_tasks": len(durations),
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min_duration,
                "max_duration": max_duration,
            }
        else:
            widget.data = {"duration_buckets": [], "total_tasks": 0}
    
    async def _refresh_orchestration_flow(self, widget: DashboardWidget) -> None:
        """Refresh orchestration flow visualization."""
        if not self.orchestrator:
            widget.data = {"nodes": [], "edges": []}
            return
        
        # Create flow chart nodes and edges
        nodes = []
        edges = []
        
        # Add orchestrator node
        nodes.append({
            "id": "orchestrator",
            "label": "Orchestrator",
            "type": "orchestrator",
            "status": "active",
        })
        
        # Add agent nodes
        registry_status = AgentRegistry.get_registry_status()
        for i, (agent_type, count) in enumerate(registry_status["agents_by_type"].items()):
            if count > 0:
                node_id = f"agents_{agent_type}"
                nodes.append({
                    "id": node_id,
                    "label": f"{agent_type.replace('_', ' ').title()} ({count})",
                    "type": "agent_group",
                    "count": count,
                })
                
                # Add edge from orchestrator to agent group
                edges.append({
                    "from": "orchestrator",
                    "to": node_id,
                    "label": "orchestrates",
                })
        
        widget.data = {
            "nodes": nodes,
            "edges": edges,
            "layout": "hierarchical",
        }


class PerformanceDashboard(BaseDashboard):
    """Dashboard for monitoring system performance."""
    
    def __init__(self, 
                 monitoring_interface=None,
                 logger: Optional[RunLogger] = None):
        
        self.monitoring_interface = monitoring_interface
        super().__init__(
            dashboard_id="performance",
            dashboard_type=DashboardType.PERFORMANCE,
            title="Performance Monitoring Dashboard",
            logger=logger
        )
    
    def _initialize_widgets(self) -> None:
        """Initialize performance monitoring widgets."""
        # System metrics widget
        self.add_widget(DashboardWidget(
            widget_id="system_metrics",
            widget_type="metrics_grid",
            title="System Metrics",
            refresh_interval=2
        ))
        
        # Performance trends widget
        self.add_widget(DashboardWidget(
            widget_id="performance_trends",
            widget_type="multi_line_chart",
            title="Performance Trends",
            config={"time_range": "1h"}
        ))
        
        # Agent performance comparison widget
        self.add_widget(DashboardWidget(
            widget_id="agent_performance_comparison",
            widget_type="radar_chart",
            title="Agent Performance Comparison",
        ))
        
        # Error rates widget
        self.add_widget(DashboardWidget(
            widget_id="error_rates",
            widget_type="area_chart",
            title="Error Rates Over Time",
            config={"color": "red"}
        ))
        
        # Resource utilization widget
        self.add_widget(DashboardWidget(
            widget_id="resource_utilization",
            widget_type="gauge_chart",
            title="Resource Utilization",
        ))
        
        # Active alerts widget
        self.add_widget(DashboardWidget(
            widget_id="active_alerts",
            widget_type="alert_list",
            title="Active Performance Alerts",
            config={"severity_filter": "warning"}
        ))
    
    async def _refresh_widget(self, widget: DashboardWidget) -> None:
        """Refresh performance dashboard widget data."""
        try:
            if widget.widget_id == "system_metrics":
                await self._refresh_system_metrics(widget)
            elif widget.widget_id == "performance_trends":
                await self._refresh_performance_trends(widget)
            elif widget.widget_id == "agent_performance_comparison":
                await self._refresh_agent_performance_comparison(widget)
            elif widget.widget_id == "error_rates":
                await self._refresh_error_rates(widget)
            elif widget.widget_id == "resource_utilization":
                await self._refresh_resource_utilization(widget)
            elif widget.widget_id == "active_alerts":
                await self._refresh_active_alerts(widget)
            
            widget.last_updated = time.time()
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to refresh widget {widget.widget_id}: {e}")
    
    async def _refresh_system_metrics(self, widget: DashboardWidget) -> None:
        """Refresh system metrics."""
        if self.monitoring_interface:
            current_metrics = self.monitoring_interface.current_metrics
        else:
            current_metrics = {}
        
        # Get basic system information
        registry_status = AgentRegistry.get_registry_status()
        
        metrics = {
            "agents_total": registry_status.get("total_agents", 0),
            "agents_active": registry_status.get("active_agents", 0),
            "memory_usage_mb": current_metrics.get("memory_usage", 0),  # Would get actual memory
            "cpu_usage_percent": current_metrics.get("cpu_usage", 0),   # Would get actual CPU
            "response_time_avg_ms": current_metrics.get("avg_response_time", 0),
            "requests_per_minute": current_metrics.get("requests_per_minute", 0),
            "uptime_seconds": current_metrics.get("uptime", 0),
            "error_rate_percent": current_metrics.get("error_rate", 0),
        }
        
        widget.data = {
            "metrics": metrics,
            "thresholds": {
                "memory_warning": 512,      # MB
                "cpu_warning": 80,          # %
                "response_time_warning": 2000,  # ms
                "error_rate_warning": 5,    # %
            }
        }
    
    async def _refresh_performance_trends(self, widget: DashboardWidget) -> None:
        """Refresh performance trends chart."""
        if self.monitoring_interface:
            performance_metrics = self.monitoring_interface.performance_metrics
        else:
            performance_metrics = []
        
        # Generate time series data
        current_time = time.time()
        timestamps = []
        response_times = []
        error_rates = []
        
        # Use last hour of data
        time_range_seconds = 3600  # 1 hour
        
        if performance_metrics:
            # Use actual metrics
            recent_metrics = [
                m for m in performance_metrics 
                if m["timestamp"] > current_time - time_range_seconds
            ]
            
            for metric in recent_metrics[-60:]:  # Last 60 data points
                timestamps.append(metric["timestamp"])
                response_times.append(metric.get("avg_response_time", 0))
                error_rates.append(metric.get("error_rate", 0))
        else:
            # Generate simulated data
            for i in range(60):
                timestamp = current_time - (i * 60)  # 1 minute intervals
                timestamps.append(timestamp)
                response_times.append(100 + (i % 20))  # Simulated
                error_rates.append(1 + (i % 5))        # Simulated
        
        timestamps.reverse()
        response_times.reverse()
        error_rates.reverse()
        
        widget.data = {
            "timestamps": timestamps,
            "series": [
                {
                    "name": "Response Time (ms)",
                    "data": response_times,
                    "color": "#3498db",
                },
                {
                    "name": "Error Rate (%)",
                    "data": error_rates,
                    "color": "#e74c3c",
                }
            ],
            "time_range": widget.config.get("time_range", "1h"),
        }
    
    async def _refresh_agent_performance_comparison(self, widget: DashboardWidget) -> None:
        """Refresh agent performance comparison radar chart."""
        all_agents = AgentRegistry.get_all_agents()
        
        agent_performance_data = []
        
        for agent in all_agents[:6]:  # Limit to 6 agents for readability
            if hasattr(agent, 'get_performance_metrics'):
                metrics = agent.get_performance_metrics()
                
                # Normalize metrics to 0-100 scale for radar chart
                performance_data = {
                    "agent_name": agent.name,
                    "metrics": [
                        metrics.get('success_rate_percent', 0),
                        min(100, (1000 / max(metrics.get('average_response_time_ms', 1000), 1)) * 100),  # Inverted response time
                        min(100, metrics.get('total_tasks', 0) * 10),  # Task completion (scaled)
                        100 - min(100, metrics.get('failed_tasks', 0) * 20),  # Reliability (inverted failure rate)
                        75,  # Quality score (placeholder)
                        80,  # Efficiency score (placeholder)
                    ],
                    "labels": [
                        "Success Rate",
                        "Response Speed", 
                        "Task Volume",
                        "Reliability",
                        "Quality",
                        "Efficiency"
                    ]
                }
                
                agent_performance_data.append(performance_data)
        
        widget.data = {
            "agents": agent_performance_data,
            "metrics_labels": ["Success Rate", "Response Speed", "Task Volume", "Reliability", "Quality", "Efficiency"],
        }
    
    async def _refresh_error_rates(self, widget: DashboardWidget) -> None:
        """Refresh error rates chart."""
        # Simulate error rate data over time
        current_time = time.time()
        timestamps = []
        error_counts = []
        
        for i in range(60):  # Last 60 minutes
            timestamp = current_time - (i * 60)
            timestamps.append(timestamp)
            # Simulate error count (would use actual error metrics)
            error_counts.append(max(0, 5 - (i % 7)))
        
        timestamps.reverse()
        error_counts.reverse()
        
        widget.data = {
            "timestamps": timestamps,
            "error_counts": error_counts,
            "total_errors": sum(error_counts),
            "avg_errors_per_minute": sum(error_counts) / len(error_counts),
        }
    
    async def _refresh_resource_utilization(self, widget: DashboardWidget) -> None:
        """Refresh resource utilization gauges."""
        # Simulate resource utilization data
        widget.data = {
            "cpu_usage": 45.2,      # Would get actual CPU usage
            "memory_usage": 67.8,   # Would get actual memory usage
            "disk_usage": 23.5,     # Would get actual disk usage
            "network_usage": 12.1,  # Would get actual network usage
            "agent_capacity": 78.9, # Percentage of agent capacity used
        }
    
    async def _refresh_active_alerts(self, widget: DashboardWidget) -> None:
        """Refresh active performance alerts."""
        if self.monitoring_interface:
            alert_states = self.monitoring_interface.alert_states
        else:
            alert_states = {}
        
        alerts = []
        for metric_name, alert_data in alert_states.items():
            alerts.append({
                "metric": metric_name,
                "severity": self._determine_alert_severity(alert_data),
                "message": f"{metric_name} exceeded threshold",
                "triggered_at": alert_data.get("triggered_at"),
                "current_value": alert_data.get("current_value"),
                "threshold": alert_data.get("threshold"),
                "acknowledged": alert_data.get("acknowledged", False),
            })
        
        # Sort by severity and time
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        alerts.sort(key=lambda x: (
            severity_order.get(x["severity"], 3),
            -(x["triggered_at"] or 0)
        ))
        
        widget.data = {
            "alerts": alerts,
            "total_count": len(alerts),
            "critical_count": sum(1 for a in alerts if a["severity"] == "critical"),
            "warning_count": sum(1 for a in alerts if a["severity"] == "warning"),
        }
    
    def _determine_alert_severity(self, alert_data: Dict[str, Any]) -> str:
        """Determine alert severity based on threshold exceeded."""
        current_value = alert_data.get("current_value", 0)
        threshold = alert_data.get("threshold", 0)
        
        if threshold == 0:
            return "info"
        
        exceeded_by = (current_value - threshold) / threshold
        
        if exceeded_by > 0.5:  # 50% over threshold
            return "critical"
        elif exceeded_by > 0.2:  # 20% over threshold
            return "warning"
        else:
            return "info"