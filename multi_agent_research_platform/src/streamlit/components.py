"""
Streamlit Components

Reusable UI components for the Multi-Agent Research Platform Streamlit interface.
Provides specialized widgets, charts, and interactive elements.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable
import time

from ..agents import AgentRegistry, OrchestrationStrategy, TaskPriority, AgentCapability


class AgentCard:
    """Component for displaying agent information in a card format."""
    
    @staticmethod
    def render(agent, show_details: bool = True, show_controls: bool = False):
        """Render an agent card."""
        with st.container():
            # Main agent info
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                # Agent name and type
                st.markdown(f"**{agent.name}**")
                agent_type_display = agent.agent_type.value.replace('_', ' ').title()
                st.caption(f"{agent_type_display} Agent")
            
            with col2:
                # Status indicator
                if agent.is_active:
                    st.success("🟢 Active")
                else:
                    st.error("🔴 Inactive")
            
            with col3:
                # Task count
                st.metric("Tasks", agent.total_tasks_completed)
            
            if show_details:
                # Capabilities
                capabilities = agent.get_capabilities()
                if capabilities:
                    cap_names = [cap.value.replace('_', ' ').title() for cap in capabilities]
                    st.caption(f"**Capabilities:** {', '.join(cap_names[:3])}")
                    if len(cap_names) > 3:
                        st.caption(f"...and {len(cap_names) - 3} more")
                
                # Performance metrics if available
                if hasattr(agent, 'get_performance_metrics'):
                    metrics = agent.get_performance_metrics()
                    if metrics:
                        col1, col2 = st.columns(2)
                        with col1:
                            success_rate = metrics.get('success_rate_percent', 0)
                            st.caption(f"Success Rate: {success_rate:.1f}%")
                        with col2:
                            avg_time = metrics.get('average_response_time_ms', 0)
                            st.caption(f"Avg Time: {avg_time:.0f}ms")
            
            if show_controls:
                # Agent controls
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📊 Details", key=f"details_{agent.agent_id}"):
                        st.session_state[f"show_agent_details_{agent.agent_id}"] = True
                
                with col2:
                    if agent.is_active:
                        if st.button("⏸️ Pause", key=f"pause_{agent.agent_id}"):
                            # Deactivate agent
                            pass
                    else:
                        if st.button("▶️ Resume", key=f"resume_{agent.agent_id}"):
                            # Activate agent
                            pass
                
                with col3:
                    if st.button("🗑️ Remove", key=f"remove_{agent.agent_id}"):
                        st.session_state[f"confirm_remove_{agent.agent_id}"] = True
            
            st.divider()


class TaskForm:
    """Component for task input and configuration."""
    
    @staticmethod
    def render(on_submit: Callable[[Dict[str, Any]], None]):
        """Render task execution form."""
        with st.form("task_form"):
            st.subheader("🎯 Research Task")
            
            # Task input
            task_input = st.text_area(
                "Research Question or Task",
                placeholder="Enter your research question or task description...",
                help="Describe what you want the agents to research or accomplish",
                height=100
            )
            
            # Configuration columns
            col1, col2 = st.columns(2)
            
            with col1:
                # Strategy selection
                strategy = st.selectbox(
                    "Orchestration Strategy",
                    options=[s.value for s in OrchestrationStrategy],
                    index=0,
                    help="How should agents collaborate on this task?"
                )
                
                # Priority selection
                priority = st.selectbox(
                    "Task Priority",
                    options=[p.value for p in TaskPriority],
                    index=1,  # Medium
                    help="Priority level for task execution"
                )
            
            with col2:
                # Timeout setting
                timeout = st.slider(
                    "Timeout (seconds)",
                    min_value=30,
                    max_value=300,
                    value=120,
                    step=30,
                    help="Maximum time allowed for task execution"
                )
                
                # Context options
                include_context = st.checkbox(
                    "Include conversation history",
                    value=True,
                    help="Provide previous task results as context"
                )
            
            # Advanced options
            with st.expander("Advanced Options"):
                max_agents = st.slider(
                    "Maximum Agents",
                    min_value=1,
                    max_value=10,
                    value=5,
                    help="Maximum number of agents to use for this task"
                )
                
                required_capabilities = st.multiselect(
                    "Required Capabilities",
                    options=[cap.value for cap in AgentCapability],
                    help="Specific capabilities required for this task"
                )
                
                custom_context = st.text_area(
                    "Additional Context",
                    placeholder="Any additional context or constraints...",
                    help="Extra information to help agents understand the task"
                )
            
            # Submit button
            submitted = st.form_submit_button(
                "🚀 Execute Task",
                type="primary",
                disabled=not task_input.strip()
            )
            
            if submitted and task_input.strip():
                task_config = {
                    "task": task_input.strip(),
                    "strategy": OrchestrationStrategy(strategy),
                    "priority": TaskPriority(priority),
                    "timeout": timeout,
                    "include_context": include_context,
                    "max_agents": max_agents,
                    "required_capabilities": [AgentCapability(cap) for cap in required_capabilities],
                    "custom_context": custom_context.strip() if custom_context.strip() else None
                }
                
                on_submit(task_config)


class PerformanceChart:
    """Component for displaying performance metrics and charts."""
    
    @staticmethod
    def render_success_rate_chart(task_history: List[Dict]):
        """Render success rate over time chart."""
        if not task_history:
            st.info("No task history available for chart")
            return
        
        # Prepare data
        df = pd.DataFrame([
            {
                'timestamp': task['timestamp'],
                'success': task['success'],
                'strategy': task['strategy']
            }
            for task in task_history
        ])
        
        # Create success rate chart
        fig = px.scatter(
            df,
            x='timestamp',
            y='success',
            color='strategy',
            title="Task Success Rate Over Time",
            labels={'success': 'Success (1=Success, 0=Failure)', 'timestamp': 'Time'},
            hover_data=['strategy']
        )
        
        fig.update_layout(
            yaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['Failure', 'Success']),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_execution_time_chart(task_history: List[Dict]):
        """Render execution time distribution chart."""
        if not task_history:
            st.info("No task history available for chart")
            return
        
        # Prepare data
        execution_times = [task['result'].execution_time_ms for task in task_history]
        
        # Create histogram
        fig = px.histogram(
            x=execution_times,
            nbins=20,
            title="Execution Time Distribution",
            labels={'x': 'Execution Time (ms)', 'y': 'Frequency'}
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_agent_utilization_chart(orchestration_results: List):
        """Render agent utilization chart."""
        if not orchestration_results:
            st.info("No orchestration results available for chart")
            return
        
        # Count agent usage
        agent_usage = {}
        for result in orchestration_results:
            for agent_id in result.agents_used:
                agent = AgentRegistry.get_agent(agent_id)
                agent_name = agent.name if agent else agent_id
                agent_usage[agent_name] = agent_usage.get(agent_name, 0) + 1
        
        if not agent_usage:
            st.info("No agent usage data available")
            return
        
        # Create bar chart
        fig = px.bar(
            x=list(agent_usage.keys()),
            y=list(agent_usage.values()),
            title="Agent Utilization Frequency",
            labels={'x': 'Agent', 'y': 'Times Used'}
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_strategy_performance_chart(task_history: List[Dict]):
        """Render strategy performance comparison."""
        if not task_history:
            st.info("No task history available for chart")
            return
        
        # Prepare data
        strategy_stats = {}
        for task in task_history:
            strategy = task['strategy']
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {'success_count': 0, 'total_count': 0, 'total_time': 0}
            
            strategy_stats[strategy]['total_count'] += 1
            strategy_stats[strategy]['total_time'] += task['result'].execution_time_ms
            
            if task['success']:
                strategy_stats[strategy]['success_count'] += 1
        
        # Calculate success rates and average times
        strategies = []
        success_rates = []
        avg_times = []
        
        for strategy, stats in strategy_stats.items():
            strategies.append(strategy)
            success_rates.append(stats['success_count'] / stats['total_count'] * 100)
            avg_times.append(stats['total_time'] / stats['total_count'])
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            specs=[[{"secondary_y": True}]],
            subplot_titles=["Strategy Performance Comparison"]
        )
        
        # Add success rate bars
        fig.add_trace(
            go.Bar(x=strategies, y=success_rates, name="Success Rate (%)", marker_color="green"),
            secondary_y=False,
        )
        
        # Add average execution time line
        fig.add_trace(
            go.Scatter(x=strategies, y=avg_times, mode="lines+markers", name="Avg Time (ms)", line_color="red"),
            secondary_y=True,
        )
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Success Rate (%)", secondary_y=False)
        fig.update_yaxes(title_text="Average Execution Time (ms)", secondary_y=True)
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


class SystemMetrics:
    """Component for displaying system metrics and health."""
    
    @staticmethod
    def render_metrics_grid(system_metrics: Dict[str, Any]):
        """Render system metrics in a grid layout."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Tasks",
                value=system_metrics.get('total_tasks', 0),
                delta=None
            )
        
        with col2:
            total = system_metrics.get('total_tasks', 0)
            successful = system_metrics.get('successful_tasks', 0)
            success_rate = (successful / total * 100) if total > 0 else 0
            
            st.metric(
                label="Success Rate",
                value=f"{success_rate:.1f}%",
                delta=None
            )
        
        with col3:
            agent_count = len(AgentRegistry.get_all_agents())
            st.metric(
                label="Active Agents",
                value=agent_count,
                delta=None
            )
        
        with col4:
            avg_time = system_metrics.get('avg_response_time', 0)
            st.metric(
                label="Avg Response Time",
                value=f"{avg_time:.0f}ms" if avg_time > 0 else "N/A",
                delta=None
            )
    
    @staticmethod
    def render_system_health():
        """Render system health indicators."""
        st.subheader("🏥 System Health")
        
        # System components status
        components = [
            ("Agent Registry", True, "All agents accessible"),
            ("Session Service", True, "Sessions active"),
            ("Memory Service", True, "Memory operations normal"),
            ("Orchestrator", True, "Task coordination active"),
        ]
        
        for component, status, description in components:
            col1, col2, col3 = st.columns([2, 1, 3])
            
            with col1:
                st.write(f"**{component}**")
            
            with col2:
                if status:
                    st.success("✅ OK")
                else:
                    st.error("❌ Error")
            
            with col3:
                st.write(description)


class TaskResultDisplay:
    """Component for displaying task execution results."""
    
    @staticmethod
    def render(task_record: Dict[str, Any]):
        """Render task result with expandable details."""
        result = task_record["result"]
        
        # Main result display
        if result.success:
            st.success("✅ Task completed successfully!")
            
            # Primary result
            st.markdown("### 📋 Results")
            st.markdown(
                f'<div style="background-color: #f8f9fa; border-left: 4px solid #007bff; '
                f'padding: 1rem; margin: 1rem 0; border-radius: 0 8px 8px 0;">'
                f'{result.primary_result}'
                f'</div>',
                unsafe_allow_html=True
            )
            
        else:
            st.error("❌ Task execution failed")
            if result.error:
                st.error(f"**Error:** {result.error}")
        
        # Execution metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Execution Time", f"{result.execution_time_ms:.0f}ms")
        
        with col2:
            st.metric("Agents Used", len(result.agents_used))
        
        with col3:
            st.metric("Strategy", result.strategy_used.value)
        
        with col4:
            if result.consensus_score is not None:
                st.metric("Consensus", f"{result.consensus_score:.1%}")
            else:
                st.metric("Consensus", "N/A")
        
        # Detailed results in expander
        with st.expander("🔍 Detailed Results"):
            if result.agent_results:
                st.write("**Individual Agent Contributions:**")
                
                for agent_id, agent_result in result.agent_results.items():
                    agent = AgentRegistry.get_agent(agent_id)
                    agent_name = agent.name if agent else agent_id
                    
                    with st.container():
                        st.write(f"**{agent_name}:**")
                        
                        # Show result with character limit
                        result_text = str(agent_result.result)
                        if len(result_text) > 300:
                            st.text(result_text[:300] + "...")
                            with st.expander(f"Full result from {agent_name}"):
                                st.text(result_text)
                        else:
                            st.text(result_text)
                        
                        # Agent execution metrics
                        if agent_result.execution_time_ms:
                            st.caption(f"Execution time: {agent_result.execution_time_ms:.0f}ms")
                        
                        st.divider()
            
            # Metadata
            if result.metadata:
                st.write("**Additional Metadata:**")
                st.json(result.metadata)


class ProgressIndicator:
    """Component for showing task execution progress."""
    
    @staticmethod
    def show_progress(message: str = "Processing...", estimated_time: Optional[int] = None):
        """Show progress indicator with optional estimated time."""
        if estimated_time:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(estimated_time):
                progress = (i + 1) / estimated_time
                progress_bar.progress(progress)
                
                remaining = estimated_time - i - 1
                status_text.text(f"{message} (ETA: {remaining}s)")
                
                time.sleep(1)
            
            progress_bar.empty()
            status_text.empty()
        else:
            with st.spinner(message):
                return True


# Utility functions for components

def format_timestamp(timestamp: datetime) -> str:
    """Format timestamp for display."""
    now = datetime.now()
    diff = now - timestamp
    
    if diff.days > 0:
        return timestamp.strftime("%Y-%m-%d %H:%M")
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    else:
        return "Just now"


def get_status_color(status: str) -> str:
    """Get color for status display."""
    status_colors = {
        'active': '#28a745',
        'inactive': '#6c757d',
        'running': '#007bff',
        'completed': '#28a745',
        'failed': '#dc3545',
        'error': '#dc3545',
        'success': '#28a745',
        'pending': '#ffc107',
    }
    
    return status_colors.get(status.lower(), '#6c757d')


def create_download_link(data: str, filename: str, link_text: str) -> str:
    """Create a download link for data."""
    import base64
    
    b64 = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'
    return href