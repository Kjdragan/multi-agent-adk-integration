"""
Multi-Agent Research Platform - Streamlit Interface

Production-ready Streamlit interface for the multi-agent research platform.
Provides a user-friendly interface for research tasks, agent management,
and result visualization.
"""

import streamlit as st
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Add project root to path for imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents import AgentRegistry, AgentOrchestrator, AgentFactory, AgentSuite, OrchestrationStrategy, TaskPriority
from src.agents.llm_agent import LLMRole
from src.agents.custom_agent import CustomAgentType
from src.platform_logging import RunLogger, LogLevel
from src.services import SessionService, MemoryService


# Page configuration
st.set_page_config(
    page_title="Multi-Agent Research Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        padding: 1rem 0;
        border-bottom: 2px solid #f0f2f6;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    .agent-card {
        border: 1px solid #e1e5e9;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        background: white;
    }
    .status-active { color: #28a745; font-weight: bold; }
    .status-inactive { color: #6c757d; font-weight: bold; }
    .status-running { color: #007bff; font-weight: bold; }
    .status-completed { color: #28a745; font-weight: bold; }
    .status-error { color: #dc3545; font-weight: bold; }
    .task-result {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .sidebar-section {
        margin-bottom: 2rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


class StreamlitApp:
    """Main Streamlit application for the multi-agent research platform."""
    
    def __init__(self):
        self.logger = None
        self.session_service = None
        self.memory_service = None
        self.orchestrator = None
        self.factory = None
        
        # Initialize session state
        self._initialize_session_state()
        
        # Initialize services
        self._initialize_services()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.agents_created = False
            st.session_state.task_history = []
            st.session_state.current_agents = []
            st.session_state.orchestration_results = []
            st.session_state.system_metrics = {
                'total_tasks': 0,
                'successful_tasks': 0,
                'active_agents': 0,
                'avg_response_time': 0
            }
    
    def _initialize_services(self):
        """Initialize core services."""
        try:
            if not st.session_state.initialized:
                # Initialize logger
                self.logger = RunLogger(
                    level=LogLevel.INFO,
                    console_output=True,
                    file_output=False  # Disable file logging in Streamlit
                )
                
                # Initialize services
                self.session_service = SessionService()
                self.memory_service = MemoryService()
                
                # Initialize orchestrator and factory
                self.orchestrator = AgentOrchestrator(
                    logger=self.logger,
                    session_service=self.session_service
                )
                
                self.factory = AgentFactory(
                    logger=self.logger,
                    session_service=self.session_service
                )
                
                st.session_state.initialized = True
                
        except Exception as e:
            st.error(f"Failed to initialize services: {e}")
    
    def run(self):
        """Main application entry point."""
        self._render_header()
        self._render_sidebar()
        self._render_main_content()
    
    def _render_header(self):
        """Render the main header."""
        st.markdown('<div class="main-header">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.title("🤖 Multi-Agent Research Platform")
            st.caption("Powered by Google Agent Development Kit (ADK)")
        
        with col2:
            # System status indicator
            status = "🟢 Online" if st.session_state.initialized else "🔴 Initializing"
            st.metric("System Status", status)
        
        with col3:
            # Quick stats
            agent_count = len(AgentRegistry.get_all_agents())
            st.metric("Active Agents", agent_count)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_sidebar(self):
        """Render the sidebar with navigation and controls."""
        with st.sidebar:
            st.header("🚀 Quick Start")
            
            # Agent creation section
            with st.expander("Create Agents", expanded=not st.session_state.agents_created):
                self._render_agent_creation()
            
            # Task execution section  
            with st.expander("Run Research Task", expanded=True):
                self._render_task_execution()
            
            # System controls
            st.header("⚙️ System Controls")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Refresh", help="Refresh agent status"):
                    st.rerun()
            
            with col2:
                if st.button("🧹 Clear History", help="Clear task history"):
                    st.session_state.task_history = []
                    st.session_state.orchestration_results = []
                    st.success("History cleared!")
    
    def _render_agent_creation(self):
        """Render agent creation interface."""
        st.subheader("Create Agent Team")
        
        # Predefined team types
        team_options = {
            "Research Team": {
                "suite": AgentSuite.RESEARCH_TEAM,
                "description": "Complete research team with researcher, analyst, and writer"
            },
            "Analysis Team": {
                "suite": AgentSuite.ANALYSIS_TEAM, 
                "description": "Data analysis focused team with analysts and processors"
            },
            "Content Team": {
                "suite": AgentSuite.CONTENT_TEAM,
                "description": "Content creation team with writers and editors"
            },
            "Custom Setup": {
                "suite": None,
                "description": "Create individual agents with custom configurations"
            }
        }
        
        selected_team = st.selectbox(
            "Choose Team Type",
            options=list(team_options.keys()),
            help="Select a predefined team or create custom agents"
        )
        
        team_config = team_options[selected_team]
        st.info(team_config["description"])
        
        if selected_team == "Custom Setup":
            self._render_custom_agent_creation()
        else:
            if st.button(f"Create {selected_team}", type="primary"):
                self._create_agent_team(team_config["suite"])
    
    def _render_custom_agent_creation(self):
        """Render custom agent creation interface."""
        st.write("**Custom Agent Configuration**")
        
        agent_type = st.selectbox(
            "Agent Type",
            ["LLM Agent", "Custom Agent"],
            help="Choose the type of agent to create"
        )
        
        agent_name = st.text_input("Agent Name", placeholder="e.g., Research Assistant")
        
        if agent_type == "LLM Agent":
            role = st.selectbox(
                "LLM Role",
                [role.value for role in LLMRole],
                help="Specialized role for the LLM agent"
            )
        else:
            custom_type = st.selectbox(
                "Custom Agent Type", 
                [agent_type.value for agent_type in CustomAgentType],
                help="Specialized type for custom agent"
            )
            
            domain = st.text_input("Domain", placeholder="e.g., Machine Learning, Finance")
        
        if st.button("Create Custom Agent", disabled=not agent_name):
            try:
                if agent_type == "LLM Agent":
                    agent = self.factory.create_llm_agent(
                        role=LLMRole(role),
                        name=agent_name
                    )
                else:
                    agent = self.factory.create_custom_agent(
                        agent_type=CustomAgentType(custom_type),
                        domain=domain,
                        name=agent_name
                    )
                
                # Activate agent asynchronously
                asyncio.run(agent.activate())
                
                st.success(f"✅ Created {agent_name} successfully!")
                st.session_state.agents_created = True
                st.rerun()
                
            except Exception as e:
                st.error(f"Failed to create agent: {e}")
    
    def _create_agent_team(self, suite_type: AgentSuite):
        """Create a predefined agent team."""
        try:
            with st.spinner("Creating agent team..."):
                agents = self.factory.create_agent_suite(suite_type)
                
                # Activate all agents
                for agent in agents:
                    asyncio.run(agent.activate())
                
                st.success(f"✅ Created {len(agents)} agents successfully!")
                st.session_state.agents_created = True
                st.rerun()
                
        except Exception as e:
            st.error(f"Failed to create team: {e}")
    
    def _render_task_execution(self):
        """Render task execution interface."""
        st.subheader("Research Task")
        
        # Check if agents exist
        agents = AgentRegistry.get_all_agents()
        if not agents:
            st.warning("⚠️ Create agents first before running tasks")
            return
        
        # Task input
        task_input = st.text_area(
            "Research Question or Task",
            placeholder="e.g., What are the latest developments in quantum computing?",
            help="Describe the research task you want the agents to perform"
        )
        
        # Orchestration settings
        col1, col2 = st.columns(2)
        
        with col1:
            strategy = st.selectbox(
                "Strategy",
                [s.value for s in OrchestrationStrategy],
                index=0,  # Default to first strategy
                help="How should agents collaborate on this task?"
            )
        
        with col2:
            priority = st.selectbox(
                "Priority",
                [p.value for p in TaskPriority],
                index=1,  # Default to medium
                help="Task priority level"
            )
        
        # Advanced options
        with st.expander("Advanced Options"):
            timeout = st.slider("Timeout (seconds)", 30, 300, 120)
            include_context = st.checkbox("Include conversation history", value=True)
        
        # Execute button
        if st.button("🚀 Execute Task", type="primary", disabled=not task_input.strip()):
            self._execute_research_task(
                task=task_input.strip(),
                strategy=OrchestrationStrategy(strategy),
                priority=TaskPriority(priority),
                timeout=timeout,
                include_context=include_context
            )
    
    def _execute_research_task(self, task: str, strategy: OrchestrationStrategy, 
                              priority: TaskPriority, timeout: int, include_context: bool):
        """Execute a research task using agent orchestration."""
        try:
            with st.spinner("🤖 Agents are working on your task..."):
                # Prepare context
                context = {}
                if include_context and st.session_state.task_history:
                    context["previous_tasks"] = st.session_state.task_history[-3:]  # Last 3 tasks
                
                # Execute task
                result = asyncio.run(
                    self.orchestrator.orchestrate_task(
                        task=task,
                        strategy=strategy,
                        context=context,
                        priority=priority,
                        timeout_seconds=timeout
                    )
                )
                
                # Store results
                task_record = {
                    "timestamp": datetime.now(),
                    "task": task,
                    "strategy": strategy.value,
                    "priority": priority.value,
                    "result": result,
                    "success": result.success
                }
                
                st.session_state.task_history.append(task_record)
                st.session_state.orchestration_results.append(result)
                
                # Update metrics
                st.session_state.system_metrics['total_tasks'] += 1
                if result.success:
                    st.session_state.system_metrics['successful_tasks'] += 1
                
                # Display results
                self._display_task_result(task_record)
                
        except Exception as e:
            st.error(f"Task execution failed: {e}")
    
    def _display_task_result(self, task_record: Dict):
        """Display task execution results."""
        result = task_record["result"]
        
        if result.success:
            st.success("✅ Task completed successfully!")
            
            # Main result
            st.markdown("### 📋 Results")
            st.markdown('<div class="task-result">', unsafe_allow_html=True)
            st.write(result.primary_result)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Additional details
            with st.expander("📊 Execution Details"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Execution Time", f"{result.execution_time_ms:.0f}ms")
                
                with col2:
                    st.metric("Agents Used", len(result.agents_used))
                
                with col3:
                    if result.consensus_score is not None:
                        st.metric("Consensus Score", f"{result.consensus_score:.1%}")
                
                # Agent contributions
                if result.agent_results:
                    st.write("**Agent Contributions:**")
                    for agent_id, agent_result in result.agent_results.items():
                        agent = AgentRegistry.get_agent(agent_id)
                        agent_name = agent.name if agent else agent_id
                        
                        with st.container():
                            st.write(f"**{agent_name}:**")
                            st.text(str(agent_result.result)[:200] + "..." if len(str(agent_result.result)) > 200 else str(agent_result.result))
        else:
            st.error("❌ Task failed")
            if result.error:
                st.error(f"Error: {result.error}")
    
    def _render_main_content(self):
        """Render the main content area."""
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🤖 Agents", "📝 Task History", "📈 Analytics"])
        
        with tab1:
            self._render_dashboard()
        
        with tab2:
            self._render_agents_view()
        
        with tab3:
            self._render_task_history()
        
        with tab4:
            self._render_analytics()
    
    def _render_dashboard(self):
        """Render the main dashboard."""
        st.header("System Overview")
        
        # System metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Tasks",
                st.session_state.system_metrics['total_tasks']
            )
        
        with col2:
            total = st.session_state.system_metrics['total_tasks']
            successful = st.session_state.system_metrics['successful_tasks']
            success_rate = (successful / total * 100) if total > 0 else 0
            st.metric(
                "Success Rate",
                f"{success_rate:.1f}%"
            )
        
        with col3:
            agent_count = len(AgentRegistry.get_all_agents())
            st.metric("Active Agents", agent_count)
        
        with col4:
            # Calculate average response time from recent results
            recent_results = st.session_state.orchestration_results[-10:]
            if recent_results:
                avg_time = sum(r.execution_time_ms for r in recent_results) / len(recent_results)
                st.metric("Avg Response Time", f"{avg_time:.0f}ms")
            else:
                st.metric("Avg Response Time", "N/A")
        
        # Recent activity
        if st.session_state.task_history:
            st.subheader("Recent Activity")
            
            recent_tasks = st.session_state.task_history[-5:]  # Last 5 tasks
            for task in reversed(recent_tasks):
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.write(f"**{task['task'][:60]}...**" if len(task['task']) > 60 else f"**{task['task']}**")
                    
                    with col2:
                        status_class = "status-completed" if task['success'] else "status-error"
                        status_text = "✅ Success" if task['success'] else "❌ Failed"
                        st.markdown(f'<span class="{status_class}">{status_text}</span>', unsafe_allow_html=True)
                    
                    with col3:
                        st.write(task['timestamp'].strftime("%H:%M:%S"))
                    
                    st.divider()
        else:
            st.info("🚀 Run your first research task to see activity here!")
    
    def _render_agents_view(self):
        """Render the agents management view."""
        st.header("Agent Management")
        
        agents = AgentRegistry.get_all_agents()
        
        if not agents:
            st.info("No agents created yet. Use the sidebar to create your first agents!")
            return
        
        # Agent statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            active_count = sum(1 for agent in agents if agent.is_active)
            st.metric("Active Agents", f"{active_count}/{len(agents)}")
        
        with col2:
            total_tasks = sum(agent.total_tasks_completed for agent in agents)
            st.metric("Total Tasks Completed", total_tasks)
        
        with col3:
            agent_types = len(set(agent.agent_type for agent in agents))
            st.metric("Agent Types", agent_types)
        
        # Agent list
        st.subheader("Agent Details")
        
        for agent in agents:
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                
                with col1:
                    st.write(f"**{agent.name}**")
                    st.caption(f"{agent.agent_type.value.replace('_', ' ').title()}")
                
                with col2:
                    status_class = "status-active" if agent.is_active else "status-inactive"
                    status_text = "🟢 Active" if agent.is_active else "🔴 Inactive"
                    st.markdown(f'<span class="{status_class}">{status_text}</span>', unsafe_allow_html=True)
                
                with col3:
                    st.metric("Tasks", agent.total_tasks_completed)
                
                with col4:
                    # Agent capabilities
                    capabilities = agent.get_capabilities()
                    if capabilities:
                        cap_text = f"{len(capabilities)} capabilities"
                        st.write(cap_text)
                    else:
                        st.write("No capabilities")
                
                # Agent details in expander
                with st.expander(f"Details for {agent.name}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Agent ID:**", agent.agent_id)
                        st.write("**Type:**", agent.agent_type.value)
                        st.write("**Status:**", "Active" if agent.is_active else "Inactive")
                    
                    with col2:
                        st.write("**Tasks Completed:**", agent.total_tasks_completed)
                        if agent.last_task_time:
                            last_task = datetime.fromtimestamp(agent.last_task_time)
                            st.write("**Last Task:**", last_task.strftime("%Y-%m-%d %H:%M:%S"))
                        
                        # Performance metrics if available
                        if hasattr(agent, 'get_performance_metrics'):
                            metrics = agent.get_performance_metrics()
                            if metrics:
                                st.write("**Success Rate:**", f"{metrics.get('success_rate_percent', 0):.1f}%")
                    
                    # Capabilities
                    if capabilities:
                        st.write("**Capabilities:**")
                        cap_names = [cap.value.replace('_', ' ').title() for cap in capabilities]
                        st.write(", ".join(cap_names))
                
                st.divider()
    
    def _render_task_history(self):
        """Render task history view."""
        st.header("Task History")
        
        if not st.session_state.task_history:
            st.info("No tasks executed yet. Run a research task to see history here!")
            return
        
        # History controls
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write(f"**{len(st.session_state.task_history)} tasks executed**")
        
        with col2:
            if st.button("📥 Export History"):
                # Create downloadable JSON
                history_data = []
                for task in st.session_state.task_history:
                    task_data = task.copy()
                    task_data['timestamp'] = task_data['timestamp'].isoformat()
                    # Simplify result for export
                    if 'result' in task_data:
                        result = task_data['result']
                        task_data['result'] = {
                            'success': result.success,
                            'primary_result': result.primary_result,
                            'execution_time_ms': result.execution_time_ms,
                            'agents_used': result.agents_used
                        }
                    history_data.append(task_data)
                
                st.download_button(
                    "Download JSON",
                    data=json.dumps(history_data, indent=2),
                    file_name=f"task_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        # Task list
        for i, task in enumerate(reversed(st.session_state.task_history)):
            with st.expander(f"Task {len(st.session_state.task_history) - i}: {task['task'][:50]}..."):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write("**Task:**", task['task'])
                    st.write("**Strategy:**", task['strategy'])
                    st.write("**Priority:**", task['priority'])
                    
                    result = task['result']
                    if result.success:
                        st.success("✅ Completed successfully")
                        st.write("**Result:**")
                        st.text(result.primary_result)
                    else:
                        st.error("❌ Failed")
                        if result.error:
                            st.write("**Error:**", result.error)
                
                with col2:
                    st.write("**Timestamp:**", task['timestamp'].strftime("%Y-%m-%d %H:%M:%S"))
                    st.write("**Execution Time:**", f"{result.execution_time_ms:.0f}ms")
                    st.write("**Agents Used:**", len(result.agents_used))
                    
                    if result.consensus_score is not None:
                        st.write("**Consensus:**", f"{result.consensus_score:.1%}")
    
    def _render_analytics(self):
        """Render analytics and charts."""
        st.header("Analytics & Insights")
        
        if not st.session_state.task_history:
            st.info("Execute some tasks to see analytics here!")
            return
        
        # Task success rate over time
        st.subheader("📈 Task Success Rate")
        
        task_data = []
        for task in st.session_state.task_history:
            task_data.append({
                'timestamp': task['timestamp'],
                'success': task['success'],
                'execution_time': task['result'].execution_time_ms,
                'strategy': task['strategy']
            })
        
        df = pd.DataFrame(task_data)
        
        # Success rate chart
        fig = px.scatter(
            df, 
            x='timestamp', 
            y='success',
            color='strategy',
            title="Task Success Over Time",
            labels={'success': 'Success (True/False)', 'timestamp': 'Time'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Execution time analysis
        st.subheader("⏱️ Execution Time Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Execution time histogram
            fig = px.histogram(
                df,
                x='execution_time',
                nbins=20,
                title="Execution Time Distribution",
                labels={'execution_time': 'Execution Time (ms)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Strategy performance
            strategy_stats = df.groupby('strategy').agg({
                'success': 'mean',
                'execution_time': 'mean'
            }).round(3)
            
            strategy_stats.columns = ['Success Rate', 'Avg Execution Time (ms)']
            st.write("**Strategy Performance:**")
            st.dataframe(strategy_stats)
        
        # Agent utilization
        if st.session_state.orchestration_results:
            st.subheader("🤖 Agent Utilization")
            
            agent_usage = {}
            for result in st.session_state.orchestration_results:
                for agent_id in result.agents_used:
                    agent = AgentRegistry.get_agent(agent_id)
                    agent_name = agent.name if agent else agent_id
                    agent_usage[agent_name] = agent_usage.get(agent_name, 0) + 1
            
            if agent_usage:
                fig = px.bar(
                    x=list(agent_usage.keys()),
                    y=list(agent_usage.values()),
                    title="Agent Usage Frequency",
                    labels={'x': 'Agent', 'y': 'Times Used'}
                )
                st.plotly_chart(fig, use_container_width=True)


def main():
    """Main entry point for the Streamlit app."""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()