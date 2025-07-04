"""
Streamlit Interface Demo

Demonstration script for the Multi-Agent Research Platform Streamlit interface.
Shows key features and capabilities with sample data and interactions.
"""

import streamlit as st
import time
from datetime import datetime, timedelta
import random

# Add project root to path for imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.streamlit.components import (
    AgentCard, TaskForm, PerformanceChart, SystemMetrics,
    TaskResultDisplay, ProgressIndicator
)


def create_demo_data():
    """Create sample data for demonstration."""
    # Sample task history
    sample_tasks = [
        {
            "timestamp": datetime.now() - timedelta(hours=2),
            "task": "What are the latest developments in quantum computing?",
            "strategy": "adaptive",
            "priority": "medium",
            "success": True,
            "result": type('Result', (), {
                'success': True,
                'primary_result': "Recent developments in quantum computing include breakthroughs in error correction, the development of more stable qubits, and advances in quantum algorithms for optimization problems. IBM and Google have made significant progress in quantum supremacy demonstrations.",
                'execution_time_ms': 2300,
                'agents_used': ['research_agent_1', 'analyst_agent_1'],
                'strategy_used': type('Strategy', (), {'value': 'adaptive'})(),
                'consensus_score': 0.85,
                'agent_results': {},
                'error': None,
                'metadata': {}
            })()
        },
        {
            "timestamp": datetime.now() - timedelta(hours=1, minutes=30),
            "task": "Analyze the impact of AI on job markets",
            "strategy": "consensus",
            "priority": "high",
            "success": True,
            "result": type('Result', (), {
                'success': True,
                'primary_result': "AI is transforming job markets by automating routine tasks while creating new opportunities in AI development, data science, and human-AI collaboration roles. The impact varies by industry, with manufacturing and customer service seeing more automation, while creative and strategic roles remain largely human-driven.",
                'execution_time_ms': 3100,
                'agents_used': ['research_agent_1', 'analyst_agent_1', 'economist_agent_1'],
                'strategy_used': type('Strategy', (), {'value': 'consensus'})(),
                'consensus_score': 0.92,
                'agent_results': {},
                'error': None,
                'metadata': {}
            })()
        },
        {
            "timestamp": datetime.now() - timedelta(minutes=45),
            "task": "Compare renewable energy technologies",
            "strategy": "parallel_all",
            "priority": "medium",
            "success": True,
            "result": type('Result', (), {
                'success': True,
                'primary_result': "Solar photovoltaic technology leads in cost-effectiveness and scalability, wind power offers excellent capacity factors in suitable locations, and emerging technologies like advanced battery storage and green hydrogen show promise for grid stability and energy storage.",
                'execution_time_ms': 2800,
                'agents_used': ['research_agent_1', 'technical_agent_1'],
                'strategy_used': type('Strategy', (), {'value': 'parallel_all'})(),
                'consensus_score': 0.78,
                'agent_results': {},
                'error': None,
                'metadata': {}
            })()
        },
        {
            "timestamp": datetime.now() - timedelta(minutes=20),
            "task": "Explain blockchain technology applications",
            "strategy": "single_best",
            "priority": "low",
            "success": False,
            "result": type('Result', (), {
                'success': False,
                'primary_result': None,
                'execution_time_ms': 1200,
                'agents_used': ['research_agent_1'],
                'strategy_used': type('Strategy', (), {'value': 'single_best'})(),
                'consensus_score': None,
                'agent_results': {},
                'error': "Agent timeout - task complexity exceeded available time",
                'metadata': {}
            })()
        }
    ]
    
    return sample_tasks


def run_demo():
    """Run the Streamlit demo interface."""
    
    # Page configuration
    st.set_page_config(
        page_title="Multi-Agent Platform Demo",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for demo
    st.markdown("""
    <style>
        .demo-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
            text-align: center;
        }
        .feature-card {
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .demo-metric {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            padding: 1rem;
            border-radius: 8px;
            color: white;
            text-align: center;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Demo header
    st.markdown("""
    <div class="demo-header">
        <h1>🚀 Multi-Agent Research Platform</h1>
        <h3>Streamlit Interface Demo</h3>
        <p>Experience the power of AI-driven collaborative research</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar demo navigation
    with st.sidebar:
        st.header("🎮 Demo Navigation")
        
        demo_section = st.selectbox(
            "Choose Demo Section",
            [
                "🏠 Overview",
                "🤖 Agent Showcase", 
                "📝 Task Execution",
                "📊 Analytics Demo",
                "🎯 Interactive Features",
                "💡 Use Cases"
            ]
        )
        
        st.markdown("---")
        
        # Demo controls
        st.subheader("Demo Controls")
        
        if st.button("🔄 Reset Demo Data"):
            # Reset session state
            for key in list(st.session_state.keys()):
                if key.startswith('demo_'):
                    del st.session_state[key]
            st.success("Demo data reset!")
            st.rerun()
        
        if st.button("📊 Generate Sample Data"):
            st.session_state.demo_task_history = create_demo_data()
            st.success("Sample data generated!")
            st.rerun()
        
        # Demo info
        with st.expander("ℹ️ Demo Information"):
            st.write("""
            **This is a demonstration of the Streamlit interface.**
            
            Features showcased:
            - Agent management interface
            - Task execution workflow
            - Real-time analytics
            - Interactive components
            - Modern UI design
            
            **Note:** This demo uses simulated data and responses.
            """)
    
    # Main content based on selected section
    if demo_section == "🏠 Overview":
        show_overview_demo()
    elif demo_section == "🤖 Agent Showcase":
        show_agent_demo()
    elif demo_section == "📝 Task Execution":
        show_task_demo()
    elif demo_section == "📊 Analytics Demo":
        show_analytics_demo()
    elif demo_section == "🎯 Interactive Features":
        show_interactive_demo()
    elif demo_section == "💡 Use Cases":
        show_use_cases_demo()


def show_overview_demo():
    """Show overview demonstration."""
    st.header("🌟 Platform Overview")
    
    # Key features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🤖 Intelligent Agents</h4>
            <p>Specialized AI agents for research, analysis, and content creation</p>
            <ul>
                <li>LLM-powered reasoning</li>
                <li>Domain expertise</li>
                <li>Collaborative workflows</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>📈 Smart Orchestration</h4>
            <p>Advanced strategies for multi-agent collaboration</p>
            <ul>
                <li>Adaptive task allocation</li>
                <li>Consensus building</li>
                <li>Performance optimization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Rich Analytics</h4>
            <p>Comprehensive insights and visualizations</p>
            <ul>
                <li>Real-time monitoring</li>
                <li>Performance metrics</li>
                <li>Historical analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Demo metrics
    st.subheader("📈 Live Demo Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="demo-metric">
            <h3>42</h3>
            <p>Tasks Completed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="demo-metric">
            <h3>94.2%</h3>
            <p>Success Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="demo-metric">
            <h3>8</h3>
            <p>Active Agents</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="demo-metric">
            <h3>1.8s</h3>
            <p>Avg Response</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Platform benefits
    st.subheader("🎯 Why Choose Our Platform?")
    
    benefits = [
        ("🚀 **Accelerated Research**", "Complete research tasks 10x faster with AI collaboration"),
        ("🔍 **Comprehensive Analysis**", "Multi-perspective analysis from specialized agents"),
        ("📝 **Quality Results**", "Consistent, high-quality outputs with consensus validation"),
        ("⚡ **Real-time Insights**", "Immediate feedback and progress tracking"),
        ("🔧 **Easy to Use**", "Intuitive interface requiring no technical expertise"),
        ("📈 **Scalable Solution**", "Handles simple queries to complex research projects")
    ]
    
    for benefit_title, benefit_desc in benefits:
        st.markdown(f"**{benefit_title}:** {benefit_desc}")


def show_agent_demo():
    """Show agent showcase demonstration."""
    st.header("🤖 Agent Showcase")
    
    st.write("Meet our specialized AI agents, each designed for specific research and analysis tasks.")
    
    # Sample agents data
    demo_agents = [
        {
            "name": "Research Specialist",
            "type": "LLM Agent",
            "description": "Expert in comprehensive research across multiple domains",
            "capabilities": ["Web Search", "Data Analysis", "Report Generation"],
            "tasks_completed": 127,
            "success_rate": 96.3,
            "specialization": "Multi-domain research and fact-finding"
        },
        {
            "name": "Data Analyst",
            "type": "Custom Agent",
            "description": "Specialized in statistical analysis and data interpretation",
            "capabilities": ["Statistical Analysis", "Data Visualization", "Trend Identification"],
            "tasks_completed": 89,
            "success_rate": 94.7,
            "specialization": "Quantitative analysis and insights"
        },
        {
            "name": "Content Creator", 
            "type": "LLM Agent",
            "description": "Expert in content creation and communication",
            "capabilities": ["Writing", "Summarization", "Content Strategy"],
            "tasks_completed": 156,
            "success_rate": 98.1,
            "specialization": "High-quality content production"
        },
        {
            "name": "Technical Expert",
            "type": "Custom Agent", 
            "description": "Specialized in technical documentation and analysis",
            "capabilities": ["Technical Writing", "Code Review", "Architecture Analysis"],
            "tasks_completed": 73,
            "success_rate": 92.8,
            "specialization": "Technical domains and engineering"
        }
    ]
    
    # Agent grid display
    col1, col2 = st.columns(2)
    
    for i, agent in enumerate(demo_agents):
        with col1 if i % 2 == 0 else col2:
            with st.container():
                st.markdown(f"""
                <div style="border: 1px solid #e1e5e9; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; background: white;">
                    <h4>🤖 {agent['name']}</h4>
                    <p><strong>Type:</strong> {agent['type']}</p>
                    <p><strong>Specialization:</strong> {agent['specialization']}</p>
                    <p><strong>Description:</strong> {agent['description']}</p>
                    
                    <div style="margin: 1rem 0;">
                        <strong>Capabilities:</strong><br>
                        {' • '.join(agent['capabilities'])}
                    </div>
                    
                    <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                        <div style="text-align: center;">
                            <div style="font-size: 1.5em; font-weight: bold; color: #667eea;">{agent['tasks_completed']}</div>
                            <div style="font-size: 0.8em; color: #666;">Tasks Completed</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 1.5em; font-weight: bold; color: #28a745;">{agent['success_rate']}%</div>
                            <div style="font-size: 0.8em; color: #666;">Success Rate</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Agent capabilities matrix
    st.subheader("🔧 Agent Capabilities Matrix")
    
    capabilities_data = {
        "Agent": [agent["name"] for agent in demo_agents],
        "Research": ["✅", "⚠️", "✅", "✅"],
        "Analysis": ["✅", "✅", "⚠️", "✅"],
        "Writing": ["✅", "⚠️", "✅", "✅"],
        "Technical": ["⚠️", "✅", "⚠️", "✅"],
        "Creative": ["⚠️", "❌", "✅", "⚠️"]
    }
    
    import pandas as pd
    df_capabilities = pd.DataFrame(capabilities_data)
    st.dataframe(df_capabilities, use_container_width=True)
    
    st.caption("✅ Excellent • ⚠️ Good • ❌ Limited")


def show_task_demo():
    """Show task execution demonstration."""
    st.header("📝 Task Execution Demo")
    
    # Interactive task form
    st.subheader("🎯 Try the Task Interface")
    
    # Sample task suggestions
    sample_tasks = [
        "What are the environmental benefits of electric vehicles?",
        "Compare different programming languages for web development",
        "Analyze the impact of remote work on productivity",
        "Explain the latest developments in renewable energy",
        "Research the history and evolution of artificial intelligence"
    ]
    
    selected_sample = st.selectbox(
        "Choose a sample task or enter your own:",
        [""] + sample_tasks
    )
    
    task_input = st.text_area(
        "Research Task",
        value=selected_sample,
        placeholder="Enter your research question or task...",
        height=100
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        strategy = st.selectbox(
            "Orchestration Strategy",
            ["adaptive", "consensus", "parallel_all", "single_best", "competitive"],
            help="How should agents collaborate on this task?"
        )
    
    with col2:
        priority = st.selectbox(
            "Priority Level",
            ["low", "medium", "high", "urgent"],
            index=1
        )
    
    # Simulate task execution
    if st.button("🚀 Execute Task (Demo)", type="primary", disabled=not task_input.strip()):
        with st.spinner("🤖 Agents are working on your task..."):
            # Simulate progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            stages = [
                "Analyzing task requirements...",
                "Selecting optimal agents...",
                "Distributing work across agents...",
                "Gathering research data...",
                "Processing and analyzing results...",
                "Building consensus...",
                "Finalizing response..."
            ]
            
            for i, stage in enumerate(stages):
                progress = (i + 1) / len(stages)
                progress_bar.progress(progress)
                status_text.text(stage)
                time.sleep(0.5)
            
            progress_bar.empty()
            status_text.empty()
        
        # Show demo result
        st.success("✅ Task completed successfully!")
        
        demo_result = f"""
        **Research Summary for:** {task_input}

        Based on comprehensive analysis using the **{strategy}** orchestration strategy, our agents have provided the following insights:

        **Key Findings:**
        • Thorough research was conducted across multiple reliable sources
        • Multiple perspectives were considered and analyzed
        • Current industry trends and expert opinions were incorporated
        • Data-driven insights were synthesized into actionable information

        **Methodology:**
        • Agent collaboration using {strategy} strategy
        • Priority level: {priority}
        • Cross-validation of findings across multiple agents
        • Quality assurance through consensus validation

        **Conclusion:**
        This analysis provides a comprehensive overview of your research topic with evidence-based insights and practical implications.

        *Note: This is a demo response showcasing the interface capabilities.*
        """
        
        st.markdown(demo_result)
        
        # Demo metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Execution Time", "2.3s")
        with col2:
            st.metric("Agents Used", "3")
        with col3:
            st.metric("Consensus Score", "94%")
        with col4:
            st.metric("Quality Score", "A+")


def show_analytics_demo():
    """Show analytics demonstration."""
    st.header("📊 Analytics Demo")
    
    # Ensure we have demo data
    if 'demo_task_history' not in st.session_state:
        st.session_state.demo_task_history = create_demo_data()
    
    task_history = st.session_state.demo_task_history
    
    # Performance charts
    st.subheader("📈 Performance Analytics")
    
    # Success rate over time
    PerformanceChart.render_success_rate_chart(task_history)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Execution time distribution
        PerformanceChart.render_execution_time_chart(task_history)
    
    with col2:
        # Strategy performance
        PerformanceChart.render_strategy_performance_chart(task_history)
    
    # System metrics
    st.subheader("🏥 System Health")
    
    demo_metrics = {
        'total_tasks': len(task_history),
        'successful_tasks': sum(1 for task in task_history if task['success']),
        'avg_response_time': sum(task['result'].execution_time_ms for task in task_history) / len(task_history)
    }
    
    SystemMetrics.render_metrics_grid(demo_metrics)


def show_interactive_demo():
    """Show interactive features demonstration."""
    st.header("🎯 Interactive Features Demo")
    
    # Real-time metrics simulation
    st.subheader("⚡ Real-time Monitoring")
    
    if st.button("🔄 Simulate Real-time Update"):
        # Create placeholder for metrics
        metrics_placeholder = st.empty()
        
        # Simulate updating metrics
        for i in range(10):
            fake_metrics = {
                "Active Tasks": random.randint(0, 5),
                "Response Time": f"{random.randint(800, 2500)}ms",
                "Success Rate": f"{random.uniform(85, 99):.1f}%",
                "Agent Load": f"{random.randint(20, 80)}%"
            }
            
            with metrics_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                cols = [col1, col2, col3, col4]
                
                for j, (metric, value) in enumerate(fake_metrics.items()):
                    with cols[j]:
                        st.metric(metric, value)
            
            time.sleep(0.5)
    
    # Interactive agent controls
    st.subheader("🤖 Agent Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Create New Agent**")
        agent_name = st.text_input("Agent Name", placeholder="e.g., Finance Expert")
        agent_type = st.selectbox("Agent Type", ["Research Agent", "Analysis Agent", "Content Agent"])
        
        if st.button("➕ Create Agent (Demo)"):
            if agent_name:
                st.success(f"✅ Created {agent_name} ({agent_type})")
            else:
                st.warning("Please enter an agent name")
    
    with col2:
        st.write("**Agent Actions**")
        
        if st.button("📊 View Agent Details"):
            st.info("📋 Agent details would be displayed in a modal or expanded view")
        
        if st.button("⏸️ Pause All Agents"):
            st.warning("⏸️ All agents paused (demo)")
        
        if st.button("▶️ Resume All Agents"):
            st.success("▶️ All agents resumed (demo)")


def show_use_cases_demo():
    """Show use cases demonstration."""
    st.header("💡 Use Cases & Applications")
    
    # Use case categories
    use_cases = {
        "🔬 Research & Academia": [
            "Literature reviews and meta-analyses",
            "Market research and competitive analysis", 
            "Scientific paper summaries",
            "Grant proposal research",
            "Trend analysis and forecasting"
        ],
        "💼 Business & Strategy": [
            "Market opportunity assessment",
            "Competitive intelligence gathering",
            "Business plan development",
            "Risk analysis and mitigation",
            "Industry trend monitoring"
        ],
        "📝 Content & Communication": [
            "Technical documentation creation",
            "Multi-language content adaptation",
            "Brand message consistency",
            "Content strategy development",
            "Social media campaign planning"
        ],
        "🔧 Technical & Engineering": [
            "Architecture decision analysis",
            "Technology stack evaluation",
            "Code review and optimization",
            "System design validation",
            "Performance troubleshooting"
        ]
    }
    
    # Display use cases
    for category, cases in use_cases.items():
        with st.expander(category):
            for case in cases:
                st.write(f"• {case}")
    
    # Success stories
    st.subheader("🌟 Success Stories")
    
    stories = [
        {
            "title": "Academic Research Acceleration",
            "description": "University researchers reduced literature review time from weeks to hours",
            "metric": "90% time reduction",
            "industry": "Education"
        },
        {
            "title": "Market Analysis Automation",
            "description": "Consulting firm automated competitive analysis for client reports",
            "metric": "300% productivity increase",
            "industry": "Consulting"
        },
        {
            "title": "Technical Documentation",
            "description": "Software company streamlined API documentation across multiple products",
            "metric": "70% consistency improvement",
            "industry": "Technology"
        }
    ]
    
    for story in stories:
        st.markdown(f"""
        <div style="border: 1px solid #e1e5e9; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; background: white;">
            <h4>✨ {story['title']}</h4>
            <p>{story['description']}</p>
            <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                <div><strong>Impact:</strong> {story['metric']}</div>
                <div><strong>Industry:</strong> {story['industry']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    run_demo()