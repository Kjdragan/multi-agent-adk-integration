"""
Multi-Agent System Demo

Demonstrates the capabilities of the multi-agent research platform including
LLM agents, workflow agents, custom agents, and orchestration.
"""

import asyncio
import time
from typing import Dict, Any, Optional

from .factory import AgentFactory, AgentSuite, create_agent, create_research_team
from .orchestrator import AgentOrchestrator, OrchestrationStrategy, TaskPriority
from .llm_agent import LLMRole
from .custom_agent import CustomAgentType
from ..platform_logging import RunLogger


class MultiAgentDemo:
    """Demonstration of multi-agent system capabilities."""
    
    def __init__(self, logger: Optional[RunLogger] = None):
        self.logger = logger
        self.factory = AgentFactory(logger=logger)
        self.orchestrator = AgentOrchestrator(logger=logger)
        self.agents = []
        
    async def run_comprehensive_demo(self) -> Dict[str, Any]:
        """Run comprehensive multi-agent system demonstration."""
        print("🤖 Multi-Agent Research Platform Demo")
        print("=" * 50)
        
        demo_results = {
            "individual_agents": {},
            "team_coordination": {},
            "orchestration_strategies": {},
            "workflow_execution": {},
            "performance_metrics": {},
        }
        
        try:
            # Demo 1: Individual Agent Capabilities
            print("\n1. Individual Agent Capabilities")
            print("-" * 30)
            demo_results["individual_agents"] = await self._demo_individual_agents()
            
            # Demo 2: Team Coordination
            print("\n2. Agent Team Coordination")
            print("-" * 30)
            demo_results["team_coordination"] = await self._demo_team_coordination()
            
            # Demo 3: Orchestration Strategies
            print("\n3. Orchestration Strategies")
            print("-" * 30)
            demo_results["orchestration_strategies"] = await self._demo_orchestration_strategies()
            
            # Demo 4: Workflow Execution
            print("\n4. Workflow Execution")
            print("-" * 30)
            demo_results["workflow_execution"] = await self._demo_workflow_execution()
            
            # Demo 5: Performance Metrics
            print("\n5. Performance Metrics")
            print("-" * 30)
            demo_results["performance_metrics"] = self._demo_performance_metrics()
            
            print("\n✅ Multi-Agent Demo Completed Successfully!")
            
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            demo_results["error"] = str(e)
        
        return demo_results
    
    async def _demo_individual_agents(self) -> Dict[str, Any]:
        """Demonstrate individual agent capabilities."""
        results = {}
        
        # Create LLM agents with different roles
        print("Creating LLM agents with specialized roles...")
        
        researcher = self.factory.create_llm_agent(
            role=LLMRole.RESEARCHER,
            name="AI Research Specialist"
        )
        
        analyst = self.factory.create_llm_agent(
            role=LLMRole.ANALYST,
            name="Data Analysis Expert"
        )
        
        synthesizer = self.factory.create_llm_agent(
            role=LLMRole.SYNTHESIZER,
            name="Information Synthesizer"
        )
        
        self.agents.extend([researcher, analyst, synthesizer])
        
        # Activate agents
        await researcher.activate()
        await analyst.activate()
        await synthesizer.activate()
        
        # Test each agent with appropriate tasks
        print("Testing researcher agent...")
        research_result = await researcher.execute_task(
            "Research the latest developments in artificial intelligence and machine learning"
        )
        results["researcher"] = {
            "success": research_result.success,
            "capabilities": [cap.value for cap in researcher.get_capabilities()],
            "execution_time_ms": research_result.execution_time_ms,
        }
        print(f"  ✓ Research completed: {research_result.success}")
        
        print("Testing analyst agent...")
        analysis_result = await analyst.execute_task(
            "Analyze the trends and patterns in AI research funding over the past 5 years"
        )
        results["analyst"] = {
            "success": analysis_result.success,
            "capabilities": [cap.value for cap in analyst.get_capabilities()],
            "execution_time_ms": analysis_result.execution_time_ms,
        }
        print(f"  ✓ Analysis completed: {analysis_result.success}")
        
        print("Testing synthesizer agent...")
        synthesis_result = await synthesizer.execute_task(
            "Create a comprehensive summary combining AI research findings and funding trends"
        )
        results["synthesizer"] = {
            "success": synthesis_result.success,
            "capabilities": [cap.value for cap in synthesizer.get_capabilities()],
            "execution_time_ms": synthesis_result.execution_time_ms,
        }
        print(f"  ✓ Synthesis completed: {synthesis_result.success}")
        
        # Create custom agents
        print("Creating custom specialized agents...")
        
        fact_checker = self.factory.create_custom_agent(
            agent_type=CustomAgentType.FACT_CHECKER,
            name="AI Fact Checker"
        )
        
        content_creator = self.factory.create_custom_agent(
            agent_type=CustomAgentType.CONTENT_CREATOR,
            domain="technology",
            name="Tech Content Creator"
        )
        
        self.agents.extend([fact_checker, content_creator])
        
        await fact_checker.activate()
        await content_creator.activate()
        
        print("Testing fact checker...")
        fact_check_result = await fact_checker.execute_task(
            "Verify the accuracy of claims about recent AI breakthroughs"
        )
        results["fact_checker"] = {
            "success": fact_check_result.success,
            "specialization": fact_checker.config.agent_type.value,
            "execution_time_ms": fact_check_result.execution_time_ms,
        }
        print(f"  ✓ Fact checking completed: {fact_check_result.success}")
        
        print("Testing content creator...")
        content_result = await content_creator.execute_task(
            "Create an engaging article about the future of AI in healthcare"
        )
        results["content_creator"] = {
            "success": content_result.success,
            "specialization": content_creator.config.agent_type.value,
            "domain": content_creator.config.domain,
            "execution_time_ms": content_result.execution_time_ms,
        }
        print(f"  ✓ Content creation completed: {content_result.success}")
        
        return results
    
    async def _demo_team_coordination(self) -> Dict[str, Any]:
        """Demonstrate agent team coordination."""
        results = {}
        
        print("Creating research team...")
        research_team = self.factory.create_agent_suite(AgentSuite.RESEARCH_TEAM)
        
        # Activate team
        for agent in research_team:
            await agent.activate()
        
        self.agents.extend(research_team)
        
        print(f"Research team created with {len(research_team)} agents:")
        for agent in research_team:
            print(f"  - {agent.name} ({agent.agent_type.value})")
        
        # Simulate coordinated research task
        research_topic = "The impact of quantum computing on cybersecurity"
        
        print(f"Coordinating research on: {research_topic}")
        
        team_results = {}
        for agent in research_team:
            if hasattr(agent, 'execute_task'):
                result = await agent.execute_task(
                    f"As part of a research team, contribute your expertise to analyzing: {research_topic}"
                )
                team_results[agent.name] = {
                    "success": result.success,
                    "agent_type": agent.agent_type.value,
                    "execution_time_ms": result.execution_time_ms,
                }
                print(f"  ✓ {agent.name} completed their contribution")
        
        results["research_team"] = {
            "team_size": len(research_team),
            "topic": research_topic,
            "individual_results": team_results,
            "coordination_success": all(r["success"] for r in team_results.values()),
        }
        
        return results
    
    async def _demo_orchestration_strategies(self) -> Dict[str, Any]:
        """Demonstrate different orchestration strategies."""
        results = {}
        
        task = "Analyze the potential risks and benefits of artificial general intelligence (AGI)"
        
        print(f"Testing orchestration strategies for task: {task[:60]}...")
        
        # Test different strategies
        strategies_to_test = [
            OrchestrationStrategy.SINGLE_BEST,
            OrchestrationStrategy.PARALLEL_ALL,
            OrchestrationStrategy.CONSENSUS,
            OrchestrationStrategy.COMPETITIVE,
            OrchestrationStrategy.ADAPTIVE,
        ]
        
        for strategy in strategies_to_test:
            print(f"Testing {strategy.value} strategy...")
            
            try:
                start_time = time.time()
                orchestration_result = await self.orchestrator.orchestrate_task(
                    task=task,
                    strategy=strategy,
                    priority=TaskPriority.MEDIUM
                )
                execution_time = (time.time() - start_time) * 1000
                
                results[strategy.value] = {
                    "success": orchestration_result.success,
                    "agents_used": len(orchestration_result.agents_used),
                    "agent_names": orchestration_result.agents_used,
                    "execution_time_ms": execution_time,
                    "consensus_score": orchestration_result.consensus_score,
                }
                
                print(f"  ✓ {strategy.value}: {orchestration_result.success} ({len(orchestration_result.agents_used)} agents)")
                
            except Exception as e:
                results[strategy.value] = {
                    "success": False,
                    "error": str(e),
                }
                print(f"  ✗ {strategy.value}: Failed - {e}")
        
        return results
    
    async def _demo_workflow_execution(self) -> Dict[str, Any]:
        """Demonstrate workflow execution capabilities."""
        results = {}
        
        print("Creating workflow agent...")
        workflow_agent = self.factory.create_workflow_agent(
            name="Research Workflow Coordinator"
        )
        
        await workflow_agent.activate()
        self.agents.append(workflow_agent)
        
        # Test complex workflow
        complex_task = """
        Create a comprehensive research report on renewable energy technologies:
        1. Research current renewable energy technologies
        2. Analyze market trends and adoption rates
        3. Evaluate environmental impact
        4. Synthesize findings into a structured report
        """
        
        print("Executing complex multi-step workflow...")
        
        workflow_result = await workflow_agent.execute_task(complex_task)
        
        results["complex_workflow"] = {
            "success": workflow_result.success,
            "workflow_details": workflow_result.result if workflow_result.success else None,
            "execution_time_ms": workflow_result.execution_time_ms,
            "error": workflow_result.error,
        }
        
        print(f"  ✓ Workflow execution: {workflow_result.success}")
        
        # Test simple task workflow
        simple_task = "Summarize the key benefits of solar energy"
        
        print("Executing simple task workflow...")
        
        simple_result = await workflow_agent.execute_task(simple_task)
        
        results["simple_workflow"] = {
            "success": simple_result.success,
            "execution_time_ms": simple_result.execution_time_ms,
        }
        
        print(f"  ✓ Simple workflow: {simple_result.success}")
        
        return results
    
    def _demo_performance_metrics(self) -> Dict[str, Any]:
        """Demonstrate performance metrics collection."""
        results = {}
        
        print("Collecting performance metrics...")
        
        # Agent registry metrics
        from .base import AgentRegistry
        registry_status = AgentRegistry.get_registry_status()
        
        results["registry_status"] = registry_status
        print(f"  • Total agents registered: {registry_status['total_agents']}")
        print(f"  • Active agents: {registry_status['active_agents']}")
        
        # Factory metrics
        factory_status = self.factory.get_factory_status()
        results["factory_status"] = factory_status
        print(f"  • Agents created by factory: {factory_status['total_agents_created']}")
        print(f"  • Teams created: {factory_status['teams_created']}")
        
        # Orchestrator metrics
        orchestrator_status = self.orchestrator.get_orchestration_status()
        results["orchestrator_status"] = orchestrator_status
        print(f"  • Tasks orchestrated: {orchestrator_status['total_orchestrated']}")
        print(f"  • Success rate: {orchestrator_status['success_rate']:.1f}%")
        
        # Individual agent metrics
        agent_metrics = {}
        for agent in self.agents:
            if hasattr(agent, 'get_performance_metrics'):
                metrics = agent.get_performance_metrics()
                agent_metrics[agent.name] = metrics
            else:
                # Basic metrics for all agents
                agent_metrics[agent.name] = {
                    "agent_type": agent.agent_type.value,
                    "is_active": agent.is_active,
                    "total_tasks": agent.total_tasks_completed,
                }
        
        results["agent_metrics"] = agent_metrics
        print(f"  • Individual agent metrics collected for {len(agent_metrics)} agents")
        
        return results
    
    async def cleanup(self):
        """Clean up demo resources."""
        print("\nCleaning up demo resources...")
        
        # Deactivate all agents
        for agent in self.agents:
            try:
                await agent.deactivate()
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to deactivate agent {agent.name}: {e}")
        
        print(f"  ✓ Deactivated {len(self.agents)} agents")


# Convenience functions for running demos

async def run_quick_demo(logger: Optional[RunLogger] = None) -> Dict[str, Any]:
    """Run a quick demonstration of key features."""
    print("🚀 Quick Multi-Agent Demo")
    print("=" * 30)
    
    # Create a simple research team
    factory = AgentFactory(logger=logger)
    orchestrator = AgentOrchestrator(logger=logger)
    
    # Create agents
    researcher = factory.create_llm_agent(LLMRole.RESEARCHER, name="Quick Researcher")
    analyst = factory.create_llm_agent(LLMRole.ANALYST, name="Quick Analyst")
    
    # Activate agents
    await researcher.activate()
    await analyst.activate()
    
    # Test orchestration
    task = "Research the benefits of renewable energy and provide analysis"
    
    result = await orchestrator.orchestrate_task(
        task=task,
        strategy=OrchestrationStrategy.PARALLEL_ALL,
        priority=TaskPriority.HIGH
    )
    
    # Cleanup
    await researcher.deactivate()
    await analyst.deactivate()
    
    print(f"✅ Quick demo completed: {result.success}")
    
    return {
        "task": task,
        "strategy": result.strategy_used.value,
        "agents_used": len(result.agents_used),
        "success": result.success,
        "execution_time_ms": result.execution_time_ms,
    }


async def run_full_demo(logger: Optional[RunLogger] = None) -> Dict[str, Any]:
    """Run comprehensive demonstration."""
    demo = MultiAgentDemo(logger=logger)
    
    try:
        results = await demo.run_comprehensive_demo()
        return results
    finally:
        await demo.cleanup()


if __name__ == "__main__":
    print("Multi-Agent Research Platform Demo")
    print("=" * 40)
    print()
    print("Available demos:")
    print("1. Quick Demo - Basic functionality test")
    print("2. Full Demo - Comprehensive feature demonstration")
    print()
    print("Run with:")
    print("  python -m src.agents.demo")
    print()
    print("Or import and use:")
    print("  from src.agents.demo import run_quick_demo, run_full_demo")
    print("  result = await run_quick_demo()")