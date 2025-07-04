#!/usr/bin/env python3
"""
Gemini 2.5 Models Integration Demo

This demo showcases the new Gemini 2.5 model capabilities including:
- Intelligent model selection based on task complexity
- Thinking budgets for enhanced reasoning
- Structured output for consistent results
- Task-optimized agent creation
"""

import asyncio
import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.factory import AgentFactory
from src.agents.llm_agent import LLMRole
from src.config.gemini_models import (
    GeminiModel, TaskComplexity, analyze_task_complexity, 
    select_model_for_task
)
from src.platform_logging import RunLogger


async def demo_intelligent_model_selection():
    """Demonstrate intelligent model selection based on task complexity."""
    print("🧠 Gemini 2.5 Intelligent Model Selection Demo")
    print("=" * 60)
    
    # Test tasks with different complexity levels
    test_tasks = [
        {
            "task": "What is the capital of France?",
            "expected_complexity": TaskComplexity.SIMPLE
        },
        {
            "task": "Analyze the economic implications of renewable energy adoption in developing countries, considering infrastructure costs, job creation, and environmental impact.",
            "expected_complexity": TaskComplexity.COMPLEX
        },
        {
            "task": "Conduct a comprehensive strategic analysis of market positioning for a multinational corporation entering the Asian semiconductor market, including risk assessment, competitive analysis, and regulatory compliance across multiple jurisdictions.",
            "expected_complexity": TaskComplexity.CRITICAL
        }
    ]
    
    for i, test in enumerate(test_tasks, 1):
        print(f"\n📝 Task {i}: {test['task'][:80]}...")
        
        # Analyze task complexity
        detected_complexity = analyze_task_complexity(test['task'])
        print(f"   Detected Complexity: {detected_complexity.value}")
        print(f"   Expected Complexity: {test['expected_complexity'].value}")
        print(f"   ✅ Match: {detected_complexity == test['expected_complexity']}")
        
        # Get optimal model configuration
        generation_config = select_model_for_task(
            task=test['task'],
            preferences={
                "enable_thinking": True,
                "enable_structured_output": detected_complexity in [TaskComplexity.COMPLEX, TaskComplexity.CRITICAL]
            }
        )
        
        print(f"   🤖 Selected Model: {generation_config.model.name}")
        print(f"   🧠 Thinking Enabled: {generation_config.thinking_config.enabled if generation_config.thinking_config else False}")
        print(f"   📊 Structured Output: {generation_config.structured_output.enabled if generation_config.structured_output else False}")
        
        # Cost estimation
        cost_factors = generation_config.estimate_cost_factors()
        print(f"   💰 Estimated Cost Factor: {cost_factors['total_factor']:.2f}")


async def demo_task_optimized_agents():
    """Demonstrate creating task-optimized agents."""
    print("\n🚀 Task-Optimized Agent Creation Demo")
    print("=" * 60)
    
    # Initialize agent factory
    logger = RunLogger()
    factory = AgentFactory(logger=logger)
    
    # Research task scenario
    research_task = """
    Investigate the impact of artificial intelligence on healthcare delivery systems, 
    focusing on diagnostic accuracy, patient outcomes, cost effectiveness, and ethical considerations. 
    Provide a comprehensive analysis with evidence-based recommendations.
    """
    
    print(f"📋 Research Task: {research_task.strip()}")
    
    # Create task-optimized research agent
    print("\n🔬 Creating Task-Optimized Research Agent...")
    researcher = factory.create_task_optimized_agent(
        task_description=research_task,
        role=LLMRole.RESEARCHER,
        context={"priority": "comprehensive", "depth": "detailed"},
        name="AI Healthcare Research Specialist"
    )
    
    print(f"   ✅ Created: {researcher.name}")
    print(f"   🎯 Role: {researcher.config.role.value}")
    print(f"   🤖 Auto-Optimize: {researcher.config.auto_optimize_model}")
    print(f"   🧠 Thinking: {researcher.config.enable_thinking}")
    print(f"   📊 Structured Output: {researcher.config.enable_structured_output}")
    print(f"   📐 Temperature: {researcher.config.temperature}")
    
    # Create task-optimized research team
    print("\n👥 Creating Task-Optimized Research Team...")
    research_team = factory.create_research_team_for_task(
        task_description=research_task,
        context={"domain": "healthcare", "urgency": "standard"},
        team_size="standard"
    )
    
    print(f"   ✅ Team Size: {len(research_team)} agents")
    for agent in research_team:
        print(f"      - {agent.name}: {agent.config.role.value}")


async def demo_model_comparison():
    """Demonstrate model comparison for different scenarios."""
    print("\n⚖️  Model Comparison Demo")
    print("=" * 60)
    
    # Different task scenarios
    scenarios = [
        {
            "name": "Quick Q&A",
            "task": "What are the main benefits of cloud computing?",
            "preferences": {"priority_speed": True}
        },
        {
            "name": "Cost-Effective Analysis", 
            "task": "Analyze market trends for e-commerce growth",
            "preferences": {"priority_cost": True}
        },
        {
            "name": "Comprehensive Research",
            "task": "Conduct thorough research on quantum computing applications in cryptography",
            "preferences": {"enable_thinking": True, "enable_structured_output": True}
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")
        print(f"   Task: {scenario['task']}")
        
        # Get model recommendation
        generation_config = select_model_for_task(
            task=scenario['task'],
            preferences=scenario['preferences']
        )
        
        print(f"   🎯 Recommended Model: {generation_config.model.name}")
        print(f"   📈 Intelligence Level: {generation_config.model.intelligence_level}/10")
        print(f"   ⚡ Speed Level: {generation_config.model.speed_level}/10")
        print(f"   💵 Cost Level: {generation_config.model.cost_level}/10")
        
        # Feature configuration
        features = []
        if generation_config.thinking_config and generation_config.thinking_config.enabled:
            budget = generation_config.thinking_config.budget_tokens
            features.append(f"Thinking (budget: {budget if budget and budget > 0 else 'auto'})")
        
        if generation_config.structured_output and generation_config.structured_output.enabled:
            features.append("Structured Output")
        
        print(f"   🔧 Features: {', '.join(features) if features else 'Standard'}")


async def demo_structured_output_schemas():
    """Demonstrate structured output schemas."""
    print("\n📋 Structured Output Schemas Demo")
    print("=" * 60)
    
    from src.config.gemini_models import StructuredOutputConfig
    
    # Analysis schema
    analysis_schema = StructuredOutputConfig.get_analysis_schema()
    print("📊 Analysis Schema:")
    print(json.dumps(analysis_schema, indent=2))
    
    # Research schema
    print("\n🔬 Research Schema:")
    research_schema = StructuredOutputConfig.get_research_schema()
    print(json.dumps(research_schema, indent=2))


async def main():
    """Run all demos."""
    print("🎉 Gemini 2.5 Models Integration Demo")
    print("="*80)
    print("This demo showcases the advanced capabilities of the Multi-Agent Research Platform")
    print("with the latest Gemini 2.5 models, including intelligent model selection,")
    print("thinking budgets, and structured output.")
    print()
    
    try:
        await demo_intelligent_model_selection()
        await demo_task_optimized_agents()
        await demo_model_comparison()
        await demo_structured_output_schemas()
        
        print("\n" + "="*80)
        print("🎊 Demo completed successfully!")
        print("The Multi-Agent Research Platform now supports:")
        print("  ✅ Gemini 2.5 Pro, Flash, and Flash-Lite models")
        print("  ✅ Intelligent model selection based on task complexity")
        print("  ✅ Thinking budgets for enhanced reasoning")
        print("  ✅ Structured output for consistent results")
        print("  ✅ Task-optimized agent and team creation")
        print("  ✅ Cost and performance optimization")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("This is expected if API keys are not configured.")
        print("The platform will work correctly when proper credentials are provided.")


if __name__ == "__main__":
    asyncio.run(main())