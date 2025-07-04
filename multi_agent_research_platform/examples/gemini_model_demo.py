#!/usr/bin/env python3
"""
Standalone Gemini 2.5 Model Selection Demo

This demo showcases the intelligent model selection system without
requiring the full agent framework to be configured.
"""

import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.gemini_models import (
    GeminiModel, TaskComplexity, ModelSelector, 
    analyze_task_complexity, select_model_for_task,
    StructuredOutputConfig
)


def demo_task_complexity_analysis():
    """Demonstrate task complexity analysis."""
    print("🧠 Task Complexity Analysis Demo")
    print("=" * 50)
    
    test_tasks = [
        "What is 2+2?",
        "Explain photosynthesis",
        "Analyze the economic impact of climate change on global supply chains",
        "Conduct a comprehensive strategic risk assessment for a multinational merger"
    ]
    
    for task in test_tasks:
        complexity = analyze_task_complexity(task)
        print(f"\nTask: {task}")
        print(f"Complexity: {complexity.value.upper()}")


def demo_model_selection():
    """Demonstrate intelligent model selection."""
    print("\n🤖 Intelligent Model Selection Demo")
    print("=" * 50)
    
    scenarios = [
        {
            "task": "Quick fact check: What year was Python created?",
            "preferences": {"priority_speed": True}
        },
        {
            "task": "Budget analysis of marketing campaign effectiveness",
            "preferences": {"priority_cost": True}
        },
        {
            "task": "Comprehensive research on quantum computing applications in cryptography",
            "preferences": {"enable_thinking": True, "enable_structured_output": True}
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📝 Scenario {i}:")
        print(f"Task: {scenario['task']}")
        print(f"Preferences: {scenario['preferences']}")
        
        config = select_model_for_task(
            task=scenario['task'],
            preferences=scenario['preferences']
        )
        
        print(f"Selected Model: {config.model.name}")
        print(f"Intelligence Level: {config.model.intelligence_level}/10")
        print(f"Speed Level: {config.model.speed_level}/10")
        print(f"Cost Level: {config.model.cost_level}/10")
        
        if config.thinking_config:
            print(f"Thinking Enabled: {config.thinking_config.enabled}")
        
        if config.structured_output:
            print(f"Structured Output: {config.structured_output.enabled}")


def demo_model_registry():
    """Demonstrate model registry capabilities."""
    print("\n📊 Model Registry Demo")
    print("=" * 50)
    
    selector = ModelSelector()
    registry = selector.registry
    
    print("Available Gemini 2.5 Models:")
    for model_id, model_config in registry.get_all_models().items():
        print(f"\n🤖 {model_config.name} ({model_id.value})")
        print(f"   Intelligence: {model_config.intelligence_level}/10")
        print(f"   Speed: {model_config.speed_level}/10") 
        print(f"   Cost: {model_config.cost_level}/10")
        print(f"   Max Input: {model_config.max_input_tokens:,} tokens")
        print(f"   Max Output: {model_config.max_output_tokens:,} tokens")
        print(f"   Thinking Support: {model_config.supports_thinking}")
        print(f"   Capabilities: {len(model_config.capabilities)} features")


def demo_structured_schemas():
    """Demonstrate structured output schemas."""
    print("\n📋 Structured Output Schemas Demo")
    print("=" * 50)
    
    print("Analysis Schema:")
    analysis_schema = StructuredOutputConfig.get_analysis_schema()
    print(json.dumps(analysis_schema, indent=2))
    
    print("\nResearch Schema:")
    research_schema = StructuredOutputConfig.get_research_schema()
    print(json.dumps(research_schema, indent=2))


def demo_cost_estimation():
    """Demonstrate cost estimation."""
    print("\n💰 Cost Estimation Demo")
    print("=" * 50)
    
    # Test different configurations
    configs = [
        {
            "name": "Basic Query",
            "task": "What is machine learning?",
            "preferences": {"priority_speed": True}
        },
        {
            "name": "Complex Analysis",
            "task": "Analyze the implications of AGI on society",
            "preferences": {"enable_thinking": True, "enable_structured_output": True}
        }
    ]
    
    for config_test in configs:
        print(f"\n💡 {config_test['name']}:")
        
        generation_config = select_model_for_task(
            task=config_test['task'],
            preferences=config_test['preferences']
        )
        
        cost_factors = generation_config.estimate_cost_factors()
        
        print(f"Task: {config_test['task']}")
        print(f"Model: {generation_config.model.name}")
        print(f"Base Cost Factor: {cost_factors['base_cost_factor']:.2f}")
        print(f"Thinking Multiplier: {cost_factors['thinking_multiplier']:.2f}")
        print(f"Structured Multiplier: {cost_factors['structured_multiplier']:.2f}")
        print(f"Total Cost Factor: {cost_factors['total_factor']:.2f}")


def main():
    """Run all demos."""
    print("🎉 Gemini 2.5 Model Selection System Demo")
    print("="*60)
    print("This demo showcases the intelligent model selection capabilities")
    print("integrated into the Multi-Agent Research Platform.")
    print()
    
    try:
        demo_task_complexity_analysis()
        demo_model_selection()
        demo_model_registry()
        demo_structured_schemas()
        demo_cost_estimation()
        
        print("\n" + "="*60)
        print("🎊 Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  ✅ Automatic task complexity analysis")
        print("  ✅ Intelligent model selection (Pro/Flash/Flash-Lite)")
        print("  ✅ Thinking budget configuration")
        print("  ✅ Structured output schemas")
        print("  ✅ Cost optimization")
        print("  ✅ Performance vs. cost tradeoffs")
        print("\nThe platform can now:")
        print("  🔍 Auto-select optimal models based on task requirements")
        print("  🧠 Enable thinking for complex reasoning tasks")
        print("  📊 Structure outputs for consistent JSON responses")
        print("  💰 Optimize costs while maintaining quality")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()