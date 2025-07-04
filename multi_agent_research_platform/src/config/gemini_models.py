# Gemini Model Configuration and Management
# Supports the latest Gemini 2.5 models with thinking budgets and structured output

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Union
import json


class GeminiModel(str, Enum):
    """Latest Gemini 2.5 model identifiers."""
    # Ordered from smartest/slowest to fastest/least intelligent
    PRO = "gemini-2.5-pro"                           # Most advanced reasoning
    FLASH = "gemini-2.5-flash"                       # Best price/performance balance  
    FLASH_LITE = "gemini-2.5-flash-lite-preview-06-17"  # Fastest, low latency


class ModelCapability(str, Enum):
    """Model capabilities and features."""
    THINKING = "thinking"
    STRUCTURED_OUTPUT = "structured_output"
    GROUNDING = "grounding"
    CODE_EXECUTION = "code_execution"
    FUNCTION_CALLING = "function_calling"
    MULTIMODAL = "multimodal"
    CONTEXT_CACHING = "context_caching"
    TUNING = "tuning"
    LIVE_API = "live_api"


class TaskComplexity(str, Enum):
    """Task complexity levels for model selection."""
    SIMPLE = "simple"           # Basic Q&A, simple tasks
    MEDIUM = "medium"           # Research, analysis, moderate complexity
    COMPLEX = "complex"         # Multi-step reasoning, synthesis
    CRITICAL = "critical"       # High-stakes, maximum accuracy needed


@dataclass
class ThinkingBudgetConfig:
    """Configuration for model thinking budget."""
    enabled: bool = True
    budget_tokens: Optional[int] = None  # -1 for auto, 0 to disable, positive for manual
    
    # Budget ranges per model
    min_tokens: int = 1
    max_tokens: int = 8192
    default_tokens: int = -1  # Auto mode
    
    def get_budget_for_complexity(self, complexity: TaskComplexity) -> int:
        """Get thinking budget based on task complexity."""
        if not self.enabled or self.budget_tokens == 0:
            return 0
        
        if self.budget_tokens and self.budget_tokens > 0:
            return self.budget_tokens
        
        # Auto-select based on complexity
        complexity_budgets = {
            TaskComplexity.SIMPLE: min(self.max_tokens // 4, 2048),
            TaskComplexity.MEDIUM: min(self.max_tokens // 2, 4096), 
            TaskComplexity.COMPLEX: min(self.max_tokens * 3 // 4, 8192),
            TaskComplexity.CRITICAL: self.max_tokens
        }
        
        return complexity_budgets.get(complexity, self.default_tokens)


@dataclass
class StructuredOutputConfig:
    """Configuration for structured output."""
    enabled: bool = False
    response_mime_type: str = "application/json"
    response_schema: Optional[Dict[str, Any]] = None
    
    # Common schemas for reuse
    @staticmethod
    def get_analysis_schema() -> Dict[str, Any]:
        """Schema for analysis results."""
        return {
            "type": "OBJECT",
            "properties": {
                "summary": {"type": "STRING"},
                "key_findings": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"}
                },
                "confidence_score": {
                    "type": "NUMBER",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "sources_used": {
                    "type": "ARRAY", 
                    "items": {"type": "STRING"}
                },
                "recommendations": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"}
                }
            },
            "required": ["summary", "key_findings", "confidence_score"]
        }
    
    @staticmethod
    def get_research_schema() -> Dict[str, Any]:
        """Schema for research results."""
        return {
            "type": "OBJECT",
            "properties": {
                "research_topic": {"type": "STRING"},
                "findings": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "finding": {"type": "STRING"},
                            "source": {"type": "STRING", "nullable": True},
                            "confidence": {"type": "NUMBER", "minimum": 0.0, "maximum": 1.0}
                        },
                        "required": ["finding", "confidence"]
                    }
                },
                "conclusion": {"type": "STRING"},
                "gaps_identified": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"}
                }
            },
            "required": ["research_topic", "findings", "conclusion"]
        }


@dataclass 
class ModelConfig:
    """Configuration for a specific Gemini model."""
    model_id: GeminiModel
    name: str
    description: str
    
    # Performance characteristics
    intelligence_level: int  # 1-10 scale
    speed_level: int        # 1-10 scale
    cost_level: int         # 1-10 scale (1=cheapest)
    
    # Technical specifications
    max_input_tokens: int
    max_output_tokens: int
    supports_thinking: bool
    thinking_budget: ThinkingBudgetConfig
    
    # Capabilities
    capabilities: List[ModelCapability] = field(default_factory=list)
    
    # Deployment requirements
    requires_global_endpoint: bool = False
    launch_stage: str = "GA"  # GA, Preview, etc.
    
    # Default parameters
    default_temperature: float = 1.0
    default_top_p: float = 0.95
    default_top_k: int = 64
    
    def is_suitable_for_complexity(self, complexity: TaskComplexity) -> bool:
        """Check if model is suitable for given task complexity."""
        complexity_requirements = {
            TaskComplexity.SIMPLE: 3,      # Any model works
            TaskComplexity.MEDIUM: 6,      # Flash or better
            TaskComplexity.COMPLEX: 8,     # Pro preferred
            TaskComplexity.CRITICAL: 9     # Pro required
        }
        
        required_level = complexity_requirements.get(complexity, 5)
        return self.intelligence_level >= required_level
    
    def get_endpoint_location(self) -> str:
        """Get required endpoint location for this model."""
        return "global" if self.requires_global_endpoint else "us-central1"


class GeminiModelRegistry:
    """Registry of available Gemini models with their configurations."""
    
    def __init__(self):
        self._models = self._initialize_models()
    
    def _initialize_models(self) -> Dict[GeminiModel, ModelConfig]:
        """Initialize model configurations."""
        models = {}
        
        # Gemini 2.5 Pro - Most advanced reasoning
        models[GeminiModel.PRO] = ModelConfig(
            model_id=GeminiModel.PRO,
            name="Gemini 2.5 Pro",
            description="Most advanced reasoning Gemini model, capable of solving complex problems",
            intelligence_level=10,
            speed_level=4,
            cost_level=8,
            max_input_tokens=1_048_576,
            max_output_tokens=65_535,
            supports_thinking=True,
            thinking_budget=ThinkingBudgetConfig(
                enabled=True,
                min_tokens=128,
                max_tokens=32_768,
                default_tokens=-1
            ),
            capabilities=[
                ModelCapability.THINKING,
                ModelCapability.STRUCTURED_OUTPUT,
                ModelCapability.GROUNDING,
                ModelCapability.CODE_EXECUTION,
                ModelCapability.FUNCTION_CALLING,
                ModelCapability.MULTIMODAL,
                ModelCapability.CONTEXT_CACHING
            ],
            requires_global_endpoint=False,
            launch_stage="GA"
        )
        
        # Gemini 2.5 Flash - Best balance of price and performance
        models[GeminiModel.FLASH] = ModelConfig(
            model_id=GeminiModel.FLASH,
            name="Gemini 2.5 Flash", 
            description="Best model in terms of price and performance, with thinking capabilities",
            intelligence_level=8,
            speed_level=7,
            cost_level=4,
            max_input_tokens=1_048_576,
            max_output_tokens=65_535,
            supports_thinking=True,
            thinking_budget=ThinkingBudgetConfig(
                enabled=True,
                min_tokens=1,
                max_tokens=24_576,
                default_tokens=-1
            ),
            capabilities=[
                ModelCapability.THINKING,
                ModelCapability.STRUCTURED_OUTPUT,
                ModelCapability.GROUNDING,
                ModelCapability.CODE_EXECUTION,
                ModelCapability.TUNING,
                ModelCapability.FUNCTION_CALLING,
                ModelCapability.MULTIMODAL,
                ModelCapability.CONTEXT_CACHING,
                ModelCapability.LIVE_API
            ],
            requires_global_endpoint=False,
            launch_stage="GA"
        )
        
        # Gemini 2.5 Flash-Lite - Fastest, optimized for low latency
        models[GeminiModel.FLASH_LITE] = ModelConfig(
            model_id=GeminiModel.FLASH_LITE,
            name="Gemini 2.5 Flash-Lite",
            description="Most balanced model optimized for low latency use cases",
            intelligence_level=6,
            speed_level=10,
            cost_level=2,
            max_input_tokens=1_048_576,
            max_output_tokens=65_536,
            supports_thinking=True,
            thinking_budget=ThinkingBudgetConfig(
                enabled=True,
                min_tokens=512,
                max_tokens=24_576,
                default_tokens=-1
            ),
            capabilities=[
                ModelCapability.THINKING,
                ModelCapability.STRUCTURED_OUTPUT,
                ModelCapability.GROUNDING,
                ModelCapability.CODE_EXECUTION,
                ModelCapability.FUNCTION_CALLING,
                ModelCapability.MULTIMODAL,
                ModelCapability.CONTEXT_CACHING
            ],
            requires_global_endpoint=True,  # Preview model requires global endpoint
            launch_stage="Preview"
        )
        
        return models
    
    def get_model(self, model_id: GeminiModel) -> ModelConfig:
        """Get model configuration by ID."""
        return self._models[model_id]
    
    def get_all_models(self) -> Dict[GeminiModel, ModelConfig]:
        """Get all available models."""
        return self._models.copy()
    
    def get_models_by_capability(self, capability: ModelCapability) -> List[ModelConfig]:
        """Get models that support a specific capability."""
        return [
            model for model in self._models.values()
            if capability in model.capabilities
        ]
    
    def recommend_model_for_task(
        self,
        complexity: TaskComplexity,
        priority_speed: bool = False,
        priority_cost: bool = False,
        required_capabilities: Optional[List[ModelCapability]] = None
    ) -> ModelConfig:
        """Recommend best model for a task based on requirements."""
        suitable_models = []
        
        for model in self._models.values():
            # Check complexity suitability
            if not model.is_suitable_for_complexity(complexity):
                continue
            
            # Check required capabilities
            if required_capabilities:
                if not all(cap in model.capabilities for cap in required_capabilities):
                    continue
            
            suitable_models.append(model)
        
        if not suitable_models:
            # Fallback to most capable model
            return self._models[GeminiModel.PRO]
        
        # Sort by priority
        if priority_speed:
            return max(suitable_models, key=lambda m: m.speed_level)
        elif priority_cost:
            return min(suitable_models, key=lambda m: m.cost_level)
        else:
            # Balance intelligence and speed
            return max(suitable_models, key=lambda m: m.intelligence_level + m.speed_level)


@dataclass
class GeminiGenerationConfig:
    """Complete configuration for Gemini model generation."""
    model: ModelConfig
    
    # Generation parameters
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    candidate_count: int = 1
    max_output_tokens: Optional[int] = None
    
    # Advanced features
    thinking_config: Optional[ThinkingBudgetConfig] = None
    structured_output: Optional[StructuredOutputConfig] = None
    
    # System instructions
    system_instruction: Optional[str] = None
    
    def to_generate_content_config(self) -> Dict[str, Any]:
        """Convert to format expected by Google GenAI SDK."""
        config = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "candidate_count": self.candidate_count
        }
        
        if self.max_output_tokens:
            config["max_output_tokens"] = self.max_output_tokens
        
        # Add thinking configuration
        if (self.thinking_config and 
            self.thinking_config.enabled and 
            self.model.supports_thinking):
            
            thinking_budget = self.thinking_config.budget_tokens
            if thinking_budget is None:
                thinking_budget = -1  # Auto mode
            
            config["thinking_config"] = {
                "include_thoughts": True
            }
            
            if thinking_budget != -1:
                # Manual budget control (future API support)
                config["thinking_budget_tokens"] = thinking_budget
        
        # Add structured output configuration
        if self.structured_output and self.structured_output.enabled:
            config["response_mime_type"] = self.structured_output.response_mime_type
            if self.structured_output.response_schema:
                config["response_schema"] = self.structured_output.response_schema
        
        return config
    
    def estimate_cost_factors(self) -> Dict[str, float]:
        """Estimate relative cost factors for this configuration."""
        base_cost = self.model.cost_level / 10.0
        
        # Thinking adds cost
        thinking_multiplier = 1.0
        if (self.thinking_config and 
            self.thinking_config.enabled and 
            self.thinking_config.budget_tokens):
            
            budget = self.thinking_config.budget_tokens
            if budget > 0:
                # Higher thinking budget = higher cost
                thinking_multiplier = 1.0 + (budget / 10000)
        
        # Structured output minimal impact
        structured_multiplier = 1.02 if (self.structured_output and self.structured_output.enabled) else 1.0
        
        return {
            "base_cost_factor": base_cost,
            "thinking_multiplier": thinking_multiplier,
            "structured_multiplier": structured_multiplier,
            "total_factor": base_cost * thinking_multiplier * structured_multiplier
        }


class ModelSelector:
    """Intelligent model selection based on task requirements."""
    
    def __init__(self):
        self.registry = GeminiModelRegistry()
    
    def analyze_task_complexity(self, task: str, context: Optional[Dict[str, Any]] = None) -> TaskComplexity:
        """Analyze task complexity to determine appropriate model."""
        task_lower = task.lower()
        
        # Simple indicators
        simple_indicators = [
            "what is", "define", "list", "name", "simple question",
            "basic", "quick", "short answer"
        ]
        
        # Medium indicators
        medium_indicators = [
            "explain", "describe", "concept", "process", "how does", "how do",
            "overview", "summary", "background", "introduction"
        ]
        
        # Complex indicators  
        complex_indicators = [
            "analyze", "compare", "evaluate", "synthesize", "research",
            "comprehensive", "detailed analysis", "multi-step", "workflow",
            "strategy", "plan", "complex reasoning"
        ]
        
        # Critical indicators
        critical_indicators = [
            "critical", "mission critical", "high stakes", "thorough analysis",
            "comprehensive evaluation", "strategic planning", "risk assessment"
        ]
        
        # Check context for complexity hints
        if context:
            if context.get("priority") == "critical":
                return TaskComplexity.CRITICAL
            if context.get("depth") == "comprehensive":
                return TaskComplexity.COMPLEX
        
        # Analyze task text
        task_words = task_lower.split()
        
        if any(indicator in task_lower for indicator in critical_indicators):
            return TaskComplexity.CRITICAL
        elif any(indicator in task_lower for indicator in complex_indicators):
            return TaskComplexity.COMPLEX
        elif any(indicator in task_lower for indicator in medium_indicators):
            return TaskComplexity.MEDIUM
        elif any(indicator in task_lower for indicator in simple_indicators):
            return TaskComplexity.SIMPLE
        elif len(task_words) > 50:  # Long tasks are likely complex
            return TaskComplexity.COMPLEX
        elif len(task_words) < 5:  # Very short tasks are likely simple
            return TaskComplexity.SIMPLE
        else:
            return TaskComplexity.MEDIUM
    
    def select_optimal_model(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> GeminiGenerationConfig:
        """Select optimal model and configuration for a task."""
        
        # Analyze task complexity
        complexity = self.analyze_task_complexity(task, context)
        
        # Extract preferences
        prefs = preferences or {}
        priority_speed = prefs.get("priority_speed", False)
        priority_cost = prefs.get("priority_cost", False)
        enable_thinking = prefs.get("enable_thinking", True)
        enable_structured = prefs.get("enable_structured_output", False)
        required_capabilities = prefs.get("required_capabilities", [])
        
        # Get model recommendation
        model = self.registry.recommend_model_for_task(
            complexity=complexity,
            priority_speed=priority_speed,
            priority_cost=priority_cost,
            required_capabilities=required_capabilities
        )
        
        # Configure thinking budget
        thinking_config = None
        if enable_thinking and model.supports_thinking:
            thinking_config = ThinkingBudgetConfig(
                enabled=True,
                min_tokens=model.thinking_budget.min_tokens,
                max_tokens=model.thinking_budget.max_tokens,
                budget_tokens=model.thinking_budget.get_budget_for_complexity(complexity)
            )
        
        # Configure structured output
        structured_config = None
        if enable_structured and ModelCapability.STRUCTURED_OUTPUT in model.capabilities:
            schema_type = prefs.get("output_schema_type", "analysis")
            if schema_type == "analysis":
                schema = StructuredOutputConfig.get_analysis_schema()
            elif schema_type == "research":
                schema = StructuredOutputConfig.get_research_schema()
            else:
                schema = prefs.get("custom_schema")
            
            if schema:
                structured_config = StructuredOutputConfig(
                    enabled=True,
                    response_schema=schema
                )
        
        # Create generation config
        return GeminiGenerationConfig(
            model=model,
            temperature=prefs.get("temperature", model.default_temperature),
            top_p=prefs.get("top_p", model.default_top_p),
            top_k=prefs.get("top_k", model.default_top_k),
            thinking_config=thinking_config,
            structured_output=structured_config,
            system_instruction=prefs.get("system_instruction")
        )
    
    def compare_model_options(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Compare different model options for a task."""
        complexity = self.analyze_task_complexity(task, context)
        
        options = []
        for model_id, model in self.registry.get_all_models().items():
            if not model.is_suitable_for_complexity(complexity):
                continue
            
            # Create configs for different priorities
            speed_config = self.select_optimal_model(
                task, context, {"priority_speed": True}
            )
            cost_config = self.select_optimal_model(
                task, context, {"priority_cost": True}
            )
            balanced_config = self.select_optimal_model(
                task, context, {}
            )
            
            options.append({
                "model": model,
                "complexity_match": complexity,
                "speed_optimized": speed_config,
                "cost_optimized": cost_config, 
                "balanced": balanced_config,
                "estimated_costs": {
                    "speed": speed_config.estimate_cost_factors(),
                    "cost": cost_config.estimate_cost_factors(),
                    "balanced": balanced_config.estimate_cost_factors()
                }
            })
        
        return sorted(options, key=lambda x: x["model"].intelligence_level, reverse=True)


# Global model selector instance
model_selector = ModelSelector()

# Convenience functions
def get_model_config(model_id: GeminiModel) -> ModelConfig:
    """Get configuration for a specific model."""
    return model_selector.registry.get_model(model_id)

def select_model_for_task(
    task: str,
    context: Optional[Dict[str, Any]] = None,
    preferences: Optional[Dict[str, Any]] = None
) -> GeminiGenerationConfig:
    """Select optimal model configuration for a task."""
    return model_selector.select_optimal_model(task, context, preferences)

def analyze_task_complexity(task: str, context: Optional[Dict[str, Any]] = None) -> TaskComplexity:
    """Analyze task complexity level."""
    return model_selector.analyze_task_complexity(task, context)