# Test utilities and helper functions
# Provides common utilities for testing across all test suites

import asyncio
import json
import tempfile
import time
from typing import Dict, Any, List, Optional, Callable, Awaitable
from unittest.mock import Mock, AsyncMock
from contextlib import asynccontextmanager
from pathlib import Path

from src.agents.base import Agent, AgentResult, AgentCapability
from src.agents.orchestrator import OrchestrationResult, OrchestrationStrategy


class MockAgent(Agent):
    """Mock agent implementation for testing."""
    
    def __init__(self, agent_id: str, name: str, capabilities: List[AgentCapability], **kwargs):
        super().__init__(agent_id, name, "mock", capabilities)
        self.execution_delay = kwargs.get("execution_delay", 0.1)
        self.success_rate = kwargs.get("success_rate", 1.0)
        self.response_template = kwargs.get("response_template", "Mock response from {name}")
        self.metadata_template = kwargs.get("metadata_template", {})
        self.call_count = 0
        
    async def execute_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Execute mock task with configurable behavior."""
        self.call_count += 1
        
        # Simulate processing time
        await asyncio.sleep(self.execution_delay)
        
        # Determine success based on success rate
        import random
        success = random.random() <= self.success_rate
        
        if success:
            result = self.response_template.format(name=self.name, task=task, call_count=self.call_count)
            metadata = self.metadata_template.copy()
            metadata.update({
                "call_count": self.call_count,
                "context_provided": context is not None
            })
            
            return AgentResult(
                agent_id=self.agent_id,
                result=result,
                success=True,
                execution_time_ms=int(self.execution_delay * 1000),
                metadata=metadata
            )
        else:
            return AgentResult(
                agent_id=self.agent_id,
                result="",
                success=False,
                execution_time_ms=int(self.execution_delay * 1000),
                error_message=f"Simulated failure (success rate: {self.success_rate})"
            )
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Return agent capabilities."""
        return self.capabilities


class TestDataGenerator:
    """Generate test data for various scenarios."""
    
    @staticmethod
    def generate_agent_configs(count: int = 5) -> List[Dict[str, Any]]:
        """Generate agent configurations for testing."""
        roles = ["researcher", "analyst", "writer", "critic", "synthesizer"]
        configs = []
        
        for i in range(count):
            role = roles[i % len(roles)]
            config = {
                "agent_type": "llm",
                "name": f"Test {role.title()} {i+1}",
                "config": {
                    "role": role,
                    "temperature": 0.5 + (i * 0.1),
                    "max_tokens": 1000 + (i * 500),
                    "timeout_seconds": 60 + (i * 30)
                }
            }
            configs.append(config)
        
        return configs
    
    @staticmethod
    def generate_test_tasks(count: int = 10) -> List[str]:
        """Generate test tasks for various scenarios."""
        task_templates = [
            "Research the topic of {topic}",
            "Analyze the data about {topic}",
            "Write a summary of {topic}",
            "Review and critique the content about {topic}",
            "Synthesize information on {topic}",
            "Compare different approaches to {topic}",
            "Evaluate the effectiveness of {topic}",
            "Predict future trends in {topic}",
            "Identify key challenges in {topic}",
            "Recommend solutions for {topic}"
        ]
        
        topics = [
            "artificial intelligence",
            "renewable energy",
            "climate change",
            "space exploration",
            "biotechnology",
            "quantum computing",
            "sustainable agriculture",
            "digital transformation",
            "cybersecurity",
            "healthcare innovation"
        ]
        
        tasks = []
        for i in range(count):
            template = task_templates[i % len(task_templates)]
            topic = topics[i % len(topics)]
            task = template.format(topic=topic)
            tasks.append(task)
        
        return tasks
    
    @staticmethod
    def generate_workflow_configs(complexity: str = "simple") -> Dict[str, Any]:
        """Generate workflow configurations for testing."""
        if complexity == "simple":
            return {
                "steps": [
                    {
                        "id": "research",
                        "agent_type": "llm",
                        "role": "researcher",
                        "task": "Research the topic",
                        "dependencies": []
                    },
                    {
                        "id": "analyze",
                        "agent_type": "llm",
                        "role": "analyst",
                        "task": "Analyze research findings",
                        "dependencies": ["research"]
                    }
                ]
            }
        elif complexity == "parallel":
            return {
                "parallel_tasks": [
                    {
                        "id": "research_tech",
                        "agent_type": "llm",
                        "role": "researcher",
                        "task": "Research technical aspects"
                    },
                    {
                        "id": "research_market",
                        "agent_type": "llm", 
                        "role": "researcher",
                        "task": "Research market aspects"
                    },
                    {
                        "id": "research_social",
                        "agent_type": "llm",
                        "role": "researcher", 
                        "task": "Research social aspects"
                    }
                ],
                "aggregation": {
                    "agent_type": "llm",
                    "role": "synthesizer",
                    "task": "Synthesize all research findings"
                }
            }
        elif complexity == "complex":
            return {
                "phases": [
                    {
                        "name": "initial_research",
                        "parallel_tasks": [
                            {
                                "id": "literature_review",
                                "agent_type": "llm",
                                "role": "researcher",
                                "task": "Conduct literature review"
                            },
                            {
                                "id": "market_analysis",
                                "agent_type": "llm",
                                "role": "analyst", 
                                "task": "Perform market analysis"
                            }
                        ]
                    },
                    {
                        "name": "synthesis",
                        "sequential_tasks": [
                            {
                                "id": "data_synthesis",
                                "agent_type": "llm",
                                "role": "synthesizer",
                                "task": "Synthesize research and analysis",
                                "dependencies": ["initial_research"]
                            },
                            {
                                "id": "quality_review",
                                "agent_type": "custom",
                                "agent_subtype": "fact_checker",
                                "task": "Review synthesis quality",
                                "dependencies": ["data_synthesis"]
                            }
                        ]
                    },
                    {
                        "name": "finalization",
                        "sequential_tasks": [
                            {
                                "id": "final_report",
                                "agent_type": "llm",
                                "role": "writer",
                                "task": "Write final report",
                                "dependencies": ["synthesis"]
                            }
                        ]
                    }
                ]
            }
        
        return {}


class AsyncTestHelper:
    """Helper for async testing scenarios."""
    
    @staticmethod
    async def wait_for_condition(
        condition: Callable[[], Awaitable[bool]],
        timeout: float = 5.0,
        interval: float = 0.1,
        timeout_message: str = "Condition not met within timeout"
    ) -> bool:
        """Wait for an async condition to become true."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if await condition():
                return True
            await asyncio.sleep(interval)
        
        raise TimeoutError(timeout_message)
    
    @staticmethod
    async def run_with_timeout(
        coro: Awaitable,
        timeout: float = 10.0,
        timeout_message: str = "Operation timed out"
    ):
        """Run coroutine with timeout."""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(timeout_message)
    
    @staticmethod
    async def gather_with_error_handling(
        *coroutines,
        return_exceptions: bool = True,
        max_failures: Optional[int] = None
    ):
        """Gather coroutines with enhanced error handling."""
        results = await asyncio.gather(*coroutines, return_exceptions=return_exceptions)
        
        if max_failures is not None:
            failure_count = sum(1 for result in results if isinstance(result, Exception))
            if failure_count > max_failures:
                raise RuntimeError(f"Too many failures: {failure_count} > {max_failures}")
        
        return results


class MockAPIClient:
    """Mock API client for external service testing."""
    
    def __init__(self, response_data: Dict[str, Any] = None, latency: float = 0.1):
        self.response_data = response_data or {"status": "success", "data": "mock_response"}
        self.latency = latency
        self.call_count = 0
        self.call_history = []
    
    async def make_request(self, endpoint: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make mock API request."""
        self.call_count += 1
        self.call_history.append({"endpoint": endpoint, "data": data, "timestamp": time.time()})
        
        # Simulate network latency
        await asyncio.sleep(self.latency)
        
        # Return mock response
        response = self.response_data.copy()
        response["call_count"] = self.call_count
        response["endpoint"] = endpoint
        
        return response
    
    def get_call_history(self) -> List[Dict[str, Any]]:
        """Get history of API calls."""
        return self.call_history.copy()
    
    def reset(self):
        """Reset call history and count."""
        self.call_count = 0
        self.call_history.clear()


@asynccontextmanager
async def temporary_agent_registry():
    """Async context manager for temporary agent registry."""
    from src.agents.base import AgentRegistry
    
    # Save current state
    original_agents = AgentRegistry._agents.copy()
    original_metrics = AgentRegistry._agent_metrics.copy()
    
    try:
        # Clear registry for test
        AgentRegistry._agents.clear()
        AgentRegistry._agent_metrics.clear()
        yield AgentRegistry
    finally:
        # Restore original state
        AgentRegistry._agents = original_agents
        AgentRegistry._agent_metrics = original_metrics


class PerformanceProfiler:
    """Performance profiling utility for tests."""
    
    def __init__(self):
        self.profiles = {}
        self.current_profile = None
        self.start_time = None
    
    def start_profile(self, name: str):
        """Start profiling a section."""
        self.current_profile = name
        self.start_time = time.time()
        
        if name not in self.profiles:
            self.profiles[name] = {
                "calls": 0,
                "total_time": 0,
                "min_time": float('inf'),
                "max_time": 0,
                "times": []
            }
    
    def end_profile(self):
        """End current profiling section."""
        if self.current_profile and self.start_time:
            duration = time.time() - self.start_time
            profile = self.profiles[self.current_profile]
            
            profile["calls"] += 1
            profile["total_time"] += duration
            profile["min_time"] = min(profile["min_time"], duration)
            profile["max_time"] = max(profile["max_time"], duration)
            profile["times"].append(duration)
            
            self.current_profile = None
            self.start_time = None
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary."""
        summary = {}
        
        for name, profile in self.profiles.items():
            if profile["calls"] > 0:
                summary[name] = {
                    "calls": profile["calls"],
                    "total_time": profile["total_time"],
                    "avg_time": profile["total_time"] / profile["calls"],
                    "min_time": profile["min_time"],
                    "max_time": profile["max_time"]
                }
                
                if len(profile["times"]) >= 2:
                    import statistics
                    summary[name]["median_time"] = statistics.median(profile["times"])
                    summary[name]["std_dev"] = statistics.stdev(profile["times"])
        
        return summary


class TestFileManager:
    """Manage temporary test files and directories."""
    
    def __init__(self):
        self.temp_files = []
        self.temp_dirs = []
    
    def create_temp_file(self, content: str = "", suffix: str = ".txt") -> str:
        """Create temporary file with content."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        self.temp_files.append(temp_path)
        return temp_path
    
    def create_temp_json_file(self, data: Dict[str, Any]) -> str:
        """Create temporary JSON file."""
        content = json.dumps(data, indent=2)
        return self.create_temp_file(content, suffix=".json")
    
    def create_temp_dir(self) -> str:
        """Create temporary directory."""
        import tempfile
        
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def cleanup(self):
        """Clean up all temporary files and directories."""
        import os
        import shutil
        
        for file_path in self.temp_files:
            try:
                os.unlink(file_path)
            except OSError:
                pass
        
        for dir_path in self.temp_dirs:
            try:
                shutil.rmtree(dir_path)
            except OSError:
                pass
        
        self.temp_files.clear()
        self.temp_dirs.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class TestResultValidator:
    """Validate test results against expected criteria."""
    
    @staticmethod
    def validate_agent_result(
        result: AgentResult,
        expected_success: bool = True,
        min_content_length: int = 0,
        max_execution_time_ms: Optional[int] = None,
        required_metadata_keys: Optional[List[str]] = None
    ) -> bool:
        """Validate AgentResult against criteria."""
        if result.success != expected_success:
            return False
        
        if len(result.result) < min_content_length:
            return False
        
        if max_execution_time_ms and result.execution_time_ms > max_execution_time_ms:
            return False
        
        if required_metadata_keys:
            for key in required_metadata_keys:
                if key not in result.metadata:
                    return False
        
        return True
    
    @staticmethod
    def validate_orchestration_result(
        result: OrchestrationResult,
        expected_success: bool = True,
        min_agents_used: int = 1,
        min_consensus_score: Optional[float] = None,
        required_strategy: Optional[OrchestrationStrategy] = None
    ) -> bool:
        """Validate OrchestrationResult against criteria."""
        if result.success != expected_success:
            return False
        
        if len(result.agents_used) < min_agents_used:
            return False
        
        if min_consensus_score and result.consensus_score < min_consensus_score:
            return False
        
        if required_strategy and result.strategy_used != required_strategy:
            return False
        
        return True
    
    @staticmethod
    def validate_performance_metrics(
        metrics: Dict[str, Any],
        max_avg_response_time_ms: Optional[float] = None,
        min_success_rate_percent: Optional[float] = None,
        max_error_rate_percent: Optional[float] = None
    ) -> bool:
        """Validate performance metrics against criteria."""
        if max_avg_response_time_ms:
            avg_time = metrics.get("response_time_stats", {}).get("mean_ms", 0)
            if avg_time > max_avg_response_time_ms:
                return False
        
        if min_success_rate_percent:
            success_rate = metrics.get("success_rate_percent", 0)
            if success_rate < min_success_rate_percent:
                return False
        
        if max_error_rate_percent:
            error_rate = 100 - metrics.get("success_rate_percent", 100)
            if error_rate > max_error_rate_percent:
                return False
        
        return True


# Export commonly used utilities
__all__ = [
    "MockAgent",
    "TestDataGenerator", 
    "AsyncTestHelper",
    "MockAPIClient",
    "temporary_agent_registry",
    "PerformanceProfiler",
    "TestFileManager",
    "TestResultValidator"
]