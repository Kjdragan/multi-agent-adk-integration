# Performance tests for load testing and scalability analysis
# Tests system performance under various load conditions

import pytest
import asyncio
import time
import psutil
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
from unittest.mock import Mock, AsyncMock, patch

from src.agents import AgentFactory, AgentOrchestrator, AgentRegistry
from src.agents.base import AgentResult, AgentCapability
from src.agents.orchestrator import OrchestrationStrategy


class PerformanceMetrics:
    """Utility class for collecting and analyzing performance metrics."""
    
    def __init__(self):
        self.response_times = []
        self.memory_usage = []
        self.cpu_usage = []
        self.error_count = 0
        self.success_count = 0
        self.start_time = None
        self.end_time = None
    
    def start_measurement(self):
        """Start performance measurement."""
        self.start_time = time.time()
        self.memory_usage.append(psutil.virtual_memory().percent)
        self.cpu_usage.append(psutil.cpu_percent())
    
    def record_response(self, response_time: float, success: bool):
        """Record a response time and outcome."""
        self.response_times.append(response_time)
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
    
    def end_measurement(self):
        """End performance measurement."""
        self.end_time = time.time()
        self.memory_usage.append(psutil.virtual_memory().percent)
        self.cpu_usage.append(psutil.cpu_percent())
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        total_time = self.end_time - self.start_time if self.end_time and self.start_time else 0
        total_requests = self.success_count + self.error_count
        
        return {
            "total_time_seconds": total_time,
            "total_requests": total_requests,
            "successful_requests": self.success_count,
            "failed_requests": self.error_count,
            "success_rate_percent": (self.success_count / total_requests * 100) if total_requests > 0 else 0,
            "requests_per_second": total_requests / total_time if total_time > 0 else 0,
            "response_time_stats": {
                "min_ms": min(self.response_times) * 1000 if self.response_times else 0,
                "max_ms": max(self.response_times) * 1000 if self.response_times else 0,
                "mean_ms": statistics.mean(self.response_times) * 1000 if self.response_times else 0,
                "median_ms": statistics.median(self.response_times) * 1000 if self.response_times else 0,
                "p95_ms": self._percentile(self.response_times, 95) * 1000 if len(self.response_times) >= 20 else 0,
                "p99_ms": self._percentile(self.response_times, 99) * 1000 if len(self.response_times) >= 100 else 0
            },
            "resource_usage": {
                "memory_usage_percent": {
                    "start": self.memory_usage[0] if self.memory_usage else 0,
                    "end": self.memory_usage[-1] if len(self.memory_usage) > 1 else 0,
                    "peak": max(self.memory_usage) if self.memory_usage else 0
                },
                "cpu_usage_percent": {
                    "start": self.cpu_usage[0] if self.cpu_usage else 0,
                    "end": self.cpu_usage[-1] if len(self.cpu_usage) > 1 else 0,
                    "peak": max(self.cpu_usage) if self.cpu_usage else 0
                }
            }
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        lower = int(index)
        upper = min(lower + 1, len(sorted_data) - 1)
        weight = index - lower
        return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight


@pytest.fixture
def performance_metrics():
    """Provide performance metrics collector."""
    return PerformanceMetrics()


class TestAgentLoadTesting:
    """Load testing for individual agents."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_single_agent_load(self, agent_factory, performance_metrics):
        """Test single agent under increasing load."""
        agent = await agent_factory.create_llm_agent(
            role="researcher",
            name="Load Test Agent"
        )
        
        # Mock agent execution with realistic timing
        def mock_execute_task(*args, **kwargs):
            # Simulate processing time with some variance
            import random
            processing_time = random.uniform(0.1, 0.3)
            return AgentResult(
                agent_id=agent.agent_id,
                result=f"Load test result {random.randint(1000, 9999)}",
                success=random.random() > 0.05,  # 95% success rate
                execution_time_ms=int(processing_time * 1000)
            )
        
        agent.execute_task = AsyncMock(side_effect=mock_execute_task)
        
        # Test different load levels
        load_levels = [10, 25, 50, 100]
        results = {}
        
        for load in load_levels:
            metrics = PerformanceMetrics()
            metrics.start_measurement()
            
            # Create concurrent tasks
            tasks = []
            for i in range(load):
                task = agent.execute_task(f"Load test task {i}")
                tasks.append(task)
            
            # Execute all tasks concurrently
            start_time = time.time()
            results_batch = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # Record metrics
            for result in results_batch:
                if isinstance(result, AgentResult):
                    metrics.record_response(
                        response_time=end_time - start_time,
                        success=result.success
                    )
                else:
                    metrics.record_response(
                        response_time=end_time - start_time,
                        success=False
                    )
            
            metrics.end_measurement()
            results[load] = metrics.get_summary()
        
        # Verify performance characteristics
        for load in load_levels:
            summary = results[load]
            assert summary["success_rate_percent"] >= 90  # At least 90% success rate
            assert summary["response_time_stats"]["mean_ms"] < 1000  # Under 1 second average
        
        # Performance should degrade gracefully with load
        assert results[100]["response_time_stats"]["mean_ms"] > results[10]["response_time_stats"]["mean_ms"]
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_agent_memory_usage_under_load(self, agent_factory):
        """Test agent memory usage under sustained load."""
        agent = await agent_factory.create_llm_agent(
            role="researcher",
            name="Memory Test Agent"
        )
        
        # Mock agent with memory accumulation simulation
        memory_usage = []
        
        def mock_execute_with_memory(*args, **kwargs):
            # Simulate memory usage growth
            memory_usage.append(len(memory_usage) * 1024)  # Simulate growing memory
            return AgentResult(
                agent_id=agent.agent_id,
                result="Memory test result",
                success=True,
                execution_time_ms=100
            )
        
        agent.execute_task = AsyncMock(side_effect=mock_execute_with_memory)
        
        # Execute many tasks to test memory behavior
        initial_memory = psutil.virtual_memory().used
        
        for batch in range(10):
            tasks = [agent.execute_task(f"Batch {batch} task {i}") for i in range(20)]
            await asyncio.gather(*tasks)
            
            current_memory = psutil.virtual_memory().used
            memory_growth = current_memory - initial_memory
            
            # Memory growth should be reasonable (less than 100MB)
            assert memory_growth < 100 * 1024 * 1024
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_agent_concurrent_execution_limits(self, agent_factory):
        """Test agent behavior at concurrency limits."""
        agent = await agent_factory.create_llm_agent(
            role="researcher",
            name="Concurrency Test Agent"
        )
        
        # Track concurrent executions
        active_executions = 0
        max_concurrent = 0
        
        async def mock_execute_with_tracking(*args, **kwargs):
            nonlocal active_executions, max_concurrent
            
            active_executions += 1
            max_concurrent = max(max_concurrent, active_executions)
            
            # Simulate processing time
            await asyncio.sleep(0.1)
            
            active_executions -= 1
            
            return AgentResult(
                agent_id=agent.agent_id,
                result="Concurrency test result",
                success=True,
                execution_time_ms=100
            )
        
        agent.execute_task = mock_execute_with_tracking
        
        # Launch many concurrent tasks
        tasks = [agent.execute_task(f"Concurrent task {i}") for i in range(100)]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Verify all tasks completed successfully
        assert all(result.success for result in results)
        assert len(results) == 100
        
        # Check that concurrency was properly managed
        total_time = end_time - start_time
        assert total_time < 30  # Should complete within reasonable time
        
        # Maximum concurrent executions should be reasonable
        assert max_concurrent <= 20  # Reasonable concurrency limit


class TestOrchestrationScalability:
    """Scalability testing for agent orchestration."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_orchestration_scaling_with_agent_count(self, agent_factory, test_agent_registry):
        """Test orchestration performance with increasing agent counts."""
        orchestrator = AgentOrchestrator()
        
        # Test with different numbers of agents
        agent_counts = [5, 10, 25, 50]
        performance_results = {}
        
        for count in agent_counts:
            # Create agents
            agents = []
            for i in range(count):
                agent = await agent_factory.create_llm_agent(
                    role="researcher",
                    name=f"Scale Test Agent {i}"
                )
                
                # Mock fast execution
                agent.execute_task = AsyncMock(return_value=AgentResult(
                    agent_id=agent.agent_id,
                    result=f"Result from agent {i}",
                    success=True,
                    execution_time_ms=50
                ))
                
                agents.append(agent)
                test_agent_registry.register_agent(agent)
            
            # Test orchestration performance
            start_time = time.time()
            
            result = await orchestrator.orchestrate_task(
                task="Scale test task",
                strategy=OrchestrationStrategy.PARALLEL_ALL,
                requirements=[AgentCapability.RESEARCH]
            )
            
            end_time = time.time()
            
            performance_results[count] = {
                "execution_time": end_time - start_time,
                "success": result.success,
                "agents_used": len(result.agents_used) if result.agents_used else 0
            }
            
            # Cleanup for next iteration
            for agent in agents:
                test_agent_registry.unregister_agent(agent.agent_id)
        
        # Verify scalability characteristics
        for count in agent_counts:
            result = performance_results[count]
            assert result["success"]
            assert result["execution_time"] < 10  # Should complete within 10 seconds
        
        # Execution time should scale sub-linearly with agent count
        time_5 = performance_results[5]["execution_time"]
        time_50 = performance_results[50]["execution_time"]
        assert time_50 < time_5 * 5  # Should be more efficient than linear scaling
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_concurrent_orchestration_requests(self, agent_factory, agent_orchestrator, test_agent_registry):
        """Test handling multiple concurrent orchestration requests."""
        # Create agent pool
        agents = []
        for i in range(10):
            agent = await agent_factory.create_llm_agent(
                role="researcher",
                name=f"Concurrent Orchestration Agent {i}"
            )
            
            agent.execute_task = AsyncMock(return_value=AgentResult(
                agent_id=agent.agent_id,
                result=f"Concurrent result {i}",
                success=True,
                execution_time_ms=100
            ))
            
            agents.append(agent)
            test_agent_registry.register_agent(agent)
        
        # Create multiple concurrent orchestration requests
        request_counts = [10, 25, 50]
        
        for request_count in request_counts:
            metrics = PerformanceMetrics()
            metrics.start_measurement()
            
            # Create concurrent orchestration requests
            orchestration_tasks = []
            for i in range(request_count):
                task = agent_orchestrator.orchestrate_task(
                    task=f"Concurrent orchestration request {i}",
                    strategy=OrchestrationStrategy.ADAPTIVE,
                    requirements=[AgentCapability.RESEARCH]
                )
                orchestration_tasks.append(task)
            
            # Execute all requests concurrently
            start_time = time.time()
            results = await asyncio.gather(*orchestration_tasks, return_exceptions=True)
            end_time = time.time()
            
            # Record metrics
            for result in results:
                if hasattr(result, 'success'):
                    metrics.record_response(
                        response_time=end_time - start_time,
                        success=result.success
                    )
                else:
                    metrics.record_response(
                        response_time=end_time - start_time,
                        success=False
                    )
            
            metrics.end_measurement()
            summary = metrics.get_summary()
            
            # Verify performance requirements
            assert summary["success_rate_percent"] >= 95
            assert summary["response_time_stats"]["mean_ms"] < 5000  # Under 5 seconds
            assert summary["requests_per_second"] > 5  # At least 5 requests per second
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_workflow_scalability(self, agent_factory):
        """Test workflow scalability with increasing complexity."""
        workflow_agent = await agent_factory.create_workflow_agent(
            workflow_type="parallel",
            name="Scalability Test Workflow"
        )
        
        # Test workflows with increasing step counts
        step_counts = [5, 10, 25, 50]
        scalability_results = {}
        
        for step_count in step_counts:
            # Create workflow configuration
            workflow_config = {
                "parallel_tasks": [
                    {
                        "id": f"step_{i}",
                        "agent_type": "llm",
                        "role": "researcher",
                        "task": f"Scalability test step {i}"
                    }
                    for i in range(step_count)
                ]
            }
            
            # Mock step execution
            def mock_step_execution(*args, **kwargs):
                return AgentResult(
                    agent_id="mock_agent",
                    result="Mock step result",
                    success=True,
                    execution_time_ms=50
                )
            
            with patch.object(workflow_agent, '_execute_step', side_effect=mock_step_execution):
                start_time = time.time()
                result = await workflow_agent.execute_workflow(workflow_config)
                end_time = time.time()
                
                scalability_results[step_count] = {
                    "execution_time": end_time - start_time,
                    "success": result.success,
                    "steps_completed": len(result.task_results) if hasattr(result, 'task_results') else 0
                }
        
        # Verify scalability
        for step_count in step_counts:
            result = scalability_results[step_count]
            assert result["success"]
            assert result["execution_time"] < step_count * 0.1  # Should be much faster than sequential
        
        # Parallel execution should be significantly faster than sequential
        time_5 = scalability_results[5]["execution_time"]
        time_50 = scalability_results[50]["execution_time"]
        assert time_50 < time_5 * 3  # Should scale much better than linear


class TestMemoryAndResourceUsage:
    """Test memory usage and resource consumption patterns."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_memory_leak_detection(self, agent_factory):
        """Test for memory leaks during extended operation."""
        agent = await agent_factory.create_llm_agent(
            role="researcher",
            name="Memory Leak Test Agent"
        )
        
        # Mock agent execution
        agent.execute_task = AsyncMock(return_value=AgentResult(
            agent_id=agent.agent_id,
            result="Memory test result",
            success=True,
            execution_time_ms=50
        ))
        
        # Record initial memory usage
        initial_memory = psutil.virtual_memory().used
        memory_samples = [initial_memory]
        
        # Execute many tasks in batches
        for batch in range(20):
            tasks = [agent.execute_task(f"Batch {batch} task {i}") for i in range(50)]
            await asyncio.gather(*tasks)
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Sample memory usage
            current_memory = psutil.virtual_memory().used
            memory_samples.append(current_memory)
            
            # Check for excessive memory growth
            memory_growth = current_memory - initial_memory
            growth_mb = memory_growth / (1024 * 1024)
            
            # Memory growth should be reasonable (less than 50MB)
            assert growth_mb < 50, f"Excessive memory growth: {growth_mb:.2f}MB"
        
        # Memory usage should stabilize (no significant trend)
        final_memory = memory_samples[-1]
        final_growth = (final_memory - initial_memory) / (1024 * 1024)
        assert final_growth < 100, f"Total memory growth too high: {final_growth:.2f}MB"
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_resource_cleanup(self, agent_factory, test_agent_registry):
        """Test proper resource cleanup when agents are destroyed."""
        initial_agent_count = len(test_agent_registry.list_agent_ids())
        initial_memory = psutil.virtual_memory().used
        
        # Create and destroy many agents
        for cycle in range(10):
            agents = []
            
            # Create agents
            for i in range(10):
                agent = await agent_factory.create_llm_agent(
                    role="researcher",
                    name=f"Cleanup Test Agent {cycle}_{i}"
                )
                agents.append(agent)
                test_agent_registry.register_agent(agent)
            
            # Use agents briefly
            for agent in agents:
                agent.execute_task = AsyncMock(return_value=AgentResult(
                    agent_id=agent.agent_id,
                    result="Cleanup test",
                    success=True,
                    execution_time_ms=10
                ))
                await agent.execute_task("Test task")
            
            # Clean up agents
            for agent in agents:
                test_agent_registry.unregister_agent(agent.agent_id)
                del agent
            
            # Force garbage collection
            import gc
            gc.collect()
        
        # Verify cleanup
        final_agent_count = len(test_agent_registry.list_agent_ids())
        final_memory = psutil.virtual_memory().used
        
        assert final_agent_count == initial_agent_count
        
        memory_growth = (final_memory - initial_memory) / (1024 * 1024)
        assert memory_growth < 20, f"Memory not properly cleaned up: {memory_growth:.2f}MB growth"


class TestPerformanceBenchmarks:
    """Performance benchmarks and baseline measurements."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_agent_creation_performance(self, agent_factory):
        """Benchmark agent creation performance."""
        creation_times = []
        
        # Create many agents and measure creation time
        for i in range(100):
            start_time = time.time()
            
            agent = await agent_factory.create_llm_agent(
                role="researcher",
                name=f"Benchmark Agent {i}"
            )
            
            end_time = time.time()
            creation_times.append(end_time - start_time)
        
        # Analyze creation performance
        avg_creation_time = statistics.mean(creation_times)
        max_creation_time = max(creation_times)
        p95_creation_time = statistics.quantiles(creation_times, n=20)[18]  # 95th percentile
        
        # Performance requirements
        assert avg_creation_time < 0.1  # Average under 100ms
        assert max_creation_time < 0.5  # Maximum under 500ms
        assert p95_creation_time < 0.2   # 95th percentile under 200ms
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_task_execution_throughput(self, agent_factory):
        """Benchmark task execution throughput."""
        agent = await agent_factory.create_llm_agent(
            role="researcher",
            name="Throughput Test Agent"
        )
        
        # Mock fast execution
        agent.execute_task = AsyncMock(return_value=AgentResult(
            agent_id=agent.agent_id,
            result="Throughput test result",
            success=True,
            execution_time_ms=25
        ))
        
        # Measure throughput over time
        test_duration = 10  # seconds
        start_time = time.time()
        completed_tasks = 0
        
        while time.time() - start_time < test_duration:
            # Execute batch of tasks
            batch_size = 20
            tasks = [agent.execute_task(f"Throughput task {i}") for i in range(batch_size)]
            results = await asyncio.gather(*tasks)
            
            completed_tasks += len([r for r in results if r.success])
        
        total_time = time.time() - start_time
        throughput = completed_tasks / total_time
        
        # Performance requirement: at least 50 tasks per second
        assert throughput >= 50, f"Throughput too low: {throughput:.2f} tasks/second"
    
    @pytest.mark.asyncio
    @pytest.mark.slow  
    async def test_orchestration_latency_benchmark(self, agent_factory, agent_orchestrator, test_agent_registry):
        """Benchmark orchestration latency."""
        # Create test agents
        agents = []
        for i in range(5):
            agent = await agent_factory.create_llm_agent(
                role="researcher", 
                name=f"Latency Test Agent {i}"
            )
            
            agent.execute_task = AsyncMock(return_value=AgentResult(
                agent_id=agent.agent_id,
                result=f"Latency test result {i}",
                success=True,
                execution_time_ms=20
            ))
            
            agents.append(agent)
            test_agent_registry.register_agent(agent)
        
        # Measure orchestration latency
        latencies = []
        
        for i in range(100):
            start_time = time.time()
            
            result = await agent_orchestrator.orchestrate_task(
                task=f"Latency benchmark task {i}",
                strategy=OrchestrationStrategy.SINGLE_BEST,
                requirements=[AgentCapability.RESEARCH]
            )
            
            end_time = time.time()
            
            if result.success:
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Analyze latency performance
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        p99_latency = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
        
        # Performance requirements
        assert avg_latency < 100, f"Average latency too high: {avg_latency:.2f}ms"
        assert p95_latency < 200, f"P95 latency too high: {p95_latency:.2f}ms"
        assert p99_latency < 500, f"P99 latency too high: {p99_latency:.2f}ms"


@pytest.mark.slow
class TestStressAndEndurance:
    """Stress tests and endurance testing."""
    
    @pytest.mark.asyncio
    async def test_sustained_load_endurance(self, agent_factory):
        """Test system endurance under sustained load."""
        agent = await agent_factory.create_llm_agent(
            role="researcher",
            name="Endurance Test Agent"
        )
        
        # Mock execution with slight delay
        async def mock_execute(*args, **kwargs):
            await asyncio.sleep(0.01)  # 10ms processing time
            return AgentResult(
                agent_id=agent.agent_id,
                result="Endurance test result",
                success=True,
                execution_time_ms=10
            )
        
        agent.execute_task = mock_execute
        
        # Run sustained load for extended period
        test_duration = 60  # 1 minute
        start_time = time.time()
        error_count = 0
        success_count = 0
        
        while time.time() - start_time < test_duration:
            # Execute continuous batches
            batch_tasks = [agent.execute_task(f"Endurance task {i}") for i in range(10)]
            
            try:
                results = await asyncio.gather(*batch_tasks)
                success_count += len([r for r in results if r.success])
            except Exception:
                error_count += 10
            
            # Brief pause to prevent overwhelming
            await asyncio.sleep(0.1)
        
        total_requests = success_count + error_count
        success_rate = success_count / total_requests if total_requests > 0 else 0
        
        # System should maintain high success rate under sustained load
        assert success_rate >= 0.95, f"Success rate too low under sustained load: {success_rate:.2%}"
        assert total_requests > 1000, f"Not enough requests processed: {total_requests}"
    
    @pytest.mark.asyncio
    async def test_burst_load_handling(self, agent_factory, test_agent_registry):
        """Test system handling of sudden burst loads."""
        # Create agent pool
        agents = []
        for i in range(10):
            agent = await agent_factory.create_llm_agent(
                role="researcher",
                name=f"Burst Test Agent {i}"
            )
            
            agent.execute_task = AsyncMock(return_value=AgentResult(
                agent_id=agent.agent_id,
                result=f"Burst test result {i}",
                success=True,
                execution_time_ms=50
            ))
            
            agents.append(agent)
            test_agent_registry.register_agent(agent)
        
        orchestrator = AgentOrchestrator()
        
        # Test increasing burst sizes
        burst_sizes = [50, 100, 200, 500]
        
        for burst_size in burst_sizes:
            start_time = time.time()
            
            # Create sudden burst of requests
            burst_tasks = [
                orchestrator.orchestrate_task(
                    task=f"Burst task {i}",
                    strategy=OrchestrationStrategy.ADAPTIVE,
                    requirements=[AgentCapability.RESEARCH]
                )
                for i in range(burst_size)
            ]
            
            # Execute burst
            results = await asyncio.gather(*burst_tasks, return_exceptions=True)
            end_time = time.time()
            
            # Analyze burst handling
            successful_results = [r for r in results if hasattr(r, 'success') and r.success]
            success_rate = len(successful_results) / len(results)
            burst_duration = end_time - start_time
            
            # System should handle bursts gracefully
            assert success_rate >= 0.90, f"Burst success rate too low for size {burst_size}: {success_rate:.2%}"
            assert burst_duration < burst_size * 0.1, f"Burst took too long for size {burst_size}: {burst_duration:.2f}s"