# Examples and Tutorials

This comprehensive examples guide demonstrates practical usage patterns and common workflows for the Multi-Agent Research Platform.

## 🚀 Quick Start Examples

### Basic Agent Creation and Task Execution

```python
from src.agents import AgentFactory, AgentOrchestrator, OrchestrationStrategy

# Initialize factory and orchestrator
factory = AgentFactory()
orchestrator = AgentOrchestrator()

# Create a research agent
research_agent = factory.create_llm_agent(
    role="researcher",
    name="Market Research Specialist"
)

# Execute a simple task
task = "Research the current state of renewable energy adoption globally"
result = await orchestrator.orchestrate_task(
    task=task,
    strategy=OrchestrationStrategy.SINGLE_BEST
)

print(f"Result: {result.primary_result}")
print(f"Success: {result.success}")
print(f"Execution time: {result.execution_time_ms}ms")
```

### Using Multiple Agents with Consensus

```python
# Create a research team
research_team = factory.create_agent_suite(
    suite_type=AgentSuite.RESEARCH_TEAM,
    custom_configs={
        "domain": "technology",
        "specialization": "artificial_intelligence"
    }
)

# Execute complex task with consensus
complex_task = """
Analyze the potential impact of artificial intelligence on employment 
in the next decade. Consider both positive and negative effects, 
provide statistical projections where possible, and suggest policy 
recommendations for workforce adaptation.
"""

result = await orchestrator.orchestrate_task(
    task=complex_task,
    strategy=OrchestrationStrategy.CONSENSUS,
    requirements=["research", "analysis", "writing"],
    context={
        "deadline": "2024-12-31T23:59:59Z",
        "target_audience": "policy_makers",
        "required_depth": "comprehensive"
    }
)

print(f"Consensus Score: {result.consensus_score}")
print(f"Agents Used: {len(result.agents_used)}")
for agent_id, agent_result in result.agent_results.items():
    print(f"{agent_id}: {agent_result.success}")
```

## 📊 Research and Analysis Examples

### Market Research Workflow

```python
async def market_research_workflow(product_category: str, target_market: str):
    """Complete market research workflow example."""
    
    # Step 1: Create specialized agents
    researcher = factory.create_llm_agent(role="researcher", name="Market Researcher")
    analyst = factory.create_llm_agent(role="analyst", name="Data Analyst")
    writer = factory.create_llm_agent(role="writer", name="Report Writer")
    
    # Step 2: Primary research
    research_task = f"""
    Conduct comprehensive market research for {product_category} in {target_market}.
    Include market size, key players, trends, opportunities, and challenges.
    """
    
    research_result = await researcher.execute_task(research_task)
    
    # Step 3: Data analysis
    analysis_task = f"""
    Analyze the following market research data and provide insights:
    
    {research_result.result}
    
    Focus on:
    - Market growth projections
    - Competitive landscape analysis
    - SWOT analysis
    - Investment opportunities
    """
    
    analysis_result = await analyst.execute_task(analysis_task)
    
    # Step 4: Report generation
    report_task = f"""
    Create an executive summary report based on this research and analysis:
    
    Research: {research_result.result}
    Analysis: {analysis_result.result}
    
    Format as a professional business report with clear sections and actionable insights.
    """
    
    report_result = await writer.execute_task(report_task)
    
    return {
        "research": research_result,
        "analysis": analysis_result,
        "report": report_result,
        "summary": {
            "total_execution_time": sum([
                research_result.execution_time_ms,
                analysis_result.execution_time_ms,
                report_result.execution_time_ms
            ]),
            "success_rate": sum([
                research_result.success,
                analysis_result.success,
                report_result.success
            ]) / 3
        }
    }

# Usage
result = await market_research_workflow("electric vehicles", "North America")
print(f"Market Research Report:\n{result['report'].result}")
```

### Academic Research Assistant

```python
async def academic_research_assistant(research_question: str, field: str):
    """Academic research workflow with fact-checking and citation."""
    
    # Create academic research team
    researcher = factory.create_custom_agent("researcher", name="Academic Researcher")
    fact_checker = factory.create_custom_agent("fact_checker", name="Fact Checker")
    synthesizer = factory.create_llm_agent(role="synthesizer", name="Literature Synthesizer")
    
    # Step 1: Literature search and initial research
    research_task = f"""
    Conduct academic research on: {research_question}
    
    Field: {field}
    Requirements:
    - Find peer-reviewed sources
    - Include recent publications (last 5 years preferred)
    - Provide proper citations
    - Identify key researchers and institutions
    """
    
    research_result = await researcher.execute_task(research_task)
    
    # Step 2: Fact verification
    fact_check_task = f"""
    Verify the accuracy of the following research findings:
    
    {research_result.result}
    
    Check:
    - Citation accuracy
    - Statistical claims
    - Methodological soundness
    - Source credibility
    """
    
    fact_check_result = await fact_checker.execute_task(fact_check_task)
    
    # Step 3: Synthesis and meta-analysis
    synthesis_task = f"""
    Synthesize the verified research into a comprehensive analysis:
    
    Original Research: {research_result.result}
    Fact Check Results: {fact_check_result.result}
    
    Provide:
    - Literature review summary
    - Identification of research gaps
    - Methodological comparison
    - Future research directions
    """
    
    synthesis_result = await synthesizer.execute_task(synthesis_task)
    
    return {
        "research_question": research_question,
        "field": field,
        "findings": synthesis_result.result,
        "verification_score": fact_check_result.metadata.get("confidence_score", 0),
        "citations_verified": fact_check_result.metadata.get("citations_checked", 0),
        "quality_metrics": {
            "comprehensiveness": research_result.metadata.get("source_count", 0),
            "credibility": fact_check_result.metadata.get("credibility_score", 0),
            "synthesis_quality": synthesis_result.metadata.get("synthesis_score", 0)
        }
    }

# Usage
result = await academic_research_assistant(
    research_question="What are the effects of remote work on employee productivity and well-being?",
    field="organizational_psychology"
)
```

### Data Analysis Pipeline

```python
async def data_analysis_pipeline(dataset_description: str, analysis_goals: list):
    """Comprehensive data analysis using multiple agents."""
    
    # Create analysis team
    data_analyst = factory.create_custom_agent("data_analyst", name="Statistical Analyst")
    domain_expert = factory.create_custom_agent("domain_expert", name="Domain Expert")
    critic = factory.create_custom_agent("critic", name="Quality Reviewer")
    
    # Step 1: Initial data exploration
    exploration_task = f"""
    Perform exploratory data analysis on: {dataset_description}
    
    Analysis goals: {', '.join(analysis_goals)}
    
    Provide:
    - Data structure overview
    - Statistical summaries
    - Pattern identification
    - Quality assessment
    - Recommended analysis approaches
    """
    
    exploration_result = await data_analyst.execute_task(exploration_task)
    
    # Step 2: Domain-specific insights
    domain_task = f"""
    Review the following data analysis from a domain expertise perspective:
    
    {exploration_result.result}
    
    Provide:
    - Domain-specific interpretations
    - Business/research implications
    - Contextual considerations
    - Additional analysis suggestions
    """
    
    domain_result = await domain_expert.execute_task(domain_task)
    
    # Step 3: Quality review and validation
    review_task = f"""
    Review the following analysis for quality and accuracy:
    
    Statistical Analysis: {exploration_result.result}
    Domain Insights: {domain_result.result}
    
    Evaluate:
    - Methodological soundness
    - Statistical validity
    - Interpretation accuracy
    - Missing considerations
    - Improvement recommendations
    """
    
    review_result = await critic.execute_task(review_task)
    
    return {
        "dataset": dataset_description,
        "goals": analysis_goals,
        "statistical_analysis": exploration_result.result,
        "domain_insights": domain_result.result,
        "quality_review": review_result.result,
        "overall_score": review_result.metadata.get("quality_score", 0)
    }

# Usage
analysis = await data_analysis_pipeline(
    dataset_description="Customer transaction data with demographics, purchase history, and satisfaction scores",
    analysis_goals=[
        "identify customer segments",
        "predict churn probability",
        "optimize pricing strategy",
        "improve customer satisfaction"
    ]
)
```

## 🔄 Workflow Examples

### Sequential Workflow with Dependencies

```python
async def content_creation_pipeline(topic: str, content_type: str, target_audience: str):
    """Sequential content creation workflow."""
    
    # Create workflow agent
    workflow_agent = factory.create_workflow_agent(
        workflow_type="sequential",
        name="Content Creation Workflow"
    )
    
    # Define workflow steps
    workflow_config = {
        "steps": [
            {
                "id": "research",
                "agent_type": "llm",
                "role": "researcher",
                "task": f"Research comprehensive information about {topic} for {target_audience}",
                "dependencies": [],
                "timeout": 60
            },
            {
                "id": "outline",
                "agent_type": "llm", 
                "role": "writer",
                "task": f"Create detailed outline for {content_type} based on research",
                "dependencies": ["research"],
                "timeout": 30
            },
            {
                "id": "content",
                "agent_type": "custom",
                "agent_subtype": "content_creator",
                "task": f"Write engaging {content_type} following the outline",
                "dependencies": ["outline"],
                "timeout": 90
            },
            {
                "id": "review",
                "agent_type": "llm",
                "role": "critic",
                "task": "Review content for quality, accuracy, and engagement",
                "dependencies": ["content"],
                "timeout": 45
            },
            {
                "id": "finalize",
                "agent_type": "llm",
                "role": "writer",
                "task": "Finalize content based on review feedback",
                "dependencies": ["review"],
                "timeout": 60
            }
        ],
        "context": {
            "topic": topic,
            "content_type": content_type,
            "target_audience": target_audience
        }
    }
    
    # Execute workflow
    result = await workflow_agent.execute_workflow(workflow_config)
    
    return {
        "topic": topic,
        "content_type": content_type,
        "target_audience": target_audience,
        "final_content": result.steps["finalize"]["result"],
        "workflow_metrics": {
            "total_time": result.total_execution_time,
            "steps_completed": len(result.completed_steps),
            "success_rate": result.success_rate,
            "quality_score": result.steps["review"]["metadata"].get("quality_score", 0)
        }
    }

# Usage
content = await content_creation_pipeline(
    topic="sustainable business practices",
    content_type="blog_post",
    target_audience="small_business_owners"
)
```

### Parallel Processing with Aggregation

```python
async def competitive_analysis(company_name: str, competitors: list, analysis_dimensions: list):
    """Parallel competitive analysis across multiple dimensions."""
    
    # Create parallel workflow
    workflow_agent = factory.create_workflow_agent(
        workflow_type="parallel",
        name="Competitive Analysis Workflow"
    )
    
    # Generate parallel tasks for each competitor and dimension
    parallel_tasks = []
    for competitor in competitors:
        for dimension in analysis_dimensions:
            parallel_tasks.append({
                "id": f"{competitor}_{dimension}",
                "agent_type": "llm",
                "role": "analyst",
                "task": f"Analyze {competitor} vs {company_name} in terms of {dimension}",
                "timeout": 45
            })
    
    # Add aggregation step
    workflow_config = {
        "parallel_tasks": parallel_tasks,
        "aggregation": {
            "agent_type": "llm",
            "role": "synthesizer",
            "task": "Synthesize all competitive analyses into comprehensive report",
            "timeout": 120
        },
        "context": {
            "company": company_name,
            "competitors": competitors,
            "dimensions": analysis_dimensions
        }
    }
    
    result = await workflow_agent.execute_workflow(workflow_config)
    
    # Process results by competitor and dimension
    analysis_matrix = {}
    for competitor in competitors:
        analysis_matrix[competitor] = {}
        for dimension in analysis_dimensions:
            task_id = f"{competitor}_{dimension}"
            analysis_matrix[competitor][dimension] = result.task_results.get(task_id, {})
    
    return {
        "company": company_name,
        "competitive_matrix": analysis_matrix,
        "synthesis": result.aggregation_result,
        "insights": {
            "strongest_competitor": result.metadata.get("strongest_competitor"),
            "key_differentiators": result.metadata.get("differentiators", []),
            "opportunities": result.metadata.get("opportunities", []),
            "threats": result.metadata.get("threats", [])
        }
    }

# Usage
competitive_analysis_result = await competitive_analysis(
    company_name="TechStartup Inc",
    competitors=["CompetitorA", "CompetitorB", "CompetitorC"],
    analysis_dimensions=["pricing", "features", "market_position", "customer_satisfaction", "innovation"]
)
```

## 🎯 Specialized Use Cases

### Scientific Literature Review

```python
async def scientific_literature_review(topic: str, timeframe: str, methodology_focus: str = None):
    """Automated scientific literature review with quality assessment."""
    
    # Create specialized research team
    researcher = factory.create_custom_agent(
        "researcher", 
        name="Scientific Researcher",
        config={
            "domain": "academic_science",
            "specialization": "literature_review"
        }
    )
    
    fact_checker = factory.create_custom_agent("fact_checker", name="Scientific Fact Checker")
    synthesizer = factory.create_llm_agent(role="synthesizer", name="Literature Synthesizer")
    
    # Step 1: Systematic literature search
    search_task = f"""
    Conduct systematic literature review on: {topic}
    
    Parameters:
    - Timeframe: {timeframe}
    - Focus: {methodology_focus or 'general'}
    
    Requirements:
    - Search multiple academic databases
    - Apply inclusion/exclusion criteria
    - Assess study quality
    - Extract key findings
    - Note methodology limitations
    """
    
    search_result = await researcher.execute_task(search_task)
    
    # Step 2: Quality assessment and bias detection
    quality_task = f"""
    Assess the quality and potential bias in this literature review:
    
    {search_result.result}
    
    Evaluate:
    - Study design quality
    - Sample sizes and power
    - Potential publication bias
    - Methodological consistency
    - Statistical validity
    """
    
    quality_result = await fact_checker.execute_task(quality_task)
    
    # Step 3: Meta-analysis and synthesis
    synthesis_task = f"""
    Synthesize findings from quality-assessed literature:
    
    Literature Review: {search_result.result}
    Quality Assessment: {quality_result.result}
    
    Provide:
    - Comprehensive meta-analysis
    - Strength of evidence assessment
    - Identification of knowledge gaps
    - Clinical/practical implications
    - Future research recommendations
    """
    
    synthesis_result = await synthesizer.execute_task(synthesis_task)
    
    return {
        "topic": topic,
        "timeframe": timeframe,
        "methodology_focus": methodology_focus,
        "literature_count": search_result.metadata.get("studies_included", 0),
        "quality_score": quality_result.metadata.get("overall_quality", 0),
        "synthesis": synthesis_result.result,
        "evidence_strength": synthesis_result.metadata.get("evidence_grade", "Unknown"),
        "recommendations": synthesis_result.metadata.get("recommendations", [])
    }

# Usage
review = await scientific_literature_review(
    topic="effectiveness of mindfulness interventions for anxiety reduction",
    timeframe="2019-2024",
    methodology_focus="randomized_controlled_trials"
)
```

### Business Intelligence Dashboard

```python
async def generate_business_intelligence_report(company_data: dict, metrics: list, time_period: str):
    """Generate comprehensive BI report with visualizations and insights."""
    
    # Create BI analysis team
    data_analyst = factory.create_custom_agent("data_analyst", name="BI Analyst")
    domain_expert = factory.create_custom_agent(
        "domain_expert", 
        name="Business Expert",
        config={"domain": "business_intelligence"}
    )
    writer = factory.create_llm_agent(role="writer", name="Report Writer")
    
    # Step 1: Data analysis and KPI calculation
    analysis_task = f"""
    Analyze business data and calculate KPIs:
    
    Data: {company_data}
    Metrics: {metrics}
    Period: {time_period}
    
    Calculate:
    - Trend analysis
    - Performance indicators
    - Comparative analysis
    - Forecasting
    - Anomaly detection
    """
    
    analysis_result = await data_analyst.execute_task(analysis_task)
    
    # Step 2: Business interpretation
    interpretation_task = f"""
    Interpret business analysis from strategic perspective:
    
    {analysis_result.result}
    
    Provide:
    - Strategic insights
    - Performance assessment
    - Risk identification
    - Opportunity analysis
    - Actionable recommendations
    """
    
    interpretation_result = await domain_expert.execute_task(interpretation_task)
    
    # Step 3: Executive report generation
    report_task = f"""
    Create executive BI report:
    
    Analysis: {analysis_result.result}
    Insights: {interpretation_result.result}
    
    Format:
    - Executive summary
    - Key findings
    - Performance highlights
    - Strategic recommendations
    - Action items with priorities
    """
    
    report_result = await writer.execute_task(report_task)
    
    return {
        "company_data": company_data,
        "time_period": time_period,
        "metrics_analyzed": metrics,
        "kpi_results": analysis_result.metadata.get("kpis", {}),
        "performance_score": interpretation_result.metadata.get("performance_score", 0),
        "executive_report": report_result.result,
        "action_items": interpretation_result.metadata.get("action_items", []),
        "risk_factors": interpretation_result.metadata.get("risks", [])
    }

# Usage
bi_report = await generate_business_intelligence_report(
    company_data={
        "revenue": [100000, 110000, 125000, 130000],
        "expenses": [80000, 85000, 90000, 95000],
        "customers": [500, 550, 600, 650],
        "satisfaction": [4.2, 4.3, 4.1, 4.4]
    },
    metrics=["profitability", "growth_rate", "customer_acquisition", "retention"],
    time_period="Q1-Q4 2024"
)
```

## 🌐 Integration Examples

### REST API Integration

```python
import requests
import asyncio

class PlatformAPIClient:
    """Example client for REST API integration."""
    
    def __init__(self, base_url: str = "http://localhost:8081", api_key: str = None):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["X-API-Key"] = api_key
    
    async def create_research_agent(self, name: str, domain: str = None):
        """Create a specialized research agent."""
        data = {
            "agent_type": "llm",
            "name": name,
            "config": {
                "role": "researcher",
                "domain": domain or "general",
                "temperature": 0.7,
                "max_tokens": 4000
            }
        }
        
        response = requests.post(
            f"{self.base_url}/api/v1/agents",
            json=data,
            headers=self.headers
        )
        
        return response.json()
    
    async def execute_research_task(self, task: str, strategy: str = "adaptive"):
        """Execute a research task using orchestration."""
        data = {
            "task": task,
            "strategy": strategy,
            "priority": "medium",
            "requirements": ["research", "analysis"],
            "timeout_seconds": 120
        }
        
        response = requests.post(
            f"{self.base_url}/api/v1/orchestration/task",
            json=data,
            headers=self.headers
        )
        
        return response.json()
    
    async def get_system_status(self):
        """Get comprehensive system status."""
        response = requests.get(f"{self.base_url}/status", headers=self.headers)
        return response.json()

# Usage example
async def api_integration_example():
    client = PlatformAPIClient()
    
    # Create agent
    agent = await client.create_research_agent(
        name="Market Research Specialist",
        domain="business_analytics"
    )
    print(f"Created agent: {agent['agent_id']}")
    
    # Execute task
    result = await client.execute_research_task(
        task="Analyze the current trends in artificial intelligence adoption in healthcare",
        strategy="consensus"
    )
    
    print(f"Task result: {result['primary_result']}")
    print(f"Execution time: {result['execution_time_ms']}ms")

# Run the example
await api_integration_example()
```

### External Tool Integration

```python
async def integrate_external_tools():
    """Example of integrating external tools with agents."""
    
    # Create agent with external tool access
    analyst = factory.create_custom_agent(
        "data_analyst",
        name="Advanced Data Analyst",
        config={
            "enable_code_execution": True,
            "enable_web_search": True,
            "external_tools": ["pandas", "matplotlib", "requests"]
        }
    )
    
    # Task using external tools
    analysis_task = """
    Perform the following data analysis:
    
    1. Fetch stock price data for AAPL from a financial API
    2. Calculate moving averages (20-day, 50-day)
    3. Generate trend analysis
    4. Create visualization
    5. Provide investment insights
    
    Use appropriate external tools and libraries.
    """
    
    result = await analyst.execute_task(analysis_task)
    
    return {
        "analysis": result.result,
        "tools_used": result.metadata.get("tools_used", []),
        "execution_details": result.metadata.get("execution_trace", [])
    }

# Usage
external_analysis = await integrate_external_tools()
```

## 📈 Performance Optimization Examples

### Caching and Performance

```python
from functools import lru_cache
import asyncio

class OptimizedResearchPlatform:
    """Example of performance-optimized research workflows."""
    
    def __init__(self):
        self.factory = AgentFactory()
        self.orchestrator = AgentOrchestrator()
        self.cache = {}
    
    @lru_cache(maxsize=100)
    async def cached_research(self, query: str, strategy: str):
        """Cache research results for repeated queries."""
        cache_key = f"{query}_{strategy}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        result = await self.orchestrator.orchestrate_task(
            task=query,
            strategy=strategy
        )
        
        self.cache[cache_key] = result
        return result
    
    async def batch_research_tasks(self, tasks: list, max_concurrent: int = 5):
        """Process multiple research tasks concurrently."""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_task(task):
            async with semaphore:
                return await self.orchestrator.orchestrate_task(
                    task=task,
                    strategy=OrchestrationStrategy.ADAPTIVE
                )
        
        # Execute tasks concurrently
        results = await asyncio.gather(*[
            process_task(task) for task in tasks
        ])
        
        return results
    
    async def optimized_workflow(self, complex_query: str):
        """Optimized workflow with intelligent agent selection."""
        
        # Analyze query complexity
        complexity_score = await self.analyze_query_complexity(complex_query)
        
        # Select optimal strategy based on complexity
        if complexity_score < 0.3:
            strategy = OrchestrationStrategy.SINGLE_BEST
        elif complexity_score < 0.7:
            strategy = OrchestrationStrategy.PARALLEL_ALL
        else:
            strategy = OrchestrationStrategy.CONSENSUS
        
        # Use cached result if available
        result = await self.cached_research(complex_query, strategy.value)
        
        return result
    
    async def analyze_query_complexity(self, query: str) -> float:
        """Analyze query complexity to optimize strategy selection."""
        
        # Simple complexity analysis
        factors = {
            "length": len(query.split()) / 100,
            "question_words": sum(1 for word in query.lower().split() 
                                if word in ["what", "how", "why", "analyze", "compare"]) / 10,
            "technical_terms": sum(1 for word in query.split() 
                                 if len(word) > 10) / 20
        }
        
        return min(sum(factors.values()), 1.0)

# Usage
platform = OptimizedResearchPlatform()

# Batch processing
tasks = [
    "What are the benefits of renewable energy?",
    "How does machine learning work?",
    "Analyze the impact of remote work on productivity",
    "Compare different programming languages for web development"
]

batch_results = await platform.batch_research_tasks(tasks, max_concurrent=3)

# Optimized workflow
complex_result = await platform.optimized_workflow(
    "Provide a comprehensive analysis of the socioeconomic implications of artificial intelligence adoption across different industries, including policy recommendations"
)
```

## 🧪 Testing Examples

### Unit Testing Agents

```python
import pytest
from unittest.mock import Mock, AsyncMock

class TestAgentBehavior:
    """Example unit tests for agent functionality."""
    
    @pytest.fixture
    async def mock_agent(self):
        factory = AgentFactory()
        agent = factory.create_llm_agent(role="researcher")
        return agent
    
    @pytest.mark.asyncio
    async def test_agent_task_execution(self, mock_agent):
        """Test basic agent task execution."""
        
        task = "What is artificial intelligence?"
        result = await mock_agent.execute_task(task)
        
        assert result.success
        assert result.result is not None
        assert result.execution_time_ms > 0
        assert len(result.result) > 50  # Reasonable response length
    
    @pytest.mark.asyncio
    async def test_orchestration_strategies(self):
        """Test different orchestration strategies."""
        
        orchestrator = AgentOrchestrator()
        task = "Explain quantum computing"
        
        strategies = [
            OrchestrationStrategy.SINGLE_BEST,
            OrchestrationStrategy.CONSENSUS,
            OrchestrationStrategy.PARALLEL_ALL
        ]
        
        for strategy in strategies:
            result = await orchestrator.orchestrate_task(task, strategy)
            assert result.success
            assert result.strategy_used == strategy
    
    @pytest.mark.asyncio
    async def test_agent_performance_metrics(self, mock_agent):
        """Test agent performance tracking."""
        
        # Execute multiple tasks
        tasks = [
            "What is machine learning?",
            "Explain neural networks",
            "Define artificial intelligence"
        ]
        
        for task in tasks:
            await mock_agent.execute_task(task)
        
        # Check performance metrics
        metrics = mock_agent.get_performance_metrics()
        
        assert metrics["total_tasks"] == 3
        assert metrics["success_rate_percent"] > 0
        assert metrics["average_response_time_ms"] > 0

# Integration Testing
class TestIntegrationWorkflows:
    """Example integration tests for complex workflows."""
    
    @pytest.mark.asyncio
    async def test_research_workflow_integration(self):
        """Test complete research workflow."""
        
        # Setup
        factory = AgentFactory()
        orchestrator = AgentOrchestrator()
        
        # Create research team
        team = factory.create_agent_suite(AgentSuite.RESEARCH_TEAM)
        
        # Execute research task
        task = "Analyze renewable energy market trends"
        result = await orchestrator.orchestrate_task(
            task=task,
            strategy=OrchestrationStrategy.CONSENSUS
        )
        
        # Assertions
        assert result.success
        assert len(result.agents_used) >= 2
        assert result.consensus_score > 0.7
        assert "renewable energy" in result.primary_result.lower()
    
    @pytest.mark.asyncio
    async def test_api_endpoint_integration(self):
        """Test API endpoint integration."""
        
        import httpx
        
        async with httpx.AsyncClient() as client:
            # Test agent creation
            agent_data = {
                "agent_type": "llm",
                "name": "Test Agent",
                "config": {"role": "researcher"}
            }
            
            response = await client.post(
                "http://localhost:8081/api/v1/agents",
                json=agent_data
            )
            
            assert response.status_code == 200
            agent = response.json()
            assert agent["name"] == "Test Agent"
            
            # Test task execution
            task_data = {
                "task": "Test research question",
                "strategy": "single_best"
            }
            
            response = await client.post(
                "http://localhost:8081/api/v1/orchestration/task",
                json=task_data
            )
            
            assert response.status_code == 200
            result = response.json()
            assert result["success"]

# Run tests
pytest.main(["-v", __file__])
```

## 🎯 Advanced Use Cases

### Multi-Language Content Generation

```python
async def multilingual_content_creation(content_brief: str, target_languages: list):
    """Create content in multiple languages with cultural adaptation."""
    
    # Create content team
    creator = factory.create_custom_agent("content_creator", name="Content Creator")
    translator = factory.create_custom_agent("translator", name="Multi-Language Translator")
    cultural_expert = factory.create_custom_agent(
        "domain_expert",
        name="Cultural Expert",
        config={"domain": "cultural_adaptation"}
    )
    
    # Step 1: Create base content
    base_content_task = f"""
    Create engaging content based on this brief:
    
    {content_brief}
    
    Requirements:
    - Clear, engaging writing
    - Cultural neutrality for translation
    - Structured format
    - Call-to-action included
    """
    
    base_content = await creator.execute_task(base_content_task)
    
    # Step 2: Translate and adapt for each language
    localized_content = {}
    
    for language in target_languages:
        # Translation
        translation_task = f"""
        Translate the following content to {language}:
        
        {base_content.result}
        
        Requirements:
        - Natural, fluent translation
        - Preserve meaning and tone
        - Adapt idioms and expressions
        - Maintain formatting
        """
        
        translated = await translator.execute_task(translation_task)
        
        # Cultural adaptation
        adaptation_task = f"""
        Review and adapt this {language} content for cultural appropriateness:
        
        {translated.result}
        
        Consider:
        - Cultural sensitivities
        - Local customs and values
        - Regional preferences
        - Business practices
        """
        
        adapted = await cultural_expert.execute_task(adaptation_task)
        
        localized_content[language] = {
            "translated": translated.result,
            "culturally_adapted": adapted.result,
            "adaptation_notes": adapted.metadata.get("adaptation_notes", [])
        }
    
    return {
        "original_brief": content_brief,
        "base_content": base_content.result,
        "localized_versions": localized_content,
        "languages_covered": target_languages,
        "quality_metrics": {
            "translation_confidence": sum(
                content.get("translation_confidence", 0.8) 
                for content in localized_content.values()
            ) / len(target_languages),
            "cultural_adaptation_score": sum(
                content.get("cultural_score", 0.8) 
                for content in localized_content.values()
            ) / len(target_languages)
        }
    }

# Usage
multilingual_campaign = await multilingual_content_creation(
    content_brief="Create marketing content for a new sustainable fashion brand targeting young professionals",
    target_languages=["Spanish", "French", "German", "Japanese"]
)
```

---

These examples demonstrate the versatility and power of the Multi-Agent Research Platform across various domains and use cases. Each example includes practical code that can be adapted for specific needs and extended with additional functionality.