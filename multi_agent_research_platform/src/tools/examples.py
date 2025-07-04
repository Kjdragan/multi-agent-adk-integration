"""
Examples and Demonstrations of ADK Built-in Tools Integration

Comprehensive examples showing how to use the integrated ADK built-in tools
with the platform's context management patterns, logging, and service architecture.
"""

import asyncio
from typing import Any, Dict, List, Optional

from google.adk.tools.tool_context import ToolContext
from google.adk.agents.invocation_context import InvocationContext

from .factory import ToolFactory, ToolSuite, get_all_tools
from .base import ToolType
from .google_search import SearchQuery
from .code_execution import CodeExecutionConfig, ExecutionSafety, CodeLanguage
from .vertex_search import VertexSearchConfig, SearchType as VertexSearchType
from .bigquery import BigQueryConfig
from ..context import (
    ToolContextPattern,
    MemoryAccessPattern,
    AgentContextPattern,
)
from ..platform_logging import RunLogger
from ..services import SessionService, MemoryService, ArtifactService
from ..config.tools import ToolsConfig


class ToolIntegrationDemo:
    """
    Demonstration class showing comprehensive ADK built-in tools integration.
    
    Shows how to use all tools together with context management patterns,
    cross-tool data flow, and sophisticated orchestration.
    """
    
    def __init__(self,
                 config: Optional[ToolsConfig] = None,
                 logger: Optional[RunLogger] = None,
                 session_service: Optional[SessionService] = None,
                 memory_service: Optional[MemoryService] = None,
                 artifact_service: Optional[ArtifactService] = None):
        
        self.config = config
        self.logger = logger
        self.session_service = session_service
        self.memory_service = memory_service
        self.artifact_service = artifact_service
        
        # Create tool factory
        self.tool_factory = ToolFactory(
            config=config,
            logger=logger,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service,
        )
        
        # Create comprehensive tool suite
        self.tools = self.tool_factory.create_tool_suite(ToolSuite.COMPREHENSIVE)
        
        # Context patterns for orchestration
        self.tool_pattern = ToolContextPattern(
            logger=logger,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service,
        )
        
        self.memory_pattern = MemoryAccessPattern(
            logger=logger,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service,
        )
    
    async def demo_research_workflow(self, 
                                   context: ToolContext, 
                                   research_topic: str) -> Dict[str, Any]:
        """
        Demonstrate a comprehensive research workflow using multiple tools.
        
        Flow: Google Search -> Vertex AI Search -> Code Analysis -> Data Storage
        """
        if self.logger:
            self.logger.info(f"Starting research workflow for: {research_topic}")
        
        workflow_results = {
            "topic": research_topic,
            "search_results": None,
            "semantic_results": None,
            "analysis_code": None,
            "analysis_results": None,
            "stored_data": None,
        }
        
        try:
            # Step 1: Google Search for initial information
            google_search = self.tools[ToolType.GOOGLE_SEARCH]
            search_result = google_search.search_with_memory_integration(
                query=research_topic,
                context=context,
                memory_queries=[f"research on {research_topic}", f"studies about {research_topic}"],
                max_results=10,
                safe_search=True,
            )
            workflow_results["search_results"] = search_result
            
            # Step 2: Vertex AI semantic search for deeper insights
            vertex_search = self.tools[ToolType.VERTEX_SEARCH]
            semantic_result = vertex_search.enterprise_search_with_access_control(
                query=f"academic research {research_topic}",
                context=context,
                user_id="research_user",
                search_type=VertexSearchType.SEMANTIC,
                max_results=5,
            )
            workflow_results["semantic_results"] = semantic_result
            
            # Step 3: Generate analysis code based on findings
            code_executor = self.tools[ToolType.CODE_EXECUTION]
            analysis_code = self._generate_analysis_code(research_topic, search_result, semantic_result)
            
            code_result = code_executor.execute_with_artifact_management(
                code=analysis_code,
                context=context,
                language=CodeLanguage.PYTHON,
                safety_level=ExecutionSafety.MODERATE,
                artifact_patterns=["*.json", "*.csv"],
            )
            workflow_results["analysis_code"] = analysis_code
            workflow_results["analysis_results"] = code_result
            
            # Step 4: Store results in BigQuery for future analysis
            if code_result.get("overall_success"):
                bigquery = self.tools[ToolType.BIGQUERY]
                storage_result = await self._store_research_data(
                    bigquery, context, research_topic, workflow_results
                )
                workflow_results["stored_data"] = storage_result
            
            if self.logger:
                self.logger.info(f"Completed research workflow for: {research_topic}")
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Research workflow failed: {e}")
            workflow_results["error"] = str(e)
        
        return workflow_results
    
    async def demo_development_workflow(self, 
                                      context: ToolContext, 
                                      problem_description: str) -> Dict[str, Any]:
        """
        Demonstrate a development workflow: Research -> Code -> Test -> Analyze.
        
        Flow: Search for solutions -> Generate code -> Execute tests -> Analyze performance
        """
        if self.logger:
            self.logger.info(f"Starting development workflow for: {problem_description}")
        
        workflow_results = {
            "problem": problem_description,
            "research_phase": None,
            "code_generation": None,
            "code_execution": None,
            "performance_analysis": None,
        }
        
        try:
            # Phase 1: Research existing solutions
            google_search = self.tools[ToolType.GOOGLE_SEARCH]
            research_result = google_search.search_with_memory_integration(
                query=f"python solution {problem_description}",
                context=context,
                memory_queries=[f"code for {problem_description}", "python examples"],
                max_results=8,
                site_restrict="github.com",
            )
            workflow_results["research_phase"] = research_result
            
            # Phase 2: Generate solution code based on research
            solution_code = self._generate_solution_code(problem_description, research_result)
            workflow_results["code_generation"] = {"code": solution_code}
            
            # Phase 3: Execute and test the solution
            code_executor = self.tools[ToolType.CODE_EXECUTION]
            execution_result = code_executor.execute_with_artifact_management(
                code=solution_code,
                context=context,
                language=CodeLanguage.PYTHON,
                safety_level=ExecutionSafety.MODERATE,
                timeout_seconds=60,
            )
            workflow_results["code_execution"] = execution_result
            
            # Phase 4: Analyze performance if execution succeeded
            if execution_result.get("overall_success"):
                bigquery = self.tools[ToolType.BIGQUERY]
                performance_analysis = await self._analyze_code_performance(
                    bigquery, context, solution_code, execution_result
                )
                workflow_results["performance_analysis"] = performance_analysis
            
            if self.logger:
                self.logger.info(f"Completed development workflow for: {problem_description}")
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Development workflow failed: {e}")
            workflow_results["error"] = str(e)
        
        return workflow_results
    
    async def demo_analytics_workflow(self, 
                                    context: ToolContext, 
                                    dataset_description: str) -> Dict[str, Any]:
        """
        Demonstrate an analytics workflow: Query -> Analyze -> Visualize -> Report.
        
        Flow: BigQuery data -> Code analysis -> Vertex AI insights -> Final report
        """
        if self.logger:
            self.logger.info(f"Starting analytics workflow for: {dataset_description}")
        
        workflow_results = {
            "dataset": dataset_description,
            "data_query": None,
            "statistical_analysis": None,
            "insights_search": None,
            "final_report": None,
        }
        
        try:
            # Phase 1: Query data from BigQuery
            bigquery = self.tools[ToolType.BIGQUERY]
            query_sql = self._generate_analytics_query(dataset_description)
            
            query_result = bigquery.analyze_query_performance(
                sql=query_sql,
                context=context,
                explain_plan=True,
            )
            workflow_results["data_query"] = query_result
            
            # Phase 2: Statistical analysis with code
            if query_result.get("query_validation", {}).get("syntax_valid"):
                code_executor = self.tools[ToolType.CODE_EXECUTION]
                analysis_code = self._generate_statistical_analysis_code(dataset_description, query_sql)
                
                analysis_result = code_executor.execute_with_artifact_management(
                    code=analysis_code,
                    context=context,
                    language=CodeLanguage.PYTHON,
                    safety_level=ExecutionSafety.MODERATE,
                    artifact_patterns=["*.png", "*.json", "*.csv"],
                )
                workflow_results["statistical_analysis"] = analysis_result
            
            # Phase 3: Search for domain insights
            vertex_search = self.tools[ToolType.VERTEX_SEARCH]
            insights_result = vertex_search.semantic_search(
                query=f"data analysis insights {dataset_description}",
                context=context,
                max_results=5,
                content_type="documents",
            )
            workflow_results["insights_search"] = insights_result
            
            # Phase 4: Generate final report
            final_report = self._generate_analytics_report(workflow_results)
            workflow_results["final_report"] = final_report
            
            if self.logger:
                self.logger.info(f"Completed analytics workflow for: {dataset_description}")
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Analytics workflow failed: {e}")
            workflow_results["error"] = str(e)
        
        return workflow_results
    
    def demo_context_patterns_integration(self, context: ToolContext) -> Dict[str, Any]:
        """
        Demonstrate integration with context management patterns.
        
        Shows how tools work with sophisticated context patterns for
        state management, authentication, and memory access.
        """
        if self.logger:
            self.logger.info("Demonstrating context patterns integration")
        
        demo_results = {
            "tool_context_pattern": None,
            "memory_access_pattern": None,
            "cross_tool_state": None,
        }
        
        try:
            # Demonstrate tool context pattern
            tool_pattern_result = self.tool_pattern.execute(
                context,
                tool_name="google_search",
                auth_config={"type": "google_search_api"},
                memory_query="previous searches",
                required_artifacts=["search_results.json"],
            )
            demo_results["tool_context_pattern"] = tool_pattern_result
            
            # Demonstrate memory access pattern
            memory_pattern_result = self.memory_pattern.execute(
                context,
                search_queries=["tool usage", "search results", "code executions"],
                result_limit=10,
                similarity_threshold=0.6,
            )
            demo_results["memory_access_pattern"] = memory_pattern_result
            
            # Demonstrate cross-tool state management
            cross_tool_state = self._demonstrate_cross_tool_state(context)
            demo_results["cross_tool_state"] = cross_tool_state
            
            if self.logger:
                self.logger.info("Completed context patterns integration demo")
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Context patterns demo failed: {e}")
            demo_results["error"] = str(e)
        
        return demo_results
    
    def _generate_analysis_code(self, 
                              topic: str, 
                              search_results: Dict[str, Any], 
                              semantic_results: Dict[str, Any]) -> str:
        """Generate Python code for analyzing research results."""
        return f'''
import json
import pandas as pd
from collections import Counter

# Research Topic: {topic}
topic = "{topic}"

# Process search results
search_data = {json.dumps(search_results.get("search_results", {}), indent=2)}
semantic_data = {json.dumps(semantic_results.get("search_results", {}), indent=2)}

# Extract key information
search_urls = []
search_titles = []
if "results" in search_data:
    for result in search_data["results"]:
        search_urls.append(result.get("url", ""))
        search_titles.append(result.get("title", ""))

# Analyze domains
from urllib.parse import urlparse
domains = [urlparse(url).netloc for url in search_urls if url]
domain_counts = Counter(domains)

# Create analysis report
analysis_report = {{
    "topic": topic,
    "total_search_results": len(search_titles),
    "unique_domains": len(domain_counts),
    "top_domains": dict(domain_counts.most_common(5)),
    "semantic_results_count": len(semantic_data.get("results", [])),
}}

# Save results
with open("research_analysis.json", "w") as f:
    json.dump(analysis_report, f, indent=2)

print("Research analysis completed:")
print(json.dumps(analysis_report, indent=2))
        '''.strip()
    
    def _generate_solution_code(self, 
                              problem: str, 
                              research_results: Dict[str, Any]) -> str:
        """Generate solution code based on problem and research."""
        return f'''
# Solution for: {problem}
import time
import json

def solve_problem():
    """
    Solution implementation based on research findings.
    """
    print("Starting solution for: {problem}")
    
    # Implementation placeholder
    start_time = time.time()
    
    # Based on research, implement solution logic here
    result = {{
        "problem": "{problem}",
        "solution_approach": "research-based implementation",
        "execution_time": 0.0,
        "success": True,
    }}
    
    # Simulate processing
    time.sleep(0.1)
    
    result["execution_time"] = time.time() - start_time
    
    print(f"Solution completed in {{result['execution_time']:.3f}} seconds")
    return result

# Execute solution
if __name__ == "__main__":
    solution_result = solve_problem()
    
    # Save results
    with open("solution_results.json", "w") as f:
        json.dump(solution_result, f, indent=2)
    
    print("Solution results:", solution_result)
        '''.strip()
    
    def _generate_analytics_query(self, dataset_description: str) -> str:
        """Generate BigQuery SQL for analytics."""
        # This is a placeholder - in real implementation, this would be more sophisticated
        return f'''
-- Analytics query for: {dataset_description}
SELECT 
    DATE(created_at) as date,
    COUNT(*) as record_count,
    AVG(CAST(value AS FLOAT64)) as avg_value,
    MAX(CAST(value AS FLOAT64)) as max_value,
    MIN(CAST(value AS FLOAT64)) as min_value
FROM 
    `project.dataset.table`
WHERE 
    created_at >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
    AND description LIKE '%{dataset_description.lower()}%'
GROUP BY 
    DATE(created_at)
ORDER BY 
    date DESC
LIMIT 100
        '''.strip()
    
    def _generate_statistical_analysis_code(self, 
                                          dataset_description: str, 
                                          query_sql: str) -> str:
        """Generate statistical analysis code."""
        return f'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# Statistical analysis for: {dataset_description}

# Simulate data (in real implementation, this would come from BigQuery results)
dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
data = {{
    'date': dates,
    'record_count': np.random.randint(100, 1000, 30),
    'avg_value': np.random.normal(50, 10, 30),
    'max_value': np.random.normal(80, 15, 30),
    'min_value': np.random.normal(20, 5, 30)
}}

df = pd.DataFrame(data)

# Statistical analysis
stats = {{
    "dataset": "{dataset_description}",
    "total_records": df['record_count'].sum(),
    "mean_avg_value": df['avg_value'].mean(),
    "std_avg_value": df['avg_value'].std(),
    "correlation_count_value": df['record_count'].corr(df['avg_value']),
    "trend_analysis": "stable" if df['avg_value'].std() < 5 else "variable"
}}

# Create visualization
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(df['date'], df['record_count'])
plt.title('Record Count Over Time')
plt.xticks(rotation=45)

plt.subplot(2, 2, 2)
plt.plot(df['date'], df['avg_value'])
plt.title('Average Value Trend')
plt.xticks(rotation=45)

plt.subplot(2, 2, 3)
plt.hist(df['avg_value'], bins=10, alpha=0.7)
plt.title('Average Value Distribution')

plt.subplot(2, 2, 4)
plt.scatter(df['record_count'], df['avg_value'])
plt.xlabel('Record Count')
plt.ylabel('Average Value')
plt.title('Count vs Value Correlation')

plt.tight_layout()
plt.savefig('analytics_visualization.png', dpi=150, bbox_inches='tight')
plt.close()

# Save statistics
with open('statistical_analysis.json', 'w') as f:
    json.dump(stats, f, indent=2)

print("Statistical analysis completed:")
print(json.dumps(stats, indent=2))
        '''.strip()
    
    async def _store_research_data(self, 
                                 bigquery_tool, 
                                 context: ToolContext, 
                                 topic: str, 
                                 results: Dict[str, Any]) -> Dict[str, Any]:
        """Store research workflow results in BigQuery."""
        # This is a placeholder for BigQuery storage
        storage_sql = f'''
CREATE TABLE IF NOT EXISTS research_platform.workflow_results (
    id STRING,
    topic STRING,
    workflow_type STRING,
    created_at TIMESTAMP,
    results JSON
);

INSERT INTO research_platform.workflow_results 
VALUES (
    GENERATE_UUID(),
    "{topic}",
    "research_workflow",
    CURRENT_TIMESTAMP(),
    JSON '{json.dumps({"summary": "research completed", "status": "success"})}'
);
        '''
        
        try:
            storage_result = bigquery_tool.query(storage_sql, context, dry_run=True)
            return {"storage_attempted": True, "result": storage_result.to_dict()}
        except Exception as e:
            return {"storage_attempted": False, "error": str(e)}
    
    async def _analyze_code_performance(self, 
                                      bigquery_tool, 
                                      context: ToolContext, 
                                      code: str, 
                                      execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code performance using BigQuery."""
        # This would integrate with actual performance metrics storage
        analysis_sql = '''
        SELECT 
            execution_time_ms,
            memory_usage_mb,
            success_rate,
            DATE(execution_date) as date
        FROM code_performance.executions 
        WHERE language = 'python'
        ORDER BY execution_date DESC 
        LIMIT 100
        '''
        
        try:
            performance_analysis = bigquery_tool.analyze_query_performance(
                sql=analysis_sql,
                context=context,
                explain_plan=True,
            )
            return performance_analysis
        except Exception as e:
            return {"analysis_attempted": False, "error": str(e)}
    
    def _generate_analytics_report(self, workflow_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final analytics report from workflow results."""
        return {
            "report_type": "analytics_workflow",
            "dataset": workflow_results.get("dataset"),
            "summary": {
                "data_query_success": workflow_results.get("data_query", {}).get("query_validation", {}).get("syntax_valid", False),
                "analysis_completed": workflow_results.get("statistical_analysis", {}).get("overall_success", False),
                "insights_found": len(workflow_results.get("insights_search", [])),
            },
            "recommendations": [
                "Continue monitoring data trends",
                "Implement automated analysis pipeline",
                "Expand dataset coverage",
            ],
            "generated_at": "2024-01-01T00:00:00Z",
        }
    
    def _demonstrate_cross_tool_state(self, context: ToolContext) -> Dict[str, Any]:
        """Demonstrate how state is shared across tool executions."""
        try:
            # Set shared state
            if hasattr(context, 'state'):
                context.state['cross_tool_demo'] = {
                    "search_queries": [],
                    "code_executions": [],
                    "analysis_results": [],
                }
            
            return {
                "state_sharing_enabled": hasattr(context, 'state'),
                "initial_state_set": True,
                "cross_tool_coordination": "active",
            }
        except Exception as e:
            return {"error": str(e)}


# Convenience functions for running demos

async def run_research_demo(context: ToolContext, 
                          topic: str = "machine learning trends") -> Dict[str, Any]:
    """Run the research workflow demonstration."""
    demo = ToolIntegrationDemo()
    return await demo.demo_research_workflow(context, topic)


async def run_development_demo(context: ToolContext, 
                             problem: str = "data processing optimization") -> Dict[str, Any]:
    """Run the development workflow demonstration."""
    demo = ToolIntegrationDemo()
    return await demo.demo_development_workflow(context, problem)


async def run_analytics_demo(context: ToolContext, 
                           dataset: str = "user engagement metrics") -> Dict[str, Any]:
    """Run the analytics workflow demonstration."""
    demo = ToolIntegrationDemo()
    return await demo.demo_analytics_workflow(context, dataset)


def run_context_demo(context: ToolContext) -> Dict[str, Any]:
    """Run the context patterns demonstration."""
    demo = ToolIntegrationDemo()
    return demo.demo_context_patterns_integration(context)


if __name__ == "__main__":
    print("ADK Built-in Tools Integration Examples")
    print("=" * 50)
    print()
    print("This module provides comprehensive examples of:")
    print("1. Google Search + Vertex AI Search research workflows")
    print("2. Code Execution + BigQuery development workflows") 
    print("3. Analytics workflows with cross-tool integration")
    print("4. Context management patterns integration")
    print()
    print("Use the async functions to run demonstrations:")
    print("- run_research_demo(context, topic)")
    print("- run_development_demo(context, problem)")
    print("- run_analytics_demo(context, dataset)")
    print("- run_context_demo(context)")