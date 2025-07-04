"""
BigQuery Built-in Tool Integration

Provides comprehensive integration with ADK's built-in BigQuery capabilities,
including SQL query execution, result processing, and data analysis features.
"""

import time
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from google.adk.tools.tool_context import ToolContext

from .base import (
    BaseTool, 
    BuiltInToolMixin, 
    ToolResult, 
    ToolType, 
    ToolExecutionStatus,
    ToolAuthConfig,
)
from ..platform_logging import RunLogger
from ..services import SessionService, MemoryService, ArtifactService


class QueryType(str, Enum):
    """Types of BigQuery operations."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    CREATE = "create"
    DROP = "drop"
    ALTER = "alter"
    DESCRIBE = "describe"
    EXPLAIN = "explain"


class QueryComplexity(str, Enum):
    """Query complexity levels for safety and monitoring."""
    SIMPLE = "simple"          # Basic SELECT queries
    MODERATE = "moderate"      # Joins, aggregations
    COMPLEX = "complex"        # Complex analytics, multiple CTEs
    ANALYTICAL = "analytical"  # Advanced analytics, ML functions


@dataclass
class BigQueryConfig:
    """Configuration for BigQuery operations."""
    project_id: str
    dataset_id: Optional[str] = None
    location: str = "US"
    max_results: int = 1000
    timeout_seconds: int = 300
    use_cache: bool = True
    use_legacy_sql: bool = False
    dry_run: bool = False
    maximum_bytes_billed: Optional[int] = None
    job_id_prefix: Optional[str] = None
    labels: Optional[Dict[str, str]] = None
    query_parameters: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for BigQuery API."""
        config = {
            "project_id": self.project_id,
            "location": self.location,
            "max_results": self.max_results,
            "timeout": self.timeout_seconds,
            "use_query_cache": self.use_cache,
            "use_legacy_sql": self.use_legacy_sql,
            "dry_run": self.dry_run,
        }
        
        # Add optional parameters
        for attr in ["dataset_id", "maximum_bytes_billed", "job_id_prefix", "labels", "query_parameters"]:
            value = getattr(self, attr)
            if value is not None:
                config[attr] = value
        
        return config


@dataclass
class BigQueryResult:
    """Result of BigQuery operation with comprehensive information."""
    query_hash: str
    query_type: QueryType
    execution_status: str  # "success", "error", "timeout", "cancelled"
    rows: List[Dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    total_bytes_processed: int = 0
    total_bytes_billed: int = 0
    cache_hit: bool = False
    execution_time_ms: float = 0.0
    slot_ms: int = 0
    job_id: Optional[str] = None
    schema: List[Dict[str, Any]] = field(default_factory=list)
    query_plan: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """Check if query execution was successful."""
        return self.execution_status == "success"
    
    @property
    def has_data(self) -> bool:
        """Check if query returned data."""
        return self.row_count > 0
    
    @property
    def cost_estimate_usd(self) -> float:
        """Estimate query cost in USD (simplified calculation)."""
        # Simplified cost calculation: $5 per TB processed
        tb_processed = self.total_bytes_processed / (1024**4)  # Convert to TB
        return tb_processed * 5.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "query_hash": self.query_hash,
            "query_type": self.query_type.value,
            "execution_status": self.execution_status,
            "success": self.success,
            "rows": self.rows,
            "row_count": self.row_count,
            "total_bytes_processed": self.total_bytes_processed,
            "total_bytes_billed": self.total_bytes_billed,
            "cache_hit": self.cache_hit,
            "execution_time_ms": self.execution_time_ms,
            "slot_ms": self.slot_ms,
            "job_id": self.job_id,
            "schema": self.schema,
            "query_plan": self.query_plan,
            "error_message": self.error_message,
            "warnings": self.warnings,
            "has_data": self.has_data,
            "cost_estimate_usd": self.cost_estimate_usd,
            "metadata": self.metadata,
        }


class BigQueryTool(BaseTool, BuiltInToolMixin):
    """
    BigQuery built-in tool integration.
    
    Provides comprehensive BigQuery capabilities including SQL query execution,
    result processing, cost monitoring, and integration with platform services.
    """
    
    def __init__(self,
                 default_config: Optional[BigQueryConfig] = None,
                 project_id: Optional[str] = None,
                 location: Optional[str] = None,
                 logger: Optional[RunLogger] = None,
                 session_service: Optional[SessionService] = None,
                 memory_service: Optional[MemoryService] = None,
                 artifact_service: Optional[ArtifactService] = None,
                 auth_config: Optional[ToolAuthConfig] = None):
        
        super().__init__(
            tool_type=ToolType.BIGQUERY,
            tool_name="bigquery",
            logger=logger,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service,
            auth_config=auth_config,
        )
        
        # BigQuery configuration
        self.project_id = project_id
        self.location = location or "US"
        self.default_config = default_config or BigQueryConfig(
            project_id=project_id or "default-project",
            location=self.location
        )
        
        # Safety and monitoring settings
        self.max_bytes_per_query = 100 * 1024**3  # 100 GB limit
        self.max_execution_time = 600  # 10 minutes
        self.cost_threshold_usd = 100.0  # $100 threshold
        
        # Query complexity detection patterns
        self.complexity_patterns = {
            QueryComplexity.SIMPLE: [
                r"SELECT .+ FROM \w+",
                r"SELECT \* FROM \w+",
            ],
            QueryComplexity.MODERATE: [
                r"JOIN", r"GROUP BY", r"ORDER BY", r"HAVING", 
                r"UNION", r"CASE WHEN", r"WITH \w+"
            ],
            QueryComplexity.COMPLEX: [
                r"WITH .+ AS \(.+\)", r"WINDOW", r"PARTITION BY",
                r"LAG\(", r"LEAD\(", r"ROW_NUMBER\("
            ],
            QueryComplexity.ANALYTICAL: [
                r"ML\.", r"GEOGRAPHY\.", r"ST_", r"APPROX_",
                r"PERCENTILE_", r"CORR\(", r"COVAR"
            ],
        }
    
    def get_tool_config(self) -> Dict[str, Any]:
        """Get BigQuery tool configuration."""
        return {
            "tool_type": "bigquery",
            "built_in": True,
            "requires_auth": True,  # BigQuery always requires authentication
            "project_id": self.project_id,
            "location": self.location,
            "supported_query_types": [qtype.value for qtype in QueryType],
            "complexity_levels": [level.value for level in QueryComplexity],
            "default_config": self.default_config.to_dict(),
            "safety_limits": {
                "max_bytes_per_query": self.max_bytes_per_query,
                "max_execution_time_seconds": self.max_execution_time,
                "cost_threshold_usd": self.cost_threshold_usd,
            },
        }
    
    def execute_tool(self, context: ToolContext, **kwargs) -> ToolResult:
        """Execute BigQuery operation with comprehensive monitoring."""
        start_time = time.time()
        
        # Parse query parameters
        query = kwargs.get("query", "")
        config = self._parse_bigquery_config(**kwargs)
        
        if not query.strip():
            return ToolResult(
                tool_type=self.tool_type,
                status=ToolExecutionStatus.FAILED,
                error="Empty SQL query provided",
            )
        
        # Generate query hash for tracking
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        
        try:
            # Get enhanced context for full capabilities
            enhanced_context = self.get_enhanced_context(context)
            enhanced_context.start_execution()
            
            # Analyze query complexity and safety
            query_analysis = self._analyze_query(query)
            if not query_analysis["safe"]:
                return ToolResult(
                    tool_type=self.tool_type,
                    status=ToolExecutionStatus.FAILED,
                    error=f"Query safety check failed: {', '.join(query_analysis['issues'])}",
                    metadata={
                        "query_hash": query_hash,
                        "safety_issues": query_analysis["issues"],
                    },
                )
            
            # Get ADK built-in BigQuery tool
            builtin_bq = self.get_builtin_tool_instance(context, "bigquery")
            if not builtin_bq:
                # Fallback to context method
                if not hasattr(context, 'query_bigquery'):
                    return ToolResult(
                        tool_type=self.tool_type,
                        status=ToolExecutionStatus.FAILED,
                        error="BigQuery capability not available",
                    )
                builtin_bq = context
            
            # Log query execution
            self.log_tool_usage("execute_query", {
                "query_hash": query_hash,
                "query_type": query_analysis["query_type"].value,
                "complexity": query_analysis["complexity"].value,
                "dry_run": config.dry_run,
                "estimated_bytes": query_analysis.get("estimated_bytes", 0),
            })
            
            # Execute query
            bq_result = self._execute_query_safely(builtin_bq, query, config, query_hash, query_analysis)
            
            # Store results as artifacts if significant data
            if bq_result.row_count > 100 and self.artifact_service:
                self._store_query_results_as_artifacts(bq_result, enhanced_context)
            
            # Store query and results in memory
            if self.memory_service and bq_result.success:
                self._store_query_in_memory(query, bq_result, enhanced_context)
            
            total_execution_time = (time.time() - start_time) * 1000
            
            result = ToolResult(
                tool_type=self.tool_type,
                status=ToolExecutionStatus.COMPLETED,
                data={
                    "query_result": bq_result.to_dict(),
                    "query": query,
                    "config": config.to_dict(),
                    "query_analysis": query_analysis,
                },
                metadata={
                    "query_hash": query_hash,
                    "query_type": query_analysis["query_type"].value,
                    "complexity": query_analysis["complexity"].value,
                    "total_execution_time_ms": total_execution_time,
                    "query_execution_time_ms": bq_result.execution_time_ms,
                    "bytes_processed": bq_result.total_bytes_processed,
                    "cost_estimate_usd": bq_result.cost_estimate_usd,
                    "cache_hit": bq_result.cache_hit,
                },
                execution_time_ms=total_execution_time,
            )
            
            # Complete execution tracking
            enhanced_context.complete_execution(result)
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_msg = f"BigQuery execution failed: {str(e)}"
            
            if self.logger:
                self.logger.error(error_msg, 
                                query_hash=query_hash,
                                execution_time_ms=execution_time)
            
            return ToolResult(
                tool_type=self.tool_type,
                status=ToolExecutionStatus.FAILED,
                error=error_msg,
                execution_time_ms=execution_time,
                metadata={"query_hash": query_hash},
            )
    
    def query(self, 
             sql: str,
             context: ToolContext,
             max_results: int = 1000,
             **query_options) -> BigQueryResult:
        """
        Convenient query method with direct result access.
        
        Args:
            sql: SQL query string
            context: Tool execution context
            max_results: Maximum number of rows to return
            **query_options: Additional query options
            
        Returns:
            BigQuery execution result with comprehensive information
        """
        result = self.execute_with_context(
            context,
            query=sql,
            max_results=max_results,
            **query_options
        )
        
        if result.success and "query_result" in result.tool_result.data:
            result_data = result.tool_result.data["query_result"]
            return BigQueryResult(**result_data)
        
        # Return failed result
        return BigQueryResult(
            query_hash=hashlib.sha256(sql.encode()).hexdigest()[:16],
            query_type=QueryType.SELECT,
            execution_status="error",
            error_message=result.tool_result.error or "Query execution failed",
        )
    
    def analyze_query_performance(self,
                                 sql: str,
                                 context: ToolContext,
                                 explain_plan: bool = True,
                                 **analysis_options) -> Dict[str, Any]:
        """
        Analyze query performance and provide optimization suggestions.
        
        Args:
            sql: SQL query to analyze
            context: Tool execution context
            explain_plan: Whether to include execution plan
            **analysis_options: Additional analysis options
            
        Returns:
            Dictionary with performance analysis and suggestions
        """
        # Run dry run first to get execution plan
        dry_run_result = self.execute_with_context(
            context,
            query=sql,
            dry_run=True,
            **analysis_options
        )
        
        # Run actual query for timing information
        actual_result = self.execute_with_context(
            context,
            query=sql,
            dry_run=False,
            max_results=100,  # Limit for analysis
            **analysis_options
        )
        
        analysis = {
            "query_validation": {
                "syntax_valid": dry_run_result.success,
                "estimated_bytes": dry_run_result.tool_result.data.get("query_result", {}).get("total_bytes_processed", 0),
                "estimated_cost_usd": 0.0,
            },
            "performance_metrics": {},
            "optimization_suggestions": [],
            "execution_plan": None,
        }
        
        if dry_run_result.success:
            dry_run_data = dry_run_result.tool_result.data.get("query_result", {})
            estimated_bytes = dry_run_data.get("total_bytes_processed", 0)
            analysis["query_validation"]["estimated_cost_usd"] = (estimated_bytes / (1024**4)) * 5.0
        
        if actual_result.success:
            actual_data = actual_result.tool_result.data.get("query_result", {})
            analysis["performance_metrics"] = {
                "execution_time_ms": actual_data.get("execution_time_ms", 0),
                "bytes_processed": actual_data.get("total_bytes_processed", 0),
                "bytes_billed": actual_data.get("total_bytes_billed", 0),
                "slot_ms": actual_data.get("slot_ms", 0),
                "cache_hit": actual_data.get("cache_hit", False),
            }
            
            # Generate optimization suggestions
            analysis["optimization_suggestions"] = self._generate_optimization_suggestions(
                sql, actual_data
            )
            
            if explain_plan:
                analysis["execution_plan"] = actual_data.get("query_plan")
        
        return analysis
    
    def _parse_bigquery_config(self, **kwargs) -> BigQueryConfig:
        """Parse and validate BigQuery configuration."""
        project_id = kwargs.get("project_id", self.default_config.project_id)
        
        return BigQueryConfig(
            project_id=project_id,
            dataset_id=kwargs.get("dataset_id", self.default_config.dataset_id),
            location=kwargs.get("location", self.default_config.location),
            max_results=min(kwargs.get("max_results", self.default_config.max_results), 10000),
            timeout_seconds=min(kwargs.get("timeout_seconds", self.default_config.timeout_seconds), self.max_execution_time),
            use_cache=kwargs.get("use_cache", self.default_config.use_cache),
            use_legacy_sql=kwargs.get("use_legacy_sql", self.default_config.use_legacy_sql),
            dry_run=kwargs.get("dry_run", self.default_config.dry_run),
            maximum_bytes_billed=kwargs.get("maximum_bytes_billed", self.default_config.maximum_bytes_billed),
            job_id_prefix=kwargs.get("job_id_prefix", self.default_config.job_id_prefix),
            labels=kwargs.get("labels", self.default_config.labels),
            query_parameters=kwargs.get("query_parameters", self.default_config.query_parameters),
        )
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query for safety, complexity, and performance characteristics."""
        import re
        
        query_upper = query.upper().strip()
        analysis = {
            "safe": True,
            "issues": [],
            "query_type": QueryType.SELECT,
            "complexity": QueryComplexity.SIMPLE,
            "estimated_bytes": 0,
            "table_count": 0,
            "join_count": 0,
            "has_aggregation": False,
        }
        
        # Determine query type
        for qtype in QueryType:
            if query_upper.startswith(qtype.value.upper()):
                analysis["query_type"] = qtype
                break
        
        # Safety checks for dangerous operations
        dangerous_patterns = [
            r"DELETE.*FROM.*WHERE\s+1\s*=\s*1",  # Delete all
            r"DROP\s+TABLE",
            r"TRUNCATE\s+TABLE",
            r"ALTER\s+TABLE.*DROP",
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, query_upper):
                analysis["safe"] = False
                analysis["issues"].append(f"Dangerous operation detected: {pattern}")
        
        # Detect complexity
        for complexity, patterns in self.complexity_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_upper):
                    if complexity.value > analysis["complexity"].value:
                        analysis["complexity"] = complexity
                    break
        
        # Count tables and joins
        table_matches = re.findall(r"FROM\s+(\w+)", query_upper)
        join_matches = re.findall(r"JOIN\s+(\w+)", query_upper)
        
        analysis["table_count"] = len(set(table_matches))
        analysis["join_count"] = len(join_matches)
        
        # Check for aggregation
        aggregation_patterns = [
            r"GROUP BY", r"COUNT\(", r"SUM\(", r"AVG\(", 
            r"MAX\(", r"MIN\(", r"HAVING"
        ]
        analysis["has_aggregation"] = any(
            re.search(pattern, query_upper) 
            for pattern in aggregation_patterns
        )
        
        # Estimate resource usage (simplified)
        if analysis["table_count"] > 5:
            analysis["issues"].append("Query involves many tables (>5)")
        
        if analysis["join_count"] > 3:
            analysis["issues"].append("Query has many joins (>3)")
        
        if len(query) > 10000:  # 10KB query
            analysis["issues"].append("Query is very long (>10KB)")
        
        return analysis
    
    def _execute_query_safely(self, 
                             bq_tool: Any, 
                             query: str, 
                             config: BigQueryConfig,
                             query_hash: str,
                             query_analysis: Dict[str, Any]) -> BigQueryResult:
        """Execute BigQuery query using ADK built-in tool with safety measures."""
        execution_start = time.time()
        
        try:
            # Prepare query parameters
            query_params = config.to_dict()
            query_params["query"] = query
            
            # Execute using ADK built-in tool
            if hasattr(bq_tool, 'query'):
                raw_result = bq_tool.query(**query_params)
            elif hasattr(bq_tool, 'query_bigquery'):
                raw_result = bq_tool.query_bigquery(**query_params)
            elif hasattr(bq_tool, 'execute_query'):
                raw_result = bq_tool.execute_query(**query_params)
            else:
                raise ValueError("No BigQuery execution method available")
            
            execution_time = (time.time() - execution_start) * 1000
            
            # Process result based on BigQuery format
            return self._process_bigquery_result(
                raw_result, query_hash, query_analysis, execution_time
            )
            
        except Exception as e:
            execution_time = (time.time() - execution_start) * 1000
            
            return BigQueryResult(
                query_hash=query_hash,
                query_type=query_analysis["query_type"],
                execution_status="error",
                error_message=str(e),
                execution_time_ms=execution_time,
            )
    
    def _process_bigquery_result(self, 
                               raw_result: Any, 
                               query_hash: str,
                               query_analysis: Dict[str, Any],
                               execution_time: float) -> BigQueryResult:
        """Process raw BigQuery result into platform format."""
        try:
            # Handle different BigQuery result formats
            if hasattr(raw_result, 'to_dataframe'):
                # BigQuery QueryJob result
                rows = raw_result.to_dataframe().to_dict('records') if raw_result.total_rows > 0 else []
                
                return BigQueryResult(
                    query_hash=query_hash,
                    query_type=query_analysis["query_type"],
                    execution_status="success",
                    rows=rows,
                    row_count=raw_result.total_rows or 0,
                    total_bytes_processed=raw_result.total_bytes_processed or 0,
                    total_bytes_billed=raw_result.total_bytes_billed or 0,
                    cache_hit=raw_result.cache_hit,
                    execution_time_ms=execution_time,
                    slot_ms=raw_result.slot_millis or 0,
                    job_id=raw_result.job_id,
                    schema=[{"name": field.name, "type": field.field_type} for field in raw_result.schema] if raw_result.schema else [],
                    metadata={
                        "location": getattr(raw_result, 'location', 'unknown'),
                        "creation_time": str(getattr(raw_result, 'created', '')),
                        "labels": getattr(raw_result, 'labels', {}),
                    }
                )
            
            elif isinstance(raw_result, dict):
                # Dictionary result format
                return BigQueryResult(
                    query_hash=query_hash,
                    query_type=query_analysis["query_type"],
                    execution_status="success" if not raw_result.get("error") else "error",
                    rows=raw_result.get("rows", []),
                    row_count=raw_result.get("row_count", len(raw_result.get("rows", []))),
                    total_bytes_processed=raw_result.get("bytes_processed", 0),
                    total_bytes_billed=raw_result.get("bytes_billed", 0),
                    cache_hit=raw_result.get("cache_hit", False),
                    execution_time_ms=execution_time,
                    slot_ms=raw_result.get("slot_ms", 0),
                    job_id=raw_result.get("job_id"),
                    schema=raw_result.get("schema", []),
                    error_message=raw_result.get("error"),
                    warnings=raw_result.get("warnings", []),
                )
            
            else:
                # Fallback for unknown result types
                return BigQueryResult(
                    query_hash=query_hash,
                    query_type=query_analysis["query_type"],
                    execution_status="success",
                    rows=[{"result": str(raw_result)}] if raw_result is not None else [],
                    row_count=1 if raw_result is not None else 0,
                    execution_time_ms=execution_time,
                )
                
        except Exception as e:
            return BigQueryResult(
                query_hash=query_hash,
                query_type=query_analysis["query_type"],
                execution_status="error",
                error_message=f"Error processing BigQuery result: {str(e)}",
                execution_time_ms=execution_time,
            )
    
    def _generate_optimization_suggestions(self, 
                                         query: str, 
                                         performance_data: Dict[str, Any]) -> List[str]:
        """Generate query optimization suggestions based on performance data."""
        suggestions = []
        
        # Check execution time
        execution_time = performance_data.get("execution_time_ms", 0)
        if execution_time > 30000:  # > 30 seconds
            suggestions.append("Consider optimizing query for better performance (execution time > 30s)")
        
        # Check bytes processed
        bytes_processed = performance_data.get("total_bytes_processed", 0)
        if bytes_processed > 10 * 1024**3:  # > 10 GB
            suggestions.append("Query processes large amount of data - consider adding filters or partitioning")
        
        # Check cache usage
        if not performance_data.get("cache_hit", False):
            suggestions.append("Query did not use cache - consider enabling query caching for repeated queries")
        
        # Query-specific suggestions
        query_upper = query.upper()
        
        if "SELECT *" in query_upper:
            suggestions.append("Avoid SELECT * - specify only needed columns")
        
        if "ORDER BY" in query_upper and "LIMIT" not in query_upper:
            suggestions.append("Consider adding LIMIT when using ORDER BY for large datasets")
        
        if query_upper.count("JOIN") > 3:
            suggestions.append("Multiple JOINs detected - ensure proper indexing and consider query restructuring")
        
        return suggestions
    
    def _store_query_results_as_artifacts(self, 
                                        bq_result: BigQueryResult,
                                        context) -> None:
        """Store large query results as artifacts."""
        try:
            if bq_result.rows:
                import json
                
                # Create artifact content
                artifact_content = {
                    "query_hash": bq_result.query_hash,
                    "execution_time": bq_result.execution_time_ms,
                    "row_count": bq_result.row_count,
                    "schema": bq_result.schema,
                    "rows": bq_result.rows[:1000],  # Limit to first 1000 rows
                }
                
                artifact_name = f"bigquery_result_{bq_result.query_hash}.json"
                
                if hasattr(context, 'save_artifact'):
                    context.save_artifact(artifact_name, json.dumps(artifact_content, indent=2))
                    
                    if self.logger:
                        self.logger.debug(
                            f"Stored BigQuery result artifact: {artifact_name}",
                            query_hash=bq_result.query_hash,
                            row_count=bq_result.row_count,
                        )
                        
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error storing BigQuery result artifacts: {e}")
    
    def _store_query_in_memory(self,
                             query: str,
                             bq_result: BigQueryResult,
                             context) -> None:
        """Store successful query and results in memory for future reference."""
        try:
            # Create summary of results
            result_summary = ""
            if bq_result.rows:
                if bq_result.row_count <= 5:
                    # Show all rows for small results
                    result_summary = f"Results:\n{bq_result.rows}"
                else:
                    # Show first few rows for large results
                    result_summary = f"Results (first 3 of {bq_result.row_count}):\n{bq_result.rows[:3]}"
            
            memory_text = f"""
BigQuery Execution (#{bq_result.query_hash}):
Query Type: {bq_result.query_type.value}
Status: {bq_result.execution_status}

SQL Query:
{query}

Execution Stats:
- Execution Time: {bq_result.execution_time_ms:.2f}ms
- Rows Returned: {bq_result.row_count}
- Bytes Processed: {bq_result.total_bytes_processed:,}
- Cost Estimate: ${bq_result.cost_estimate_usd:.4f}
- Cache Hit: {bq_result.cache_hit}

{result_summary}
""".strip()
            
            if hasattr(context, 'store_memory'):
                context.store_memory(
                    text=memory_text,
                    metadata={
                        "type": "bigquery_execution",
                        "query_hash": bq_result.query_hash,
                        "query_type": bq_result.query_type.value,
                        "execution_status": bq_result.execution_status,
                        "row_count": bq_result.row_count,
                        "execution_time_ms": bq_result.execution_time_ms,
                        "bytes_processed": bq_result.total_bytes_processed,
                        "cost_estimate_usd": bq_result.cost_estimate_usd,
                    }
                )
                
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error storing BigQuery execution in memory: {e}")