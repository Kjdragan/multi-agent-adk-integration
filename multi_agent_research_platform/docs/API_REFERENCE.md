# API Reference Documentation

This comprehensive API reference provides detailed documentation for all REST endpoints in the Multi-Agent Research Platform. The API is built with FastAPI and provides automatic OpenAPI documentation at `/docs`.

## 📋 API Overview

### Base Information

- **Base URL**: `http://localhost:8081/api/v1`
- **Protocol**: HTTP/HTTPS
- **Format**: JSON
- **Authentication**: API Key (optional, configurable)
- **Documentation**: Available at `/docs` (Swagger UI) and `/redoc`

### API Versioning

The API uses URL-based versioning:
- **Current Version**: `v1`
- **Full Path**: `/api/v1/{endpoint}`
- **Version Policy**: Backward compatibility maintained within major versions

### Response Format

All API responses follow a consistent structure:

**Success Response**:
```json
{
  "success": true,
  "data": {...},
  "timestamp": "2024-01-15T10:30:00Z",
  "execution_time_ms": 150
}
```

**Error Response**:
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid agent type specified",
    "details": {...}
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## 🤖 Agent Management API

### List All Agents

**Endpoint**: `GET /api/v1/agents`

**Description**: Retrieve all registered agents with their current status and metrics.

**Parameters**: None

**Response**:
```json
[
  {
    "agent_id": "agent_123",
    "name": "Research Specialist",
    "agent_type": "llm",
    "capabilities": ["research", "analysis", "writing"],
    "is_active": true,
    "total_tasks_completed": 15,
    "status": {
      "current_state": "idle",
      "last_task_time": "2024-01-15T10:25:00Z",
      "performance_metrics": {
        "success_rate_percent": 96.7,
        "average_response_time_ms": 1200,
        "total_tasks": 15,
        "failed_tasks": 0
      }
    }
  }
]
```

**Example**:
```bash
curl -X GET "http://localhost:8081/api/v1/agents" \
  -H "Content-Type: application/json"
```

### Get Specific Agent

**Endpoint**: `GET /api/v1/agents/{agent_id}`

**Description**: Retrieve detailed information about a specific agent.

**Parameters**:
- `agent_id` (path): Unique agent identifier

**Response**:
```json
{
  "agent_id": "agent_123",
  "name": "Research Specialist",
  "agent_type": "llm",
  "capabilities": ["research", "analysis", "writing"],
  "is_active": true,
  "total_tasks_completed": 15,
  "configuration": {
    "role": "researcher",
    "model": "gemini-2.5-flash",
    "temperature": 0.7,
    "max_tokens": 4000
  },
  "status": {...},
  "performance_history": [...]
}
```

**Example**:
```bash
curl -X GET "http://localhost:8081/api/v1/agents/agent_123" \
  -H "Content-Type: application/json"
```

**Error Responses**:
- `404`: Agent not found
- `500`: Internal server error

### Create New Agent

**Endpoint**: `POST /api/v1/agents`

**Description**: Create and register a new agent with specified configuration.

**Request Body**:
```json
{
  "agent_type": "llm",
  "name": "Custom Research Agent",
  "config": {
    "role": "researcher",
    "domain": "technology",
    "model": "gemini-2.5-flash",
    "temperature": 0.7,
    "max_tokens": 4000,
    "timeout_seconds": 30
  }
}
```

**Response**:
```json
{
  "agent_id": "agent_456",
  "name": "Custom Research Agent",
  "agent_type": "llm",
  "capabilities": ["research", "analysis"],
  "is_active": true,
  "total_tasks_completed": 0,
  "status": {
    "current_state": "ready",
    "created_at": "2024-01-15T10:30:00Z"
  }
}
```

**Example**:
```bash
curl -X POST "http://localhost:8081/api/v1/agents" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "llm",
    "name": "Research Assistant",
    "config": {
      "role": "researcher",
      "domain": "artificial_intelligence"
    }
  }'
```

**Error Responses**:
- `400`: Invalid agent configuration
- `409`: Agent name already exists
- `500`: Creation failed

### Create Agent Team

**Endpoint**: `POST /api/v1/agents/teams`

**Description**: Create a predefined team of agents optimized for specific use cases.

**Request Body**:
```json
{
  "suite_type": "research_team",
  "custom_configs": {
    "domain": "healthcare",
    "specialization": "medical_research"
  }
}
```

**Response**:
```json
[
  {
    "agent_id": "agent_789",
    "name": "Research Team - Researcher",
    "agent_type": "llm",
    "capabilities": ["research", "analysis"],
    "is_active": true
  },
  {
    "agent_id": "agent_790",
    "name": "Research Team - Analyst", 
    "agent_type": "custom",
    "capabilities": ["data_analysis", "interpretation"],
    "is_active": true
  }
]
```

**Available Suite Types**:
- `research_team`: Research, analysis, and writing agents
- `analysis_team`: Data analysis and fact-checking agents
- `content_team`: Content creation and translation agents
- `development_team`: Code review and technical agents

**Example**:
```bash
curl -X POST "http://localhost:8081/api/v1/agents/teams" \
  -H "Content-Type: application/json" \
  -d '{
    "suite_type": "research_team",
    "custom_configs": {
      "domain": "technology"
    }
  }'
```

### Execute Agent Task

**Endpoint**: `POST /api/v1/agents/{agent_id}/task`

**Description**: Execute a specific task using an individual agent.

**Parameters**:
- `agent_id` (path): Target agent identifier

**Request Body**:
```json
{
  "task": "Research the latest developments in quantum computing",
  "context": {
    "priority": "high",
    "deadline": "2024-01-15T18:00:00Z",
    "previous_research": "..."
  }
}
```

**Response**:
```json
{
  "success": true,
  "result": "Quantum computing has seen significant advancements...",
  "error": null,
  "execution_time_ms": 2340,
  "metadata": {
    "model_used": "gemini-2.5-flash",
    "tokens_used": 1234,
    "confidence_score": 0.92
  }
}
```

**Example**:
```bash
curl -X POST "http://localhost:8081/api/v1/agents/agent_123/task" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Analyze market trends for electric vehicles",
    "context": {
      "focus": "environmental_impact",
      "region": "north_america"
    }
  }'
```

### Agent Control Operations

#### Activate Agent

**Endpoint**: `POST /api/v1/agents/{agent_id}/activate`

**Description**: Activate a deactivated agent.

**Response**:
```json
{
  "success": true,
  "agent_id": "agent_123",
  "status": "activated"
}
```

#### Deactivate Agent

**Endpoint**: `POST /api/v1/agents/{agent_id}/deactivate`

**Description**: Deactivate an active agent.

**Response**:
```json
{
  "success": true,
  "agent_id": "agent_123", 
  "status": "deactivated"
}
```

### Agent Registry Status

**Endpoint**: `GET /api/v1/agents/registry/status`

**Description**: Get overall agent registry statistics.

**Response**:
```json
{
  "total_agents": 8,
  "active_agents": 6,
  "agents_by_type": {
    "llm": 5,
    "custom": 2,
    "workflow": 1
  },
  "agents_by_capability": {
    "research": 6,
    "analysis": 5,
    "writing": 4,
    "translation": 2
  },
  "performance_summary": {
    "total_tasks_completed": 127,
    "average_success_rate": 94.2,
    "average_response_time_ms": 1456
  }
}
```

## 🎯 Orchestration API

### Execute Orchestrated Task

**Endpoint**: `POST /api/v1/orchestration/task`

**Description**: Execute a task using multi-agent orchestration with specified strategy.

**Request Body**:
```json
{
  "task": "Comprehensive analysis of renewable energy market trends",
  "strategy": "consensus",
  "priority": "high",
  "requirements": ["research", "analysis", "data_processing"],
  "context": {
    "region": "global",
    "time_frame": "2020-2024",
    "focus_areas": ["solar", "wind", "hydroelectric"]
  },
  "timeout_seconds": 120
}
```

**Response**:
```json
{
  "success": true,
  "task_id": "task_789",
  "strategy_used": "consensus",
  "agents_used": ["agent_123", "agent_456", "agent_789"],
  "primary_result": "The renewable energy market has experienced...",
  "consensus_score": 0.89,
  "execution_time_ms": 3450,
  "error": null,
  "metadata": {
    "total_agents_considered": 6,
    "agents_filtered_by_capability": 3,
    "orchestration_overhead_ms": 120
  },
  "agent_results": {
    "agent_123": {
      "success": true,
      "result": "Research findings indicate...",
      "execution_time_ms": 2100
    },
    "agent_456": {
      "success": true,
      "result": "Analysis shows...",
      "execution_time_ms": 1890
    }
  }
}
```

**Available Strategies**:
- `adaptive`: Dynamic strategy selection
- `consensus`: Multi-agent collaboration and agreement
- `parallel_all`: All agents work simultaneously
- `single_best`: Best agent selection
- `competitive`: Agent competition for best result
- `iterative`: Multi-round refinement
- `cascade`: Sequential agent chain
- `random`: Random selection (testing)
- `weighted`: Probability-based selection

**Example**:
```bash
curl -X POST "http://localhost:8081/api/v1/orchestration/task" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Market analysis for AI startups",
    "strategy": "parallel_all",
    "priority": "medium",
    "requirements": ["research", "analysis"]
  }'
```

### Get Orchestration Status

**Endpoint**: `GET /api/v1/orchestration/status`

**Description**: Retrieve current orchestration system status and metrics.

**Response**:
```json
{
  "active_tasks": 2,
  "queued_tasks": 1,
  "completed_tasks": 156,
  "total_orchestrated": 159,
  "success_rate": 94.3,
  "available_agents": 6,
  "active_agents": 4,
  "strategy_success_rates": {
    "adaptive": 96.2,
    "consensus": 91.8,
    "parallel_all": 93.5,
    "single_best": 97.1
  },
  "performance_metrics": {
    "average_execution_time_ms": 2340,
    "average_agents_per_task": 2.8,
    "peak_concurrent_tasks": 5
  }
}
```

### List Available Strategies

**Endpoint**: `GET /api/v1/orchestration/strategies`

**Description**: Get all available orchestration strategies with descriptions.

**Response**:
```json
{
  "strategies": [
    {
      "name": "adaptive",
      "description": "Dynamically selects optimal strategy based on task",
      "best_for": ["general_purpose", "unknown_complexity"]
    },
    {
      "name": "consensus", 
      "description": "Multiple agents collaborate to build consensus",
      "best_for": ["important_decisions", "validation_required"]
    }
  ],
  "priorities": ["low", "medium", "high", "urgent"],
  "capabilities": [
    "research", "analysis", "writing", "translation",
    "code_execution", "data_processing"
  ]
}
```

### Get Agent Workload

**Endpoint**: `GET /api/v1/orchestration/workload`

**Description**: Retrieve current agent workload distribution and capacity.

**Response**:
```json
{
  "agents": [
    {
      "agent_id": "agent_123",
      "name": "Research Specialist",
      "current_load": 2,
      "max_capacity": 5,
      "utilization_percent": 40,
      "queue_length": 1,
      "estimated_availability": "2024-01-15T10:35:00Z"
    }
  ],
  "total_capacity": 25,
  "total_utilization": 12,
  "system_utilization_percent": 48,
  "bottlenecks": ["agent_456"]
}
```

## 🔍 Debug API

### Get Debug Status

**Endpoint**: `GET /api/v1/debug/status`

**Description**: Retrieve debug interface status and capabilities.

**Response**:
```json
{
  "is_active": true,
  "step_debugging_enabled": true,
  "agent_inspection_enabled": true,
  "live_logs_enabled": true,
  "performance_profiling_enabled": true,
  "active_debug_sessions": 1,
  "active_breakpoints": 2,
  "step_mode_enabled": false,
  "captured_logs_count": 1567,
  "execution_traces_count": 234
}
```

### Inspect Agent

**Endpoint**: `POST /api/v1/debug/inspect`

**Description**: Perform detailed inspection of agent state and configuration.

**Request Body**:
```json
{
  "agent_id": "agent_123",
  "include_performance": true,
  "include_memory": false
}
```

**Response**:
```json
{
  "agent_id": "agent_123",
  "name": "Research Specialist",
  "type": "llm",
  "status": {
    "current_state": "processing",
    "current_task": "market_research_analysis",
    "start_time": "2024-01-15T10:30:00Z"
  },
  "capabilities": ["research", "analysis", "writing"],
  "is_active": true,
  "total_tasks_completed": 15,
  "performance_metrics": {
    "success_rate_percent": 96.7,
    "average_response_time_ms": 1200,
    "total_execution_time_ms": 18000,
    "memory_usage_mb": 45.6
  },
  "configuration": {
    "model": "gemini-2.5-flash",
    "temperature": 0.7,
    "max_tokens": 4000,
    "timeout_seconds": 30
  }
}
```

### Get Captured Logs

**Endpoint**: `GET /api/v1/debug/logs`

**Description**: Retrieve captured log entries with filtering options.

**Query Parameters**:
- `limit` (int): Maximum number of logs to return (default: 100)
- `level` (string): Filter by log level (DEBUG, INFO, WARNING, ERROR)
- `agent_id` (string): Filter by specific agent
- `start_time` (ISO8601): Start time for log range
- `end_time` (ISO8601): End time for log range

**Response**:
```json
{
  "logs": [
    {
      "timestamp": "2024-01-15T10:30:15.123Z",
      "level": "INFO",
      "message": "Task execution started",
      "agent_id": "agent_123",
      "task_id": "task_456",
      "context": {
        "strategy": "adaptive",
        "priority": "medium"
      }
    }
  ],
  "total_count": 1567,
  "filtered_count": 45
}
```

**Example**:
```bash
curl -X GET "http://localhost:8081/api/v1/debug/logs?level=ERROR&limit=50" \
  -H "Content-Type: application/json"
```

### Breakpoint Management

#### Set Breakpoint

**Endpoint**: `POST /api/v1/debug/breakpoint`

**Description**: Set a debugging breakpoint for an agent.

**Request Body**:
```json
{
  "agent_id": "agent_123",
  "condition": "task_complexity > 0.8"
}
```

**Response**:
```json
{
  "breakpoint_id": "bp_789",
  "agent_id": "agent_123",
  "condition": "task_complexity > 0.8",
  "created_at": "2024-01-15T10:30:00Z"
}
```

#### List Breakpoints

**Endpoint**: `GET /api/v1/debug/breakpoints`

**Response**:
```json
{
  "breakpoints": {
    "bp_789": {
      "agent_id": "agent_123",
      "condition": "task_complexity > 0.8",
      "created_at": "2024-01-15T10:30:00Z",
      "hit_count": 3
    }
  },
  "total_count": 1
}
```

#### Remove Breakpoint

**Endpoint**: `DELETE /api/v1/debug/breakpoint/{breakpoint_id}`

**Response**:
```json
{
  "success": true,
  "breakpoint_id": "bp_789"
}
```

### Performance Data

**Endpoint**: `GET /api/v1/debug/performance`

**Description**: Get detailed performance profiling data.

**Response**:
```json
{
  "performance_data": {
    "total_execution_time_ms": 45678,
    "agent_performance": {
      "agent_123": {
        "total_time_ms": 12345,
        "task_count": 15,
        "average_time_ms": 823
      }
    },
    "bottlenecks": [
      {
        "component": "model_inference",
        "average_time_ms": 1456,
        "percentage_of_total": 45.2
      }
    ]
  },
  "execution_traces": [
    {
      "trace_id": "trace_123",
      "timestamp": "2024-01-15T10:30:00Z",
      "duration_ms": 2340,
      "stages": [
        {
          "name": "task_preprocessing",
          "duration_ms": 45
        },
        {
          "name": "model_inference", 
          "duration_ms": 2100
        }
      ]
    }
  ],
  "total_traces": 234
}
```

### Step Mode Control

**Endpoint**: `POST /api/v1/debug/step-mode/{enabled}`

**Description**: Enable or disable step-by-step debugging mode.

**Parameters**:
- `enabled` (boolean): True to enable, false to disable

**Response**:
```json
{
  "step_mode_enabled": true
}
```

## 📊 Monitoring API

### Get Current Metrics

**Endpoint**: `GET /api/v1/monitoring/metrics/current`

**Description**: Retrieve current real-time system metrics.

**Response**:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "agents": {
    "total": 8,
    "active": 6,
    "utilization_percent": 75
  },
  "tasks": {
    "active": 3,
    "completed_today": 45,
    "success_rate_percent": 94.2
  },
  "performance": {
    "average_response_time_ms": 1456,
    "requests_per_minute": 12,
    "error_rate_percent": 0.8
  },
  "system": {
    "cpu_usage_percent": 34.5,
    "memory_usage_mb": 512,
    "uptime_seconds": 7890
  }
}
```

### Get Performance Metrics

**Endpoint**: `GET /api/v1/monitoring/metrics/performance`

**Description**: Retrieve historical performance metrics within specified time range.

**Query Parameters**:
- `limit` (int): Maximum number of metric points (default: 100)
- `start_time` (ISO8601): Start time for metrics range
- `end_time` (ISO8601): End time for metrics range

**Response**:
```json
{
  "metrics": [
    {
      "timestamp": "2024-01-15T10:30:00Z",
      "avg_response_time_ms": 1456,
      "success_rate_percent": 94.2,
      "active_agents": 6,
      "tasks_per_minute": 8.5,
      "error_rate_percent": 0.8
    }
  ],
  "total_count": 1440,
  "filtered_count": 100,
  "aggregations": {
    "avg_response_time": 1456,
    "max_response_time": 5678,
    "min_response_time": 234
  }
}
```

### Get Active Alerts

**Endpoint**: `GET /api/v1/monitoring/alerts`

**Description**: Retrieve currently active monitoring alerts.

**Response**:
```json
{
  "alerts": {
    "high_response_time": {
      "metric": "avg_response_time_ms",
      "current_value": 5678,
      "threshold": 3000,
      "severity": "warning",
      "triggered_at": "2024-01-15T10:25:00Z",
      "acknowledged": false
    }
  },
  "alert_count": 1,
  "alert_thresholds": {
    "error_rate": 0.1,
    "response_time_ms": 3000,
    "memory_usage_mb": 1024,
    "agent_failure_rate": 0.05
  }
}
```

### Acknowledge Alert

**Endpoint**: `POST /api/v1/monitoring/alerts/acknowledge/{metric_name}`

**Description**: Acknowledge an active alert to suppress notifications.

**Parameters**:
- `metric_name` (path): Name of the metric with active alert

**Response**:
```json
{
  "success": true,
  "metric_name": "high_response_time",
  "acknowledged_at": "2024-01-15T10:30:00Z"
}
```

### Export Metrics

**Endpoint**: `GET /api/v1/monitoring/export`

**Description**: Export metrics data in specified format.

**Query Parameters**:
- `format` (string): Export format (json, csv, excel)
- `metric_type` (string): Type of metrics (performance, usage, error, all)
- `start_time` (ISO8601): Start time for export range
- `end_time` (ISO8601): End time for export range

**Response**:
```json
{
  "data": {
    "performance": [...],
    "usage": [...],
    "error": [...]
  },
  "format": "json",
  "exported_at": "2024-01-15T10:30:00Z",
  "record_count": 1440,
  "download_url": "/api/v1/monitoring/download/metrics_20240115_103000.json"
}
```

## 🌐 System Endpoints

### Health Check

**Endpoint**: `GET /health`

**Description**: Basic health check endpoint for monitoring and load balancers.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "mode": "production"
}
```

### System Status

**Endpoint**: `GET /status`

**Description**: Comprehensive system status including all components.

**Response**:
```json
{
  "web_interface": {
    "is_running": true,
    "uptime_seconds": 7890,
    "connected_clients": 5,
    "total_requests": 1234
  },
  "agents": {
    "total_agents": 8,
    "active_agents": 6,
    "agents_by_type": {...}
  },
  "debug": {
    "is_active": true,
    "active_debug_sessions": 1
  },
  "monitoring": {
    "is_active": true,
    "real_time_updates_enabled": true
  }
}
```

## 🔐 Authentication

### API Key Authentication

When authentication is enabled, include the API key in requests:

**Header Method**:
```bash
curl -X GET "http://localhost:8081/api/v1/agents" \
  -H "X-API-Key: your-api-key-here"
```

**Query Parameter Method**:
```bash
curl -X GET "http://localhost:8081/api/v1/agents?api_key=your-api-key-here"
```

### Rate Limiting

Rate limits apply per API key or IP address:
- **Default Limit**: 100 requests per minute
- **Headers**: Response includes rate limit headers
  - `X-RateLimit-Limit`: Requests allowed per window
  - `X-RateLimit-Remaining`: Requests remaining in window
  - `X-RateLimit-Reset`: UTC timestamp of window reset

**Rate Limit Exceeded Response**:
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Try again in 60 seconds.",
    "retry_after": 60
  }
}
```

## 📝 Error Codes

### Standard HTTP Status Codes

- `200`: Success
- `201`: Created
- `400`: Bad Request - Invalid input
- `401`: Unauthorized - Authentication required
- `403`: Forbidden - Insufficient permissions
- `404`: Not Found - Resource doesn't exist
- `409`: Conflict - Resource already exists
- `422`: Unprocessable Entity - Validation error
- `429`: Too Many Requests - Rate limit exceeded
- `500`: Internal Server Error
- `503`: Service Unavailable

### Custom Error Codes

**Agent Errors**:
- `AGENT_NOT_FOUND`: Specified agent doesn't exist
- `AGENT_INACTIVE`: Agent is not currently active
- `AGENT_BUSY`: Agent is at capacity
- `INVALID_AGENT_CONFIG`: Agent configuration is invalid

**Task Errors**:
- `TASK_TIMEOUT`: Task execution exceeded time limit
- `TASK_FAILED`: Task execution failed
- `INVALID_STRATEGY`: Orchestration strategy not supported
- `NO_CAPABLE_AGENTS`: No agents available with required capabilities

**System Errors**:
- `SERVICE_UNAVAILABLE`: Required service is not available
- `RESOURCE_EXHAUSTED`: System resources at capacity
- `CONFIGURATION_ERROR`: System configuration issue

## 🔧 SDK and Client Libraries

### Python SDK Example

```python
import requests
from datetime import datetime

class MultiAgentAPI:
    def __init__(self, base_url="http://localhost:8081", api_key=None):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["X-API-Key"] = api_key
    
    def list_agents(self):
        response = requests.get(f"{self.base_url}/api/v1/agents", headers=self.headers)
        return response.json()
    
    def create_agent(self, agent_type, name, config=None):
        data = {"agent_type": agent_type, "name": name}
        if config:
            data["config"] = config
        
        response = requests.post(f"{self.base_url}/api/v1/agents", 
                               json=data, headers=self.headers)
        return response.json()
    
    def execute_task(self, task, strategy="adaptive", priority="medium"):
        data = {
            "task": task,
            "strategy": strategy,
            "priority": priority
        }
        response = requests.post(f"{self.base_url}/api/v1/orchestration/task",
                               json=data, headers=self.headers)
        return response.json()

# Usage example
api = MultiAgentAPI()
agents = api.list_agents()
result = api.execute_task("Research renewable energy trends")
```

### JavaScript SDK Example

```javascript
class MultiAgentAPI {
    constructor(baseUrl = 'http://localhost:8081', apiKey = null) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Content-Type': 'application/json'
        };
        if (apiKey) {
            this.headers['X-API-Key'] = apiKey;
        }
    }
    
    async listAgents() {
        const response = await fetch(`${this.baseUrl}/api/v1/agents`, {
            headers: this.headers
        });
        return response.json();
    }
    
    async executeTask(task, strategy = 'adaptive', priority = 'medium') {
        const response = await fetch(`${this.baseUrl}/api/v1/orchestration/task`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({
                task,
                strategy,
                priority
            })
        });
        return response.json();
    }
}

// Usage example
const api = new MultiAgentAPI();
const agents = await api.listAgents();
const result = await api.executeTask('Analyze market trends');
```

## 📚 Interactive Documentation

### Swagger UI

Visit `/docs` for interactive API documentation where you can:
- Browse all available endpoints
- Test API calls directly in the browser
- View request/response schemas
- Download OpenAPI specification

### ReDoc

Visit `/redoc` for alternative documentation with:
- Clean, readable format
- Detailed endpoint descriptions
- Code examples in multiple languages
- Downloadable API specification

---

This comprehensive API reference provides all the information needed to integrate with and extend the Multi-Agent Research Platform programmatically.