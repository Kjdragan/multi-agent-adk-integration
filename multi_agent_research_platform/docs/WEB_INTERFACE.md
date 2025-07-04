# Web Debug Interface Documentation

The Web Debug Interface is a comprehensive monitoring and debugging platform designed for developers, system administrators, and power users. Built with FastAPI and featuring real-time capabilities, it provides deep insights into the multi-agent system's operation.

## 🌐 Overview

The Web Debug Interface serves as the technical command center for the Multi-Agent Research Platform, offering:

- **Real-time Monitoring**: Live dashboards with WebSocket updates
- **Debugging Tools**: Step-by-step execution analysis
- **System Health**: Comprehensive component monitoring
- **API Access**: Full REST API with OpenAPI documentation
- **Performance Analytics**: Detailed metrics and profiling

## 🚀 Getting Started

### Quick Launch

```bash
# Debug mode (recommended for development)
python src/web/launcher.py -e debug

# Production web interface
python src/web/launcher.py -e production

# Development with auto-reload
python src/web/launcher.py -e development --reload
```

### Access Points

- **Main Interface**: http://localhost:8081
- **API Documentation**: http://localhost:8081/docs
- **Interactive API**: http://localhost:8081/redoc
- **Health Check**: http://localhost:8081/health
- **System Status**: http://localhost:8081/status

## 🎛️ Interface Components

### 1. Dashboard Navigation

The interface features a modern sidebar navigation with the following sections:

#### 📊 Overview Dashboard
- System metrics and health indicators
- Recent activity timeline
- Performance summaries
- Quick statistics

#### 🤖 Agent Management
- Live agent status and monitoring
- Agent creation and configuration
- Performance metrics per agent
- Capability matrix visualization

#### 📝 Task Orchestration
- Active task monitoring
- Orchestration strategy analytics
- Task completion rates
- Execution flow visualization

#### 📈 Performance Monitoring
- Real-time system metrics
- Performance trends and charts
- Resource utilization monitoring
- Alert management

#### 🔧 Debug Console
- Step-by-step debugging
- Log analysis and filtering
- Breakpoint management
- Execution traces

#### 📋 Live Logs
- Real-time log streaming
- Advanced filtering options
- Log export capabilities
- Error highlighting

### 2. Dashboard Features

#### Overview Dashboard

**System Health Cards**
```
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│   Total Agents  │  Active Tasks   │ Avg Performance │ System Uptime   │
│       8         │       3         │     94.2%       │    2h 34m       │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

**Recent Activity Timeline**
- Task executions with timestamps
- Agent status changes
- System events and alerts
- Performance milestones

**Agent Distribution Chart**
- Visual breakdown of agent types
- Active vs inactive agents
- Capability distribution
- Load balancing visualization

#### Agent Management Dashboard

**Active Agents List**
```
Agent Name          | Type      | Status | Tasks | Success Rate | Last Active
Research Specialist | LLM       | Active |   15  |    96.7%     | 2 min ago
Data Analyst       | Custom    | Active |   12  |    91.3%     | 5 min ago
Content Creator    | LLM       | Active |    8  |    100%      | 1 min ago
```

**Agent Performance Charts**
- Success rate trends over time
- Response time analysis
- Task volume per agent
- Error rate monitoring

**Capabilities Matrix**
- Heatmap of agent capabilities
- Skill overlap analysis
- Coverage gap identification
- Optimization recommendations

#### Task Orchestration Dashboard

**Active Tasks Monitor**
```
Task ID | Description           | Strategy | Progress | Assigned Agents | ETA
tk_001  | Market Research      | Adaptive |   65%    | 3 agents       | 2m
tk_002  | Content Analysis     | Parallel |   90%    | 2 agents       | 30s
tk_003  | Code Review         | Single   |   25%    | 1 agent        | 5m
```

**Strategy Performance Analysis**
- Success rates by strategy
- Execution time comparisons
- Agent utilization patterns
- Optimization insights

**Orchestration Flow Diagram**
- Visual representation of task flow
- Agent interaction patterns
- Bottleneck identification
- Process optimization

#### Performance Monitoring Dashboard

**System Metrics Grid**
```
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│   CPU Usage     │  Memory Usage   │  Response Time  │   Error Rate    │
│     45.2%       │     67.8%       │     1.2s        │     0.3%        │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

**Performance Trends**
- Real-time charts with multiple metrics
- Historical performance data
- Trend analysis and forecasting
- Performance baseline comparisons

**Resource Utilization**
- Agent capacity monitoring
- System resource usage
- Load distribution analysis
- Scaling recommendations

**Active Alerts**
- Performance threshold alerts
- System health warnings
- Error rate notifications
- Capacity planning alerts

## 🔧 Configuration

### Environment Modes

#### Debug Mode
```bash
python src/web/launcher.py -e debug
```
**Features**:
- Detailed logging and error reporting
- Step-by-step debugging capabilities
- Real-time monitoring enabled
- All debugging tools available

#### Development Mode
```bash
python src/web/launcher.py -e development --reload
```
**Features**:
- Auto-reload on file changes
- Development tools enabled
- Relaxed security settings
- Extended logging

#### Production Mode
```bash
python src/web/launcher.py -e production
```
**Features**:
- Optimized performance
- Security hardening
- Rate limiting enabled
- Minimal logging

### Customization Options

#### Host and Port Configuration
```bash
# Custom host and port
python src/web/launcher.py --host 0.0.0.0 --port 8082

# Network accessible
python src/web/launcher.py --host 0.0.0.0 --port 8081
```

#### Feature Toggles
```python
# In web configuration
WebConfig(
    enable_real_time_updates=True,
    enable_debugging_tools=True,
    enable_performance_profiling=True,
    enable_websocket_monitoring=True
)
```

## 📡 Real-time Features

### WebSocket Connections

The interface uses WebSockets for real-time updates:

**Connection Endpoint**: `ws://localhost:8081/ws`

**Supported Events**:
- `agent_status_update`: Agent state changes
- `task_progress_update`: Task execution progress
- `orchestration_event`: Orchestration activities
- `performance_metric_update`: System metrics
- `error_notification`: Error and warning events
- `log_entry`: Real-time log streaming

### Live Monitoring

**Agent Status Monitoring**
```javascript
// Real-time agent status updates
{
  "type": "agent_status_update",
  "data": {
    "agent_id": "agent_123",
    "status": "active",
    "current_task": "research_analysis",
    "performance_metrics": {
      "success_rate": 96.7,
      "avg_response_time": 1200
    }
  }
}
```

**Task Progress Tracking**
```javascript
// Live task progress updates
{
  "type": "task_progress_update", 
  "data": {
    "task_id": "task_456",
    "progress": 65,
    "current_stage": "data_analysis",
    "estimated_completion": "2024-01-15T10:30:00Z"
  }
}
```

## 🔍 Debugging Tools

### Step-by-Step Debugging

**Enable Debug Mode**
```python
# Set debugging breakpoints
POST /api/v1/debug/breakpoint
{
  "agent_id": "agent_123",
  "condition": "task_type == 'research'"
}
```

**Debug Session Management**
- Set conditional breakpoints
- Step through agent execution
- Inspect agent state and memory
- Analyze decision-making process

### Log Analysis

**Advanced Log Filtering**
- Filter by log level (DEBUG, INFO, WARNING, ERROR)
- Search by agent ID or task ID
- Time range filtering
- Pattern matching and regex support

**Log Export**
```bash
# Export filtered logs
GET /api/v1/debug/logs?level=ERROR&agent_id=agent_123&format=json
```

### Performance Profiling

**Execution Traces**
- Detailed execution timing
- Method-level profiling
- Resource usage tracking
- Bottleneck identification

**Performance Metrics**
```python
# Get detailed performance data
GET /api/v1/debug/performance
{
  "execution_traces": [...],
  "performance_data": {
    "avg_execution_time": 1234,
    "memory_usage": 256,
    "cpu_utilization": 45.2
  }
}
```

## 🔌 REST API

### Agent Management Endpoints

#### List All Agents
```http
GET /api/v1/agents
```
**Response**:
```json
[
  {
    "agent_id": "agent_123",
    "name": "Research Specialist",
    "agent_type": "llm",
    "capabilities": ["research", "analysis"],
    "is_active": true,
    "total_tasks_completed": 15,
    "status": {...}
  }
]
```

#### Create Agent
```http
POST /api/v1/agents
Content-Type: application/json

{
  "agent_type": "llm",
  "name": "Custom Researcher",
  "config": {
    "role": "researcher",
    "domain": "technology"
  }
}
```

#### Agent Task Execution
```http
POST /api/v1/agents/{agent_id}/task
Content-Type: application/json

{
  "task": "Research the latest AI developments",
  "context": {
    "priority": "high",
    "deadline": "2024-01-15T18:00:00Z"
  }
}
```

### Orchestration Endpoints

#### Execute Orchestrated Task
```http
POST /api/v1/orchestration/task
Content-Type: application/json

{
  "task": "Comprehensive market analysis",
  "strategy": "consensus",
  "priority": "high",
  "requirements": ["research", "analysis"],
  "context": {...}
}
```

#### Get Orchestration Status
```http
GET /api/v1/orchestration/status
```

#### List Available Strategies
```http
GET /api/v1/orchestration/strategies
```

### Monitoring Endpoints

#### Get Current Metrics
```http
GET /api/v1/monitoring/metrics/current
```

#### Get Performance History
```http
GET /api/v1/monitoring/metrics/performance?limit=100&start_time=2024-01-15T00:00:00Z
```

#### Export Metrics
```http
GET /api/v1/monitoring/export?format=json&metric_type=performance
```

### Debug Endpoints

#### Get Debug Status
```http
GET /api/v1/debug/status
```

#### Inspect Agent
```http
POST /api/v1/debug/inspect
Content-Type: application/json

{
  "agent_id": "agent_123",
  "include_performance": true,
  "include_memory": false
}
```

#### Manage Breakpoints
```http
# Set breakpoint
POST /api/v1/debug/breakpoint
{
  "agent_id": "agent_123",
  "condition": "task_complexity > 0.8"
}

# List breakpoints
GET /api/v1/debug/breakpoints

# Remove breakpoint
DELETE /api/v1/debug/breakpoint/{breakpoint_id}
```

## 📊 Analytics and Reporting

### Performance Analytics

**Key Metrics Tracked**:
- Task execution times
- Agent success rates
- System resource utilization
- Error rates and patterns
- User interaction patterns

**Visualization Features**:
- Interactive charts with Plotly
- Real-time data updates
- Historical trend analysis
- Comparative performance views

### Export Capabilities

**Data Export Formats**:
- JSON for programmatic access
- CSV for spreadsheet analysis
- PDF reports for documentation
- Real-time streaming for monitoring tools

**Export Examples**:
```bash
# Export agent performance data
curl "http://localhost:8081/api/v1/monitoring/export?format=csv&metric_type=agent_performance"

# Export system logs
curl "http://localhost:8081/api/v1/debug/logs?format=json&level=ERROR&start_time=2024-01-15T00:00:00Z"
```

## 🎨 Customization

### UI Themes

The interface supports multiple themes:
- **Light Theme**: Clean, professional appearance
- **Dark Theme**: Reduced eye strain for extended use
- **High Contrast**: Accessibility-focused design

### Dashboard Layout

**Customizable Elements**:
- Widget arrangement and sizing
- Chart types and configurations
- Refresh intervals and update frequencies
- Alert thresholds and notifications

### Branding

**Customization Options**:
- Logo and company branding
- Color scheme adaptation
- Custom CSS styling
- White-label configurations

## 🛡️ Security Features

### Authentication

**Access Control**:
- API key authentication
- Session-based access control
- Role-based permissions
- IP-based restrictions

### Security Headers

**Implemented Protections**:
- CORS configuration
- CSRF protection
- Content security policies
- Rate limiting

### Audit Logging

**Tracked Activities**:
- User authentication events
- API access patterns
- Configuration changes
- System modifications

## 🔧 Troubleshooting

### Common Issues

#### Interface Won't Load
```bash
# Check if server is running
curl http://localhost:8081/health

# Verify port availability
netstat -an | grep 8081

# Check logs for errors
python src/web/launcher.py -e debug
```

#### WebSocket Connection Issues
```bash
# Test WebSocket connectivity
wscat -c ws://localhost:8081/ws

# Check browser console for errors
# Verify firewall settings
```

#### Performance Issues
```bash
# Monitor resource usage
top -p $(pgrep -f "web/launcher")

# Check system metrics
GET /api/v1/monitoring/metrics/current

# Optimize configuration
# Reduce update frequencies
# Disable unnecessary features
```

### Debug Mode Troubleshooting

**Enable Verbose Logging**:
```python
# Set debug log level
LOG_LEVEL=DEBUG python src/web/launcher.py -e debug
```

**Check Component Health**:
```bash
# System status endpoint
curl http://localhost:8081/status

# Individual component health
curl http://localhost:8081/api/v1/debug/status
```

## 🎯 Best Practices

### Performance Optimization

1. **Efficient Monitoring**:
   - Set appropriate refresh intervals
   - Use WebSocket for real-time data
   - Implement client-side caching

2. **Resource Management**:
   - Monitor system resource usage
   - Implement connection pooling
   - Use compression for large datasets

3. **User Experience**:
   - Progressive data loading
   - Responsive design patterns
   - Accessibility considerations

### Development Workflow

1. **Development Mode**:
   - Use auto-reload for faster iteration
   - Enable detailed logging
   - Test with sample data

2. **Testing**:
   - Validate API endpoints
   - Test WebSocket connections
   - Verify cross-browser compatibility

3. **Production Deployment**:
   - Enable security features
   - Configure rate limiting
   - Set up monitoring alerts

---

The Web Debug Interface provides a powerful platform for monitoring, debugging, and optimizing the Multi-Agent Research Platform. Its comprehensive feature set makes it an essential tool for developers and system administrators working with the platform.