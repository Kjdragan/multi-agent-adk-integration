# Quick Start Guide (Current Implementation)

Get up and running with the Multi-Agent Research Platform in minutes! This guide covers the **actual platform capabilities** as they exist today.

## ⚡ 5-Minute Setup

### Step 1: Install and Configure

```bash
# Clone and install
git clone <repository-url>
cd multi-agent-research-platform
uv sync  # Creates .venv automatically

# Configure environment
cp .env.example .env
# Edit .env with your Google API key:
# GOOGLE_API_KEY=your_gemini_api_key_here
```

### Step 2: Get Your Google API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key (free tier available)
3. Add to your `.env` file:
   ```bash
   GOOGLE_API_KEY=your_api_key_here
   GOOGLE_GENAI_USE_VERTEXAI=false
   ```

### Step 3: Launch the Platform

**Option A: Streamlit Interface (Recommended for new users)**
```bash
python src/streamlit/launcher.py
# Access: http://localhost:8501
```

**Option B: Web Debug Interface (For developers)**
```bash
python src/web/launcher.py -e debug
# Access: http://localhost:8081
```

**Option C: Both Interfaces Simultaneously**
```bash
python src/web/launcher.py -e debug &
python src/streamlit/launcher.py -e development &
# Web Debug: http://localhost:8081
# Streamlit: http://localhost:8501
```

🎉 **You're ready!** Skip to [Your First Research Task](#-your-first-research-task) below.

## 🎯 Your First Research Task

### Using the Streamlit Interface (Recommended)

#### 1. Create Your First Agent
1. Open http://localhost:8501
2. In the sidebar, expand **"Agent Management"**
3. Click **"Create New Agent"**
4. Select:
   - **Type**: LLM Agent
   - **Role**: Researcher
   - **Auto-optimize model**: Yes ✅
   - **Enable thinking**: Yes ✅
5. Click **"Create Agent"**
6. ✅ You should see: "Agent created successfully!"

#### 2. Execute Your First Task
1. In the main area, find **"Task Execution"**
2. Enter a research question:
   ```
   What are the main environmental benefits of solar energy compared to fossil fuels?
   ```
3. Select:
   - **Orchestration Strategy**: Adaptive (recommended)
   - **Priority**: Medium
4. Click **"🚀 Execute Task"**
5. Watch the real-time progress indicator
6. View comprehensive results with sources and analysis

#### 3. Try Advanced Features
```
Create a comprehensive analysis of artificial intelligence applications in healthcare, including current uses, benefits, challenges, and future opportunities.
```
- This will trigger **adaptive orchestration** 
- May use **multiple agents** for comprehensive analysis
- Includes **source verification** through MCP servers

### Using the Web Debug Interface (For developers)

#### 1. Check System Health
```bash
curl http://localhost:8081/health
# Should return: {"status": "healthy", "agents": X, "services": "active"}
```

#### 2. Create Agent via API
```bash
curl -X POST http://localhost:8081/api/v1/agents/create \
  -H "Content-Type: application/json" \
  -d '{
    "type": "llm",
    "role": "researcher", 
    "config": {
      "auto_optimize_model": true,
      "enable_thinking": true
    }
  }'
```

#### 3. Execute Task via API
```bash
curl -X POST http://localhost:8081/api/v1/orchestration/execute \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Analyze renewable energy trends in 2024",
    "strategy": "adaptive",
    "priority": "medium"
  }'
```

#### 4. Monitor Real-time Activity
- Open http://localhost:8081/dashboard
- View live agent activity, performance metrics
- Check WebSocket connections for real-time updates

## 🔄 Interface Comparison

### Streamlit Interface (Production)
**Best for**: Researchers, analysts, business users
```bash
python src/streamlit/launcher.py -e production
```
**Features**:
- ✅ Intuitive, user-friendly design
- ✅ Guided workflows and wizards
- ✅ Visual charts and analytics
- ✅ No technical knowledge required
- ✅ Task history and export capabilities

### Web Debug Interface (Development) 
**Best for**: Developers, system administrators
```bash
python src/web/launcher.py -e debug
```
**Features**:
- 🔧 Real-time monitoring dashboards
- 🔧 Agent performance analytics
- 🔧 WebSocket communication testing
- 🔧 API documentation at `/docs`
- 🔧 System health monitoring

## 📝 Example Research Tasks

Try these tasks to explore different platform capabilities:

### Simple Factual Research
```
What is machine learning and how does it differ from traditional programming?
```
**Expected**: Single agent execution, quick response

### Comparative Analysis
```
Compare the advantages and disadvantages of renewable energy sources: solar, wind, and hydroelectric power.
```
**Expected**: Analyst agent with structured comparison

### Comprehensive Research
```
Conduct a comprehensive analysis of the impact of artificial intelligence on the job market, including current trends, affected industries, and future predictions.
```
**Expected**: Multi-agent orchestration (researcher + analyst + synthesizer)

### Technical Deep Dive
```
Explain quantum computing principles, current limitations, and potential applications in cryptography and drug discovery.
```
**Expected**: Specialist agent with technical expertise

### Current Events Research
```
What are the latest developments in space exploration, including recent missions, discoveries, and future plans?
```
**Expected**: MCP server integration for current information

## 🎛️ Orchestration Strategies Explained

The platform automatically selects the best strategy, but you can specify:

### ADAPTIVE (Recommended for most tasks)
- **Auto-selects** the best approach based on task complexity
- **Simple tasks** → Single best agent
- **Complex tasks** → Multiple agents with coordination
- **Critical tasks** → Consensus building

### SINGLE_BEST (For simple questions)
```
What is the capital of France?
```
- Uses the most suitable agent
- Fastest execution
- Good for factual questions

### CONSENSUS (For important decisions)
```
Should our company invest in renewable energy infrastructure?
```
- Multiple agents provide input
- Builds consensus across perspectives
- Provides confidence scores

### PARALLEL_ALL (For comprehensive analysis)
```
Analyze the global economic impact of climate change from multiple perspectives.
```
- All relevant agents work simultaneously
- Comprehensive multi-perspective results
- Aggregated insights

### COMPETITIVE (For creative tasks)
```
Generate innovative marketing strategies for a sustainable fashion brand.
```
- Agents compete for best solution
- Quality-based selection
- Creative optimization

## 📊 Understanding Results

### Result Components

When your task completes, you'll see:

#### 1. **Primary Result**
- Main answer or analysis
- Formatted for easy reading
- Includes key insights and conclusions

#### 2. **Execution Metadata**
```json
{
  "strategy_used": "adaptive",
  "agents_used": ["researcher_agent_123"],
  "execution_time_ms": 2847,
  "success": true,
  "thinking_enabled": true,
  "model_used": "gemini-2.5-flash"
}
```

#### 3. **Performance Metrics**
- ⏱️ **Execution Time**: Task completion time
- 🤖 **Agents Used**: Which agents contributed
- 📊 **Consensus Score**: Agreement level (if applicable)
- ⭐ **Quality Score**: Result quality assessment

#### 4. **Source Information** (when available)
- External sources consulted via MCP servers
- Verification status of claims
- Confidence levels

### Quality Indicators

- **High Consensus Score (>80%)**: Strong agreement between agents
- **Multiple Agent Contributions**: Comprehensive analysis
- **Fast Execution (<3s)**: Efficient processing
- **Source Verification**: External validation

## 🔧 Environment Configurations

### Development Mode
```bash
python src/streamlit/launcher.py -e development
```
**Features**: Enhanced logging, debug info, auto-reload

### Production Mode  
```bash
python src/streamlit/launcher.py -e production
```
**Features**: Optimized performance, rate limiting, security

### Demo Mode
```bash
python src/streamlit/launcher.py -e demo
```
**Features**: Sample data, pre-configured agents, presentation mode

### Custom Configuration
```bash
# Custom port and theme
python src/streamlit/launcher.py -p 8502 --theme dark

# Custom host for network access
python src/streamlit/launcher.py --host 0.0.0.0 -e production
```

## 🎮 Advanced Features

### Agent Teams (Predefined Suites)

Try creating predefined agent teams through the interface:

#### Research Team
- **Researcher** + **Analyst** + **Synthesizer**
- Perfect for comprehensive research projects

#### Content Creation Team  
- **Creative** + **Communicator** + **Critic**
- Ideal for content development and optimization

#### Fact-Checking Team
- **Fact Checker** + **Critic** + **Domain Expert**
- Best for verification and validation tasks

### MCP Server Integration

The platform automatically uses external services for enhanced capabilities:

- **Perplexity**: AI-powered research and analysis
- **Tavily**: Optimized web search
- **Brave Search**: Privacy-focused search
- **Omnisearch**: Multi-source aggregation

### Memory and Context

- **Conversation History**: Agents maintain context across interactions
- **Memory Storage**: Important results stored for future reference
- **Cross-session Memory**: Information persists between sessions

## 📈 Monitoring Your Usage

### Task History
- View all previous tasks and results
- Export data as JSON or text
- Track performance over time

### Analytics Dashboard
- Success rates and execution times
- Agent utilization statistics
- Performance trends and optimization recommendations

### Real-time Monitoring (Web Interface)
- Live agent activity
- Resource usage monitoring
- System health dashboards

## 🛠️ Quick Configuration

### API Key Management
```bash
# Test your API key
python -c "
import google.generativeai as genai
import os
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
print('API key working!' if list(genai.list_models()) else 'API key failed')
"
```

### Service Backend Selection
```bash
# Use different service backends
export SESSION_SERVICE_BACKEND=database    # inmemory, database, vertexai
export MEMORY_SERVICE_BACKEND=database     # inmemory, database, vertexai
export ARTIFACT_SERVICE_BACKEND=local      # inmemory, local, gcs, s3
```

### Performance Tuning
```bash
# Optimize for speed
export PRIORITY_SPEED=true
export MAX_CONCURRENT_AGENTS=3

# Optimize for quality
export PRIORITY_COST=false
export ENABLE_THINKING_BUDGETS=true
```

## 🚨 Quick Troubleshooting

### Platform Won't Start
```bash
# Check virtual environment
which python  # Should point to .venv/bin/python

# Check dependencies
uv sync --force

# Check configuration
python -c "from src.config import get_config; print('Config OK')"
```

### API Errors
```bash
# Verify API key
echo $GOOGLE_API_KEY | head -c 10  # Should start with "AI"

# Test API connection
python -c "
import google.generativeai as genai
import os
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
print('Connected to Gemini API')
"
```

### Port Conflicts
```bash
# Use different ports
python src/streamlit/launcher.py -p 8502
python src/web/launcher.py -p 8082
```

### Slow Responses
- Check internet connection
- Verify API key quotas at [Google AI Console](https://console.cloud.google.com)
- Try simpler tasks first to test basic functionality

## 📚 What's Next?

After completing your first tasks:

1. **Explore Different Agent Roles**: Try specialist, creative, and domain expert agents
2. **Experiment with Orchestration**: Test different strategies for various task types
3. **Advanced Configuration**: Customize models, thinking budgets, and output formats
4. **Team Collaboration**: Create agent teams for complex projects
5. **API Integration**: Use the REST API for programmatic access

### Recommended Learning Path

1. ✅ **Quick Start** (you're here!)
2. 📖 **[Agent Documentation](AGENTS_a.md)** - Understand agent capabilities
3. 🖥️ **[Interface Guides](STREAMLIT_INTERFACE_a.md)** - Master the interfaces  
4. ⚙️ **[Architecture Overview](ARCHITECTURE_a.md)** - System understanding
5. 🔧 **[Advanced Configuration](CONFIGURATION_a.md)** - Customize your setup

## 💡 Pro Tips

- **Start simple** with single questions to understand the system
- **Use adaptive strategy** for most tasks - it's intelligent
- **Monitor the analytics** to understand performance patterns
- **Try different agent roles** to see their specializations
- **Check task history** to learn from previous interactions
- **Enable debug logging** while learning to see what's happening
- **Use both interfaces** - Streamlit for usage, Web for monitoring

---

**Ready to dive deeper?** Check out the [Agent Documentation](AGENTS_a.md) to understand the full capabilities of the platform's sophisticated multi-agent system!