# Quick Start Guide

Get up and running with the Multi-Agent Research Platform in just a few minutes! This guide will walk you through the essential steps to start using the platform for your first research task.

## ⚡ 5-Minute Setup

### Step 1: Install the Platform

```bash
# Clone and install
git clone <repository-url>
cd multi-agent-research-platform
uv sync  # or pip install -r requirements.txt

# Activate environment
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows
```

### Step 2: Configure API Keys

```bash
# Copy environment template
cp .env.example .env

# Edit with your keys (minimum required)
GOOGLE_API_KEY=your_gemini_api_key
OPENWEATHER_API_KEY=your_openweather_key
```

**Get API Keys:**
- **Gemini API**: [Google AI Studio](https://makersuite.google.com/app/apikey) (free)
- **OpenWeather**: [OpenWeatherMap](https://openweathermap.org/api) (free)

### Step 3: Launch the Platform

```bash
# Start the user-friendly interface
python src/streamlit/launcher.py
```

### Step 4: Access the Interface

Open your browser to: **http://localhost:8501**

🎉 **You're ready to go!** Skip to [Your First Research Task](#-your-first-research-task) below.

## 🎯 Your First Research Task

### 1. Create Your Agent Team

In the **Streamlit interface sidebar**:

1. **Expand "Create Agents"** section
2. **Choose "Research Team"** from the dropdown
3. **Click "Create Research Team"**
4. **Wait for confirmation** (✅ Created 3 agents successfully!)

### 2. Execute Your First Task

In the **"Run Research Task"** section:

1. **Enter a research question**, for example:
   ```
   What are the main benefits and challenges of renewable energy adoption?
   ```

2. **Select orchestration strategy**: 
   - Start with **"adaptive"** (recommended for beginners)

3. **Choose priority**: 
   - Use **"medium"** for most tasks

4. **Click "🚀 Execute Task"**

### 3. View Your Results

The platform will:
- Show a progress indicator while agents work
- Display comprehensive results with multiple perspectives
- Provide execution metrics and agent contributions
- Save the task to your history for future reference

## 🔄 Interface Options

### Option A: Streamlit Interface (Recommended for Beginners)

**Best for**: End users, researchers, business analysts

```bash
# Start production interface
python src/streamlit/launcher.py

# Access at: http://localhost:8501
```

**Features:**
- ✅ User-friendly design
- ✅ Guided workflows
- ✅ Visual analytics
- ✅ No technical knowledge required

### Option B: Web Debug Interface (For Advanced Users)

**Best for**: Developers, system administrators, debugging

```bash
# Start debug interface
python src/web/launcher.py -e debug

# Access at: http://localhost:8081
```

**Features:**
- 🔧 Real-time monitoring
- 🔧 Advanced debugging tools
- 🔧 System health dashboards
- 🔧 API documentation at `/docs`

## 📝 Common Research Tasks

Try these example tasks to explore the platform:

### Business Analysis
```
Analyze the competitive landscape for electric vehicle manufacturers
```

### Technology Research
```
Compare the advantages and disadvantages of different cloud computing platforms
```

### Market Research
```
What are the emerging trends in artificial intelligence for healthcare?
```

### Academic Research
```
Summarize the current state of quantum computing research and its potential applications
```

### Content Creation
```
Create an executive summary of the benefits of remote work for organizations
```

## 🎛️ Orchestration Strategies Explained

Choose the right strategy for your task:

| Strategy | Best For | Description |
|----------|----------|-------------|
| **Adaptive** 🎯 | Most tasks | Automatically selects the best approach |
| **Consensus** 🤝 | Important decisions | Multiple agents collaborate and agree |
| **Parallel All** ⚡ | Comprehensive analysis | All agents work simultaneously |
| **Single Best** 🏆 | Simple questions | Uses the most suitable agent |
| **Competitive** 🏁 | Creative tasks | Agents compete, best result wins |

**Recommendation**: Start with **"Adaptive"** - it's smart enough to choose the right approach automatically.

## 📊 Understanding Results

### Result Components

When your task completes, you'll see:

1. **Main Result**: The primary answer or analysis
2. **Execution Metrics**: 
   - ⏱️ **Execution Time**: How long the task took
   - 🤖 **Agents Used**: Which agents contributed
   - 📊 **Consensus Score**: How much agents agreed (if applicable)
   - ⭐ **Strategy Used**: Which orchestration approach was selected

3. **Individual Contributions**: See what each agent contributed
4. **Additional Context**: Supporting information and metadata

### Quality Indicators

- **High Consensus Score (>80%)**: Strong agreement between agents
- **Multiple Agent Contributions**: Comprehensive multi-perspective analysis
- **Fast Execution (<3s)**: Efficient processing
- **Detailed Results**: Thorough and well-researched responses

## 🔧 Quick Configuration

### Environment Modes

The platform supports different modes for different use cases:

```bash
# Development mode (more verbose, demo agents)
python src/streamlit/launcher.py -e development

# Production mode (optimized, secure)
python src/streamlit/launcher.py -e production

# Demo mode (sample data, presentations)
python src/streamlit/launcher.py -e demo
```

### Custom Settings

```bash
# Custom port
python src/streamlit/launcher.py -p 8502

# Custom host (for network access)
python src/streamlit/launcher.py --host 0.0.0.0

# Dark theme
python src/streamlit/launcher.py --theme dark
```

## 📈 Monitoring Your Usage

### View Task History

In the Streamlit interface:
1. Click the **"Task History"** tab
2. See all your previous tasks and results
3. Export your data as JSON if needed

### Analytics Dashboard

In the **"Analytics"** tab, explore:
- 📊 **Success rates** over time
- ⏱️ **Execution time** patterns
- 🤖 **Agent utilization** statistics
- 📈 **Performance trends**

## 🎮 Interactive Features

### Real-time Updates

The platform provides real-time feedback:
- ⚡ Live progress indicators during task execution
- 🔄 Auto-refreshing metrics and dashboards
- 💬 WebSocket connections for instant updates

### Agent Management

- ➕ **Create agents** on-demand for specific needs
- 📊 **Monitor agent performance** and utilization
- ⚙️ **Configure agent behavior** for your use cases

### Export Capabilities

- 📥 **Download results** as JSON or text
- 📊 **Export analytics** data for external analysis
- 🗃️ **Backup task history** for record keeping

## 🚀 Advanced Quick Start

### For Power Users

```bash
# Start both interfaces simultaneously
python src/streamlit/launcher.py -p 8501 &
python src/web/launcher.py -p 8081 &

# Access both:
# Streamlit: http://localhost:8501
# Web Debug: http://localhost:8081
```

### For Developers

```bash
# Development mode with auto-reload
python src/web/launcher.py -e development --reload

# API documentation available at:
# http://localhost:8081/docs
```

### For Demonstrations

```bash
# Demo mode with sample data
python src/streamlit/launcher.py -e demo

# Includes pre-loaded examples and enhanced visuals
```

## 🎯 Best Practices for Beginners

### Writing Effective Research Questions

**Good Examples:**
- ✅ "Compare the environmental impact of solar vs wind energy"
- ✅ "What are the key challenges in implementing AI in healthcare?"
- ✅ "Analyze the pros and cons of remote work for productivity"

**Avoid:**
- ❌ "Tell me about AI" (too broad)
- ❌ "Yes or no: Is solar energy good?" (too narrow)
- ❌ "What's the weather today?" (use weather agent directly)

### Choosing the Right Strategy

- **Start simple**: Use "adaptive" for most tasks
- **Need consensus**: Use "consensus" for important decisions
- **Want speed**: Use "single_best" for quick answers
- **Want depth**: Use "parallel_all" for comprehensive analysis

### Monitoring Performance

- Check the **Analytics tab** regularly to understand usage patterns
- Look for agents with high success rates for your types of tasks
- Monitor execution times to optimize your workflow

## 🛠️ Troubleshooting Quick Fixes

### Common Issues

**Platform won't start:**
```bash
# Check Python version
python --version  # Should be 3.9+

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**API errors:**
```bash
# Verify environment variables
python -c "import os; print('API Key set:', bool(os.getenv('GOOGLE_API_KEY')))"
```

**Port conflicts:**
```bash
# Use different port
python src/streamlit/launcher.py -p 8502
```

**Slow responses:**
- Check your internet connection
- Verify API key quotas/limits
- Try simpler tasks first

## 📚 What's Next?

After completing your first research task:

1. **Explore Different Agent Types**: Try creating custom agents for specific domains
2. **Experiment with Strategies**: Test different orchestration approaches
3. **Dive into Analytics**: Understand your usage patterns and optimize
4. **Read Advanced Guides**: Check out detailed documentation for specific features
5. **Join the Community**: Connect with other users and share experiences

### Recommended Reading Order

1. ✅ **Quick Start** (you're here!)
2. 📖 **[Agent Documentation](AGENTS.md)** - Understand agent capabilities
3. 🖥️ **[Streamlit Interface Guide](STREAMLIT_INTERFACE.md)** - Master the UI
4. ⚙️ **[Configuration Guide](CONFIGURATION.md)** - Customize your setup
5. 🔧 **[Web Interface Guide](WEB_INTERFACE.md)** - Advanced monitoring

## 💡 Pro Tips

- **Start with simple questions** to get familiar with the interface
- **Use the demo mode** to explore features without using API quotas
- **Check the task history** to learn from previous interactions
- **Experiment with different strategies** to see how they affect results
- **Monitor the analytics** to understand which approaches work best for you

---

**Ready to dive deeper?** Check out our [comprehensive documentation](README.md) or explore specific features in the detailed guides!

**Need help?** Visit our [Troubleshooting Guide](TROUBLESHOOTING.md) or open an issue on GitHub.