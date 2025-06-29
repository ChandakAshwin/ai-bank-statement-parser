# Bank Statement Parser AI Agent

An intelligent AI agent that can parse bank statements (PDFs or Excel files) from different banks and extract structured data with advanced learning capabilities.

## 🚀 AI Agent Features

This project has been transformed into a true AI agent with the following capabilities:

### 1. **AI Reasoning Layer**
- Uses LLM (GPT-4/GPT-3.5) to analyze document structure and determine optimal parsing strategy
- Identifies bank formats and applies specialized handling
- Provides confidence scores and recommendations
- Validates parsing results for consistency and completeness

### 2. **Memory System**
- Tracks past documents using vector database (ChromaDB)
- Learns from successful parsing patterns
- Provides recommendations based on similar documents
- Improves accuracy over time through experience

### 3. **Autonomous Behavior**
- Monitors email for new bank statements
- Integrates with cloud storage (Google Drive, Dropbox)
- Automatically processes and analyzes statements
- Generates insights and trend analysis
- Runs background monitoring with configurable intervals

### 4. **Feedback Loop & Learning**
- Learns from user corrections to improve future extractions
- Uses machine learning to predict and prevent common errors
- Tracks performance metrics and provides improvement recommendations
- Continuously adapts parsing strategies based on feedback

## 📋 Features

- **Multi-format Support**: PDF (native and scanned), Excel files
- **Bank Agnostic**: Works with various bank statement formats
- **Intelligent Parsing**: AI-driven strategy selection
- **Learning Capabilities**: Improves accuracy over time
- **Autonomous Operation**: Can run independently
- **Comprehensive Analysis**: Transaction categorization, trend analysis, anomaly detection

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd bank-statement-parser-master
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables** (optional, for AI features):
```bash
export OPENAI_API_KEY="your-openai-api-key"
export EMAIL_ADDRESS="your-email@gmail.com"
export EMAIL_PASSWORD="your-app-password"
export GOOGLE_DRIVE_FOLDER_ID="your-folder-id"
```

## 🎯 Usage

### Basic Usage
```bash
python main.py --file "path/to/statement.pdf"
```

### With AI Insights
```bash
python main.py --file "path/to/statement.pdf" --insights
```

### Autonomous Mode
```bash
python main.py --file "path/to/statement.pdf" --autonomous
```

### Provide Feedback for Learning
```bash
python main.py --file "path/to/statement.pdf" --feedback "feedback.json"
```

### Disable AI Agent (Legacy Mode)
```bash
python main.py --file "path/to/statement.pdf" --no-ai-agent
```

## 📊 AI Agent Capabilities

### Document Analysis
The AI agent analyzes each document to:
- Identify the bank and statement format
- Determine the best parsing strategy
- Predict potential challenges
- Provide confidence scores

### Learning and Memory
- **Vector Database**: Stores document embeddings for similarity search
- **Pattern Recognition**: Learns successful parsing patterns
- **Recommendation Engine**: Suggests strategies based on similar documents
- **Performance Tracking**: Monitors accuracy improvements over time

### Autonomous Features
- **Email Monitoring**: Automatically checks for new bank statements
- **Cloud Integration**: Syncs with Google Drive and Dropbox
- **Background Processing**: Runs continuous monitoring
- **Insight Generation**: Creates trend analysis and spending patterns

### Feedback Learning
- **User Corrections**: Learns from manual corrections
- **Error Prediction**: Identifies potential issues before they occur
- **Strategy Optimization**: Continuously improves parsing strategies
- **Performance Metrics**: Tracks learning progress and accuracy

## 🔧 Configuration

### AI Agent Settings (`config.py`)
```python
AI_AGENT_CONFIG = {
    'openai_api_key': 'your-api-key',
    'openai_model': 'gpt-4-turbo-preview',
    'temperature': 0.1,
    'memory_enabled': True,
    'autonomous_mode': False,
    'learning_enabled': True,
    'feedback_loop_enabled': True,
    'confidence_threshold': 0.8
}
```

### Autonomous Behavior
```python
AUTONOMOUS_CONFIG = {
    'email_monitoring': {
        'enabled': False,
        'check_interval': 300,  # 5 minutes
        'keywords': ['bank statement', 'account statement']
    },
    'cloud_integration': {
        'enabled': False,
        'sync_interval': 600  # 10 minutes
    },
    'auto_analysis': {
        'enabled': True,
        'trend_analysis': True,
        'anomaly_detection': True
    }
}
```

## 📈 Performance Metrics

The AI agent tracks various metrics:
- **Confidence Scores**: Per-document parsing confidence
- **Learning Progress**: Number of corrections and accuracy improvements
- **Success Rates**: Bank-specific and overall success rates
- **Processing Speed**: Time taken for different document types
- **Error Patterns**: Common issues and their resolutions

## 🔍 Example Output

### Basic Parsing
```
=== PARSING RESULTS ===
Parsed 45 transactions.
Confidence Score: 0.87
Closing Balance: ₹125,450.00
```

### With AI Insights
```
=== AI INSIGHTS ===
AI Agent Status: Enabled
Bank Identified: HDFC Bank
Parsing Strategy: table_extraction
Memory Suggestion: standard
Learning Progress: 12 corrections recorded
Accuracy Improvement: 0.15
Total Files Processed: 25
Average Confidence: 0.82

Recommendations:
  - Consider reviewing parsing strategies for better accuracy
  - More user feedback needed for better learning
```

## 🚨 Troubleshooting

### Common Issues

1. **Low Confidence Scores**
   - Enable AI insights for detailed analysis
   - Provide feedback to improve learning
   - Check document quality and format

2. **AI Features Not Working**
   - Verify OpenAI API key is set
   - Check internet connectivity
   - Ensure all dependencies are installed

3. **Memory Issues**
   - Check disk space for vector database
   - Verify ChromaDB installation
   - Clear memory cache if needed

### Performance Optimization

1. **For Large Documents**
   - Increase memory allocation
   - Use batch processing
   - Enable parallel processing

2. **For Better Accuracy**
   - Provide regular feedback
   - Use high-quality document scans
   - Enable all AI features

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models
- ChromaDB for vector database
- Sentence Transformers for embeddings
- All contributors and users

---

**Note**: This AI agent continuously learns and improves. The more you use it and provide feedback, the better it becomes at parsing your specific bank statements! 