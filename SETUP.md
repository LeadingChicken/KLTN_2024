# Setup Guide for LangGraph Fact-Checking System

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (optional, for version control)

## Installation Steps

### 1. Create a Virtual Environment (Recommended)

```bash
# Create a new virtual environment
python -m venv factcheck_env

# Activate the virtual environment
# On Windows:
factcheck_env\Scripts\activate
# On macOS/Linux:
source factcheck_env/bin/activate
```

### 2. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in your project directory with your API keys:

```env
# OpenAI API Key (for GPT models)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Key (for Claude models)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google API Key (for Gemini models)
GEMINI_API_KEY=your_google_api_key_here

# Together.ai API Key (for Qwen, Llama, DeepSeek models)
TOGETHER_API_KEY=your_together_api_key_here
```

**Note**: No search API key is required as the system uses DuckDuckGo search which is free and doesn't require authentication.

### 4. API Key Setup

#### OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy and paste into your `.env` file

#### Together.ai API Key

1. Go to [Together.ai](https://together.ai/)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy and paste into your `.env` file

### 5. Directory Structure

Ensure you have the following directory structure:

```
your_project/
├── LLM_auto_label_reasoning.py
├── requirements.txt
├── SETUP.md
├── .env
├── Dataset/
│   ├── entity1.csv
│   ├── entity2.csv
│   └── ...
└── Results/
    └── (will be created automatically)
```

### 6. Dataset Format

Your CSV files in the `Dataset/` folder should have this format:

```csv
Atomic Fact,True Label
"Albert Einstein was born in 1879",Supported
"Albert Einstein was a famous musician",Unsupported
"Albert Einstein invented the light bulb",Unsupported
```

## Usage

### Basic Usage

```bash
# Run the fact-checking system
python LLM_auto_label_reasoning.py
```

### Model Configuration

You can change the model in the code:

```python
# For OpenAI GPT models
llm_factory = LLMFactory(model_name="gpt-4o-mini", together=False)

# For Together.ai models (Qwen, Llama, etc.)
llm_factory = LLMFactory(model_name="Qwen/Qwen3-235B-A22B-fp8-tput", together=True)
```

## Features

### Search Engine

- **DuckDuckGo Search**: Free, privacy-focused search engine
- **No API Key Required**: Works out of the box
- **Intelligent Fallback**: Automatically tries general searches when specific queries fail
- **Content Extraction**: Extracts and processes content from search results

### Reasoning Process

- **Multi-Step Reasoning**: Initial analysis → Intelligent search → Final analysis
- **Dynamic Search Queries**: Agent decides what to search for based on context
- **Maximum 5 Searches**: Efficient resource usage with intelligent stopping
- **Detailed Logging**: Real-time console output showing reasoning steps

## Troubleshooting

### Common Issues

1. **Import Errors**

   ```bash
   # Reinstall packages if you get import errors
   pip install --upgrade -r requirements.txt
   ```

2. **API Key Errors**

   - Ensure your `.env` file is in the correct location
   - Check that API keys are valid and have sufficient credits
   - Verify API key format (no extra spaces or quotes)

3. **Search Issues**

   - DuckDuckGo search is free and doesn't require API keys
   - If searches fail, the system will automatically try general searches
   - Check your internet connection

4. **Memory Issues**

   - Reduce `max_results` in search functions if you encounter memory issues
   - Process smaller datasets at a time

5. **Rate Limiting**
   - The code includes built-in delays to respect API rate limits
   - DuckDuckGo has generous rate limits for normal usage

### Performance Optimization

1. **Reduce Search Results**

   ```python
   # In get_search_results function
   results = list(ddgs.text(query, max_results=3))  # Reduce from 5 to 3
   ```

2. **Adjust Max Iterations**

   ```python
   workflow = StateGraph(AgentState)
   # Reduce if you want faster processing
   ```

3. **Batch Processing**
   - Process entities one by one for better error handling
   - Use smaller datasets for testing

## Output Files

The system generates several output files:

1. **Individual Results**: `Results/{entity_name}_langgraph_labeled.csv`
2. **Overall Analysis**: `Results/overall_langgraph_results.json`
3. **Console Output**: Real-time progress and accuracy metrics

## Support

If you encounter any issues:

1. Check the console output for error messages
2. Verify your API keys and internet connection
3. Ensure all dependencies are properly installed
4. Check the dataset format matches the expected structure
5. DuckDuckGo search issues are usually related to internet connectivity
