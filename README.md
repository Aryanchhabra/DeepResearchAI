# Deep Research AI Agentic System

A comprehensive research assistant built with specialized agents to perform web research and draft detailed, fact-checked answers with citations.

## Overview

The Deep Research AI Agentic System is designed to automate the process of conducting web research and generating comprehensive, fact-checked answers to complex questions. The system employs a multi-agent architecture with specialized components working together to gather, analyze, and synthesize information from the web.

## Key Features

- **Multi-Agent Architecture**: Specialized agents for research and answer drafting work together to produce high-quality results
- **Web Research**: Integration with Tavily Search API for high-quality search results
- **Content Analysis**: Web crawling capabilities for deeper content extraction
- **Comprehensive Output**: Verified answers with cited sources
- **Google Gemini Integration**: Powered by Google's Gemini AI models
- **Web Interface**: Modern, responsive web application for easy interaction with the system

## System Architecture

The system consists of the following components:

1. **Research Agent**: Responsible for generating search queries, performing web searches, extracting content from web pages, and summarizing research findings.

2. **Answer Agent**: Responsible for drafting comprehensive answers based on research findings, fact-checking the drafted answers against source materials, and finalizing the answers.

3. **Implementation Options**:
   - **Standard Version**: Uses LangGraph for workflow orchestration (may require specific LangGraph versions)
   - **Simplified Version**: Direct agent implementation without LangGraph (recommended for reliability)

4. **Web Interface**: A Flask-based web application that provides a user-friendly interface for interacting with the system.

## Workflow

1. The system receives a research question from the user.
2. The Research Agent generates search queries based on the question.
3. The Research Agent performs web searches using the Tavily API.
4. The Research Agent extracts content from the search results.
5. The Research Agent summarizes the research findings.
6. The Answer Agent drafts a comprehensive answer based on the research summary.
7. The Answer Agent fact-checks the drafted answer against the source materials.
8. The Answer Agent finalizes the answer based on the fact-checking results.
9. The system returns the final answer with citations to the user.

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys (use the provided `.env.example` as a template):
   ```
   GOOGLE_API_KEY=your_google_api_key
   TAVILY_API_KEY=your_tavily_api_key
   GEMINI_MODEL=gemini-1.5-pro
   TEMPERATURE=0.1
   ```

## Usage

### Web Interface (Recommended)

The easiest way to use Deep Research AI is through the web interface:

```
python run_web.py
```

Then open your browser and navigate to `http://localhost:5000` to access the web interface.

### Command Line Interface

#### Recommended: Simplified Version

For reliable operation, use the simplified version that doesn't rely on LangGraph:
```
python simple_main.py "Your research question here"
```

#### Alternative: LangGraph Version

If you want to use the LangGraph-based implementation (note: may require specific LangGraph versions):
```
python main.py "Your research question here"
```

## Web Interface Features

The web interface provides a user-friendly way to interact with the Deep Research AI system:

- **Research Form**: Submit research questions and choose between the simplified and LangGraph implementations
- **Research History**: View and manage your past research
- **Detailed Results**: View comprehensive research results with proper formatting
- **Export Options**: Print or copy research results to clipboard

## Implementation Notes

This project provides two implementations of the same functionality:

1. **Simplified Version (`simple_main.py`)**: 
   - Direct implementation without LangGraph
   - More reliable and less prone to compatibility issues
   - Recommended for most use cases
   - Produces high-quality, comprehensive answers with proper citations

2. **LangGraph Version (`main.py`)**:
   - Uses LangGraph for workflow orchestration
   - Demonstrates advanced agent orchestration techniques
   - May require specific LangGraph versions for compatibility
   - Included to showcase knowledge of advanced orchestration frameworks

## Project Structure

- `main.py`: Entry point for the LangGraph implementation
- `simple_main.py`: Entry point for the simplified implementation (recommended)
- `run_web.py`: Entry point for the web interface
- `deep_research/`: Main package
  - `workflow.py`: LangGraph workflow orchestrator
  - `agents/`: Contains the specialized agent implementations
    - `research_agent.py`: Research Agent implementation
    - `answer_agent.py`: Answer Agent implementation
  - `utils/`: Utility functions
    - `search_utils.py`: Functions for web search
    - `web_utils.py`: Functions for web content extraction
- `web/`: Web interface
  - `app.py`: Flask application
  - `templates/`: HTML templates
  - `static/`: Static assets (CSS, JS)
  - `research_history/`: Storage for research history

## Dependencies

- LangChain: Framework for building LLM applications
- LangGraph: Framework for orchestrating multi-agent workflows (used in the standard version)
- Google Generative AI: For LLM capabilities via Gemini models
- Tavily: For web search functionality
- BeautifulSoup4: For web content extraction
- Requests: For HTTP requests
- Python-dotenv: For environment variable management
- Pydantic: For data validation
- Flask: For the web interface
- Flask-WTF: For form handling in the web interface
- Flask-CORS: For cross-origin resource sharing in the web interface
- Markdown: For converting markdown to HTML in the web interface

## Troubleshooting

If you encounter issues with the LangGraph version:
1. Try using the simplified version (`simple_main.py`) which is more reliable
2. Check your LangGraph version compatibility (this project was developed with langgraph>=0.0.20)
3. Ensure all API keys are correctly set in your `.env` file

## License

MIT 