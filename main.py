#!/usr/bin/env python
"""
Main entry point for the Deep Research AI Agentic System.
"""

import os
import sys
import logging
import argparse
from typing import Dict, Any
from dotenv import load_dotenv

# Import the workflow
from deep_research.workflow import DeepResearchWorkflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('deep_research.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def run_research(question: str) -> Dict[str, Any]:
    """
    Run the research workflow for a given question.
    
    Args:
        question: The research question
        
    Returns:
        Dictionary with the research results
    """
    logger.info(f"Starting research for question: {question}")
    
    try:
        # Initialize the workflow
        workflow = DeepResearchWorkflow()
        
        # Run the workflow
        results = workflow.run(question)
        
        return results
    except Exception as e:
        logger.error(f"Error in research workflow: {str(e)}")
        return {
            "question": question,
            "answer": f"An error occurred during research: {str(e)}",
            "sources": [],
            "status": "error",
            "errors": [str(e)]
        }

def main():
    """Main entry point for the application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Deep Research AI Agentic System")
    parser.add_argument("question", type=str, help="The research question to answer")
    args = parser.parse_args()
    
    # Check for API keys
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY not found in environment variables")
        print("Error: GOOGLE_API_KEY not found. Please set it in your .env file.")
        return
    
    if not os.getenv("TAVILY_API_KEY"):
        logger.warning("TAVILY_API_KEY not found in environment variables")
        print("Warning: TAVILY_API_KEY not found. Web search functionality may be limited.")
    
    # Run the research
    results = run_research(args.question)
    
    # Print the results
    print("\n" + "="*50)
    print(f"Research Question: {results['question']}")
    print("="*50)
    print("\nAnswer:")
    print(results['answer'])
    
    print("\nSources:")
    for i, source in enumerate(results['sources'], 1):
        print(f"{i}. {source.get('title', 'Unknown')} - {source.get('url', 'No URL')}")
    
    print("\nStatus:", results['status'])
    
    if 'errors' in results and results['errors']:
        print("\nErrors encountered:")
        for error in results['errors']:
            print(f"- {error}")
    
    print("="*50)

if __name__ == "__main__":
    main() 