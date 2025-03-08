#!/usr/bin/env python
"""
Simplified entry point for the Deep Research AI Agentic System.
This version uses direct agent calls without the LangGraph workflow.
"""

import os
import sys
import logging
import argparse
from typing import Dict, Any
from dotenv import load_dotenv

# Import agents
from deep_research.agents.research_agent import ResearchAgent
from deep_research.agents.answer_agent import AnswerAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('deep_research_simple.log')
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
        # Initialize agents
        research_agent = ResearchAgent()
        answer_agent = AnswerAgent()
        
        # Step 1: Conduct research
        research_results = research_agent.conduct_research(question)
        
        # Step 2: Draft answer
        draft_answer = answer_agent.draft_answer(research_results)
        
        # Step 3: Fact check
        fact_check = answer_agent.fact_check(draft_answer, research_results)
        
        # Step 4: Finalize answer
        final_answer = answer_agent.finalize_answer(fact_check)
        
        return {
            "question": question,
            "answer": final_answer.get("answer", "No answer generated."),
            "sources": final_answer.get("sources", []),
            "status": "success"
        }
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
    parser = argparse.ArgumentParser(description="Deep Research AI Agentic System (Simple Version)")
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