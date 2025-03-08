"""
Search utilities for the Deep Research AI Agentic System.
Handles web search functionality using the Tavily API.
"""

import os
import logging
from typing import Dict, List, Any
from tavily import TavilyClient
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Tavily client
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key)

def search_web(query: str, max_results: int = 5, search_depth: str = "basic") -> List[Dict[str, Any]]:
    """
    Search the web using Tavily API.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        search_depth: Depth of search ('basic' or 'comprehensive')
        
    Returns:
        List of search results
    """
    logger.info(f"Searching for: {query}")
    
    try:
        # Remove quotes from the query to avoid Tavily API errors
        clean_query = query.replace('"', '').replace("'", "")
        
        response = tavily_client.search(
            query=clean_query,
            search_depth=search_depth,
            max_results=max_results,
            include_answer=True,
            include_raw_content=False,
            include_images=False
        )
        
        if "results" in response:
            logger.info(f"Found {len(response['results'])} results")
            return response["results"]
        else:
            logger.warning(f"Unexpected response format from Tavily: {response}")
            return []
    except Exception as e:
        logger.error(f"Error searching with Tavily: {str(e)}")
        return []

def generate_search_queries(question: str, num_queries: int = 3) -> List[str]:
    """
    Generate search queries based on the research question.
    
    Args:
        question: The research question
        num_queries: Number of queries to generate
        
    Returns:
        List of search queries
    """
    logger.info(f"Generating search queries for question: {question}")
    
    # Generate simple variations of the question
    queries = [
        question,
        f"latest information about {question}",
        f"recent developments in {question}"
    ]
    
    # Return the specified number of queries
    return queries[:num_queries] 