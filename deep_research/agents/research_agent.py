"""
Research Agent for the Deep Research AI Agentic System.
Responsible for web search, content extraction, and research summarization.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Import utilities
from deep_research.utils.search_utils import search_web, generate_search_queries
from deep_research.utils.web_utils import extract_content, split_content

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize LLM
model = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
temperature = float(os.getenv("TEMPERATURE", "0.1"))
llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)

class ResearchAgent:
    """
    Agent responsible for conducting web research and summarizing findings.
    """
    
    def __init__(self):
        """Initialize the research agent."""
        self.name = "Research Agent"
        logger.info(f"Initializing {self.name}")
    
    def conduct_research(self, question: str) -> Dict[str, Any]:
        """
        Conduct research on a given question.
        
        Args:
            question: The research question
            
        Returns:
            Dictionary with research results
        """
        logger.info(f"Starting research on: {question}")
        
        # Step 1: Generate search queries
        search_queries = generate_search_queries(question)
        
        # Step 2: Perform searches
        all_results = []
        for query in search_queries:
            results = search_web(query)
            all_results.extend(results)
        
        # Step 3: Extract content from search results
        content_list = []
        sources = []
        
        # Get unique URLs
        unique_urls = set()
        for result in all_results:
            if "url" in result and result["url"] not in unique_urls:
                unique_urls.add(result["url"])
        
        # Extract content from each URL (limit to 5 URLs)
        for url in list(unique_urls)[:5]:
            content = extract_content(url)
            if content["content"] and "Error" not in content["title"]:
                content_list.append(content)
                sources.append({"url": url, "title": content["title"]})
        
        # Step 4: Summarize content
        if content_list:
            summary = self.summarize_content(content_list, question)
        else:
            summary = "No relevant content found."
        
        # Return research results
        return {
            "question": question,
            "summary": summary,
            "sources": sources,
            "raw_content": content_list
        }
    
    def summarize_content(self, content_list: List[Dict[str, Any]], question: str) -> str:
        """
        Summarize the content from multiple sources.
        
        Args:
            content_list: List of content dictionaries
            question: The original research question
            
        Returns:
            Summarized content
        """
        logger.info(f"Summarizing content from {len(content_list)} sources")
        
        try:
            # Prepare content for summarization
            formatted_content = []
            for item in content_list:
                source_info = f"Source: {item['title']} ({item['url']})"
                content_excerpt = item['content'][:5000]  # Limit length for summarization
                formatted_content.append(f"{source_info}\n\n{content_excerpt}")
            
            combined_content = "\n\n---\n\n".join(formatted_content)
            
            # Create a summarization prompt
            summarization_prompt = ChatPromptTemplate.from_template(
                """You are a research assistant tasked with summarizing information related to a specific question.
                
                Question: {question}
                
                Below is the content to summarize:
                
                {content}
                
                Please provide a comprehensive summary of the key information related to the question.
                Focus on facts, data, and insights that help answer the question.
                Include any relevant statistics, expert opinions, or important context.
                Maintain objectivity and avoid inserting your own opinions.
                Organize the information in a structured way with clear sections.
                
                Your summary should be detailed enough to serve as the basis for a comprehensive answer.
                """
            )
            
            chain = summarization_prompt | llm
            response = chain.invoke({"question": question, "content": combined_content})
            
            return response.content
        except Exception as e:
            logger.error(f"Error in content summarization: {str(e)}")
            return f"Error in summarization: {str(e)}" 