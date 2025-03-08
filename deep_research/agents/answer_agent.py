"""
Answer Agent for the Deep Research AI Agentic System.
Responsible for drafting comprehensive answers based on research findings.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize LLM
model = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
temperature = float(os.getenv("TEMPERATURE", "0.2"))  # Slightly higher temperature for more creative drafting
llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)

class AnswerAgent:
    """
    Agent responsible for drafting comprehensive answers based on research findings.
    """
    
    def __init__(self):
        """Initialize the answer agent."""
        self.name = "Answer Agent"
        logger.info(f"Initializing {self.name}")
    
    def draft_answer(self, research_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Draft a comprehensive answer based on research findings.
        
        Args:
            research_results: Dictionary with research results
            
        Returns:
            Dictionary with the drafted answer
        """
        logger.info("Drafting answer based on research findings")
        
        question = research_results.get("question", "")
        summary = research_results.get("summary", "")
        sources = research_results.get("sources", [])
        
        if not summary:
            logger.warning("No research summary available for drafting an answer")
            return {
                "question": question,
                "answer": "Insufficient research data to provide an answer.",
                "sources": sources
            }
        
        try:
            # Format sources for the prompt
            sources_text = ""
            for i, source in enumerate(sources, 1):
                sources_text += f"{i}. {source.get('title', 'Unknown')} - {source.get('url', 'No URL')}\n"
            
            # If no sources are available
            if not sources_text:
                sources_text = "No specific sources available."
            
            # Create a prompt for drafting the answer
            draft_prompt = ChatPromptTemplate.from_template(
                """You are a research assistant tasked with drafting a comprehensive answer to a question
                based on research findings.
                
                Question: {question}
                
                Research Summary:
                {summary}
                
                Sources:
                {sources}
                
                Please draft a comprehensive answer that:
                1. Directly addresses the original question
                2. Incorporates key information from the research summary
                3. Is well-structured with clear sections and logical flow
                4. Includes relevant facts, data, and expert opinions
                5. Maintains objectivity and avoids speculation
                6. Acknowledges any limitations in the available information
                7. Includes proper citations to the sources using [1], [2], etc.
                
                End with a "References" section listing all sources.
                
                Your answer should be detailed, informative, and backed by the research findings.
                """
            )
            
            # Invoke the LLM
            chain = draft_prompt | llm
            response = chain.invoke({
                "question": question,
                "summary": summary,
                "sources": sources_text
            })
            
            # Return the drafted answer
            return {
                "question": question,
                "answer": response.content,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error drafting answer: {str(e)}")
            return {
                "question": question,
                "answer": f"Error in answer drafting: {str(e)}",
                "sources": sources
            }
    
    def fact_check(self, draft_answer: Dict[str, Any], research_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fact check the drafted answer against the research findings.
        
        Args:
            draft_answer: Dictionary with the drafted answer
            research_results: Dictionary with research results
            
        Returns:
            Dictionary with the fact-checked answer
        """
        logger.info("Fact checking the drafted answer")
        
        question = draft_answer.get("question", "")
        answer = draft_answer.get("answer", "")
        sources = draft_answer.get("sources", [])
        raw_content = research_results.get("raw_content", [])
        
        if not answer or not raw_content:
            logger.warning("Insufficient data for fact checking")
            return draft_answer
        
        try:
            # Prepare source content for verification
            source_content = []
            for content in raw_content:
                source_info = f"Source: {content.get('title', 'Unknown')} ({content.get('url', 'No URL')})"
                content_text = content.get('content', '')[:5000]  # Limit length
                source_content.append(f"{source_info}\n\n{content_text}")
            
            combined_sources = "\n\n---\n\n".join(source_content)
            
            # Create a prompt for fact checking
            fact_check_prompt = ChatPromptTemplate.from_template(
                """You are a fact checker tasked with verifying the accuracy of an answer against source materials.
                
                Question: {question}
                
                Answer to verify:
                {answer}
                
                Source Materials:
                {sources}
                
                Please verify the accuracy of the answer against the source materials and:
                1. Identify any factual errors or misrepresentations
                2. Note any claims that cannot be verified with the provided sources
                3. Suggest corrections for any inaccuracies
                
                If the answer is accurate and well-supported by the sources, simply state that it has been verified.
                If there are issues, provide specific feedback on what needs to be corrected.
                """
            )
            
            # Invoke the LLM
            chain = fact_check_prompt | llm
            response = chain.invoke({
                "question": question,
                "answer": answer,
                "sources": combined_sources
            })
            
            # Return the fact-checked answer
            return {
                "question": question,
                "answer": answer,
                "fact_check": response.content,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error in fact checking: {str(e)}")
            return {
                "question": question,
                "answer": answer,
                "fact_check": f"Error in fact checking: {str(e)}",
                "sources": sources
            }
    
    def finalize_answer(self, fact_checked_answer: Dict[str, Any]) -> Dict[str, Any]:
        """
        Finalize the answer based on fact checking results.
        
        Args:
            fact_checked_answer: Dictionary with the fact-checked answer
            
        Returns:
            Dictionary with the final answer
        """
        logger.info("Finalizing answer")
        
        question = fact_checked_answer.get("question", "")
        answer = fact_checked_answer.get("answer", "")
        fact_check = fact_checked_answer.get("fact_check", "")
        sources = fact_checked_answer.get("sources", [])
        
        if "Error" in fact_check or not fact_check:
            logger.warning("No fact check results available or error in fact checking")
            return {
                "question": question,
                "answer": answer,
                "sources": sources
            }
        
        try:
            # Format sources for the prompt
            sources_text = ""
            for i, source in enumerate(sources, 1):
                sources_text += f"{i}. {source.get('title', 'Unknown')} - {source.get('url', 'No URL')}\n"
            
            # Create a prompt for finalizing the answer
            finalize_prompt = ChatPromptTemplate.from_template(
                """You are a research assistant tasked with finalizing an answer based on fact checking results.
                
                Question: {question}
                
                Draft Answer:
                {answer}
                
                Fact Check Results:
                {fact_check}
                
                Sources:
                {sources}
                
                Please revise the draft answer based on the fact check results to ensure accuracy.
                If the fact check identified any issues, correct them in the final answer.
                If the fact check verified the answer, you can keep it as is or make minor improvements.
                
                Ensure the final answer:
                1. Is accurate and well-supported by the sources
                2. Directly addresses the original question
                3. Is well-structured with clear sections
                4. Includes proper citations to the sources
                5. Ends with a "References" section
                
                Your final answer should be comprehensive, accurate, and properly cited.
                """
            )
            
            # Invoke the LLM
            chain = finalize_prompt | llm
            response = chain.invoke({
                "question": question,
                "answer": answer,
                "fact_check": fact_check,
                "sources": sources_text
            })
            
            # Return the final answer
            return {
                "question": question,
                "answer": response.content,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error finalizing answer: {str(e)}")
            return {
                "question": question,
                "answer": answer,  # Return the original answer if there's an error
                "sources": sources
            } 