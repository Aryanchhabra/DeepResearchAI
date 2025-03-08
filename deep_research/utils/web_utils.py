"""
Web utilities for the Deep Research AI Agentic System.
Handles web content extraction and processing.
"""

import logging
import requests
from typing import Dict, List, Any, Tuple, Optional
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_content(url: str, max_content_length: int = 10000) -> Dict[str, Any]:
    """
    Extract content from a webpage.
    
    Args:
        url: The URL to extract content from
        max_content_length: Maximum length of content to extract
        
    Returns:
        Dictionary with extracted content
    """
    logger.info(f"Extracting content from: {url}")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title
        title = None
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text(strip=True)
        
        # Extract main content
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.extract()
            
        # Get text
        text = soup.get_text(separator='\n')
        
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Remove blank lines
        content = '\n'.join(chunk for chunk in chunks if chunk)
        
        return {
            "url": url,
            "title": title or "Unknown",
            "content": content[:max_content_length]  # Limit content length
        }
    except Exception as e:
        logger.error(f"Error extracting content from {url}: {str(e)}")
        return {
            "url": url,
            "title": "Error",
            "content": f"Error extracting content: {str(e)}"
        }

def split_content(content: str, chunk_size: int = 4000, chunk_overlap: int = 200) -> List[str]:
    """
    Split content into manageable chunks.
    
    Args:
        content: The content to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of content chunks
    """
    # Simple implementation of content splitting
    chunks = []
    start = 0
    
    while start < len(content):
        end = min(start + chunk_size, len(content))
        
        # If this is not the first chunk and we're not at the end, 
        # adjust to avoid splitting in the middle of a word or sentence
        if start > 0 and end < len(content):
            # Try to find a period, question mark, or exclamation point followed by a space
            for punct in ['. ', '? ', '! ']:
                last_punct = content[start:end].rfind(punct)
                if last_punct != -1:
                    end = start + last_punct + 2  # +2 to include the punctuation and space
                    break
            
            # If no punctuation found, try to find a newline
            if end == start + chunk_size:
                last_newline = content[start:end].rfind('\n')
                if last_newline != -1:
                    end = start + last_newline + 1  # +1 to include the newline
        
        chunks.append(content[start:end])
        start = end - chunk_overlap if end < len(content) else len(content)
    
    return chunks 