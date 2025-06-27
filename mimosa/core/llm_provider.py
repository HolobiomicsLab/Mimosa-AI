import os
from typing import List, Dict, Optional
from openai import OpenAI
from anthropic import Anthropic


class LLMProvider:
    """Handles interactions with various LLM APIs.
    
    Attributes:
        deepseek_client (OpenAI): Client for Deepseek API
        openai_client (OpenAI): Client for OpenAI API
    """
    
    def __init__(self) -> None:
        """Initialize the LLM provider with API clients."""
        self.deepseek_client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def deepseek_completion(
        self, 
        history: List[Dict[str, str]], 
        verbose: bool = False
    ) -> str:
        """Generate text using Deepseek API.
        
        Args:
            history: Conversation history in OpenAI format
            verbose: Whether to print the response
            
        Returns:
            str: Generated text from Deepseek
            
        Raises:
            RuntimeError: If API request fails
        """
        try:
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-reasoner",
                messages=history,
                stream=False
            )
            thought = response.choices[0].message.content
            if verbose:
                print(thought)
            return thought
        except Exception as e:
            raise RuntimeError(f"❌ Deepseek API error: {str(e)}") from e

    def openai_completion(
        self, 
        history: List[Dict[str, str]], 
        verbose: bool = False
    ) -> str:
        """Generate text using OpenAI API.
        
        Args:
            history: Conversation history in OpenAI format
            verbose: Whether to print the response
            
        Returns:
            str: Generated text from OpenAI
            
        Raises:
            RuntimeError: If API request fails
        """
        try:
            response = self.openai_client.chat.completions.create(
                model="o3-2025-04-16",
                messages=history
            )
            if response is None:
                raise RuntimeError("❌ OpenAI response is empty")
            thought = response.choices[0].message.content
            if verbose:
                print(thought)
            return thought
        except Exception as e:
            raise RuntimeError(f"❌ OpenAI API error: {str(e)}") from e
