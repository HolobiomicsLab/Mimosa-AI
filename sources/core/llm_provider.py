import os
import json
from typing import List, Dict, Optional
from openai import OpenAI


class LLMProvider:
    """Handles interactions with various LLM APIs.

    Attributes:
        deepseek_client (OpenAI): Client for Deepseek API
        openai_client (OpenAI): Client for OpenAI API
    """

    def __init__(self) -> None:
        """Initialize the LLM provider with API clients."""
        self.deepseek_client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
        )
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def save_call(self, call: Dict[str, str], path: str) -> None:
        """
        Save the API call details to a JSON file as a list of calls.
        If the file exists, append to the list; otherwise, create a new list.

        Args:
            call: Dictionary containing API call details
            uuid_str: Unique identifier for the request
        """
        path = f"{path}/llm_calls.json"
        try:
            with open(path, "r") as f:
                calls: list = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            calls = []
        calls.append(call)
        with open(path, "w") as f:
            json.dump(calls, f, indent=2)

    def deepseek_completion(
        self,
        history: List[Dict[str, str]],
        path: str,
        verbose: bool = False,
        model="deepseek-reasoner",
    ) -> str:
        """Generate text using Deepseek API.

        Args:
            history: Conversation history in OpenAI format
            uuid_str: Unique identifier for the request
            verbose: Whether to print the response

        Returns:
            str: Generated text from Deepseek

        Raises:
            RuntimeError: If API request fails
        """
        try:
            response = self.deepseek_client.chat.completions.create(
                model=model, messages=history, stream=False
            )
            thought = response.choices[0].message.content
            if verbose:
                print(thought)
            self.save_call(
                {"model": model, "messages": history, "thought": thought}, path
            )
            return thought
        except Exception as e:
            raise RuntimeError(f"❌ Deepseek API error: {str(e)}") from e

    def openai_completion(
        self,
        history: List[Dict[str, str]],
        path: str,
        verbose: bool = False,
        model="o3-2025-04-16",
    ) -> str:
        """Generate text using OpenAI API.

        Args:
            history: Conversation history in OpenAI format
            uuid_str: Unique identifier for the request
            verbose: Whether to print the response

        Returns:
            str: Generated text from OpenAI

        Raises:
            RuntimeError: If API request fails
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=model, messages=history
            )
            if response is None:
                raise RuntimeError("❌ OpenAI response is empty")
            thought = response.choices[0].message.content
            if verbose:
                print(thought)
            self.save_call(
                {"model": model, "history": history, "thought": thought}, path
            )
            return thought
        except Exception as e:
            raise RuntimeError(f"❌ OpenAI API error: {str(e)}") from e
