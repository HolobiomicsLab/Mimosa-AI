import json
import os

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

    def save_call(self, call: dict[str, str], called_by: str, memory_path:str) -> None:
        """
        Save the API call details to a JSON file.

        Args:
            call: Dictionary containing API call details
            uuid_str: Unique identifier for the request
        """
        path = os.path.join(memory_path, f"{called_by}.json")
        with open(path, "w") as f:
            json.dump(call, f, indent=2)

    def deepseek_completion(
        self,
        history: list[dict[str, str]],
        called_by: str,
        memory_path: str,
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
            # Extract token usage information
            token_usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            if verbose:
                print(thought)
            self.save_call(
                {
                    "model": model,
                    "messages": history,
                    "thought": thought,
                    "token_usage": token_usage
                },
                called_by,
                memory_path
            )
            return thought
        except Exception as e:
            raise RuntimeError(f"❌ Deepseek API error: {str(e)}") from e

    def openai_completion(
        self,
        history: list[dict[str, str]],
        called_by: str | None = None,
        memory_path: str | None = None,
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
            # Extract token usage information
            token_usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            if verbose:
                print(thought)
            self.save_call(
                {
                    "model": model,
                    "messages": history,
                    "thought": thought,
                    "token_usage": token_usage
                },
                called_by,
                memory_path
            )
            return thought
        except Exception as e:
            raise RuntimeError(f"❌ OpenAI API error: {str(e)}") from e
