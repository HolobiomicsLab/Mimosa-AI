
import os
from anthropic import Anthropic
import openai
from openai import OpenAI

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

def deepseek_fn(history, verbose=False):
    """
    Use deepseek api to generate text.
    """
    client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=history,
            stream=False
        )
        thought = response.choices[0].message.content
        if verbose:
            print(thought)
        return thought
    except Exception as e:
        raise Exception(f"deepseek_fn: Deepseek API error: {str(e)}") from e

def openai_fn(history, verbose=False):
    """
    Use openai to generate text.
    """
    client = OpenAI(api_key=openai_api_key)

    try:
        response = client.chat.completions.create(
            model="o3-2025-04-16",
            messages=history,
        )
        if response is None:
            raise Exception("openai_fn: OpenAI response is empty.")
        thought = response.choices[0].message.content
        if verbose:
            print(thought)
        return thought
    except Exception as e:
        raise Exception(f"openai_fn: OpenAI API error: {str(e)}") from e