

import os
import threading
import time
import socket
from typing import Dict, Optional
from contextlib import contextmanager
from python_a2a import run_server
from python_a2a import A2AClient
from python_a2a.langchain import to_a2a_server
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


class ServerConfig:
    """Configuration for A2A servers"""
    def __init__(self, host: str = "0.0.0.0", start_port: int = 8000, max_port_tries: int = 100):
        self.host = host
        self.start_port = start_port
        self.max_port_tries = max_port_tries

class A2AServerWrapper:
    """Wrapper for A2A servers with lifecycle management"""
    
    def __init__(self, server, name: str, config: ServerConfig):
        self.server = server
        self.name = name
        self.config = config
        self.port: Optional[int] = None
        self.thread: Optional[threading.Thread] = None
        self._running = False

    def find_available_port(start_port: int = 8000, max_tries: int = 100) -> int:
        """Find an available port starting from start_port"""
        for port in range(start_port, start_port + max_tries):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.bind(('localhost', port))
                    return port
            except OSError:
                continue
        raise RuntimeError(f"No available ports found in range {start_port}-{start_port + max_tries}")
    
    def start(self) -> int:
        """Start the server and return the port"""
        if self._running:
            return self.port
        
        self.port = self.find_available_port(
            self.config.start_port, 
            self.config.max_port_tries
        )
        
        def server_thread():
            try:
                run_server(self.server, host=self.config.host, port=self.port)
            except Exception as e:
                print(f"Server {self.name} failed: {e}")
        
        self.thread = threading.Thread(target=server_thread, daemon=True, name=f"{self.name}-server")
        self.thread.start()        
        self._running = True
        return self.port
    
    def stop(self):
        """Stop the server"""
        if self._running:
            self._running = False
    
    @property
    def url(self) -> str:
        """Get the server URL"""
        if not self.port:
            raise RuntimeError(f"Server {self.name} is not running")
        return f"http://{self.config.host}:{self.port}"


class LangChainA2AConverter:
    """Converts LangChain components to A2A servers"""
    
    def __init__(self, config: ServerConfig = None):
        self.config = config or ServerConfig()
        self.servers: Dict[str, A2AServerWrapper] = {}
    
    def register_component(self, component, name: str) -> A2AServerWrapper:
        """Register a LangChain component as an A2A server"""
        
        if name in self.servers:
            raise ValueError(f"Server with name '{name}' already exists")
        
        a2a_server = to_a2a_server(component)
        wrapper = A2AServerWrapper(a2a_server, name, self.config)
        self.servers[name] = wrapper
        return wrapper
    
    def start_all(self) -> Dict[str, str]:
        """Start all registered servers and return their URLs"""
        urls = {}
        for name, server in self.servers.items():
            port = server.start()
            urls[name] = server.url
        return urls
    
    def stop_all(self):
        """Stop all servers"""
        for server in self.servers.values():
            server.stop()
    
    def get_client(self, name: str):
        """Get a client for a specific server"""
        
        if name not in self.servers:
            raise ValueError(f"No server named '{name}' found")
        
        server = self.servers[name]
        if not server._running:
            raise RuntimeError(f"Server '{name}' is not running")
        
        return A2AClient(server.url)


class ComponentFactory:
    """Factory for creating LangChain components"""
    
    @staticmethod
    def create_llm(api_key: str, model: str = "gpt-3.5-turbo", temperature: float = 0):
        """Create a ChatOpenAI instance"""
        return ChatOpenAI(api_key=api_key, model=model, temperature=temperature)
    
    @staticmethod
    def create_travel_chain(llm):
        """Create a travel guide chain"""
        
        template = """
        You are a helpful travel guide with extensive knowledge of destinations worldwide.
        
        Question: {query}
        
        Please provide a detailed and helpful answer:
        """
        prompt = PromptTemplate.from_template(template)
        return prompt | llm | StrOutputParser()
    
    @staticmethod
    def create_code_reviewer_chain(llm):
        """Create a code review chain"""
        
        template = """
        You are an expert code reviewer. Analyze the following code and provide constructive feedback.
        
        Code: {code}
        
        Please provide:
        1. Code quality assessment
        2. Potential issues or bugs
        3. Suggestions for improvement
        4. Best practices recommendations
        
        Review:
        """
        prompt = PromptTemplate.from_template(template)
        return prompt | llm | StrOutputParser()


class A2AServerManager:
    """High-level manager for A2A server operations"""
    
    def __init__(self, api_key: str, config: ServerConfig = None):
        self._validate_api_key(api_key)
        self.api_key = api_key
        self.config = config or ServerConfig()
        self.converter = LangChainA2AConverter(self.config)
        self.factory = ComponentFactory()
    
    def _validate_api_key(self, api_key: str):
        """Validate API key"""
        if not api_key:
            raise ValueError("API key is required")
    
    def setup_default_services(self):
        """Setup default LangChain services"""
        # Create base LLM
        llm = self.factory.create_llm(self.api_key)
        
        # Register services
        self.converter.register_component(llm, "llm")
        self.converter.register_component(
            self.factory.create_travel_chain(llm), 
            "travel_guide"
        )
        self.converter.register_component(
            self.factory.create_code_reviewer_chain(llm),
            "code_reviewer"
        )
    
    def start_services(self) -> Dict[str, str]:
        """Start all services and return URLs"""
        return self.converter.start_all()
    
    def stop_services(self):
        """Stop all services"""
        self.converter.stop_all()
    
    def get_client(self, service_name: str):
        """Get a client for a specific service"""
        return self.converter.get_client(service_name)
    
    @contextmanager
    def managed_services(self):
        """Context manager for service lifecycle"""
        urls = self.start_services()
        try:
            yield urls
        finally:
            self.stop_services()


def main():
    """Main function"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        manager = A2AServerManager(api_key)
        manager.setup_default_services()
        with manager.managed_services() as service_urls:
            for name, url in service_urls.items():
                print(f"{name}: {url}")
    

if __name__ == "__main__":
    exit(main())