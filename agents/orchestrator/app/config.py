from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Core agents
    memory_agent_url: str = "http://memory_agent:8000"
    elara_url: str = "http://elara:8002"

    # Ollama
    ollama_url: str = "http://172.17.0.1:11434"
    ollama_model: str = "mistral:latest"

    # Conversation summarization (0 = disabled)
    summarize_every_n_turns: int = 5

    # MCP tool servers
    web_search_mcp_url: str = "http://web_search_tool:8010"
    assistant_mcp_url: str = "http://assistant_tool:8011"

    # Tavily key (read here so we can surface missing-key errors early)
    tavily_api_key: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
