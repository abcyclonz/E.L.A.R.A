from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Memory agent
    memory_agent_url: str = "http://memory_agent:8000"

    # Elara conversation + learning agent
    elara_url: str = "http://elara:8002"

    # Ollama (used by memory agent extractor — Elara has its own)
    ollama_url: str = "http://172.17.0.1:11434"
    ollama_model: str = "mistral:latest"

    # Summarize conversation every N turns per speaker (0 = disabled)
    summarize_every_n_turns: int = 5

    class Config:
        env_file = ".env"


settings = Settings()
