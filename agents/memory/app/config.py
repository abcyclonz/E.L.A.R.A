from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://memory_user:memory_pass@localhost:5432/memory_db"

    # Ollama (local LLM)
    ollama_url: str = "http://172.17.0.1:11434"  # host.docker.internal = your machine from inside Docker
    ollama_model: str = "mistral"

    # Embedding model — separate from the chat model
    embedding_model: str = "nomic-embed-text"

    # Embedding dimension must match your model
    # nomic-embed-text = 768, mistral = 4096, llama3.2 = 3072
    embedding_dim: int = 768

    class Config:
        env_file = ".env"


settings = Settings()