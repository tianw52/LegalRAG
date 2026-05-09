"""Centralised settings loaded from environment / .env file.

LLM client note
---------------
We use the ``openai`` Python library as a generic HTTP client for any server
that implements the OpenAI Chat Completions API spec.  This includes:

  - HuggingFace Inference Router  (default – routes to Novita, Together, etc.)
  - vLLM           serving Qwen, Llama, Mistral, etc.  (local GPU)
  - Ollama         serving Qwen, Llama, etc.            (local CPU/GPU)
  - Alibaba DashScope (cloud Qwen)
  - OpenAI         (GPT-4o, etc.)

Swap backends by changing LLM_BASE_URL + LLM_MODEL in .env – no code changes.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """Settings for the LLM backend (any OpenAI-compatible server)."""

    # The base URL of whichever inference server you are using.
    # HF Router (default)  : https://router.huggingface.co/v1
    # vLLM (local)         : http://localhost:8000/v1
    # Ollama (local)       : http://localhost:11434/v1
    # DashScope (cloud)    : https://dashscope.aliyuncs.com/compatible-mode/v1
    # OpenAI (cloud)       : https://api.openai.com/v1
    base_url: str = Field("https://router.huggingface.co/v1", alias="LLM_BASE_URL")

    # API key sent in the Authorization header.
    # HF Router / DashScope / OpenAI need real keys.  vLLM accepts any string.
    # Set via HF_TOKEN env var or LLM_API_KEY in .env.
    api_key: str = Field("EMPTY", alias="LLM_API_KEY")

    # Model identifier as the inference server expects it.
    # HF Router style  : "Qwen/Qwen3.5-397B-A17B:novita"
    # vLLM style       : "Qwen/Qwen2.5-7B-Instruct"
    # Ollama style     : "qwen2.5:7b"
    # DashScope style  : "qwen-plus"
    # OpenAI style     : "gpt-4o"
    model: str = Field("Qwen/Qwen3.5-9B", alias="LLM_MODEL")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


class EmbeddingSettings(BaseSettings):
    provider: str = Field("sentence_transformers", alias="EMBEDDING_PROVIDER")
    model: str = Field("nlpaueb/legal-bert-base-uncased", alias="EMBEDDING_MODEL")
    dim: int = Field(768, alias="EMBEDDING_DIM")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


class OpenSearchSettings(BaseSettings):
    host: str = Field("localhost", alias="OPENSEARCH_HOST")
    port: int = Field(9200, alias="OPENSEARCH_PORT")
    user: str = Field("admin", alias="OPENSEARCH_USER")
    password: str = Field("admin", alias="OPENSEARCH_PASSWORD")
    use_ssl: bool = Field(False, alias="OPENSEARCH_USE_SSL")
    index_name: str = Field("legalrag", alias="OPENSEARCH_INDEX_NAME")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


class RetrievalSettings(BaseSettings):
    top_k: int = Field(100, alias="RETRIEVAL_TOP_K")
    rerank_top_k: int = Field(100, alias="RERANK_TOP_K")
    chunk_size: int = Field(512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(64, alias="CHUNK_OVERLAP")
    reranker_model: str = Field(
        "cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL"
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


class Settings(BaseSettings):
    """Aggregate settings object – import this everywhere."""

    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    opensearch: OpenSearchSettings = Field(default_factory=OpenSearchSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# Module-level singleton – re-instantiate in tests by patching this reference.
settings = Settings()
