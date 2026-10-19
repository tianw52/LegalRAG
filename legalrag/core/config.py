"""Centralised settings loaded from environment / .env file.

LLM client note
---------------
We use the ``openai`` Python library as a generic HTTP client for any server
that implements the OpenAI Chat Completions API spec.  This includes:

  - Ollama         serving Qwen, Llama, etc.            (local CPU/GPU; default)
  - vLLM           serving Qwen, Llama, Mistral, etc.  (local GPU)
  - HuggingFace Inference Router  (routes to Novita, Together, etc.; paid credits)
  - Alibaba DashScope (cloud Qwen)
  - OpenAI         (GPT-4o, etc.)
"""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """Settings for the LLM backend (any OpenAI-compatible server)."""

    # The base URL of whichever inference server you are using.
    # Ollama (default)     : http://localhost:11434/v1
    # vLLM (lab, tunnel)   : http://localhost:8001/v1
    # HF Router            : https://router.huggingface.co/v1
    # DashScope (cloud)    : https://dashscope.aliyuncs.com/compatible-mode/v1
    # OpenAI (cloud)       : https://api.openai.com/v1
    base_url: str = Field("http://localhost:11434/v1", alias="LLM_BASE_URL")

    # API key sent in the Authorization header.
    # HF Router / DashScope / OpenAI need real keys.  vLLM accepts any string.
    # Set via HF_TOKEN env var or LLM_API_KEY in .env.
    api_key: str = Field("EMPTY", alias="LLM_API_KEY")

    # Model identifier as the inference server expects it.
    # Ollama style     : "qwen3:8b"
    # vLLM style       : "qwen3" (--served-model-name in scripts/serve_qwen3_vllm.sh)
    # HF Router style  : "Qwen/Qwen3.5-397B-A17B:novita"
    # DashScope style  : "qwen-plus"
    # OpenAI style     : "gpt-4o"
    model: str = Field("qwen3:8b", alias="LLM_MODEL")

    # How to switch off Qwen3 "thinking" (slow hidden reasoning); depends on the server:
    #   ollama → sends reasoning_effort: "none"  (Ollama ignores /no_think and think:false)
    #   vllm   → sends chat_template_kwargs.enable_thinking: false
    #   none   → leave thinking on (slower; the reasoning is shown in traces)
    # Kept next to LLM_BASE_URL so switching servers is one .env block.
    thinking_off: Literal["ollama", "vllm", "none"] = Field("ollama", alias="LLM_THINKING_OFF")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


class EmbeddingSettings(BaseSettings):
    provider: str = Field("sentence_transformers", alias="EMBEDDING_PROVIDER")
    model: str = Field("sentence-transformers/all-mpnet-base-v2", alias="EMBEDDING_MODEL")
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
    chunk_size: int = Field(512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(64, alias="CHUNK_OVERLAP")
    reranker_model: str = Field(
        "cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL"
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


class ExploreSettings(BaseSettings):
    """Settings for the exploration app (API + web UI) over CanLegalRAGBench.

    Kept separate from the stable CanLII settings above so both indices can
    coexist.  The embedding model MUST match the one used to build the index.
    """

    index_name: str = Field("canlegal-qwen3", alias="EXPLORE_INDEX_NAME")
    embedding_provider: str = Field("sentence_transformers", alias="EXPLORE_EMBEDDING_PROVIDER")
    embedding_model: str = Field(
        "Qwen/Qwen3-Embedding-0.6B", alias="EXPLORE_EMBEDDING_MODEL"
    )
    # Reranker for the exploration app (build_reranker picks the implementation)
    reranker_model: str = Field("Qwen/Qwen3-Reranker-0.6B", alias="EXPLORE_RERANKER_MODEL")
    # LLM used for view generation; empty → fall back to LLM_MODEL
    llm_model: str = Field("", alias="EXPLORE_LLM_MODEL")
    profiles_dir: str = Field("data/app/profiles", alias="EXPLORE_PROFILES_DIR")
    cors_origins: str = Field("http://localhost:5173", alias="EXPLORE_CORS_ORIGINS")

    # No populate_by_name: it would also match env vars by field name, so
    # EMBEDDING_MODEL / LLM_MODEL from .env would leak into these fields.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class Settings(BaseSettings):
    """Aggregate settings object – import this everywhere."""

    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    opensearch: OpenSearchSettings = Field(default_factory=OpenSearchSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    explore: ExploreSettings = Field(default_factory=ExploreSettings)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# Module-level singleton – re-instantiate in tests by patching this reference.
settings = Settings()
