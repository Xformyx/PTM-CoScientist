"""PTM-CoScientist Configuration.

All configurable values are centralised here and read from environment variables.
"""

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class LLMConfig:
    """LLM provider configuration (mirrors PTM-platform's LLMClient)."""

    provider: str = os.getenv("LLM_PROVIDER", "auto")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
    ollama_model: str = os.getenv("LLM_MODEL", "gemma3:27b")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.4"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))


@dataclass
class PTMPlatformConfig:
    """Read-only connection settings for PTM-platform outputs."""

    chromadb_url: str = os.getenv("CHROMADB_URL", "http://localhost:8000")
    database_url: str = os.getenv(
        "DATABASE_URL",
        "mysql+asyncmy://ptm_reader:reader_password@localhost:3306/ptm_platform?charset=utf8mb4",
    )
    artifacts_dir: str = os.getenv("PTM_ARTIFACTS_DIR", "/data/ptm-platform/outputs")


@dataclass
class CoScientistConfig:
    """Pipeline and scientific-reasoning configuration."""

    # Tournament settings
    max_hypotheses: int = int(os.getenv("MAX_HYPOTHESES", "10"))
    tournament_rounds: int = int(os.getenv("TOURNAMENT_ROUNDS", "3"))
    elo_initial: int = int(os.getenv("ELO_INITIAL", "1500"))
    elo_k_factor: int = int(os.getenv("ELO_K_FACTOR", "32"))

    # Agent settings
    generate_candidates: int = int(os.getenv("GENERATE_CANDIDATES", "5"))
    debate_depth: int = int(os.getenv("DEBATE_DEPTH", "2"))
    evolve_top_k: int = int(os.getenv("EVOLVE_TOP_K", "3"))

    # Scientific reasoning extensions
    reflection_enabled: bool = _env_bool("REFLECTION_ENABLED", True)
    evidence_graph_enabled: bool = _env_bool("EVIDENCE_GRAPH_ENABLED", True)
    proximity_enabled: bool = _env_bool("PROXIMITY_ENABLED", True)
    meta_review_enabled: bool = _env_bool("META_REVIEW_ENABLED", True)
    max_diverse_hypotheses: int = int(os.getenv("MAX_DIVERSE_HYPOTHESES", "5"))

    # Output
    output_dir: str = os.getenv("COSCIENTIST_OUTPUT_DIR", "/data/coscientist/outputs")


@dataclass
class Settings:
    """Root settings container."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    ptm_platform: PTMPlatformConfig = field(default_factory=PTMPlatformConfig)
    coscientist: CoScientistConfig = field(default_factory=CoScientistConfig)


def get_settings() -> Settings:
    """Create settings from environment."""
    return Settings()
