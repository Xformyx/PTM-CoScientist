"""
PTM-CoScientist Configuration.

All configurable values are centralized here.
Reads from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    """LLM provider configuration (mirrors PTM-platform's LLMClient)."""

    provider: str = os.getenv("LLM_PROVIDER", "auto")  # auto | ollama | openai | gemini
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
    """Connection settings for reading PTM-platform artifacts (read-only)."""

    # ChromaDB (shared instance from PTM-platform)
    chromadb_url: str = os.getenv("CHROMADB_URL", "http://localhost:8000")

    # MySQL (read-only access to PTM-platform's orders/enriched data)
    database_url: str = os.getenv(
        "DATABASE_URL",
        "mysql+asyncmy://ptm_reader:reader_password@localhost:3306/ptm_platform?charset=utf8mb4",
    )

    # File-based artifacts (mounted volume from PTM-platform)
    artifacts_dir: str = os.getenv("PTM_ARTIFACTS_DIR", "/data/ptm-platform/outputs")


@dataclass
class CoScientistConfig:
    """Co-Scientist pipeline configuration."""

    # Tournament settings
    max_hypotheses: int = int(os.getenv("MAX_HYPOTHESES", "10"))
    tournament_rounds: int = int(os.getenv("TOURNAMENT_ROUNDS", "3"))
    elo_initial: int = int(os.getenv("ELO_INITIAL", "1500"))
    elo_k_factor: int = int(os.getenv("ELO_K_FACTOR", "32"))

    # Agent settings
    generate_candidates: int = int(os.getenv("GENERATE_CANDIDATES", "5"))
    debate_depth: int = int(os.getenv("DEBATE_DEPTH", "2"))
    evolve_top_k: int = int(os.getenv("EVOLVE_TOP_K", "3"))

    # Output
    output_dir: str = os.getenv("COSCIENTIST_OUTPUT_DIR", "/data/coscientist/outputs")


@dataclass
class Settings:
    """Root settings container."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    ptm_platform: PTMPlatformConfig = field(default_factory=PTMPlatformConfig)
    coscientist: CoScientistConfig = field(default_factory=CoScientistConfig)


def get_settings() -> Settings:
    """Factory to create settings from environment."""
    return Settings()
