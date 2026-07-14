"""
Co-Scientist Pipeline Orchestrator.

Implements the Generate → Debate → Evolve loop using LangGraph StateGraph.
Supports iterative refinement with Scientist-in-the-loop feedback.
"""

import logging
from typing import Any, Dict, List, Optional

from src.core.llm_client import LLMClient
from src.core.models import CoScientistState, Hypothesis, HypothesisStatus
from src.core.elo import rank_hypotheses
from src.connectors.chromadb_connector import ChromaDBConnector
from src.connectors.ptm_platform_connector import PTMPlatformConnector
from src.agents.generator import run_generation
from src.agents.debater import run_debate
from src.agents.evolver import run_evolution

logger = logging.getLogger(__name__)


class CoScientistPipeline:
    """
    Main orchestrator for the Generate → Debate → Evolve loop.

    Manages the iterative hypothesis refinement cycle with optional
    scientist feedback between iterations.
    """

    def __init__(
        self,
        llm: LLMClient,
        chromadb: ChromaDBConnector,
        ptm_connector: PTMPlatformConnector,
        max_iterations: int = 3,
        generate_candidates: int = 5,
        tournament_rounds: int = 3,
        evolve_top_k: int = 3,
        elo_k_factor: int = 32,
    ):
        self.llm = llm
        self.chromadb = chromadb
        self.ptm_connector = ptm_connector
        self.max_iterations = max_iterations
        self.generate_candidates = generate_candidates
        self.tournament_rounds = tournament_rounds
        self.evolve_top_k = evolve_top_k
        self.elo_k_factor = elo_k_factor

    def run(
        self,
        order_code: str,
        research_goal: str = "",
        ptm_type: str = "phosphorylation",
        rag_collections: Optional[List[str]] = None,
        scientist_feedback: Optional[List[Dict[str, str]]] = None,
        progress_callback=None,
    ) -> CoScientistState:
        """
        Execute the full Co-Scientist pipeline.

        Args:
            order_code: PTM-platform order code to load context from
            research_goal: Natural language research goal
            ptm_type: "phosphorylation" or "ubiquitylation"
            rag_collections: Specific ChromaDB collections to use
            scientist_feedback: Previous feedback from researcher
            progress_callback: Optional callback(pct, message)

        Returns:
            CoScientistState with all hypotheses and results
        """
        state = CoScientistState(
            research_goal=research_goal,
            rag_collections=rag_collections or [],
            scientist_feedback=scientist_feedback or [],
            max_iterations=self.max_iterations,
        )

        # ─── Phase 1: Load Context ───────────────────────────────────────
        if progress_callback:
            progress_callback(5, "Loading PTM-platform context")

        context = self.ptm_connector.assemble_context(order_code, ptm_type)
        state.enriched_ptm_data = context.get("top_ptms", [])
        state.kinase_modules = context.get("kinase_modules", {})
        state.signal_flow = context.get("signal_flow", {})
        state.comovement_clusters = context.get("comovement_clusters", {})
        state.experimental_context = context

        logger.info(f"[Pipeline] Loaded context: {context.get('enriched_ptm_count', 0)} PTMs")

        # ─── Phase 2: Iterative Generate → Debate → Evolve ──────────────
        all_hypotheses: List[Hypothesis] = []

        for iteration in range(self.max_iterations):
            pct_base = 10 + (iteration * 25)

            if progress_callback:
                progress_callback(pct_base, f"Iteration {iteration + 1}: Generating hypotheses")

            # GENERATE
            new_hypotheses = run_generation(
                context=context,
                llm=self.llm,
                n_candidates=self.generate_candidates,
                research_goal=self._build_goal_with_feedback(research_goal, state.scientist_feedback, iteration),
                iteration=iteration,
                chromadb=self.chromadb,
                rag_collections=rag_collections,
            )
            all_hypotheses.extend(new_hypotheses)

            if progress_callback:
                progress_callback(pct_base + 8, f"Iteration {iteration + 1}: Debating {len(all_hypotheses)} hypotheses")

            # DEBATE
            all_hypotheses = run_debate(
                hypotheses=all_hypotheses,
                llm=self.llm,
                chromadb=self.chromadb,
                tournament_rounds=self.tournament_rounds,
                k_factor=self.elo_k_factor,
                rag_collections=rag_collections,
            )

            if progress_callback:
                progress_callback(pct_base + 16, f"Iteration {iteration + 1}: Evolving top hypotheses")

            # EVOLVE (from iteration 1 onward)
            if iteration < self.max_iterations - 1:
                evolved = run_evolution(
                    hypotheses=all_hypotheses,
                    llm=self.llm,
                    top_k=self.evolve_top_k,
                    context=context,
                )
                all_hypotheses.extend(evolved)

            state.iteration = iteration + 1
            state.tournament_history.append({
                "iteration": iteration,
                "total_hypotheses": len(all_hypotheses),
                "top_elo": all_hypotheses[0].elo_rating if all_hypotheses else 0,
                "avg_elo": sum(h.elo_rating for h in all_hypotheses) / len(all_hypotheses) if all_hypotheses else 0,
            })

            logger.info(
                f"[Pipeline] Iteration {iteration + 1} complete: "
                f"{len(all_hypotheses)} hypotheses, top Elo={all_hypotheses[0].elo_rating if all_hypotheses else 0}"
            )

        # ─── Phase 3: Final ranking ─────────────────────────────────────
        state.hypotheses = rank_hypotheses(all_hypotheses)

        if progress_callback:
            progress_callback(90, "Pipeline complete")

        return state

    def _build_goal_with_feedback(self, base_goal: str, feedback: List[Dict[str, str]], iteration: int) -> str:
        """Incorporate scientist feedback into the research goal."""
        if not feedback:
            return base_goal

        parts = [base_goal]
        for fb in feedback:
            if fb.get("type") == "direction":
                parts.append(f"\nResearcher guidance: {fb['content']}")
            elif fb.get("type") == "constraint":
                parts.append(f"\nConstraint: {fb['content']}")
            elif fb.get("type") == "seed_idea":
                parts.append(f"\nSeed idea to explore: {fb['content']}")

        return "\n".join(parts)
