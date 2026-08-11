"""PTM-CoScientist scientific-reasoning pipeline orchestrator."""

from __future__ import annotations

import logging
from collections.abc import Callable

from src.agents.debater import attach_literature_evidence, run_debate
from src.agents.evolver import run_evolution
from src.agents.generator import run_generation
from src.agents.proximity import cluster_and_select_diverse_hypotheses
from src.agents.reflection import run_reflection
from src.connectors.chromadb_connector import ChromaDBConnector
from src.connectors.ptm_platform_connector import PTMPlatformConnector
from src.core.elo import rank_hypotheses
from src.core.evidence_graph import build_evidence_graph
from src.core.llm_client import LLMClient
from src.core.models import CoScientistState, Hypothesis, LabResult

logger = logging.getLogger(__name__)


class CoScientistPipeline:
    """Orchestrate evidence-grounded, iterative PTM scientific reasoning."""

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
        reflection_enabled: bool = True,
        evidence_graph_enabled: bool = True,
        proximity_enabled: bool = True,
        max_diverse_hypotheses: int = 5,
    ):
        self.llm = llm
        self.chromadb = chromadb
        self.ptm_connector = ptm_connector
        self.max_iterations = max_iterations
        self.generate_candidates = generate_candidates
        self.tournament_rounds = tournament_rounds
        self.evolve_top_k = evolve_top_k
        self.elo_k_factor = elo_k_factor
        self.reflection_enabled = reflection_enabled
        self.evidence_graph_enabled = evidence_graph_enabled
        self.proximity_enabled = proximity_enabled
        self.max_diverse_hypotheses = max_diverse_hypotheses

    def run(
        self,
        order_code: str = "",
        research_goal: str = "",
        ptm_type: str = "phosphorylation",
        rag_collections: list[str] | None = None,
        scientist_feedback: list[dict[str, str]] | None = None,
        progress_callback: Callable[[int, str], None] | None = None,
        order_codes: list[str] | None = None,
        lab_results: list[LabResult] | None = None,
        prior_hypotheses: list[Hypothesis] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> CoScientistState:
        """Execute Generate → Reflection → Debate → Evolve over PTM evidence.

        Input artifacts and ChromaDB remain read-only. Human feedback and lab
        results are carried as explicit scientific-reasoning context; neither is
        silently converted into a causal conclusion.
        """
        codes = order_codes or ([order_code] if order_code else [])
        state = CoScientistState(
            research_goal=research_goal,
            rag_collections=rag_collections or [],
            scientist_feedback=scientist_feedback or [],
            lab_results=lab_results or [],
            max_iterations=self.max_iterations,
        )

        if progress_callback:
            progress_callback(5, "Loading PTM-platform context")

        context = (
            self.ptm_connector.assemble_multi_context(codes, ptm_type)
            if len(codes) != 1
            else self.ptm_connector.assemble_context(codes[0], ptm_type)
        )
        context["research_goal"] = research_goal
        state.enriched_ptm_data = context.get("top_ptms", [])
        state.kinase_modules = context.get("kinase_modules", {})
        state.signal_flow = context.get("signal_flow", {})
        state.comovement_clusters = context.get("comovement_clusters", {})
        state.experimental_context = context
        logger.info("[Pipeline] Loaded context: %d PTMs", context.get("enriched_ptm_count", 0))

        if self.evidence_graph_enabled:
            state.evidence_graph = build_evidence_graph(context, lab_results=state.lab_results)
            logger.info("[Pipeline] Built Evidence Graph: %s", state.evidence_graph.get("summary", {}))

        # Preserve prior candidates on researcher-requested re-runs so that
        # recorded laboratory outcomes remain linked to the same hypothesis IDs.
        all_hypotheses: list[Hypothesis] = list(prior_hypotheses or [])
        for iteration in range(self.max_iterations):
            if cancel_check and cancel_check():
                logger.info("[Pipeline] Cancelled before iteration %d", iteration + 1)
                break
            pct_base = 10 + (iteration * 25)

            if progress_callback:
                progress_callback(pct_base, f"Iteration {iteration + 1}: Generating hypotheses")
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

            if cancel_check and cancel_check():
                logger.info("[Pipeline] Cancelled after Generate (iteration %d)", iteration + 1)
                break

            if self.evidence_graph_enabled:
                state.evidence_graph = build_evidence_graph(context, all_hypotheses, state.lab_results)

            # Retrieve literature before Reflection so literature_consistency and
            # evidence gaps are grounded in ChromaDB results, not an empty slate.
            if progress_callback:
                progress_callback(pct_base + 3, f"Iteration {iteration + 1}: Retrieving literature")
            all_hypotheses = attach_literature_evidence(
                all_hypotheses,
                self.chromadb,
                self.llm,
                rag_collections=rag_collections,
                only_missing=True,
            )
            if self.evidence_graph_enabled:
                state.evidence_graph = build_evidence_graph(context, all_hypotheses, state.lab_results)

            if self.reflection_enabled:
                if progress_callback:
                    progress_callback(pct_base + 5, f"Iteration {iteration + 1}: Self-critique")
                all_hypotheses = run_reflection(
                    all_hypotheses,
                    context=context,
                    evidence_graph=state.evidence_graph,
                    lab_results=state.lab_results,
                    llm=self.llm,
                )

            if cancel_check and cancel_check():
                logger.info("[Pipeline] Cancelled after Reflection (iteration %d)", iteration + 1)
                break

            if progress_callback:
                progress_callback(pct_base + 10, f"Iteration {iteration + 1}: Debating {len(all_hypotheses)} hypotheses")
            all_hypotheses = run_debate(
                hypotheses=all_hypotheses,
                llm=self.llm,
                chromadb=self.chromadb,
                tournament_rounds=self.tournament_rounds,
                k_factor=self.elo_k_factor,
                rag_collections=rag_collections,
            )

            if cancel_check and cancel_check():
                logger.info("[Pipeline] Cancelled after Debate (iteration %d)", iteration + 1)
                break

            if iteration < self.max_iterations - 1:
                if progress_callback:
                    progress_callback(pct_base + 18, f"Iteration {iteration + 1}: Evolving top hypotheses")
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
            logger.info("[Pipeline] Iteration %d complete: %d hypotheses", iteration + 1, len(all_hypotheses))

        state.hypotheses = rank_hypotheses(all_hypotheses)
        if self.proximity_enabled:
            _, state.diversity_summary = cluster_and_select_diverse_hypotheses(
                state.hypotheses,
                max_hypotheses=self.max_diverse_hypotheses,
            )
        if self.evidence_graph_enabled:
            state.evidence_graph = build_evidence_graph(context, state.hypotheses, state.lab_results)
        if progress_callback:
            progress_callback(90, "Core scientific reasoning complete")
        return state

    @staticmethod
    def _build_goal_with_feedback(base_goal: str, feedback: list[dict[str, str]], iteration: int) -> str:
        """Incorporate explicit research guidance without rewriting source evidence."""
        if not feedback:
            return base_goal
        parts = [base_goal]
        for item in feedback:
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            feedback_type = item.get("type")
            prefix = {
                "direction": "Researcher guidance",
                "constraint": "Constraint",
                "seed_idea": "Seed idea to explore",
                "hypothesis_decision": "Researcher hypothesis decision",
                "evidence_feedback": "Researcher evidence feedback",
            }.get(feedback_type, "Researcher feedback")
            parts.append(f"\n{prefix}: {content}")
        if iteration:
            parts.append(f"\nThis is refinement iteration {iteration + 1}; seek a non-duplicative, evidence-grounded angle.")
        return "\n".join(parts)
