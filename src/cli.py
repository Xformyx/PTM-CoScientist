"""
CLI Interface for PTM-CoScientist.

Provides command-line access to the Co-Scientist pipeline
for quick testing and interactive use.
"""

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.group()
def main():
    """PTM-CoScientist: AI Co-Scientist for PTM Research."""


@main.command()
@click.argument("order_code")
@click.option("--goal", "-g", default="", help="Research goal in natural language")
@click.option("--ptm-type", "-t", default="phosphorylation", help="PTM type")
@click.option("--iterations", "-i", default=3, help="Number of Generate-Debate-Evolve iterations")
@click.option("--output", "-o", default=None, help="Output directory")
def run(order_code: str, goal: str, ptm_type: str, iterations: int, output: str):
    """Run the Co-Scientist pipeline on a PTM-platform order."""
    from config.settings import get_settings
    from src.agents.experiment_designer import run_experiment_design
    from src.agents.meta_reviewer import run_meta_review
    from src.agents.proximity import cluster_and_select_diverse_hypotheses
    from src.connectors.chromadb_connector import ChromaDBConnector
    from src.connectors.ptm_platform_connector import PTMPlatformConnector
    from src.core.llm_client import LLMClient
    from src.core.pipeline import CoScientistPipeline

    settings = get_settings()

    console.print(Panel(
        f"[bold]PTM-CoScientist[/bold]\n"
        f"Order: {order_code}\n"
        f"Goal: {goal or '(auto-generated from data)'}\n"
        f"PTM Type: {ptm_type}\n"
        f"Iterations: {iterations}",
        title="Configuration",
    ))

    # Initialize components
    llm = LLMClient(
        provider=settings.llm.provider,
        model=settings.llm.ollama_model,
        ollama_url=settings.llm.ollama_url,
        openai_api_key=settings.llm.openai_api_key,
        openai_model=settings.llm.openai_model,
        gemini_api_key=settings.llm.gemini_api_key,
        gemini_model=settings.llm.gemini_model,
    )

    if not llm.is_available():
        console.print("[red]ERROR: No LLM provider available. Check configuration.[/red]")
        sys.exit(1)

    chromadb = ChromaDBConnector(settings.ptm_platform.chromadb_url)
    ptm_conn = PTMPlatformConnector(artifacts_dir=settings.ptm_platform.artifacts_dir)

    pipeline = CoScientistPipeline(
        llm=llm,
        chromadb=chromadb,
        ptm_connector=ptm_conn,
        max_iterations=iterations,
        generate_candidates=settings.coscientist.generate_candidates,
        tournament_rounds=settings.coscientist.tournament_rounds,
        evolve_top_k=settings.coscientist.evolve_top_k,
        elo_k_factor=settings.coscientist.elo_k_factor,
        reflection_enabled=settings.coscientist.reflection_enabled,
        evidence_graph_enabled=settings.coscientist.evidence_graph_enabled,
        proximity_enabled=settings.coscientist.proximity_enabled,
        max_diverse_hypotheses=settings.coscientist.max_diverse_hypotheses,
    )

    def progress(pct, msg):
        console.print(f"  [{pct:3d}%] {msg}")

    console.print("\n[bold cyan]Running pipeline...[/bold cyan]\n")
    state = pipeline.run(
        order_code=order_code,
        research_goal=goal,
        ptm_type=ptm_type,
        progress_callback=progress,
    )

    # Use proximity representatives for a diverse experimental portfolio.
    if settings.coscientist.proximity_enabled:
        selected, state.diversity_summary = cluster_and_select_diverse_hypotheses(
            state.hypotheses,
            max_hypotheses=settings.coscientist.max_diverse_hypotheses,
        )
    else:
        selected = state.hypotheses[:settings.coscientist.max_diverse_hypotheses]

    console.print("\n[bold cyan]Designing experiments...[/bold cyan]\n")
    state.experiment_designs = run_experiment_design(
        hypotheses=selected,
        llm=llm,
        experimental_context=state.experimental_context,
        top_n=len(selected),
    )
    if settings.coscientist.meta_review_enabled:
        state.meta_review = run_meta_review(
            research_goal=state.research_goal,
            hypotheses=selected,
            evidence_graph_summary=state.evidence_graph.get("summary", {}),
            experiment_designs=state.experiment_designs,
            lab_results=state.lab_results,
            scientist_feedback=state.scientist_feedback,
            llm=llm,
        )

    # Display results
    _display_results(state)

    # Save
    out_dir = Path(output or settings.coscientist.output_dir) / order_code
    out_dir.mkdir(parents=True, exist_ok=True)
    results = state.to_dict()
    results["order_code"] = order_code
    results["research_goal"] = goal
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    console.print(f"\n[green]Results saved to {out_dir / 'results.json'}[/green]")


@main.command()
def serve():
    """Start the API server."""
    import uvicorn
    console.print("[bold]Starting PTM-CoScientist API server on port 8080...[/bold]")
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8080, reload=False)


@main.command()
@click.argument("results_file")
def show(results_file: str):
    """Display results from a saved JSON file."""
    path = Path(results_file)
    if not path.exists():
        console.print(f"[red]File not found: {results_file}[/red]")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    hypotheses = data.get("hypotheses", [])
    designs = data.get("experiment_designs", [])

    console.print(Panel(f"Results: {len(hypotheses)} hypotheses, {len(designs)} experiments", title="Summary"))

    # Show top hypotheses
    table = Table(title="Top Hypotheses (by Elo)")
    table.add_column("#", width=3)
    table.add_column("Elo", width=6)
    table.add_column("Category", width=12)
    table.add_column("IF → THEN", min_width=40)
    table.add_column("Confidence", width=10)

    for i, h in enumerate(hypotheses[:10], 1):
        table.add_row(
            str(i),
            str(h.get("elo_rating", 0)),
            h.get("category", ""),
            f"{h.get('condition', '')[:40]} → {h.get('prediction', '')[:40]}",
            f"{h.get('confidence', 0):.2f}",
        )

    console.print(table)


def _display_results(state):
    """Display pipeline results in rich format."""
    console.print(f"\n[bold]Pipeline Complete[/bold] — {state.iteration} iterations\n")

    # Hypotheses table
    table = Table(title=f"Top Hypotheses ({len(state.hypotheses)} total)")
    table.add_column("#", width=3)
    table.add_column("Elo", width=6)
    table.add_column("Category", width=12)
    table.add_column("Hypothesis (IF → THEN)", min_width=50)
    table.add_column("Conf.", width=6)

    for i, h in enumerate(state.hypotheses[:10], 1):
        table.add_row(
            str(i),
            str(h.elo_rating),
            h.category.value,
            f"{h.condition[:35]}... → {h.prediction[:35]}...",
            f"{h.confidence:.2f}",
        )

    console.print(table)

    # Experiment designs
    if state.experiment_designs:
        console.print(f"\n[bold]Experiment Designs ({len(state.experiment_designs)} total)[/bold]\n")
        for i, d in enumerate(state.experiment_designs[:5], 1):
            console.print(Panel(
                f"[bold]{d.title}[/bold]\n"
                f"Approach: {d.approach}\n"
                f"Objective: {d.objective}\n"
                f"Expected: {d.expected_outcome[:100]}\n"
                f"Timeline: {d.estimated_timeline}",
                title=f"Experiment {i} [{d.priority.upper()}]",
            ))


if __name__ == "__main__":
    main()
