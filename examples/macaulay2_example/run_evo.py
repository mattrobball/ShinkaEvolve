#!/usr/bin/env python3
"""
Evolution runner for Macaulay2 resurgence optimization.

Goal: Find point configurations in affine/projective space with maximal resurgence.

Resurgence ρ(I) = sup{m/r : I^(m) ⊄ I^r} measures how symbolic powers
compare to ordinary powers. Higher resurgence indicates more interesting
algebraic/geometric properties.

Known results:
- Generic points: ρ ≈ 1
- Special configurations can have ρ > 1
- Finding ρ > 3/2 means I^(3) ⊄ I^2 (the famous containment problem)
- Fermat configurations, star configurations have been studied
"""

from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

job_config = LocalJobConfig(eval_program_path="evaluate.py")

# Use weighted strategy for balanced exploration/exploitation
strategy = "weighted"
if strategy == "uniform":
    parent_config = dict(
        parent_selection_strategy="power_law",
        exploitation_alpha=0.0,
        exploitation_ratio=1.0,
    )
elif strategy == "weighted":
    parent_config = dict(
        parent_selection_strategy="weighted",
        parent_selection_lambda=10.0,
    )
elif strategy == "power_law":
    parent_config = dict(
        parent_selection_strategy="power_law",
        exploitation_alpha=1.0,
        exploitation_ratio=0.2,
    )

db_config = DatabaseConfig(
    db_path="evolution_db.sqlite",
    num_islands=2,
    archive_size=40,
    elite_selection_ratio=0.3,
    num_archive_inspirations=4,
    num_top_k_inspirations=2,
    migration_interval=10,
    migration_rate=0.1,
    island_elitism=True,
    **parent_config,
)

search_task_sys_msg = """You are an expert in commutative algebra and algebraic geometry, specializing in symbolic powers of ideals and the containment problem.

GOAL: Find point configurations in affine 2-space (or projective 2-space) whose ideal I has maximal resurgence.

BACKGROUND:
- The resurgence ρ(I) = sup{m/r : I^(m) ⊄ I^r} measures containment failures between symbolic and ordinary powers
- For generic points, resurgence approaches 1
- Special configurations can have higher resurgence
- Finding configurations where I^(3) ⊄ I^2 (resurgence > 3/2) is particularly interesting

KNOWN HIGH-RESURGENCE CONFIGURATIONS:
1. Fermat configurations: Points defined by x^n - y^n = 0 and similar
2. Star configurations: Points at intersections of lines through a common point
3. Points on special curves (e.g., cuspidal cubics)
4. Configurations with high multiplicity structure

MACAULAY2 SYNTAX:
- Define ring: R = QQ[x,y] or R = QQ[x,y,z] for projective
- Point ideals: ideal(x-a, y-b) for point (a,b)
- Intersection: I = intersect {ideal1, ideal2, ...}
- Can also use saturations, primary decomposition, etc.

DIRECTIONS TO EXPLORE:
1. Try different numbers of points (more points = more possibilities)
2. Explore algebraic relationships between point coordinates
3. Consider points with rational or algebraic coordinates
4. Try configurations with symmetry (cyclic, dihedral groups)
5. Explore points on curves with special properties
6. Consider weighted/fat point schemes (though standard points are the focus)

The evolved code must define an ideal I in a polynomial ring R over QQ.

Be creative! This is an open research problem - novel configurations that achieve high resurgence could be mathematically significant."""

evo_config = EvolutionConfig(
    task_sys_msg=search_task_sys_msg,
    patch_types=["diff", "full", "cross"],
    patch_type_probs=[0.5, 0.4, 0.1],
    num_generations=200,
    max_parallel_jobs=3,
    max_patch_resamples=3,
    max_patch_attempts=3,
    job_type="local",
    language="macaulay2",
    llm_models=["gemini-3-pro-preview"],
    llm_kwargs=dict(
        temperatures=[0.0, 0.5, 1.0],
        reasoning_efforts=["auto", "low", "medium", "high"],
        max_tokens=32768,
    ),
    meta_rec_interval=10,
    meta_llm_models=["gemini-3-flash-preview"],
    meta_llm_kwargs=dict(temperatures=[0.0], max_tokens=16384),
    embedding_model="gemini-embedding-001",
    code_embed_sim_threshold=0.995,
    novelty_llm_models=["gemini-3-flash-preview"],
    novelty_llm_kwargs=dict(temperatures=[0.0], max_tokens=16384),
    llm_dynamic_selection="ucb1",
    llm_dynamic_selection_kwargs=dict(exploration_coef=1.0),
    init_program_path="initial.m2",
    results_dir="results_resurgence",
    use_text_feedback=True,
)


def main():
    print("=" * 60)
    print("Resurgence Optimization - Finding High-Resurgence Ideals")
    print("=" * 60)
    print()
    print("Goal: Find point configurations with maximal resurgence ρ(I)")
    print("      where ρ(I) = sup{m/r : I^(m) ⊄ I^r}")
    print()

    evo_runner = EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        verbose=True,
    )
    evo_runner.run()

    print("\n" + "=" * 60)
    print("Evolution Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
