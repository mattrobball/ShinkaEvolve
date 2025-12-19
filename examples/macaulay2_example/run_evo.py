#!/usr/bin/env python3
"""Run Macaulay2 evolution for 5 generations."""
from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig
from pathlib import Path

# Configure local job execution
job_config = LocalJobConfig(eval_program_path="evaluate.py")

# Configure database
db_config = DatabaseConfig(
    db_path="evolution_db.sqlite",
    num_islands=2,
    archive_size=20,
    exploitation_ratio=0.2,
    elite_selection_ratio=0.3,
    num_archive_inspirations=4,
    num_top_k_inspirations=2,
    migration_interval=10,
    migration_rate=0.1,
    island_elitism=True,
)

# Task system message for Macaulay2
task_sys_msg = """You are an expert in computational algebra and Macaulay2 programming.
Your task is to write efficient and correct Macaulay2 programs.

Macaulay2 is a software system for algebraic geometry and commutative algebra.
It provides a high-level programming language for mathematical computation.

Key considerations:
1. Ensure your code follows Macaulay2 syntax and conventions
2. Use appropriate data structures (rings, ideals, modules, etc.)
3. Consider computational efficiency for algebraic operations
4. Test edge cases and handle errors appropriately

Be creative and explore different approaches to solving the problem."""

# Configure evolution
evo_config = EvolutionConfig(
    task_sys_msg=task_sys_msg,
    patch_types=["diff", "full", "cross"],
    patch_type_probs=[0.6, 0.3, 0.1],
    num_generations=5,
    max_parallel_jobs=5,
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
    meta_llm_models=["gemini-2.5-flash"],
    meta_llm_kwargs=dict(temperatures=[0.0], max_tokens=16384),
    embedding_model="gemini-embedding-001",
    code_embed_sim_threshold=0.995,
    novelty_llm_models=["gemini-2.5-flash"],
    novelty_llm_kwargs=dict(temperatures=[0.0], max_tokens=16384),
    llm_dynamic_selection="ucb1",
    llm_dynamic_selection_kwargs=dict(exploration_coef=1.0),
    init_program_path="initial.m2",
    results_dir="results_cpack",
)

def main():
    print("=" * 60)
    print("Starting Macaulay2 Evolution Test - 5 Generations")
    print("=" * 60)

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
