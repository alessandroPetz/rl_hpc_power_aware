# Trace-Driven Energy Dispatch for HPC Clusters

This repository contains the simulation code and experimental material for the PECS Workshop paper at Euro-Par 2026:

**Trace-Driven Energy Dispatch for HPC Clusters: Cost--Carbon Trade-offs under Grid Power Constraints**

## Contents

- `rule_based_sims.ipynb`  
  Contains the rule-based simulations used in the paper, including the grid-only, battery-assisted, renewable-aware, and generator-supported dispatch policies.

- `rl_agent_sim3_optimize.py`  
  Contains the preliminary reinforcement-learning optimizer for the `All` scenario, based on battery, renewable generation, backup generator, and grid dispatch.

- `requirements.txt`  
  Python dependencies needed to reproduce the experiments.

## Setup

Create and activate a Conda environment, using requirements.txt
