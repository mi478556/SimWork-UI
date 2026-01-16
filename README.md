SimWork-UI 

A Python-based training and experimentation framework for agent–environment systems that supports recording, replaying, editing, and branching interactive simulations.

The system provides a reversible simulation runtime, a deterministic trace player, and a safe state-injection bridge, enabling recorded environment states to be inspected, modified, and re-executed under controlled conditions as part of the training workflow.

It supports structured session logging, clip-based trace exploration, snapshot-based environment editing, human-in-the-loop intervention, and reproducible evaluation from precise states with explicit branch provenance.

Designed to be agent-, environment-, and kernel-agnostic, the framework works with any simulation to generate deterministic, editable interaction traces and high-quality datasets for policy training, debugging, controlled perturbation studies, and surrogate-model development.

The workstation targets research and development workflows where understanding, testing, and shaping agent behavior is as important as optimizing performance.