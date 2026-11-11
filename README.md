# Cloudflight Coding Contest 2026 – Colony: One

Our solutions for the four levels (4th place).

- **Levels 1–2** – `colony.py`: exhaustive search solver for the puzzle scenes
  (`python colony.py "CK=E4 CP=D4 MK=H4 MP=A4" --mode win --depth 2`).
- **Level 3** – `level-3/colony-one-env/build_model.py`: hand-built ONNX model
  (material value + capture-biased policy), no training.
- **Level 4** – in `level-3/colony-one-env/`:
  - `build_model4.py`, `build_model5.py`: hand-built evaluators with threat maps,
    2-action Kernel reach, guards and firewall awareness.
  - `train/gen.py`: self-play data under the referee MCTS.
  - `train/train.py`: residual CNN trained as a correction on top of the v5 evaluator,
    exported as one merged ONNX graph.
  - `train/finetune.py`, `train/tweak.py`: warm-start fine-tuning and policy sharpening.
  - `model.onnx`: final submitted model (net5, policy ×1.5).

The game engine (`colony_one/`, `tools/`, `play/`) is the contest's published environment.
