# Colony: One – Cloudflight Coding Contest 2026

Solutions to all four levels of the Cloudflight Coding Contest 2026 (October 2026).
**Final result: 4th place** in the live Level 4 tournament against the other teams' models.

Colony: One is a two-player strategy game on a shrinking 8×8 board. Each team plays
two actions per turn with pieces that move and shoot in different patterns, and the
goal is to destroy the enemy Kernel. Level 4 was a live, Glicko-2-rated tournament:
each team submits one ONNX model, and the organisers' Monte-Carlo Tree Search (MCTS)
uses it to evaluate positions and choose moves.

## Levels

| Level | Task | Approach |
|---|---|---|
| 1 | Reach a tile safe from the enemy in two actions | Exhaustive search over both teams' turns (`colony.py`) |
| 2 | Force a win with Kernel + Ping | Minimax search for forced wins, with memoisation (`colony.py`) |
| 3 | Submit a model that beats a greedy bot | Hand-built ONNX model: material value + capture-biased policy |
| 4 | Live tournament against all teams | Hand-built evaluator, then self-play training on top of it |

## Level 4: how the final model was built

The model never chooses moves itself: it scores positions (value) and suggests
moves (policy), and the referee's MCTS does the lookahead with 100 simulations per action.

1. **Hand-built evaluator (v4, v5).** Fixed convolution kernels compute, for every
   piece type, the tiles it can attack. On top of material, the value adds threats,
   two-action reach on Kernels (move then shoot), bodyguard checks and a penalty
   for standing on the ring the firewall is about to remove. These were added after
   reading the tournament replays to see how the model was losing.
2. **Self-play data.** About 335,000 positions from games played under the referee's
   MCTS, storing the search's visit counts and the game result.
3. **Learned correction.** A small residual CNN (4 blocks, 64 channels) learns a
   correction to the hand-built value (`value = tanh(heuristic + net)`) and a policy
   head. Network and evaluator are exported together as one ONNX graph built from
   allow-listed operators only.
4. **Iteration.** Each new model was only kept if it clearly beat the previous one
   in a few hundred local games, playing both colours.

| Model | Change | Local result vs previous |
|---|---|---|
| v4 | Threat maps on top of material | 67% vs Level 3 model |
| v5 | Two-action Kernel reach, guards, firewall | 68% vs v4 |
| net1 | First trained correction | 64% vs v5 |
| net2 | Longer training | 75% vs net1 |
| net3 | More data and epochs | 62% vs net2 |
| net4 | Wider network (64 channels) | 70% vs net3 |
| net5 (final) | Warm-start fine-tune + sharper policy | 64% vs net4 |

## Repository layout

```
colony.py                        Levels 1–2 solver (own rules engine + search)
level-3/colony-one-env/
  build_model.py                 Level 3 hand-built model
  build_model4.py, build_model5.py   Level 4 hand-built evaluators
  train/gen.py                   Parallel self-play data generation
  train/train.py                 Training + export of the merged ONNX model
  train/finetune.py              Warm-start fine-tuning from an exported model
  train/tweak.py                 Policy/value scaling of an exported model
  model.onnx                     Final submitted model
  colony_one/, tools/, play/, docs/   Game engine and tools provided by the organisers
```

## Running it

Requires Python 3.11–3.14 (64-bit).

```bash
python colony.py "CK=E4 CP=D4 MK=H4 MP=A4" --mode win --depth 2
```

```bash
cd level-3/colony-one-env
python -m venv .venv && source .venv/bin/activate
pip install -e . torch
python tools/validate_submission.py model.onnx
python play/model_vs_model.py --model-a model.onnx --model-b play/random.onnx --games 20
```

To retrain: `python build_model5.py model_v5.onnx fw=150`, then
`python train/gen.py model_v5.onnx 2000 40` and
`python train/train.py model_v5.onnx out.onnx 6`.

## Credits

The game engine, MCTS and validation tools in `colony_one/`, `tools/` and `play/`
were published by the contest organisers (see their `LICENSE`). Everything else is our own work.
