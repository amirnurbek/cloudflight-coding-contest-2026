"""Self-play data: heuristic model under referee MCTS (100 sims). Saves obs, visit policy, outcome."""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(args):
    wid, n_games, model = args
    from colony_one import mcts
    from colony_one.env import ColonyOneEnv
    from colony_one.onnx_runner import OnnxEvaluator
    ev = OnnxEvaluator(ROOT / model)
    cfg = mcts.MCTSConfig(simulations=100, batch_size=8)
    rng = np.random.default_rng(wid * 7919 + 13)
    O, P, Z = [], [], []
    for _ in range(n_games):
        env = ColonyOneEnv()
        obs, pol, teams = [], [], []
        while not env.terminated:
            team = env.current_team
            root = mcts.run(env.board, team, env.actions_left, env.round_number, ev, cfg, rng=rng)
            temp = mcts.warmup_temperature(mcts.move_number(team, env.actions_left, env.round_number),
                                           cfg.warmup_temperatures, cfg.warmup_step_actions)
            if rng.random() < 0.1:
                temp = max(temp, 1.0)  # extra exploration
            obs.append(np.rint(env.observation() * 8).astype(np.uint8))
            ids, cnt = mcts.visit_distribution(root)
            pi = np.full(64, -1, np.int16); pp = np.zeros(64, np.float16)
            order = np.argsort(cnt)[::-1][:64]; tot = max(1, sum(cnt))
            pi[:len(order)] = np.array(ids)[order]; pp[:len(order)] = np.array(cnt)[order] / tot
            pol.append((pi, pp)); teams.append(team)
            idx = mcts.select_action_index(root, temp, rng=rng)
            env.step(idx)
        for t in teams:
            Z.append(1.0 if t == env.winner else -1.0)
        O += obs; P += pol
    np.savez_compressed(ROOT / "train/data" / f"{model.replace('/', '_')}_{wid}.npz",
                        obs=np.array(O, dtype=np.uint8), pidx=np.array([a for a, _ in P], dtype=np.int16),
                        pp=np.array([b for _, b in P], dtype=np.float16), z=np.array(Z, dtype=np.float32))
    return len(Z)


if __name__ == "__main__":
    model, games, chunks = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    per = games // chunks
    with Pool(10) as p:
        total = sum(p.map(worker, [(i, per, model) for i in range(chunks)]))
    print("positions", total)
