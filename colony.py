#!/usr/bin/env python3
"""Colony: ONE solver (levels 1 + 2 rules).

Board: columns A-H, rows 1-8. Pieces: K = Kernel, P = Ping. Teams: C = CYAN, M = MAGENTA.
Scene syntax:  "CK=E4 CP=D4 MK=H4 MP=A4"   (team letter + piece letter = tile)

Usage:
  colony.py "CK=E4 MK=H4 MP=A4" --mode safe            # level-1 style: list safe end tiles
  colony.py "CK=E4 CP=D4 MK=H4 MP=A4" --mode win --depth 2   # forced win within 2 CYAN turns
Optional: --round N (default 1), --cut R (rings already removed, default 0), --to M (side to move)
"""
import argparse
import sys
from functools import lru_cache

C, M = "C", "M"
OTHER = {C: M, M: C}
KDIRS = [(dc, dr) for dc in (-1, 0, 1) for dr in (-1, 0, 1) if (dc, dr) != (0, 0)]
ODIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
DDIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
SHRINK_ROUNDS = (8, 16, 24, 26)


def ring(c, r):
    return min(c, r, 7 - c, 7 - r)


def alive(c, r, cut):
    return 0 <= c < 8 and 0 <= r < 8 and ring(c, r) >= cut


def name(c, r):
    return "ABCDEFGH"[c] + str(r + 1)


def parse_tile(s):
    return "ABCDEFGH".index(s[0].upper()), int(s[1:]) - 1


# A state is (pieces, cut, rnd); pieces is a frozenset of (team, kind, c, r).

def board(pieces):
    return {(c, r): (t, k) for t, k, c, r in pieces}


def protected(b, c, r):
    """Kernel at (c,r) is protected if a friendly piece is in any of 8 adjacent tiles."""
    team = b[(c, r)][0]
    for dc, dr in KDIRS:
        o = b.get((c + dc, r + dr))
        if o and o[0] == team:
            return (c + dc, r + dr)
    return None


def actions(pieces, cut, team):
    """Yield (description, new_pieces, winner) for every legal single action of `team`."""
    b = board(pieces)
    for (c, r), (t, k) in list(b.items()):
        if t != team:
            continue
        if k == "K":
            for dc, dr in KDIRS:
                x, y = c + dc, r + dr
                if not alive(x, y, cut):
                    continue
                o = b.get((x, y))
                if o and o[0] == team:
                    continue
                yield from resolve(b, team, (c, r), k, (x, y), f"K {name(c, r)}->{name(x, y)}")
        else:
            for dc, dr in ODIRS:
                for n in (1, 2):
                    x, y = c + dc * n, r + dr * n
                    if not alive(x, y, cut) or (x, y) in b:
                        break  # empty tiles only, no jumping
                    nb = dict(b)
                    del nb[(c, r)]
                    nb[(x, y)] = (t, k)
                    yield (f"P {name(c, r)}->{name(x, y)}", frozenset((tt, kk, cc, rr) for (cc, rr), (tt, kk) in nb.items()), None)
            for dc, dr in DDIRS:
                for n in (1, 2):
                    x, y = c + dc * n, r + dr * n
                    if not alive(x, y, cut):
                        break
                    o = b.get((x, y))
                    if o is None:
                        continue
                    if o[0] != team:
                        yield from resolve(b, team, (c, r), k, (x, y), f"P {name(c, r)}x{name(x, y)}")
                    break  # first piece blocks the diagonal


def resolve(b, team, src, kind, dst, desc):
    """Apply an attack/move of piece at src onto dst (handles capture + bodyguard)."""
    target = b.get(dst)
    nb = dict(b)
    winner = None
    if target and target[1] == "K":
        guard = protected(b, *dst)
        if guard:
            del nb[guard]  # guard dies, attacker stays on its tile
            return [(desc + f" (guard {name(*guard)} destroyed)",
                     frozenset((t, k, c, r) for (c, r), (t, k) in nb.items()), None)]
        winner = team
    del nb[src]
    nb[dst] = (team, kind)
    return [(desc, frozenset((t, k, c, r) for (c, r), (t, k) in nb.items()), winner)]


def end_of_turn(pieces, cut, rnd, team):
    """Kernel endgame + firewall. Returns (pieces, cut, rnd, winner)."""
    if len(pieces) == 2:
        kc = {t: (c, r) for t, k, c, r in pieces}
        (c1, r1), (c2, r2) = kc[C], kc[M]
        if max(abs(c1 - c2), abs(r1 - r2)) <= 2:
            return pieces, cut, rnd, OTHER[team]
        rc, rm = ring(c1, r1), ring(c2, r2)
        return pieces, cut, rnd, (C if rc > rm else M)
    if team == M:
        if rnd in SHRINK_ROUNDS:
            cut += 1
            dead = [p for p in pieces if ring(p[2], p[3]) < cut]
            pieces = frozenset(p for p in pieces if ring(p[2], p[3]) >= cut)
            lost = {t for t, k, c, r in dead if k == "K"}
            if lost == {C, M}:
                return pieces, cut, rnd, M
            if lost:
                return pieces, cut, rnd, OTHER[lost.pop()]
        rnd += 1
        if rnd > 26:
            return pieces, cut, rnd, M
    return pieces, cut, rnd, None


def turns(pieces, cut, rnd, team):
    """All distinct outcomes of a 2-action turn: {(pieces,cut,rnd): (path, winner)}."""
    out = {}
    for d1, p1, w1 in actions(pieces, cut, team):
        if w1:
            out.setdefault((p1, cut, rnd, w1), ([d1], w1))
            continue
        for d2, p2, w2 in actions(p1, cut, team):
            if w2:
                out.setdefault((p2, cut, rnd, w2), ([d1, d2], w2))
                continue
            np_, nc, nr, w = end_of_turn(p2, cut, rnd, team)
            out.setdefault((np_, nc, nr, w), ([d1, d2], w))
    return out


def can_win(pieces, cut, rnd, n):
    """Does CYAN (to move) have a forced win within n CYAN turns? Returns list of winning paths."""
    wins = []
    for (np_, nc, nr, w), (path, _) in turns(pieces, cut, rnd, C).items():
        if w == C:
            wins.append(path)
        elif w is None and n > 1 and survives_and_wins(np_, nc, nr, n - 1):
            wins.append(path)
    return wins


@lru_cache(maxsize=None)
def survives_and_wins(pieces, cut, rnd, n):
    replies = turns(pieces, cut, rnd, M)
    if not replies:
        return False
    for (np_, nc, nr, w), _ in replies.items():
        if w == M:
            return False
        if w == C:
            continue
        if not can_win(np_, nc, nr, n):
            return False
    return True


def safe_turns(pieces, cut, rnd):
    """Level-1 style: CYAN turns after which MAGENTA cannot win in its next turn (and CYAN hasn't lost)."""
    res = []
    for (np_, nc, nr, w), (path, _) in turns(pieces, cut, rnd, C).items():
        if w == M:
            continue
        if w == C:
            res.append((path, "wins immediately", None))
            continue
        if all(w2 != M for (_, _, _, w2) in turns(np_, nc, nr, M)):
            ck = [(c, r) for t, k, c, r in np_ if t == C and k == "K"][0]
            res.append((path, "safe", ck))
    return res


def parse_scene(s):
    pieces = set()
    for tok in s.split():
        key, tile = tok.split("=")
        c, r = parse_tile(tile)
        pieces.add((key[0].upper(), key[1].upper(), c, r))
    return frozenset(pieces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--mode", choices=["safe", "win"], default="win")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--cut", type=int, default=0)
    a = ap.parse_args()
    pieces = parse_scene(a.scene)
    if a.mode == "safe":
        res = safe_turns(pieces, a.cut, a.round)
        tiles = sorted({name(*ck) for _, _, ck in res if ck})
        print(f"{len(res)} safe CYAN turns; safe end tiles for the Kernel: {', '.join(tiles)}")
        for path, tag, ck in sorted(res, key=lambda x: (name(*x[2]) if x[2] else '', x[0])):
            print(f"  {' ; '.join(path)}   -> K ends on {name(*ck) if ck else '-'} [{tag}]")
    else:
        for depth in range(1, a.depth + 1):
            wins = can_win(pieces, a.cut, a.round, depth)
            if wins:
                print(f"Forced win within {depth} CYAN turn(s): {len(wins)} opening turns")
                for p in sorted(wins):
                    print("  " + " ; ".join(p))
                return
            print(f"No forced win within {depth} CYAN turn(s)")
        sys.exit(1)


if __name__ == "__main__":
    main()
