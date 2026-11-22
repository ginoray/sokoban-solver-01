[![Build the best Sokoban solver](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![Rank](https://make.fun/badge/sokoban-solver-01-run-20261121/ginoray/sokoban-solver-01/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![Reward](https://make.fun/badge/sokoban-solver-01-run-20261121/ginoray/sokoban-solver-01/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)

# Room-by-room push-optimal Sokoban solver

`main.ts` (Node 22, no dependencies) answers a set of five Sokoban levels: it reads each
`/input/<id>/level.txt` (XSB) and writes the player's moves to `/output/<id>/solution.txt`,
in as few pushes as the level allows.

## How it works

The levels are a one-wide hallway with one to four small rooms off it. A box pushed out of
a room into the hallway can never be pushed back (the player would have to stand on the
hallway's far side, which is wall), so every box stays in its room and a level's
push-optimal count is the sum of its rooms' optima. The solver exploits exactly that:

1. **Split.** Squares from which a lone box can reach no goal are dead; the hallway's
   squares are all dead. Cutting the dead squares of the player's row out of the floor
   leaves one connected part per room.
2. **Solve each room** with an A\* search over pushes, the other rooms' boxes standing as
   walls and the player walking on from where the last room left them:
   - a state is the room's sorted box squares plus the player's reachable region, named
     by its lowest square — states that differ only in where the player stands inside
     the region merge into one;
   - the heuristic assigns boxes to distinct goals at their true single-box push
     distances (flooded backwards from each goal over the push graph, other boxes
     ignored). It never overstates the pushes still needed, so the first solution the
     search settles on is push-optimal for the room;
   - pushes are pruned when they land a box on a dead square, complete a frozen 2 x 2
     block with a box off its goal, or leave a box frozen — every push of it blocked by
     walls or by boxes already frozen, iterated to a fixed point.
3. **Concatenate** the rooms' move strings; walking between pushes and between rooms is
   free, only pushes are scored.

The search is budgeted by states examined, never by the clock, and every tie is broken by
a fixed order, so two runs write byte-identical answers. On the kit's practice set it
solves all five instances in their known optimal pushes (13, 16, 29, 47, 60) in well
under a second, against the 60-second budget for a set.

## Layout

- `main.ts` — the whole solver: parser, room split, push A\* with deadlock pruning, and
  the move reconstruction.
- `LICENSE` — MIT.
