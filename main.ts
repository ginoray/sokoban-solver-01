// sokoban-solver-01: a push-optimal Sokoban solver for make.fun's generated levels.
//
// A level is a one-wide hallway with one to four small rooms off it. A box pushed out of a
// room into the hallway can never come back (pushing it through the door again would need
// the player on the hallway's far side, which is wall), so every box stays in its room and
// the level's push optimum is the sum of its rooms' optima. This solver therefore splits
// the level into its rooms and solves each room on its own with an A* search over pushes:
//
//   - a state is the room's sorted box squares plus the player's reachable region, named
//     by the region's lowest square (where exactly the player stands inside the region
//     cannot change what pushes are possible, so states that differ only in that merge);
//   - the heuristic is the cheapest assignment of boxes to distinct goals, where moving
//     one box to one goal costs its true single-box push distance (a flood backwards from
//     the goal over the push graph, the other boxes ignored). Ignoring the other boxes can
//     only shorten a box's trip, so the assignment cost never overstates the pushes still
//     needed and the first solution A* settles on is push-optimal for the room;
//   - pushes that land a box on a dead square (one from which no goal is reachable in the
//     single-box flood), complete a frozen 2 x 2 block of walls and boxes with a box off
//     its goal, or leave any box frozen — every push of it blocked by walls or by boxes
//     already frozen, iterated to a fixed point — are pruned: none can appear in a
//     solution.
//
// The search is budgeted by states examined, never by the clock, and every tie is broken
// by a fixed order, so two runs of this program write byte-identical answers.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The most states one board search may keep before it gives up: work, never the clock. */
const STATE_CAP = 800_000;
const INF = 1_000_000_000;

/** The four steps: [dx, dy, walk letter, push letter]. */
const STEP: ReadonlyArray<readonly [number, number, string, string]> = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

/** Parse XSB text: short rows and the board's outer frame count as wall. */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let x = 0; x < w; x++) {
    wall[x] = 1;
    wall[(h - 1) * w + x] = 1;
  }
  for (let y = 0; y < h; y++) {
    wall[y * w] = 1;
    wall[y * w + w - 1] = 1;
  }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

/**
 * For every goal, the fewest pushes that bring a lone box from each square onto it (the
 * other boxes ignored), flooded backwards from the goal: a box now on `t` may have been
 * pushed there from `t - d`, with the player standing on `t - 2d`. Squares no goal's flood
 * reaches are dead: a box pushed onto one can never be solved.
 */
function goalDistances(wall: Uint8Array, w: number, goals: number[]): { dist: Int32Array; live: Uint8Array } {
  const size = wall.length;
  const dist = new Int32Array(size * goals.length).fill(INF);
  const live = new Uint8Array(size);
  const queue = new Int32Array(size);
  goals.forEach((g, gi) => {
    const base = gi * size;
    let head = 0;
    let tail = 0;
    queue[tail++] = g;
    dist[base + g] = 0;
    while (head < tail) {
      const t = queue[head++]!;
      const nd = dist[base + t]! + 1;
      for (let d = 0; d < 4; d++) {
        const delta = STEP[d]![1] * w + STEP[d]![0];
        const from = t - delta;
        const stand = t - 2 * delta;
        if (from < 0 || stand < 0 || from >= size || stand >= size) continue;
        if (wall[from] || wall[stand] || dist[base + from] !== INF) continue;
        dist[base + from] = nd;
        queue[tail++] = from;
      }
    }
    for (let s = 0; s < size; s++) if (dist[base + s] !== INF) live[s] = 1;
  });
  return { dist, live };
}

/**
 * The level's rooms, or null when the level is not a set of rooms. The hallway is the
 * player's row; its squares are dead (a box there reaches no goal), so cutting the dead
 * squares of that row out of the floor leaves one part per room, joined to the hallway
 * only through its door.
 */
function splitRooms(level: Level): Array<{ boxes: number[]; goals: number[] }> | null {
  const { w, wall, goals, boxes, player } = level;
  const { live } = goalDistances(wall, w, goals);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const parts: Array<{ boxes: number[]; goals: number[] }> = [];
  const stack: number[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const id = parts.length;
    const mine = { boxes: [] as number[], goals: [] as number[] };
    parts.push(mine);
    part[s] = id;
    stack.push(s);
    while (stack.length > 0) {
      const p = stack.pop()!;
      for (let d = 0; d < 4; d++) {
        const q = p + STEP[d]![1] * w + STEP[d]![0];
        if (q < 0 || q >= wall.length || wall[q] || cut[q] || part[q] >= 0) continue;
        part[q] = id;
        stack.push(q);
      }
    }
  }
  for (const b of boxes) parts[part[b]!]!.boxes.push(b);
  for (const g of goals) {
    if (part[g]! < 0) return null;
    parts[part[g]!]!.goals.push(g);
  }
  if (parts.some((p) => p.boxes.length !== p.goals.length)) return null;
  return parts.filter((p) => p.boxes.length > 0);
}

/** All orderings of [0, n): the assignments of n boxes to n goals, cheapest first tried. */
function permutations(n: number): number[][] {
  if (n <= 1) return [Array.from({ length: n }, (_, i) => i)];
  const out: number[][] = [];
  const used = new Array<boolean>(n).fill(false);
  const cur: number[] = [];
  const walk = (): void => {
    if (cur.length === n) {
      out.push(cur.slice());
      return;
    }
    for (let i = 0; i < n; i++) {
      if (used[i]) continue;
      used[i] = true;
      cur.push(i);
      walk();
      cur.pop();
      used[i] = false;
    }
  };
  walk();
  return out;
}

interface Board {
  w: number;
  wall: Uint8Array; // static walls: the board's walls plus any boxes held fixed
  goals: number[];
  boxes: number[];
  player: number;
}

interface Solved {
  pushes: number;
  moves: string;
  endPlayer: number;
}

/**
 * The push-optimal solution of one board (a room, or a whole level when it is no set of
 * rooms), or null when the search gives up. A* over pushes; see the header for the
 * heuristic and the prunes.
 */
function solveBoard(board: Board): Solved | null {
  const { w, wall, goals } = board;
  const size = wall.length;
  const n = board.boxes.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const { dist, live } = goalDistances(wall, w, goals);
  if (board.boxes.some((b) => !live[b])) return null;
  const perms = permutations(n);
  const size0 = size;

  // The admissible estimate: the cheapest assignment of the boxes to distinct goals at
  // single-box push distances; for many boxes, the sum of each box's own cheapest goal
  // (a weaker bound of the same kind) stands in for the assignment. The estimate depends
  // only on the boxes, never on where the player stands, so it is memoised by them: many
  // states share one box configuration and differ only in the player's region.
  const hCache = new Map<number | string, number>();
  const boxesKey = (bs: number[]): number | string => {
    if (n <= 6) {
      let k = 0;
      for (const b of bs) k = k * size + b;
      return k;
    }
    return bs.join(',');
  };
  const rawEstimate = (bs: number[]): number => {
    if (n <= 7) {
      let best = INF;
      for (const perm of perms) {
        let sum = 0;
        for (let i = 0; i < n; i++) {
          const d = dist[perm[i]! * size0 + bs[i]!]!;
          if (d >= INF) {
            sum = INF;
            break;
          }
          sum += d;
          if (sum >= best) break;
        }
        if (sum < best) best = sum;
      }
      return best;
    }
    let sum = 0;
    for (const b of bs) {
      let m = INF;
      for (let gi = 0; gi < n; gi++) m = Math.min(m, dist[gi * size0 + b]!);
      if (m >= INF) return INF;
      sum += m;
    }
    return sum;
  };
  const estimate = (bs: number[]): number => {
    const k = boxesKey(bs);
    const hit = hCache.get(k);
    if (hit !== undefined) return hit;
    const v = rawEstimate(bs);
    hCache.set(k, v);
    return v;
  };

  // Flood the player's reachable region around `occupied`, marking `arr` with `stamp`;
  // returns the region's lowest square, the region's name in a state's key. Two mark
  // arrays are kept: `seen` holds the region of the state being expanded (the pushes it
  // offers are read from it), `mark` the region of each child that push makes.
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const floodStack = new Int32Array(size);
  let stamp = 0;
  const regionOf = (arr: Int32Array, occupied: Uint8Array, from: number): number => {
    stamp++;
    const my = stamp;
    let low = from;
    let top = 0;
    floodStack[top++] = from;
    arr[from] = my;
    while (top > 0) {
      const p = floodStack[--top]!;
      if (p < low) low = p;
      for (let d = 0; d < 4; d++) {
        const q = p + STEP[d]![1] * w + STEP[d]![0];
        if (q < 0 || q >= size || wall[q] || occupied[q] || arr[q] === my) continue;
        arr[q] = my;
        floodStack[top++] = q;
      }
    }
    return low;
  };

  // A frozen 2 x 2: four squares of walls and boxes with a box off its goal in it.
  const frozenBlock = (occupied: Uint8Array, t: number): boolean => {
    for (const a of [t, t - 1, t - w, t - w - 1]) {
      const b = a + 1;
      const c = a + w;
      const e = c + 1;
      if (a < 0 || e >= size) continue;
      if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[e] || occupied[e])) continue;
      if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[e] && !goal[e])) return true;
    }
    return false;
  };

  // Frozen boxes, past the 2 x 2 case: a box is frozen when every push of it is blocked
  // by a wall or by a box already known frozen (any other box may yet move out of the
  // way, so it never blocks). Freezing is iterated to a fixed point — a line of boxes
  // along a wall freezes one by one from the ends — and a frozen box off its goal can
  // never be solved, so the state is dead. `occupied` must hold the boxes `bs`.
  const frz = new Int32Array(size);
  let frzStamp = 0;
  const freezeDead = (occupied: Uint8Array, bs: number[]): boolean => {
    frzStamp++;
    let changed = true;
    while (changed) {
      changed = false;
      for (const b of bs) {
        if (frz[b] === frzStamp) continue;
        let stuck = true;
        for (let d = 0; d < 4 && stuck; d++) {
          const delta = STEP[d]![1] * w + STEP[d]![0];
          const target = b + delta;
          const stand = b - delta;
          const blocked =
            wall[target] || wall[stand] || (occupied[target] === 1 && frz[target] === frzStamp) || (occupied[stand] === 1 && frz[stand] === frzStamp);
          if (!blocked) stuck = false;
        }
        if (stuck) {
          frz[b] = frzStamp;
          changed = true;
        }
      }
    }
    return bs.some((b) => frz[b] === frzStamp && !goal[b]);
  };

  const occupied = new Uint8Array(size);
  const keyOf = (bs: number[], low: number): number | string => {
    // The numeric key packs the region and the boxes in base `size`; it is exact while
    // size ** (boxes + 1) stays under 2 ** 53, which on these boards means five boxes.
    if (n <= 5) {
      let k = low;
      for (const b of bs) k = k * size + b;
      return k;
    }
    return `${low}|${bs.join(',')}`;
  };

  // The states, in the order made: boxes, the player's square, pushes so far, and the
  // parent with the push that made the state (the box's square before, and its direction).
  const stBoxes: number[][] = [board.boxes.slice()];
  const stPlayer: number[] = [board.player];
  const stG: number[] = [0];
  const stParent: number[] = [-1];
  const stFrom: number[] = [-1];
  const stDir: number[] = [-1];
  const bestG = new Map<number | string, number>();

  // The open list: a binary heap of state indices on (f, then deeper first, then older).
  const heap: number[] = [];
  const stF: number[] = [estimate(board.boxes)];
  if (stF[0]! >= INF) return null;
  const before = (a: number, b: number): boolean =>
    stF[a]! !== stF[b]! ? stF[a]! < stF[b]! : stG[a]! !== stG[b]! ? stG[a]! > stG[b]! : a < b;
  const heapPush = (s: number): void => {
    let i = heap.length;
    heap.push(s);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!before(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): number => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && before(heap[c + 1]!, heap[c]!)) c++;
        if (!before(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  for (const b of board.boxes) occupied[b] = 1;
  const rootLow = regionOf(seen, occupied, board.player);
  for (const b of board.boxes) occupied[b] = 0;
  bestG.set(keyOf(board.boxes, rootLow), 0);
  heapPush(0);

  let found = -1;
  if (board.boxes.every((b) => goal[b])) found = 0;
  while (found < 0 && heap.length > 0) {
    const s = heapPop();
    const bs = stBoxes[s]!;
    const g = stG[s]!;
    for (const b of bs) occupied[b] = 1;
    const low = regionOf(seen, occupied, stPlayer[s]!);
    if (bestG.get(keyOf(bs, low))! < g) {
      for (const b of bs) occupied[b] = 0;
      continue;
    }
    // A settled state (its pushes are the fewest known for it) with every box on a goal
    // is the optimum: no state still open can reach a goal in fewer pushes.
    if (bs.every((b) => goal[b])) {
      found = s;
      for (const b of bs) occupied[b] = 0;
      break;
    }
    const here = stamp;
    for (let bi = 0; bi < n && found < 0; bi++) {
      const b = bs[bi]!;
      for (let d = 0; d < 4; d++) {
        const delta = STEP[d]![1] * w + STEP[d]![0];
        const stand = b - delta;
        const target = b + delta;
        if (stand < 0 || target < 0 || stand >= size || target >= size) continue;
        if (seen[stand] !== here || wall[target] || occupied[target] || !live[target]) continue;
        occupied[b] = 0;
        occupied[target] = 1;
        const frozen = frozenBlock(occupied, target);
        let child = -1;
        if (!frozen) {
          const nb = bs.slice();
          nb[bi] = target;
          nb.sort((x, y) => x - y);
          if (!freezeDead(occupied, nb)) {
          const nLow = regionOf(mark, occupied, b);
          const k = keyOf(nb, nLow);
          const known = bestG.get(k);
          if (known === undefined || known > g + 1) {
            const h = estimate(nb);
            if (h < INF) {
              if (stBoxes.length >= STATE_CAP) {
                for (const x of bs) occupied[x] = 0;
                return null;
              }
              bestG.set(k, g + 1);
              child = stBoxes.length;
              stBoxes.push(nb);
              stPlayer.push(b);
              stG.push(g + 1);
              stParent.push(s);
              stFrom.push(b);
              stDir.push(d);
              stF.push(g + 1 + h);
              heapPush(child);
            }
          }
          }
        }
        occupied[target] = 0;
        occupied[b] = 1;
      }
    }
    for (const b of bs) occupied[b] = 0;
  }
  if (found < 0) return null;

  // Rebuild the moves: walk the player to each push's stand square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = stParent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = board.player;
  let boxes = board.boxes.slice();
  const walkTo = (from: number, to: number): string => {
    if (from === to) return '';
    const prev = new Int32Array(size).fill(-1);
    const how = new Int32Array(size).fill(-1);
    const queue = new Int32Array(size);
    let head = 0;
    let tail = 0;
    queue[tail++] = from;
    prev[from] = from;
    while (head < tail) {
      const p = queue[head++]!;
      if (p === to) break;
      for (let d = 0; d < 4; d++) {
        const q = p + STEP[d]![1] * w + STEP[d]![0];
        if (q < 0 || q >= size || wall[q] || occupied[q] || prev[q] !== -1) continue;
        prev[q] = p;
        how[q] = d;
        queue[tail++] = q;
      }
    }
    let s = '';
    for (let p = to; p !== from; p = prev[p]!) s = STEP[how[p]!]![2] + s;
    return s;
  };
  for (const s of path) {
    const from = stFrom[s]!;
    const d = stDir[s]!;
    const delta = STEP[d]![1] * w + STEP[d]![0];
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(player, from - delta);
    for (const b of boxes) occupied[b] = 0;
    moves += STEP[d]![3];
    player = from;
    boxes = boxes.map((b) => (b === from ? from + delta : b)).sort((x, y) => x - y);
  }
  return { pushes: path.length, moves, endPlayer: player };
}

/** One level's moves: room by room when it is a set of rooms, whole otherwise. */
function solveLevel(text: string): string {
  const level = parseLevel(text);
  const rooms = splitRooms(level);
  if (rooms === null) {
    const whole = solveBoard({ w: level.w, wall: level.wall, goals: level.goals, boxes: level.boxes, player: level.player });
    return whole === null ? '' : whole.moves;
  }
  // Each room in turn, the other rooms' boxes standing as walls: a room not yet solved by
  // its boxes, a room already solved by the boxes resting on its goals.
  const resting = rooms.map((r) => r.boxes.slice());
  let player = level.player;
  let moves = '';
  for (let ri = 0; ri < rooms.length; ri++) {
    const room = rooms[ri]!;
    const wall = level.wall.slice();
    for (let oj = 0; oj < rooms.length; oj++) {
      if (oj === ri) continue;
      for (const b of resting[oj]!) wall[b] = 1;
    }
    const solved = solveBoard({ w: level.w, wall, goals: room.goals.slice(), boxes: room.boxes.slice(), player });
    if (solved === null) return '';
    moves += solved.moves;
    player = solved.endPlayer;
    resting[ri] = room.goals.slice();
  }
  return moves;
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  let moves = '';
  try {
    moves = solveLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
  } catch {
    moves = '';
  }
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
