// Snake, ported from `sodachat/games/snake.py`, so the action head has a board
// to read in the browser.
//
// The model was trained on `model_board()` — one ASCII character per cell, head
// and body distinct — so that is the one thing here that has to be exact: same
// glyphs, same 20x20 grid, same row-per-line layout. The rules are ported with
// it (legality, growth, the stuck-without-food cutoff) so the moves it picks
// are judged on the same game the Python one plays. Food placement uses the
// browser's RNG rather than reproducing Python's; nothing downstream depends on
// the two agreeing on where an apple lands.

const EMPTY = ".";
const BODY = "#";
const HEAD = "@";
const FOOD = "*";

const DELTA = { up: [-1, 0], down: [1, 0], left: [0, -1], right: [0, 1] };
const OPPOSITE = { up: "down", down: "up", left: "right", right: "left" };

export const ACTIONS = ["up", "down", "left", "right"];

export class SnakeGame {
  constructor({ rows = 20, cols = 20, random = Math.random } = {}) {
    this.rows = rows;
    this.cols = cols;
    this.random = random;
    this.reset();
  }

  reset() {
    const c = Math.floor(this.rows / 2);
    this.snake = [[c, c]];
    this.heading = "right";
    this.score = 0;
    this.sinceFood = 0;
    this.done = false;
    this._placeFood();
    return this;
  }

  get head() {
    return this.snake[0];
  }

  _key([r, c]) {
    return r * this.cols + c;
  }

  _placeFood() {
    const taken = new Set(this.snake.map((cell) => this._key(cell)));
    const free = [];
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        if (!taken.has(this._key([r, c]))) free.push([r, c]);
      }
    }
    this.food = free.length ? free[Math.floor(this.random() * free.length)] : null;
  }

  /** Reversing into your own neck is the one move that is never offered. */
  legal(action) {
    return !(this.snake.length > 1 && action === OPPOSITE[this.heading]);
  }

  /** Where `action` lands, and whether it eats — null if it kills. */
  _result(action) {
    if (!this.legal(action)) return [null, false];
    const [dr, dc] = DELTA[action];
    const next = [this.head[0] + dr, this.head[1] + dc];
    if (next[0] < 0 || next[0] >= this.rows || next[1] < 0 || next[1] >= this.cols) {
      return [null, false];
    }
    const eating = this.food !== null && next[0] === this.food[0] && next[1] === this.food[1];
    const occupied = new Set(this.snake.map((cell) => this._key(cell)));
    // The tail moves out of the way on a non-eating step, so following it is safe.
    if (!eating) occupied.delete(this._key(this.snake[this.snake.length - 1]));
    if (occupied.has(this._key(next))) return [null, false];
    return [next, eating];
  }

  safeActions() {
    return ACTIONS.filter((a) => this._result(a)[0] !== null);
  }

  step(action) {
    if (this.done) return;
    if (!this.legal(action)) action = this.heading;
    const [next, eating] = this._result(action);
    this.heading = action;
    if (next === null) {
      this.done = true;
      return;
    }
    this.snake.unshift(next);
    if (eating) {
      this.score += 1;
      this.sinceFood = 0;
      this._placeFood();
      if (this.food === null) this.done = true;
    } else {
      this.snake.pop();
      this.sinceFood += 1;
    }
    if (this.sinceFood > 2 * this.rows * this.cols) this.done = true;
  }

  /** The board the model reads — `Game.model_board()`, character for character. */
  modelBoard() {
    const grid = Array.from({ length: this.rows }, () => new Array(this.cols).fill(EMPTY));
    for (const [r, c] of this.snake) grid[r][c] = BODY;
    grid[this.head[0]][this.head[1]] = HEAD;
    if (this.food) grid[this.food[0]][this.food[1]] = FOOD;
    return grid.map((row) => row.join("")).join("\n");
  }
}
