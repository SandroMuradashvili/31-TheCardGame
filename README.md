# BURA (Thirty-One / Cutter) — Card Game

A fully playable web implementation of the BURA card game with a clean Python Flask backend and vanilla HTML/JS frontend.

---

## Quick Start

### 1. Install dependencies

```bash
pip install flask flask-cors
```

### 2. Run the server

```bash
python server.py
```

### 3. Open the game

Navigate to: **http://localhost:5000**

---

## Files

```
├── game_engine.py    # All game rules, state management, validation
├── server.py         # Flask REST API + SimpleBot
├── index.html        # Single-page game UI (HTML/CSS/JS, no frameworks)
└── README.md         # This file
```

---

## Architecture

### `game_engine.py` — Pure Game Logic

The heart of the system. Has **zero** Flask/HTTP dependencies.

**Key classes:**
- `Card` — Suit, rank, points, repr
- `Deck` — 20-card deck, shuffle, deal, trump reveal
- `Player` (abstract) — `HumanPlayer`, `BotPlayer` subclasses
  - `BotPlayer` is an abstract class — subclass it to implement AI strategies
- `GameEngine` — Full state machine, all rule enforcement

**State machine phases:**
```
WAITING → STAKES → PLAYING → CUTTING/FORCED_CUT → CALCULATING → PLAYING …
                                                              ↓
                                                        ROUND_OVER → WAITING
                                                              ↓
                                                         GAME_OVER
```

**Key methods:**
```python
engine.start_round()              # Begin a new round
engine.offer_stake(idx, value)    # Raise the stakes
engine.accept_stake(idx)          # Accept pending stake
engine.decline_stake(idx)         # Decline (forfeit round)
engine.start_play()               # Begin play phase
engine.play_cards(idx, card_ids)  # Play 1-3 cards
engine.cut_cards(idx, card_ids)   # Attempt to cut
engine.pass_cards(idx, card_ids)  # Pass cards (don't cut)
engine.calculate(idx)             # Reveal pile and count
engine.skip_calculate(idx)        # Pass on calculating
engine.get_state(perspective_idx, debug)  # Full state snapshot
engine.get_valid_cuts(player_idx) # All legal cut combos
```

---

### `server.py` — Flask REST API

All endpoints return `{ success: true, state: {...} }` or `{ success: false, error: "..." }`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/new_game` | Create a game (`mode`, `player_name`, `target_score`) |
| GET  | `/api/game/:id/state` | Get current state (`debug`, `perspective` params) |
| POST | `/api/game/:id/start_round` | Deal cards, start round |
| POST | `/api/game/:id/offer_stake` | Offer to raise stakes |
| POST | `/api/game/:id/accept_stake` | Accept pending stake |
| POST | `/api/game/:id/decline_stake` | Decline stake offer |
| POST | `/api/game/:id/start_play` | Skip stakes, begin play |
| POST | `/api/game/:id/play` | Play cards from hand |
| POST | `/api/game/:id/cut` | Attempt to cut played cards |
| POST | `/api/game/:id/pass` | Pass cards (don't cut) |
| POST | `/api/game/:id/calculate` | Reveal pile, count points |
| POST | `/api/game/:id/skip_calculate` | Skip calculating, draw |

**Bot integration**: `SimpleBot` in `server.py` implements the `BotPlayer` abstract class. After every human action, `_bot_act_if_needed()` loops until it's the human's turn again.

---

## Adding AI Bots

Subclass `BotPlayer` in `game_engine.py`:

```python
class GreedyBot(BotPlayer):
    def choose_play(self, engine: GameEngine) -> list[str]:
        # Return list of card repr strings from self.hand
        best = max(self.hand, key=lambda c: c.points)
        return [repr(best)]

    def choose_cut_or_pass(self, engine: GameEngine):
        valid = engine.get_valid_cuts(engine.players.index(self))
        if valid:
            return ("cut", valid[0])
        n = len(engine.played_cards)
        cheapest = sorted(self.hand, key=lambda c: c.points)[:n]
        return ("pass", [repr(c) for c in cheapest])

    def choose_calculate(self, engine: GameEngine) -> bool:
        return self.pile_points >= 31

    def choose_stake(self, engine: GameEngine) -> int:
        return 0  # Never raises
```

Then swap out the bot in `server.py`:

```python
p2 = GreedyBot("p2", "GreedyBot")
```

---

## Bot vs Bot Tournaments

Since `GameEngine` is pure Python with no I/O, you can run automated tournaments:

```python
from game_engine import GameEngine
from my_bots import GreedyBot, MCTSBot

wins = {0: 0, 1: 0}
for _ in range(1000):
    engine = GameEngine(GreedyBot("p1","Greedy"), MCTSBot("p2","MCTS"), target_score=7)
    engine.start_round()
    while engine.phase not in (GamePhase.GAME_OVER,):
        # Both bots act via their choose_* methods
        bot_act(engine, 0)
        bot_act(engine, 1)
    winner = 0 if engine.players[0].game_score >= engine.target_score else 1
    wins[winner] += 1
print(wins)
```

---

## Multiplayer / WebSocket

The current in-memory session store (`games` dict in `server.py`) is single-server.

**To add real multiplayer:**
1. Replace `games` dict with Redis or database persistence
2. Add Socket.IO: `pip install flask-socketio`
3. Emit state updates on every action instead of returning state in HTTP response
4. Add player authentication / room codes

The `GameEngine` itself needs no changes — it's already player-indexed and state-serializable.

---

## AI Advisor Mode

Add a `/api/game/:id/suggest` endpoint:

```python
@app.route("/api/game/<game_id>/suggest", methods=["GET"])
def suggest(game_id):
    engine = get_game(game_id)
    # Clone state, run simulations, return best move
    return ok_response({"suggestion": advisor_bot.choose_play(engine)})
```

---

## Debug Mode

Click **⚙ Debug** in the top-right corner to reveal:
- Both players' full hands
- Pile contents and exact point totals
- Deck order (top 5 cards)
- Full move history log
- Current game state (phase, stake, active player, calculator)

---

## Game Rules Summary

- **20 cards**: A, T, K, Q, J in 4 suits. Points: A=11, T=10, K=4, Q=3, J=2
- **Trump**: revealed from bottom of deck, beats all non-trump
- **Play**: active player plays 1-3 same-suit cards
- **Cut**: opponent plays same count — same suit (higher value) OR trump (beats anything)
- **Pass**: opponent passes same count anonymously (active player can't see which cards)
- **Maliutka**: 3 same-suit non-trump cards force opponent to cut
- **Bura (3 Trumps)**: instant round win
- **Calculate**: player who last took cards may reveal pile — win if ≥31 points
- **Stakes**: 1–6, raised before play; declining forfeits round at previous stake
- **Game**: first to 7 (or 11) points wins
