"""
sim_runner.py — Mass simulation runner
────────────────────────────────────────
Runs bot-vs-bot matches in background threads.
The browser polls /api/sim/status/<sim_id> for progress.

Each "simulation" is a set of matchups (unique bot pairs).
Each matchup runs N games, collecting statistics after every game.

Thread safety:
  - Each Simulation object is only written by its own thread(s).
  - The main Flask thread only reads .status / .results (safe for small dicts).
  - We use one thread per matchup so matchups run in parallel.

Depends on: game_engine.py, bot.py
Imported by: server.py
"""

import threading
import time
import uuid
from itertools import combinations
from typing import Optional

from game_engine import GameEngine, GamePhase
from bot import BOT_REGISTRY, BotPlayer


# ─── In-memory store ──────────────────────────────────────────────────────────

simulations: dict[str, "Simulation"] = {}


# ─── Statistics helpers ───────────────────────────────────────────────────────

def _empty_bot_stats() -> dict:
    return {
        "games":              0,
        "wins":               0,
        "rounds_played":      0,
        "total_stake":        0,
        "stake_raises":       0,
        "stake_declines":     0,
        "stake_accepts":      0,
        "calculate_attempts": 0,
        "calculate_wins":     0,
        "calculate_losses":   0,
        "three_trumps":       0,
        "cuts":               0,
        "passes":             0,
        "counter_plays":      0,
        "points_at_calculate": [],   # list of totals for avg
        "round_end_reasons":  {},
    }


def _merge_stats(a: dict, b: dict) -> dict:
    """Merge b into a (for aggregate cross-matchup stats). Returns new dict."""
    result = dict(a)
    for k, v in b.items():
        if k == "points_at_calculate":
            result[k] = result.get(k, []) + v
        elif k == "round_end_reasons":
            d = dict(result.get(k, {}))
            for reason, cnt in v.items():
                d[reason] = d.get(reason, 0) + cnt
            result[k] = d
        elif isinstance(v, (int, float)):
            result[k] = result.get(k, 0) + v
        else:
            result[k] = v
    return result


def _finalise_stats(s: dict) -> dict:
    """Add derived fields (rates, averages) to a raw stats dict."""
    games   = max(s["games"], 1)
    rounds  = max(s["rounds_played"], 1)
    calcs   = max(s["calculate_attempts"], 1)
    out = dict(s)
    out["win_rate"]              = round(s["wins"] / games * 100, 2)
    out["avg_rounds_per_game"]   = round(s["rounds_played"] / games, 2)
    out["avg_stake"]             = round(s["total_stake"] / rounds, 2)
    out["stake_raise_rate"]      = round(s["stake_raises"] / rounds * 100, 2)
    out["calculate_success_rate"]= round(s["calculate_wins"] / calcs * 100, 2)
    out["cut_rate"]              = round(s["cuts"] / max(s["cuts"]+s["passes"],1)*100, 2)
    out["pass_rate"]             = round(s["passes"] / max(s["cuts"]+s["passes"],1)*100, 2)
    out["avg_points_at_calculate"]= (
        round(sum(s["points_at_calculate"]) / len(s["points_at_calculate"]), 2)
        if s["points_at_calculate"] else 0
    )
    # remove raw list for JSON cleanliness
    out.pop("points_at_calculate", None)
    return out


# ─── Single game runner ───────────────────────────────────────────────────────

def _run_one_game(BotClassA: type, BotClassB: type,
                  target_score: int, stats_a: dict, stats_b: dict):
    """
    Run one complete game between BotA and BotB.
    Mutates stats_a and stats_b in place.
    """
    bot_a = BotClassA("a", BotClassA.__name__)
    bot_b = BotClassB("b", BotClassB.__name__)
    engine = GameEngine(bot_a, bot_b, target_score=target_score)
    engine.start_round()

    max_iter = 5000
    for _ in range(max_iter):
        if engine.phase == GamePhase.GAME_OVER:
            break
        if engine.phase == GamePhase.ROUND_OVER:
            engine.start_round()
            continue
        if engine.phase == GamePhase.WAITING:
            break
        _step(engine)

    # Tally results
    stats_a["games"] += 1
    stats_b["games"] += 1

    winner_idx = None
    for i, p in enumerate(engine.players):
        if p.game_score >= engine.target_score:
            winner_idx = i
            break

    if winner_idx == 0:
        stats_a["wins"] += 1
    elif winner_idx == 1:
        stats_b["wins"] += 1

    # Parse move history for detailed stats
    for move in engine.move_history:
        pid  = move.player_id
        mtype = move.move_type
        data = move.data
        s = stats_a if pid == "a" else stats_b if pid == "b" else None

        if mtype == "round_start":
            stats_a["rounds_played"] += 1
            stats_b["rounds_played"] += 1

        if mtype == "round_end" and s is None:
            reason = data.get("reason", "unknown")
            stake  = data.get("stake", 1)
            w_id   = data.get("winner")
            ws = stats_a if w_id == "a" else stats_b if w_id == "b" else None
            if ws:
                ws["round_end_reasons"][reason] = ws["round_end_reasons"].get(reason, 0) + 1
            stats_a["total_stake"] += stake
            stats_b["total_stake"] += stake

        if s is None:
            continue

        if mtype == "stake_offer":
            s["stake_raises"] += 1
        elif mtype == "stake_decline":
            s["stake_declines"] += 1
        elif mtype == "stake_accept":
            s["stake_accepts"] += 1
        elif mtype == "calculate":
            s["calculate_attempts"] += 1
            pts = data.get("total", 0)
            s["points_at_calculate"].append(pts)
            if data.get("win"):
                s["calculate_wins"] += 1
            else:
                s["calculate_losses"] += 1
        elif mtype == "three_trumps":
            s["three_trumps"] += 1
        elif mtype == "cut":
            s["cuts"] += 1
        elif mtype == "pass":
            s["passes"] += 1
        elif mtype == "counter_play":
            s["counter_plays"] += 1


def _step(engine: GameEngine):
    """Execute one action in the engine (same logic as bvb_runner but inline)."""
    phase = engine.phase
    if phase in (GamePhase.GAME_OVER, GamePhase.WAITING, GamePhase.ROUND_OVER):
        return

    # Pending stake
    for idx, bot in enumerate(engine.players):
        if (engine.stake_offerer_idx is not None
                and engine.stake_offerer_idx != idx):
            if engine.pending_stake <= 3:
                engine.accept_stake(idx)
            else:
                engine.decline_stake(idx)
            return

    if phase == GamePhase.STAKES:
        engine.start_play()
    elif phase == GamePhase.PLAYING:
        active = engine.active_idx
        bot    = engine.players[active]
        engine.play_cards(active, bot.choose_play(engine))
    elif phase in (GamePhase.CUTTING, GamePhase.FORCED_CUT):
        cutter_idx = 1 - engine.playing_player_idx
        bot        = engine.players[cutter_idx]
        action, card_ids = bot.choose_cut_or_pass(engine)
        if action == "cut":
            engine.cut_cards(cutter_idx, card_ids)
        else:
            engine.pass_cards(cutter_idx, card_ids)
    elif phase == GamePhase.CALCULATING:
        calc_idx = engine.calculator_idx
        bot      = engine.players[calc_idx]
        if bot.choose_calculate(engine):
            engine.calculate(calc_idx)
        else:
            engine.skip_calculate(calc_idx)


# ─── Matchup runner ───────────────────────────────────────────────────────────

class Matchup:
    def __init__(self, bot_a: str, bot_b: str, games: int, target_score: int):
        self.bot_a        = bot_a
        self.bot_b        = bot_b
        self.games        = games
        self.target_score = target_score
        self.completed    = 0
        self.stats_a      = _empty_bot_stats()
        self.stats_b      = _empty_bot_stats()
        self.done         = False
        self.started_at   = None
        self.finished_at  = None

    def run(self):
        self.started_at = time.time()
        BotA = BOT_REGISTRY[self.bot_a]
        BotB = BOT_REGISTRY[self.bot_b]
        for _ in range(self.games):
            _run_one_game(BotA, BotB, self.target_score, self.stats_a, self.stats_b)
            self.completed += 1
        self.done         = True
        self.finished_at  = time.time()

    def progress(self) -> float:
        return self.completed / max(self.games, 1)

    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at or time.time()
        return end - self.started_at

    def eta(self) -> Optional[float]:
        if not self.started_at or self.completed == 0:
            return None
        rate = self.completed / self.elapsed()
        remaining = self.games - self.completed
        return remaining / rate if rate > 0 else None

    def to_status(self) -> dict:
        return {
            "bot_a":     self.bot_a,
            "bot_b":     self.bot_b,
            "games":     self.games,
            "completed": self.completed,
            "progress":  round(self.progress() * 100, 1),
            "done":      self.done,
            "elapsed":   round(self.elapsed(), 1),
            "eta":       round(self.eta(), 1) if self.eta() is not None else None,
            # live scores during run
            "wins_a":    self.stats_a["wins"],
            "wins_b":    self.stats_b["wins"],
        }

    def to_result(self) -> dict:
        return {
            "bot_a":   self.bot_a,
            "bot_b":   self.bot_b,
            "games":   self.games,
            "elapsed": round(self.elapsed(), 1),
            "stats_a": _finalise_stats(self.stats_a),
            "stats_b": _finalise_stats(self.stats_b),
        }


# ─── Simulation (collection of matchups) ──────────────────────────────────────

class Simulation:
    def __init__(self, sim_id: str, matchups: list[Matchup]):
        self.sim_id    = sim_id
        self.matchups  = matchups
        self.started_at = time.time()
        self.threads: list[threading.Thread] = []

    def start(self):
        for m in self.matchups:
            t = threading.Thread(target=m.run, daemon=True)
            t.start()
            self.threads.append(t)

    @property
    def done(self) -> bool:
        return all(m.done for m in self.matchups)

    def overall_progress(self) -> float:
        total = sum(m.games for m in self.matchups)
        done  = sum(m.completed for m in self.matchups)
        return done / max(total, 1)

    def max_eta(self) -> Optional[float]:
        etas = [m.eta() for m in self.matchups if m.eta() is not None]
        return max(etas) if etas else None

    def to_status(self) -> dict:
        return {
            "sim_id":   self.sim_id,
            "done":     self.done,
            "progress": round(self.overall_progress() * 100, 1),
            "eta":      round(self.max_eta(), 1) if self.max_eta() else None,
            "matchups": [m.to_status() for m in self.matchups],
        }

    def to_result(self) -> dict:
        # Per-matchup results
        matchup_results = [m.to_result() for m in self.matchups]

        # Aggregate per-bot across all matchups
        bot_agg: dict[str, dict] = {}
        for m in self.matchups:
            for bot_name, raw in [(m.bot_a, m.stats_a), (m.bot_b, m.stats_b)]:
                if bot_name not in bot_agg:
                    bot_agg[bot_name] = _empty_bot_stats()
                bot_agg[bot_name] = _merge_stats(bot_agg[bot_name], raw)

        bot_summary = {name: _finalise_stats(s) for name, s in bot_agg.items()}

        return {
            "sim_id":       self.sim_id,
            "matchups":     matchup_results,
            "bot_summary":  bot_summary,
            "elapsed":      round(time.time() - self.started_at, 1),
        }


# ─── Public API ───────────────────────────────────────────────────────────────

def create_simulation(bot_pairs: list[tuple[str, str]],
                      games_per_matchup: int,
                      target_score: int) -> str:
    """
    Create and start a simulation.
    bot_pairs: list of (bot_a_name, bot_b_name) tuples — no duplicates, no same-vs-same.
    Returns sim_id.
    """
    sim_id   = str(uuid.uuid4())[:8]
    matchups = [
        Matchup(a, b, games_per_matchup, target_score)
        for a, b in bot_pairs
    ]
    sim = Simulation(sim_id, matchups)
    simulations[sim_id] = sim
    sim.start()
    return sim_id


def get_simulation_status(sim_id: str) -> Optional[dict]:
    sim = simulations.get(sim_id)
    return sim.to_status() if sim else None


def get_simulation_result(sim_id: str) -> Optional[dict]:
    sim = simulations.get(sim_id)
    if sim is None:
        return None
    return sim.to_result()