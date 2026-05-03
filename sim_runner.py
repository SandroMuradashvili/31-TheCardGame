"""
sim_runner.py — Mass simulation runner
───────────────────────────────────────
Runs bot-vs-bot simulations in background threads.
Browser polls /api/sim/status/<sim_id> for progress.

Each simulation:
  - Has a unique sim_id
  - Runs N matchups (bot pairs) in parallel threads
  - Each matchup plays M games
  - Tracks detailed statistics per matchup and per bot

Depends on: game_engine.py, bot.py
Imported by: server.py
"""

import threading
import time
import uuid
from itertools import combinations
from collections import defaultdict

from game_engine import GameEngine, GamePhase, RoundEndReason
from bot import get_bot, BOT_REGISTRY


# ─── In-memory simulation store ───────────────────────────────────────────────

simulations: dict[str, dict] = {}


# ─── Statistics accumulator ───────────────────────────────────────────────────

class MatchStats:
    """Accumulates statistics for one bot-vs-bot matchup over many games."""

    def __init__(self, bot_a_id: str, bot_b_id: str):
        self.bot_a_id = bot_a_id
        self.bot_b_id = bot_b_id
        self.games_played   = 0
        self.wins           = [0, 0]       # wins[0]=botA wins, wins[1]=botB wins
        self.total_rounds   = 0
        self.total_stake    = 0
        self.stake_samples  = 0

        # Per-bot action counts [botA, botB]
        self.raises         = [0, 0]
        self.declines       = [0, 0]
        self.accepts        = [0, 0]
        self.cuts           = [0, 0]
        self.passes         = [0, 0]
        self.counters       = [0, 0]
        self.calculates     = [0, 0]
        self.calc_wins      = [0, 0]   # calculated and had 31+
        self.calc_losses    = [0, 0]   # calculated but < 31
        self.three_trumps   = [0, 0]
        self.stake_declines_received = [0, 0]  # opponent declined my raise

        # Points at calculate
        self.calc_points_sum = [0, 0]
        self.calc_attempts   = [0, 0]

    def record_game(self, game_history: list[dict], winner_idx: int,
                    player_ids: list[str], rounds: int, final_stake: int):
        self.games_played += 1
        self.wins[winner_idx] += 1
        self.total_rounds += rounds
        self.total_stake  += final_stake
        self.stake_samples += 1

        pid = {player_ids[0]: 0, player_ids[1]: 1}

        for move in game_history:
            p = pid.get(move.get("player"), -1)
            if p == -1:
                continue
            t = move.get("type")
            d = move.get("data", {})

            if t == "stake_offer":
                self.raises[p] += 1
            elif t == "stake_accept":
                self.accepts[p] += 1
            elif t == "stake_decline":
                self.declines[p] += 1
                # The other player raised and got declined
                raiser = pid.get(move.get("player"), -1)
                # actually raiser is the offerer — tracked by stake_declines_received
                opp = 1 - p
                self.stake_declines_received[opp] += 1
            elif t == "cut":
                self.cuts[p] += 1
            elif t == "pass":
                self.passes[p] += 1
            elif t == "counter_play":
                self.counters[p] += 1
            elif t == "three_trumps":
                self.three_trumps[p] += 1
            elif t == "calculate":
                self.calculates[p] += 1
                pts = d.get("total", 0)
                won = d.get("win", False)
                self.calc_points_sum[p] += pts
                self.calc_attempts[p]   += 1
                if won:
                    self.calc_wins[p] += 1
                else:
                    self.calc_losses[p] += 1

    def to_dict(self) -> dict:
        g = max(self.games_played, 1)
        r = max(self.total_rounds, 1)

        def pct(num, denom):
            return round(num / max(denom, 1) * 100, 1)

        def per_game(val):
            return round(val / g, 2)

        def avg_pts(p):
            return round(self.calc_points_sum[p] / max(self.calc_attempts[p], 1), 1)

        result = {
            "bot_a_id":    self.bot_a_id,
            "bot_b_id":    self.bot_b_id,
            "games_played": self.games_played,
            "avg_rounds_per_game": round(self.total_rounds / g, 1),
            "avg_stake":   round(self.total_stake / max(self.stake_samples, 1), 2),
        }

        for i, label in enumerate(["bot_a", "bot_b"]):
            ca = max(self.calc_attempts[i], 1)
            result[label] = {
                "wins":              self.wins[i],
                "win_rate":          pct(self.wins[i], g),
                "raises":            self.raises[i],
                "raises_per_game":   per_game(self.raises[i]),
                "declines":          self.declines[i],
                "declines_per_game": per_game(self.declines[i]),
                "cuts":              self.cuts[i],
                "cuts_per_game":     per_game(self.cuts[i]),
                "passes":            self.passes[i],
                "passes_per_game":   per_game(self.passes[i]),
                "counters":          self.counters[i],
                "counters_per_game": per_game(self.counters[i]),
                "three_trumps":      self.three_trumps[i],
                "three_trumps_per_game": per_game(self.three_trumps[i]),
                "calc_attempts":     self.calculates[i],
                "calc_attempts_per_game": per_game(self.calculates[i]),
                "calc_success_rate": pct(self.calc_wins[i], ca),
                "calc_bluff_rate":   pct(self.calc_losses[i], ca),
                "avg_pts_at_calc":   avg_pts(i),
            }

        return result


# ─── Single game runner ───────────────────────────────────────────────────────

def _run_one_game(bot_a_id: str, bot_b_id: str, target_score: int) -> dict:
    """Run one complete game. Returns winner_idx, rounds, history."""
    p1 = get_bot(bot_a_id, "p1", "BotA")
    p2 = get_bot(bot_b_id, "p2", "BotB")
    engine = GameEngine(p1, p2, target_score=target_score)
    engine.start_round()

    max_rounds = 500
    rounds = 0

    while engine.phase != GamePhase.GAME_OVER and rounds < max_rounds:
        phase = engine.phase

        if phase == GamePhase.ROUND_OVER:
            engine.start_round()
            rounds += 1
            continue

        if phase == GamePhase.WAITING:
            break

        # Handle pending stake
        stake_handled = False
        for idx, bot in enumerate(engine.players):
            if (engine.stake_offerer_idx is not None
                    and engine.stake_offerer_idx != idx):
                if hasattr(bot, 'choose_raise_stake') and not bot.choose_raise_stake(engine):
                    engine.decline_stake(idx)
                else:
                    engine.accept_stake(idx)
                stake_handled = True
                break
        if stake_handled:
            continue

        if phase == GamePhase.STAKES:
            active = engine.active_idx
            bot = engine.players[active]
            # Check if bot wants to raise stake
            if hasattr(bot, 'choose_raise_stake') and bot.choose_raise_stake(engine):
                if engine.can_raise_stake(active):
                    engine.offer_stake(active)
                    continue
            engine.start_play()

        elif phase == GamePhase.PLAYING:
            active = engine.active_idx
            bot = engine.players[active]
            engine.play_cards(active, bot.choose_play(engine))

        elif phase in (GamePhase.CUTTING, GamePhase.FORCED_CUT):
            cutter_idx = 1 - engine.playing_player_idx
            bot = engine.players[cutter_idx]
            action, card_ids = bot.choose_cut_or_pass(engine)
            if action == "counter" and phase == GamePhase.CUTTING:
                result = engine.counter_play(cutter_idx, card_ids)
                if not result:
                    # Counter failed — fall back to pass
                    n = len(engine.played_cards)
                    cheapest = sorted(bot.hand, key=lambda c: c.points)[:n]
                    engine.pass_cards(cutter_idx, [repr(c) for c in cheapest])
            elif action == "cut":
                engine.cut_cards(cutter_idx, card_ids)
            else:
                engine.pass_cards(cutter_idx, card_ids)

        elif phase == GamePhase.CALCULATING:
            calc_idx = engine.calculator_idx
            bot = engine.players[calc_idx]
            if bot.choose_calculate(engine):
                engine.calculate(calc_idx)
            else:
                engine.skip_calculate(calc_idx)

        else:
            break

    winner_idx = 0
    if engine.round_winner_idx is not None:
        winner_idx = engine.round_winner_idx
    elif engine.players[1].game_score > engine.players[0].game_score:
        winner_idx = 1

    return {
        "winner_idx":  winner_idx,
        "rounds":      rounds,
        "history":     [m.to_dict() for m in engine.move_history],
        "final_stake": engine.current_stake,
        "player_ids":  [p.player_id for p in engine.players],
    }


# ─── Matchup worker ───────────────────────────────────────────────────────────

def _run_matchup(sim_id: str, matchup_idx: int,
                 bot_a_id: str, bot_b_id: str,
                 games: int, target_score: int):
    """Worker thread for one bot-pair matchup."""
    sim = simulations[sim_id]
    stats = MatchStats(bot_a_id, bot_b_id)

    for g in range(games):
        if sim.get("cancelled"):
            break
        try:
            result = _run_one_game(bot_a_id, bot_b_id, target_score)
            stats.record_game(
                result["history"],
                result["winner_idx"],
                result["player_ids"],
                result["rounds"],
                result["final_stake"],
            )
        except Exception as e:
            import traceback
            print(f"\n[SIMULATION CRASH] {bot_a_id} vs {bot_b_id} (Game {g})")
            traceback.print_exc()
            # We still skip the game so the whole server doesn't crash,
            # but now we can clearly see the error in the terminal!

        # Update progress
        sim["matchups"][matchup_idx]["games_done"] = g + 1
        
        # Periodically push live results so the UI can show live wins
        if (g + 1) % 50 == 0:
            sim["matchups"][matchup_idx]["results"] = stats.to_dict()

    sim["matchups"][matchup_idx]["status"]  = "done"
    sim["matchups"][matchup_idx]["results"] = stats.to_dict()

    # Check if all matchups done
    all_done = all(m["status"] == "done" for m in sim["matchups"])
    if all_done:
        sim["status"]   = "done"
        sim["ended_at"] = time.time()


# ─── Public API ───────────────────────────────────────────────────────────────

def start_simulation(pairs: list[list[str]], games_per_matchup: int,
                     target_score: int = 7) -> str:
    """
    Start a simulation for the given bot pairs.
    Returns sim_id.
    """
    sim_id   = str(uuid.uuid4())[:8]
    matchups = []

    for i, (a, b) in enumerate(pairs):
        matchups.append({
            "idx":        i,
            "bot_a_id":   a,
            "bot_b_id":   b,
            "games_total": games_per_matchup,
            "games_done":  0,
            "status":     "running",
            "results":    None,
        })

    simulations[sim_id] = {
        "sim_id":       sim_id,
        "status":       "running",
        "started_at":   time.time(),
        "ended_at":     None,
        "games_per_matchup": games_per_matchup,
        "target_score": target_score,
        "matchups":     matchups,
        "cancelled":    False,
    }

    for i, (a, b) in enumerate(pairs):
        t = threading.Thread(
            target=_run_matchup,
            args=(sim_id, i, a, b, games_per_matchup, target_score),
            daemon=True,
        )
        t.start()

    return sim_id


def get_sim_status(sim_id: str) -> dict | None:
    sim = simulations.get(sim_id)
    if not sim:
        return None

    elapsed = time.time() - sim["started_at"]
    total_games   = sum(m["games_total"] for m in sim["matchups"])
    done_games    = sum(m["games_done"]  for m in sim["matchups"])
    pct = done_games / max(total_games, 1)

    eta = None
    if pct > 0.01 and sim["status"] == "running":
        eta = round((elapsed / pct) * (1 - pct))

    return {
        **sim,
        "elapsed_seconds": round(elapsed),
        "pct_done":        round(pct * 100, 1),
        "eta_seconds":     eta,
        "total_games":     total_games,
        "done_games":      done_games,
    }


def cancel_simulation(sim_id: str):
    if sim_id in simulations:
        simulations[sim_id]["cancelled"] = True