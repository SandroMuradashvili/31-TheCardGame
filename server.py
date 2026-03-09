"""
BURA (Thirty-One) Flask Game Server
RESTful API for all game actions.
"""

from flask import Flask, jsonify, request, send_from_directory, make_response
import os
import uuid

from game_engine import (
    GameEngine, HumanPlayer, BotPlayer,
    GamePhase, Suit, Rank, Card
)

# ─────────────────────────────────────────────
# SIMPLE BOT (random-ish strategy)
# ─────────────────────────────────────────────

class SimpleBot(BotPlayer):
    """Basic bot that makes legal moves. Replace with smarter AI subclasses."""

    def choose_play(self, engine: "GameEngine") -> list[str]:
        import random
        from itertools import combinations

        hand = self.hand
        # Group by suit
        by_suit: dict = {}
        for c in hand:
            by_suit.setdefault(c.suit, []).append(c)

        # Prefer playing trump if we have 3 trumps (instant win)
        trump_cards = by_suit.get(engine.trump_suit, [])
        if len(trump_cards) == 3:
            return [repr(c) for c in trump_cards]

        # Try to play highest-value single card from non-trump
        best_single = max(hand, key=lambda c: c.points)
        return [repr(best_single)]

    def choose_cut_or_pass(self, engine: "GameEngine"):
        """Returns (action, card_ids) where action is 'cut' or 'pass'."""
        valid_cuts = engine.get_valid_cuts(engine.players.index(self))
        if valid_cuts:
            # Pick the cut that uses lowest-value cards
            best = min(valid_cuts, key=lambda combo: sum(
                next(c for c in self.hand if repr(c) == cid).points
                for cid in combo
            ))
            return ("cut", best)

        # Must pass — pick lowest value cards
        import random
        n = len(engine.played_cards)
        sorted_hand = sorted(self.hand, key=lambda c: c.points)
        pass_cards = sorted_hand[:n]
        return ("pass", [repr(c) for c in pass_cards])

    def choose_calculate(self, engine: "GameEngine") -> bool:
        return self.pile_points >= 31

    def choose_stake(self, engine: "GameEngine") -> int:
        """Return 0 to decline, or new stake value to offer/accept."""
        return 0  # Conservative bot never raises


# ─────────────────────────────────────────────
# SESSION STORE (in-memory; swap for Redis/DB for multiplayer)
# ─────────────────────────────────────────────

games: dict[str, GameEngine] = {}


def get_game(game_id: str) -> GameEngine:
    if game_id not in games:
        raise KeyError(f"Game {game_id} not found")
    return games[game_id]


# ─────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────

app = Flask(__name__, static_folder=".")

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

@app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
def options_handler(path):
    return make_response("", 204)


def error_response(msg: str, code: int = 400):
    return jsonify({"success": False, "error": msg}), code


def ok_response(data: dict):
    return jsonify({"success": True, **data})


# ── Game management ──────────────────────────

@app.route("/api/new_game", methods=["POST"])
def new_game():
    """Create a new game. Supports human vs bot or human vs human."""
    body = request.get_json(silent=True) or {}
    mode = body.get("mode", "hvb")          # hvb = human vs bot, hvh = human vs human
    target = int(body.get("target_score", 7))
    player_name = body.get("player_name", "Player")

    p1 = HumanPlayer("p1", player_name)

    if mode == "hvb":
        p2 = SimpleBot("p2", "Bot")
    else:
        p2 = HumanPlayer("p2", body.get("player2_name", "Player 2"))

    engine = GameEngine(p1, p2, target_score=target)
    game_id = str(uuid.uuid4())[:8]
    games[game_id] = engine

    return ok_response({"game_id": game_id, "mode": mode})


@app.route("/api/game/<game_id>/state", methods=["GET"])
def game_state(game_id: str):
    try:
        engine = get_game(game_id)
    except KeyError:
        return error_response("Game not found", 404)

    debug = request.args.get("debug", "false").lower() == "true"
    perspective = request.args.get("perspective")
    perspective_idx = int(perspective) if perspective is not None else None

    state = engine.get_state(perspective_idx=perspective_idx, debug=debug)
    return ok_response({"state": state})


# ── Round lifecycle ──────────────────────────

@app.route("/api/game/<game_id>/start_round", methods=["POST"])
def start_round(game_id: str):
    try:
        engine = get_game(game_id)
    except KeyError:
        return error_response("Game not found", 404)

    if engine.phase not in (GamePhase.WAITING, GamePhase.ROUND_OVER):
        return error_response(f"Cannot start round from phase: {engine.phase.value}")

    engine.start_round()

    # If bot goes first in stakes, let it act
    _bot_act_if_needed(engine)

    state = engine.get_state(perspective_idx=0)
    return ok_response({"state": state})


# ── Stakes ───────────────────────────────────

@app.route("/api/game/<game_id>/offer_stake", methods=["POST"])
def offer_stake(game_id: str):
    try:
        engine = get_game(game_id)
    except KeyError:
        return error_response("Game not found", 404)

    body = request.get_json(silent=True) or {}
    player_idx = int(body.get("player_idx", 0))
    new_stake = int(body.get("stake", 2))

    if not engine.offer_stake(player_idx, new_stake):
        return error_response(engine.error)

    _bot_act_if_needed(engine)

    return ok_response({"state": engine.get_state(perspective_idx=0)})


@app.route("/api/game/<game_id>/accept_stake", methods=["POST"])
def accept_stake(game_id: str):
    try:
        engine = get_game(game_id)
    except KeyError:
        return error_response("Game not found", 404)

    body = request.get_json(silent=True) or {}
    player_idx = int(body.get("player_idx", 0))

    if not engine.accept_stake(player_idx):
        return error_response(engine.error)

    _bot_act_if_needed(engine)

    return ok_response({"state": engine.get_state(perspective_idx=0)})


@app.route("/api/game/<game_id>/decline_stake", methods=["POST"])
def decline_stake(game_id: str):
    try:
        engine = get_game(game_id)
    except KeyError:
        return error_response("Game not found", 404)

    body = request.get_json(silent=True) or {}
    player_idx = int(body.get("player_idx", 0))

    if not engine.decline_stake(player_idx):
        return error_response(engine.error)

    return ok_response({"state": engine.get_state(perspective_idx=0)})


@app.route("/api/game/<game_id>/start_play", methods=["POST"])
def start_play(game_id: str):
    try:
        engine = get_game(game_id)
    except KeyError:
        return error_response("Game not found", 404)

    if not engine.start_play():
        return error_response(engine.error)

    _bot_act_if_needed(engine)

    return ok_response({"state": engine.get_state(perspective_idx=0)})


# ── Playing ──────────────────────────────────

@app.route("/api/game/<game_id>/play", methods=["POST"])
def play_cards(game_id: str):
    try:
        engine = get_game(game_id)
    except KeyError:
        return error_response("Game not found", 404)

    body = request.get_json(silent=True) or {}
    player_idx = int(body.get("player_idx", 0))
    card_ids = body.get("cards", [])

    if not engine.play_cards(player_idx, card_ids):
        return error_response(engine.error)

    # Let bot respond
    _bot_act_if_needed(engine)

    return ok_response({"state": engine.get_state(perspective_idx=0)})


@app.route("/api/game/<game_id>/cut", methods=["POST"])
def cut_cards(game_id: str):
    try:
        engine = get_game(game_id)
    except KeyError:
        return error_response("Game not found", 404)

    body = request.get_json(silent=True) or {}
    player_idx = int(body.get("player_idx", 0))
    card_ids = body.get("cards", [])

    if not engine.cut_cards(player_idx, card_ids):
        return error_response(engine.error)

    _bot_act_if_needed(engine)

    return ok_response({"state": engine.get_state(perspective_idx=0)})


@app.route("/api/game/<game_id>/pass", methods=["POST"])
def pass_cards(game_id: str):
    try:
        engine = get_game(game_id)
    except KeyError:
        return error_response("Game not found", 404)

    body = request.get_json(silent=True) or {}
    player_idx = int(body.get("player_idx", 0))
    card_ids = body.get("cards", [])

    if not engine.pass_cards(player_idx, card_ids):
        return error_response(engine.error)

    _bot_act_if_needed(engine)

    return ok_response({"state": engine.get_state(perspective_idx=0)})


# ── Calculating ──────────────────────────────

@app.route("/api/game/<game_id>/calculate", methods=["POST"])
def calculate(game_id: str):
    try:
        engine = get_game(game_id)
    except KeyError:
        return error_response("Game not found", 404)

    body = request.get_json(silent=True) or {}
    player_idx = int(body.get("player_idx", 0))

    if not engine.calculate(player_idx):
        return error_response(engine.error)

    return ok_response({"state": engine.get_state(perspective_idx=0)})


@app.route("/api/game/<game_id>/skip_calculate", methods=["POST"])
def skip_calculate(game_id: str):
    try:
        engine = get_game(game_id)
    except KeyError:
        return error_response("Game not found", 404)

    body = request.get_json(silent=True) or {}
    player_idx = int(body.get("player_idx", 0))

    if not engine.skip_calculate(player_idx):
        return error_response(engine.error)

    _bot_act_if_needed(engine)

    return ok_response({"state": engine.get_state(perspective_idx=0)})


# ── Serve frontend ───────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


# ─────────────────────────────────────────────
# BOT AUTO-PLAY LOGIC
# ─────────────────────────────────────────────

def _bot_act_if_needed(engine: GameEngine, max_iterations: int = 20):
    """
    After each human action, keep letting the bot act until it's the human's turn
    or the game is waiting for human input.
    """
    for _ in range(max_iterations):
        phase = engine.phase

        if phase in (GamePhase.GAME_OVER, GamePhase.ROUND_OVER, GamePhase.WAITING):
            break

        bot_idx = next((i for i, p in enumerate(engine.players) if not p.is_human()), None)
        if bot_idx is None:
            break  # HvH, no bot

        bot = engine.players[bot_idx]

        # STAKES phase — bot may want to raise (or respond to human offer)
        if phase == GamePhase.STAKES:
            if engine.stake_offerer_idx is not None and engine.stake_offerer_idx != bot_idx:
                # Human offered stake — bot responds
                # Simple bot: accept if pending is ≤ 3, else decline
                if engine.pending_stake <= 3:
                    engine.accept_stake(bot_idx)
                else:
                    engine.decline_stake(bot_idx)
            elif engine.stake_offerer_idx is None:
                # No pending offer — always break so human gets a chance to raise stakes.
                # The human calls start_play() explicitly when ready.
                break
            else:
                # Bot already offered, waiting for human
                break

        # PLAYING phase — bot's turn
        elif phase == GamePhase.PLAYING:
            if engine.active_idx != bot_idx:
                break  # Human's turn
            play_ids = bot.choose_play(engine)
            engine.play_cards(bot_idx, play_ids)

        # CUTTING / FORCED_CUT — bot is the opponent
        elif phase in (GamePhase.CUTTING, GamePhase.FORCED_CUT):
            if engine.playing_player_idx == bot_idx:
                break  # Bot played, waiting for human to cut/pass
            action, card_ids = bot.choose_cut_or_pass(engine)
            if action == "cut":
                engine.cut_cards(bot_idx, card_ids)
            else:
                engine.pass_cards(bot_idx, card_ids)

        # CALCULATING — bot decides whether to calculate
        elif phase == GamePhase.CALCULATING:
            if engine.calculator_idx != bot_idx:
                break  # Human has right to calculate
            if bot.choose_calculate(engine):
                engine.calculate(bot_idx)
            else:
                engine.skip_calculate(bot_idx)

        else:
            break


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"BURA server running at http://localhost:{port}")
    app.run(debug=True, port=port)