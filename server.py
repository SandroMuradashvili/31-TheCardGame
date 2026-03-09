"""
BURA Server v2
- HvB (human vs bot) single-player
- LAN multiplayer with room codes + shareable link
- Polling-based (no socketio dependency) — client polls /state every ~800ms when waiting
"""

from flask import Flask, jsonify, request, send_from_directory, make_response
import os, uuid, random, string, socket, threading, time

from game_engine import (
    GameEngine, HumanPlayer, BotPlayer,
    GamePhase, Rank, POINT_VALUES
)


# ─────────────────────────────────────────────
# SIMPLE BOT
# ─────────────────────────────────────────────

class SimpleBot(BotPlayer):
    """Conservative bot. Subclass BotPlayer for smarter AI."""

    def choose_play(self, engine: GameEngine) -> list[str]:
        from itertools import combinations as _comb
        hand = self.hand
        trump = engine.trump_suit

        # If we have 3 trumps — instant win, play them
        trumps = [c for c in hand if c.suit == trump]
        if len(trumps) == 3:
            return [repr(c) for c in trumps]

        # Group by suit; prefer longest suit run
        by_suit: dict = {}
        for c in hand:
            by_suit.setdefault(c.suit, []).append(c)

        # Prefer playing single highest-value non-trump card to keep trump
        non_trump = [c for c in hand if c.suit != trump]
        if non_trump:
            best = max(non_trump, key=lambda c: c.points)
            return [repr(best)]

        # Only trumps left — play lowest
        best = min(hand, key=lambda c: c.points)
        return [repr(best)]

    def choose_cut_or_pass(self, engine: GameEngine):
        valid = engine.get_valid_cuts(engine.players.index(self))
        if valid:
            # Cut with the lowest-value valid combination
            def combo_value(combo):
                return sum(next(c for c in self.hand if repr(c) == cid).points for cid in combo)
            best = min(valid, key=combo_value)
            return ("cut", best)
        # Pass lowest-value cards
        n = len(engine.played_cards)
        cheapest = sorted(self.hand, key=lambda c: c.points)[:n]
        return ("pass", [repr(c) for c in cheapest])

    def choose_calculate(self, engine: GameEngine) -> bool:
        return self.pile_points >= 31

    def choose_raise_stake(self, engine: GameEngine) -> bool:
        return False  # Conservative: never raises


# ─────────────────────────────────────────────
# ROOM / SESSION STORE
# ─────────────────────────────────────────────

class Room:
    def __init__(self, room_id: str, host_name: str, mode: str, target_score: int):
        self.room_id = room_id
        self.mode = mode  # 'hvb' or 'hvh'
        self.target_score = target_score
        self.host_name = host_name
        self.guest_name: str = ""
        self.engine: GameEngine = None
        self.status = "waiting"  # waiting, playing, done
        self.created_at = time.time()

    def is_full(self) -> bool:
        return self.mode == "hvb" or bool(self.guest_name)

    def start_game(self):
        p1 = HumanPlayer("p1", self.host_name)
        if self.mode == "hvb":
            p2 = SimpleBot("p2", "Bot")
        else:
            p2 = HumanPlayer("p2", self.guest_name)
        self.engine = GameEngine(p1, p2, target_score=self.target_score)
        self.engine.start_round()
        self.status = "playing"
        # For HvB, let bot act if it goes first
        if self.mode == "hvb":
            _bot_act_if_needed(self.engine)


rooms: dict[str, Room] = {}


def make_room_code() -> str:
    """6 uppercase letters, easy to type."""
    while True:
        code = ''.join(random.choices(string.ascii_uppercase, k=6))
        if code not in rooms:
            return code


def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


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
def options_handler(path=""):
    return make_response("", 204)


def err(msg, code=400):
    return jsonify({"success": False, "error": msg}), code

def ok(**data):
    return jsonify({"success": True, **data})


# ── Room management ──────────────────────────

@app.route("/api/create_room", methods=["POST"])
def create_room():
    body = request.get_json(silent=True) or {}
    host_name = body.get("player_name", "Host")[:20]
    mode = body.get("mode", "hvb")
    target = int(body.get("target_score", 7))

    room_id = make_room_code()
    room = Room(room_id, host_name, mode, target)
    rooms[room_id] = room

    local_ip = get_local_ip()
    port = request.host.split(":")[-1] if ":" in request.host else "5000"
    join_link = f"http://{local_ip}:{port}/join/{room_id}"

    # HvB: start immediately
    if mode == "hvb":
        room.start_game()
        return ok(
            room_id=room_id,
            join_link=join_link,
            status="playing",
            player_idx=0,
        )
    else:
        return ok(
            room_id=room_id,
            join_link=join_link,
            status="waiting",
            player_idx=0,
        )


@app.route("/api/join_room", methods=["POST"])
def join_room():
    body = request.get_json(silent=True) or {}
    room_id = body.get("room_id", "").upper().strip()
    guest_name = body.get("player_name", "Guest")[:20]

    if room_id not in rooms:
        return err("Room not found. Check the code and try again.")
    room = rooms[room_id]
    if room.is_full():
        return err("Room is already full.")
    if room.mode != "hvh":
        return err("That room is not a multiplayer room.")

    room.guest_name = guest_name
    room.start_game()

    local_ip = get_local_ip()
    port = request.host.split(":")[-1] if ":" in request.host else "5000"
    join_link = f"http://{local_ip}:{port}/join/{room_id}"

    return ok(
        room_id=room_id,
        join_link=join_link,
        status="playing",
        player_idx=1,
    )


@app.route("/api/room/<room_id>/status", methods=["GET"])
def room_status(room_id: str):
    """Lightweight poll — used by host to detect when guest joins."""
    room_id = room_id.upper()
    if room_id not in rooms:
        return err("Room not found", 404)
    room = rooms[room_id]
    return ok(
        status=room.status,
        mode=room.mode,
        host_name=room.host_name,
        guest_name=room.guest_name,
        is_full=room.is_full(),
    )


@app.route("/join/<room_id>")
def join_page(room_id):
    """Serve the frontend — JS will pick up the room_id from URL."""
    return send_from_directory(".", "index.html")


# ── Game state & actions ─────────────────────

def _get_room_engine(room_id: str):
    room_id = room_id.upper()
    if room_id not in rooms:
        return None, None
    room = rooms[room_id]
    if room.engine is None:
        return room, None
    return room, room.engine


@app.route("/api/room/<room_id>/state", methods=["GET"])
def game_state(room_id: str):
    room, engine = _get_room_engine(room_id)
    if room is None:
        return err("Room not found", 404)
    if engine is None:
        return ok(state=None, room_status=room.status,
                   host_name=room.host_name, guest_name=room.guest_name)

    debug = request.args.get("debug", "false").lower() == "true"
    perspective = request.args.get("perspective")
    perspective_idx = int(perspective) if perspective is not None else None
    state = engine.get_state(perspective_idx=perspective_idx, debug=debug)
    return ok(state=state, room_status=room.status)


def _action(room_id: str, perspective_idx: int, fn, *args, **kwargs):
    """Generic action handler."""
    room, engine = _get_room_engine(room_id)
    if room is None:
        return err("Room not found", 404)
    if engine is None:
        return err("Game not started yet.")
    result = fn(*args, **kwargs)
    if not result:
        return err(engine.error or "Action failed.")
    if room.mode == "hvb":
        _bot_act_if_needed(engine)
    state = engine.get_state(perspective_idx=perspective_idx)
    return ok(state=state)


@app.route("/api/room/<room_id>/start_round", methods=["POST"])
def start_round(room_id: str):
    room, engine = _get_room_engine(room_id)
    if room is None:
        return err("Room not found", 404)
    if engine is None:
        return err("Game not started.")
    if engine.phase not in (GamePhase.WAITING, GamePhase.ROUND_OVER):
        return err(f"Cannot start round now (phase: {engine.phase.value})")
    engine.start_round()
    if room.mode == "hvb":
        _bot_act_if_needed(engine)
    body = request.get_json(silent=True) or {}
    pidx = int(body.get("player_idx", 0))
    return ok(state=engine.get_state(perspective_idx=pidx))


@app.route("/api/room/<room_id>/offer_stake", methods=["POST"])
def offer_stake(room_id: str):
    body = request.get_json(silent=True) or {}
    pidx = int(body.get("player_idx", 0))
    return _action(room_id, pidx, lambda: rooms[room_id.upper()].engine.offer_stake(pidx))


@app.route("/api/room/<room_id>/accept_stake", methods=["POST"])
def accept_stake(room_id: str):
    body = request.get_json(silent=True) or {}
    pidx = int(body.get("player_idx", 0))
    return _action(room_id, pidx, lambda: rooms[room_id.upper()].engine.accept_stake(pidx))


@app.route("/api/room/<room_id>/decline_stake", methods=["POST"])
def decline_stake(room_id: str):
    body = request.get_json(silent=True) or {}
    pidx = int(body.get("player_idx", 0))
    return _action(room_id, pidx, lambda: rooms[room_id.upper()].engine.decline_stake(pidx))


@app.route("/api/room/<room_id>/play", methods=["POST"])
def play_cards(room_id: str):
    body = request.get_json(silent=True) or {}
    pidx = int(body.get("player_idx", 0))
    cards = body.get("cards", [])
    return _action(room_id, pidx, lambda: rooms[room_id.upper()].engine.play_cards(pidx, cards))


@app.route("/api/room/<room_id>/cut", methods=["POST"])
def cut_cards(room_id: str):
    body = request.get_json(silent=True) or {}
    pidx = int(body.get("player_idx", 0))
    cards = body.get("cards", [])
    return _action(room_id, pidx, lambda: rooms[room_id.upper()].engine.cut_cards(pidx, cards))


@app.route("/api/room/<room_id>/pass", methods=["POST"])
def pass_cards(room_id: str):
    body = request.get_json(silent=True) or {}
    pidx = int(body.get("player_idx", 0))
    cards = body.get("cards", [])
    return _action(room_id, pidx, lambda: rooms[room_id.upper()].engine.pass_cards(pidx, cards))


@app.route("/api/room/<room_id>/calculate", methods=["POST"])
def calculate(room_id: str):
    body = request.get_json(silent=True) or {}
    pidx = int(body.get("player_idx", 0))
    return _action(room_id, pidx, lambda: rooms[room_id.upper()].engine.calculate(pidx))


@app.route("/api/room/<room_id>/skip_calculate", methods=["POST"])
def skip_calculate(room_id: str):
    body = request.get_json(silent=True) or {}
    pidx = int(body.get("player_idx", 0))
    return _action(room_id, pidx, lambda: rooms[room_id.upper()].engine.skip_calculate(pidx))


# ── Frontend ─────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


# ─────────────────────────────────────────────
# BOT LOGIC
# ─────────────────────────────────────────────

def _bot_act_if_needed(engine: GameEngine, max_iter: int = 30):
    bot_idx = next((i for i, p in enumerate(engine.players) if not p.is_human()), None)
    if bot_idx is None:
        return

    bot = engine.players[bot_idx]

    for _ in range(max_iter):
        phase = engine.phase
        if phase in (GamePhase.GAME_OVER, GamePhase.ROUND_OVER, GamePhase.WAITING):
            break

        if phase == GamePhase.STAKES:
            if engine.stake_offerer_idx is not None and engine.stake_offerer_idx != bot_idx:
                # Respond to human's offer
                if engine.pending_stake <= 3:
                    engine.accept_stake(bot_idx)
                else:
                    engine.decline_stake(bot_idx)
            else:
                # Bot's turn to decide: never raises (conservative), just wait for human
                break

        elif phase == GamePhase.PLAYING:
            if engine.active_idx != bot_idx:
                break
            play_ids = bot.choose_play(engine)
            engine.play_cards(bot_idx, play_ids)

        elif phase in (GamePhase.CUTTING, GamePhase.FORCED_CUT):
            if engine.playing_player_idx == bot_idx:
                break  # Human needs to cut/pass
            action, card_ids = bot.choose_cut_or_pass(engine)
            if action == "cut":
                engine.cut_cards(bot_idx, card_ids)
            else:
                engine.pass_cards(bot_idx, card_ids)

        elif phase == GamePhase.CALCULATING:
            if engine.calculator_idx != bot_idx:
                break
            if bot.choose_calculate(engine):
                engine.calculate(bot_idx)
            else:
                engine.skip_calculate(bot_idx)
        else:
            break


# ─────────────────────────────────────────────
# ENTRY
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    local_ip = get_local_ip()
    print(f"\n{'='*50}")
    print(f"  BURA Server running")
    print(f"  Local:   http://localhost:{port}")
    print(f"  Network: http://{local_ip}:{port}  ← share this for LAN play")
    print(f"{'='*50}\n")
    app.run(host="0.0.0.0", debug=True, port=port)