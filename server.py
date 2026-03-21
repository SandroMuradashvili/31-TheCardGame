"""
server.py — Flask HTTP API + static file serving
──────────────────────────────────────────────────
Entry point. Contains only:
  - Flask app setup
  - Static file routes (/, /css, /js, /join)
  - One route function per API endpoint

Game logic lives in:  game_engine.py
Bot decisions live in: bot.py
Bot turn driver in:   bot_runner.py
Room/session store in: room_manager.py

Run:
    pip install flask
    python server.py
    → http://localhost:5000
"""

import os
import socket

from flask import Flask, jsonify, request, send_from_directory, make_response

from game_engine import GamePhase
from room_manager import Room, rooms, make_room_code
from bot_runner import bot_act_if_needed


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=BASE_DIR)


# ─── CORS (allow LAN cross-origin requests) ───────────────────────────────────

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

@app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
def options_handler(path=""):
    return make_response("", 204)


# ─── Response helpers ─────────────────────────────────────────────────────────

def err(msg, code=400):
    return jsonify({"success": False, "error": msg}), code

def ok(**data):
    return jsonify({"success": True, **data})


# ─── Static file serving ──────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/join/<room_id>")
def join_page(room_id):
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/css/<path:filename>")
def serve_css(filename):
    return send_from_directory(os.path.join(BASE_DIR, "css"), filename)

@app.route("/js/<path:filename>")
def serve_js(filename):
    return send_from_directory(os.path.join(BASE_DIR, "js"), filename)


# ─── Shared helpers ───────────────────────────────────────────────────────────

def _get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def _get_room(room_id: str):
    """Look up a room by code. Returns (room, engine) or (None, None)."""
    room_id = room_id.upper()
    if room_id not in rooms:
        return None, None
    room = rooms[room_id]
    return room, room.engine


def _action(room_id: str, perspective_idx: int, fn):
    """
    Generic wrapper for single-step game actions.
    Runs fn(), triggers the bot if HvB, returns updated state.
    """
    room, engine = _get_room(room_id)
    if room is None:
        return err("Room not found", 404)
    if engine is None:
        return err("Game not started yet.")

    result = fn()
    if not result:
        return err(engine.error or "Action failed.")

    if room.mode == "hvb":
        bot_act_if_needed(engine)

    state = engine.get_state(perspective_idx=perspective_idx)
    return ok(state=state)


# ─── Room management routes ───────────────────────────────────────────────────

@app.route("/api/create_room", methods=["POST"])
def create_room():
    body        = request.get_json(silent=True) or {}
    host_name   = body.get("player_name", "Host")[:20]
    mode        = body.get("mode", "hvb")
    target      = int(body.get("target_score", 7))
    bot_instant = bool(body.get("bot_instant", False))

    room_id   = make_room_code()
    room      = Room(room_id, host_name, mode, target)
    room.bot_instant = bot_instant
    rooms[room_id] = room

    local_ip  = _get_local_ip()
    port      = request.host.split(":")[-1] if ":" in request.host else "5000"
    join_link = f"http://{local_ip}:{port}/join/{room_id}"

    if mode == "hvb":
        room.start_game()
        return ok(room_id=room_id, join_link=join_link, status="playing", player_idx=0)
    else:
        return ok(room_id=room_id, join_link=join_link, status="waiting", player_idx=0)


@app.route("/api/join_room", methods=["POST"])
def join_room():
    body       = request.get_json(silent=True) or {}
    room_id    = body.get("room_id", "").upper().strip()
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

    local_ip  = _get_local_ip()
    port      = request.host.split(":")[-1] if ":" in request.host else "5000"
    join_link = f"http://{local_ip}:{port}/join/{room_id}"

    return ok(room_id=room_id, join_link=join_link, status="playing", player_idx=1)


@app.route("/api/room/<room_id>/status", methods=["GET"])
def room_status(room_id: str):
    room_id = room_id.upper()
    if room_id not in rooms:
        return err("Room not found", 404)
    room = rooms[room_id]
    return ok(
        status=room.status, mode=room.mode,
        host_name=room.host_name, guest_name=room.guest_name,
        is_full=room.is_full(),
    )


# ─── State ────────────────────────────────────────────────────────────────────

@app.route("/api/room/<room_id>/state", methods=["GET"])
def game_state(room_id: str):
    room, engine = _get_room(room_id)
    if room is None:
        return err("Room not found", 404)
    if engine is None:
        return ok(state=None, room_status=room.status,
                  host_name=room.host_name, guest_name=room.guest_name)

    debug       = request.args.get("debug", "false").lower() == "true"
    perspective = request.args.get("perspective")
    pidx        = int(perspective) if perspective is not None else None
    state       = engine.get_state(perspective_idx=pidx, debug=debug)
    return ok(state=state, room_status=room.status)


# ─── Round lifecycle ──────────────────────────────────────────────────────────

@app.route("/api/room/<room_id>/start_round", methods=["POST"])
def start_round(room_id: str):
    room, engine = _get_room(room_id)
    if room is None:
        return err("Room not found", 404)
    if engine is None:
        return err("Game not started.")
    if engine.phase not in (GamePhase.WAITING, GamePhase.ROUND_OVER):
        return err(f"Cannot start round now (phase: {engine.phase.value})")

    engine.start_round()

    if room.mode == "hvb":
        bot_act_if_needed(engine)

    body = request.get_json(silent=True) or {}
    pidx = int(body.get("player_idx", 0))
    return ok(state=engine.get_state(perspective_idx=pidx))


# ─── Stake routes ─────────────────────────────────────────────────────────────

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


# ─── Card play routes ─────────────────────────────────────────────────────────

@app.route("/api/room/<room_id>/play", methods=["POST"])
def play_cards(room_id: str):
    body  = request.get_json(silent=True) or {}
    pidx  = int(body.get("player_idx", 0))
    cards = body.get("cards", [])
    return _action(room_id, pidx, lambda: rooms[room_id.upper()].engine.play_cards(pidx, cards))


@app.route("/api/room/<room_id>/play_only", methods=["POST"])
def play_cards_only(room_id: str):
    """
    Two-step HvB play (step 1 of 2).
    Plays human cards WITHOUT running the bot, so the frontend can
    animate the human's cards onto the table before the bot responds.
    Follow up with /bot_respond after the animation finishes.
    """
    room, engine = _get_room(room_id)
    if room is None:
        return err("Room not found", 404)
    if engine is None:
        return err("Game not started yet.")

    body  = request.get_json(silent=True) or {}
    pidx  = int(body.get("player_idx", 0))
    cards = body.get("cards", [])

    result = engine.play_cards(pidx, cards)
    if not result:
        return err(engine.error or "Action failed.")

    # Return intermediate state (before bot acts)
    return ok(state=engine.get_state(perspective_idx=pidx))


@app.route("/api/room/<room_id>/bot_respond", methods=["POST"])
def bot_respond(room_id: str):
    """
    Two-step HvB play (step 2 of 2).
    Triggers the bot and returns the resulting state.
    Called by the frontend after the human's play animation finishes.
    """
    room, engine = _get_room(room_id)
    if room is None:
        return err("Room not found", 404)
    if engine is None:
        return err("Game not started yet.")
    if room.mode != "hvb":
        return err("Only available in HvB mode.")

    body = request.get_json(silent=True) or {}
    pidx = int(body.get("player_idx", 0))

    bot_act_if_needed(engine)
    return ok(state=engine.get_state(perspective_idx=pidx))


@app.route("/api/room/<room_id>/counter_play", methods=["POST"])
def counter_play(room_id: str):
    body  = request.get_json(silent=True) or {}
    pidx  = int(body.get("player_idx", 0))
    cards = body.get("cards", [])
    return _action(room_id, pidx, lambda: rooms[room_id.upper()].engine.counter_play(pidx, cards))


@app.route("/api/room/<room_id>/cut", methods=["POST"])
def cut_cards(room_id: str):
    body  = request.get_json(silent=True) or {}
    pidx  = int(body.get("player_idx", 0))
    cards = body.get("cards", [])
    return _action(room_id, pidx, lambda: rooms[room_id.upper()].engine.cut_cards(pidx, cards))


@app.route("/api/room/<room_id>/pass", methods=["POST"])
def pass_cards(room_id: str):
    body  = request.get_json(silent=True) or {}
    pidx  = int(body.get("player_idx", 0))
    cards = body.get("cards", [])
    return _action(room_id, pidx, lambda: rooms[room_id.upper()].engine.pass_cards(pidx, cards))


# ─── Calculate routes ─────────────────────────────────────────────────────────

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


# ─── Debug / testing routes ───────────────────────────────────────────────────

@app.route("/api/room/<room_id>/set_bot_speed", methods=["POST"])
def set_bot_speed(room_id: str):
    """Toggle instant bot moves for a room (removes animation delay)."""
    room_id = room_id.upper()
    if room_id not in rooms:
        return err("Room not found", 404)
    body    = request.get_json(silent=True) or {}
    instant = bool(body.get("instant", False))
    rooms[room_id].bot_instant = instant
    return ok(bot_instant=instant)


@app.route("/api/room/<room_id>/dev/deal_specific", methods=["POST"])
def dev_deal_specific(room_id: str):
    """
    DEV ONLY — Reset the round and deal specific cards to each player.
    Works at ANY time during a round — no phase restriction.

    Body: {
        "player_idx": 0,              -- perspective for returned state
        "p0": ["AH", "KH", "QH"],    -- up to 3 cards for player 0 (you)
        "p1": ["AD", "KD", "QD"],    -- up to 3 cards for player 1 (opponent)
        "trump": "TH"                 -- optional: which card is trump
    }
    Card format: rank (A T K Q J) + suit (H D C S). e.g. "AH" = Ace of Hearts.
    Empty slots are filled randomly. Scores are preserved; everything else resets.
    """
    room, engine = _get_room(room_id)
    if room is None:
        return err("Room not found", 404)
    if engine is None:
        return err("Game not started yet.")

    body  = request.get_json(silent=True) or {}
    pidx  = int(body.get("player_idx", 0))
    p0    = body.get("p0", [])
    p1    = body.get("p1", [])
    trump = body.get("trump", None)

    result = engine.dev_deal_specific(p0, p1, trump)
    if not result:
        return err(engine.error or "dev_deal_specific failed.")

    # Run bot if it goes first in the new round
    if room.mode == "hvb":
        bot_act_if_needed(engine)

    state = engine.get_state(perspective_idx=pidx, debug=True)
    return ok(state=state)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port     = int(os.environ.get("PORT", 5000))
    local_ip = _get_local_ip()
    print(f"\n{'='*55}")
    print(f"  BURA Server")
    print(f"  Local:   http://localhost:{port}")
    print(f"  Network: http://{local_ip}:{port}  ← share for LAN play")
    print(f"{'='*55}\n")
    app.run(host="0.0.0.0", debug=True, port=port)