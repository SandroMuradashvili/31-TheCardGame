"""
server.py — Flask HTTP API + static file serving
──────────────────────────────────────────────────
Entry point. Contains only:
  - Flask app setup
  - Static file routes
  - One route function per API endpoint

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
from bvb_runner import bvb_step, bvb_run_full
from bot import get_bot, list_bots
from sim_runner import start_simulation, get_sim_status, cancel_simulation, simulations


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=BASE_DIR)


# ─── CORS ─────────────────────────────────────────────────────────────────────

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


# ─── Static files ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/stats")
def stats_page():
    return send_from_directory(BASE_DIR, "stats.html")

@app.route("/join/<room_id>")
def join_page(room_id):
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/css/<path:filename>")
def serve_css(filename):
    return send_from_directory(os.path.join(BASE_DIR, "css"), filename)

@app.route("/js/<path:filename>")
def serve_js(filename):
    return send_from_directory(os.path.join(BASE_DIR, "js"), filename)


# ─── Helpers ──────────────────────────────────────────────────────────────────

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
    room_id = room_id.upper()
    if room_id not in rooms:
        return None, None
    room = rooms[room_id]
    return room, room.engine


def _action(room_id: str, perspective_idx: int, fn):
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


# ─── Bot list ─────────────────────────────────────────────────────────────────

@app.route("/api/bots/list", methods=["GET"])
def bots_list():
    return ok(bots=list_bots())


# ─── Room management ─────────────────────────────────────────────────────────

@app.route("/api/create_room", methods=["POST"])
def create_room():
    body      = request.get_json(silent=True) or {}
    host_name = body.get("player_name", "Host")[:20]
    mode      = body.get("mode", "hvb")
    target    = int(body.get("target_score", 7))
    room_id   = make_room_code()
    room      = Room(room_id, host_name, mode, target)
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
    return ok(status=room.status, mode=room.mode,
              host_name=room.host_name, guest_name=room.guest_name,
              is_full=room.is_full())


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

    body = request.get_json(silent=True) or {}
    pidx = int(body.get("player_idx", 0))

    # If round is already running (HvH race: both players clicked "Next Round"),
    # just return current state instead of erroring.
    if engine.phase not in (GamePhase.WAITING, GamePhase.ROUND_OVER):
        return ok(state=engine.get_state(perspective_idx=pidx))

    engine.start_round()
    if room.mode == "hvb":
        bot_act_if_needed(engine)

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
    return ok(state=engine.get_state(perspective_idx=pidx))


@app.route("/api/room/<room_id>/bot_respond", methods=["POST"])
def bot_respond(room_id: str):
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


# ─── Debug / dev routes ───────────────────────────────────────────────────────

@app.route("/api/room/<room_id>/dev/deal_specific", methods=["POST"])
def dev_deal_specific(room_id: str):
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

    if room.mode == "hvb":
        bot_act_if_needed(engine)

    state = engine.get_state(perspective_idx=pidx, debug=True)
    return ok(state=state)


# ─── BvB Spectator routes ─────────────────────────────────────────────────────

@app.route("/api/create_bvb", methods=["POST"])
def create_bvb():
    """Create a BvB spectator room with chosen bot types."""
    body     = request.get_json(silent=True) or {}
    target   = int(body.get("target_score", 7))
    bot_a_id = body.get("bot_a", "simple")
    bot_b_id = body.get("bot_b", "aggressive")

    from game_engine import GameEngine
    room_id = make_room_code()
    room    = Room(room_id, f"Bot A", "bvb", target)
    rooms[room_id] = room

    bots = list_bots()
    name_a = next((b["name"] for b in bots if b["id"] == bot_a_id), bot_a_id)
    name_b = next((b["name"] for b in bots if b["id"] == bot_b_id), bot_b_id)

    p1 = get_bot(bot_a_id, "p1", name_a)
    p2 = get_bot(bot_b_id, "p2", name_b)
    room.engine = GameEngine(p1, p2, target_score=target)
    room.engine.start_round()
    room.status = "playing"

    # Store bot ids on room for reference
    room.bot_a_id = bot_a_id
    room.bot_b_id = bot_b_id

    state = room.engine.get_state(debug=True)
    return ok(room_id=room_id, state=state, bot_a=bot_a_id, bot_b=bot_b_id)


@app.route("/api/room/<room_id>/bvb_step", methods=["POST"])
def bvb_step_route(room_id: str):
    """Advance BvB game by exactly one action."""
    room, engine = _get_room(room_id)
    if room is None:
        return err("Room not found", 404)
    if engine is None:
        return err("Game not started.")

    action = bvb_step(engine)
    state  = engine.get_state(debug=True)
    return ok(state=state, action=action)


@app.route("/api/room/<room_id>/bvb_run", methods=["POST"])
def bvb_run_route(room_id: str):
    """Run entire remaining game headlessly. Used for instant mode."""
    room, engine = _get_room(room_id)
    if room is None:
        return err("Room not found", 404)
    if engine is None:
        return err("Game not started.")

    body      = request.get_json(silent=True) or {}
    max_steps = int(body.get("max_steps", 5000))

    try:
        history = bvb_run_full(engine, max_steps)
        state   = engine.get_state(debug=True)
        return ok(state=state, history=history)
    except Exception as e:
        return err(f"Simulation error: {str(e)}")


# ─── Mass Simulation routes ───────────────────────────────────────────────────

@app.route("/api/sim/start", methods=["POST"])
def sim_start():
    body     = request.get_json(silent=True) or {}
    bot_ids  = body.get("bot_ids", [])
    games    = int(body.get("games_per_matchup", 1000))
    target   = int(body.get("target_score", 7))

    if len(bot_ids) < 2:
        return err("Need at least 2 bots.")
    if games < 1 or games > 100000:
        return err("games_per_matchup must be 1–100000.")

    # Validate bot ids
    from bot import BOT_REGISTRY
    for bid in bot_ids:
        if bid not in BOT_REGISTRY:
            return err(f"Unknown bot: {bid}")

    sim_id = start_simulation(bot_ids, games, target)
    return ok(sim_id=sim_id)


@app.route("/api/sim/status/<sim_id>", methods=["GET"])
def sim_status(sim_id: str):
    status = get_sim_status(sim_id)
    if status is None:
        return err("Simulation not found", 404)
    return ok(**status)


@app.route("/api/sim/cancel/<sim_id>", methods=["POST"])
def sim_cancel(sim_id: str):
    cancel_simulation(sim_id)
    return ok(cancelled=True)


@app.route("/api/sim/list", methods=["GET"])
def sim_list():
    result = []
    for sid, sim in simulations.items():
        result.append({
            "sim_id":   sid,
            "status":   sim["status"],
            "matchups": len(sim["matchups"]),
        })
    return ok(simulations=result)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port     = int(os.environ.get("PORT", 5000))
    local_ip = _get_local_ip()
    print(f"\n{'='*55}")
    print(f"  BURA Server")
    print(f"  Local:   http://localhost:{port}")
    print(f"  Network: http://{local_ip}:{port}  ← share for LAN play")
    print(f"  Stats:   http://localhost:{port}/stats")
    print(f"{'='*55}\n")
    app.run(host="0.0.0.0", debug=True, port=port)