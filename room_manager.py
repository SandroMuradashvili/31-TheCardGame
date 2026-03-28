"""
room_manager.py — Room store and Room lifecycle
───────────────────────────────────────────────
Owns:
  - Room class (state for one game session)
  - rooms dict (in-memory store of all active rooms)
  - make_room_code() helper

Depends on: game_engine.py, bot.py

Imported by: server.py (all Flask routes call into here)
"""

import random
import string
import time
from bot import get_bot, BOT_REGISTRY, SimpleBot

from game_engine import GameEngine, HumanPlayer, GamePhase
from bot import SimpleBot


# ─── In-memory store ──────────────────────────────────────────────────────────
# All active rooms, keyed by 6-letter code.
# Rooms are never explicitly deleted — old rooms just sit idle.
rooms: dict[str, "Room"] = {}


def make_room_code() -> str:
    """Generate a unique 6-letter uppercase room code."""
    while True:
        code = ''.join(random.choices(string.ascii_uppercase, k=6))
        if code not in rooms:
            return code


# ─── Room ─────────────────────────────────────────────────────────────────────

class Room:
    """
    Holds everything associated with one game session:
      - metadata (mode, players, target score)
      - the GameEngine instance
      - bot_instant flag for test/debug speed
    """

    def __init__(self, room_id: str, host_name: str, mode: str, target_score: int):
        self.room_id      = room_id
        self.mode         = mode           # 'hvb' (vs bot) | 'hvh' (vs human)
        self.target_score = target_score
        self.host_name    = host_name
        self.guest_name   = ""
        self.engine: GameEngine = None
        self.status       = "waiting"      # waiting | playing | done
        self.created_at   = time.time()

    def is_full(self) -> bool:
        """HvB is always full; HvH needs a guest to join."""
        return self.mode == "hvb" or bool(self.guest_name)


    def start_game(self, bot_id: str = "simple"):
        """
               Instantiate the GameEngine and deal the first round.
               For HvB, also immediately runs the bot if it goes first.
               """
        p1 = HumanPlayer("p1", self.host_name)
        if self.mode == "hvb":
            from bot import BOT_REGISTRY
            name = getattr(BOT_REGISTRY.get(bot_id, SimpleBot), "DISPLAY_NAME", bot_id)
            p2 = get_bot(bot_id, "p2", name)
        else:
            p2 = HumanPlayer("p2", self.guest_name)

        self.engine = GameEngine(p1, p2, target_score=self.target_score)
        self.engine.start_round()
        self.status = "playing"

        if self.mode == "hvb":
            from bot_runner import bot_act_if_needed
            bot_act_if_needed(self.engine)