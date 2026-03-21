"""
bot_runner.py — Bot turn driver
────────────────────────────────
Contains bot_act_if_needed(), which steps the bot through
every phase it's responsible for after a human action.

Separated from bot.py (AI decisions) and server.py (HTTP routing)
so the driving loop is easy to read and modify independently.

Depends on: game_engine.py, bot.py
Imported by: room_manager.py, server.py
"""

from game_engine import GameEngine, GamePhase


def bot_act_if_needed(engine: GameEngine, max_iter: int = 30):
    """
    Run the bot through as many consecutive turns as it owns.
    Stops when:
      - It's the human's turn
      - The round/game ends
      - max_iter is hit (safety guard against infinite loops)

    Call this after every human action in HvB mode.
    """
    bot_idx = next(
        (i for i, p in enumerate(engine.players) if not p.is_human()),
        None
    )
    if bot_idx is None:
        return  # No bot in this room (HvH mode)

    bot = engine.players[bot_idx]

    for _ in range(max_iter):
        phase = engine.phase

        # Terminal / waiting states — nothing to do
        if phase in (GamePhase.GAME_OVER, GamePhase.ROUND_OVER, GamePhase.WAITING):
            break

        # ── Stakes ───────────────────────────────────────────────────────────
        if phase == GamePhase.STAKES:
            if (engine.stake_offerer_idx is not None
                    and engine.stake_offerer_idx != bot_idx):
                # Human raised — bot responds: accept cheap raises, decline expensive ones
                if engine.pending_stake <= 3:
                    engine.accept_stake(bot_idx)
                else:
                    engine.decline_stake(bot_idx)

            elif engine.stake_offerer_idx is None and engine.active_idx == bot_idx:
                # Bot leads first — skip negotiation and just start playing
                engine.start_play()

            else:
                # Waiting for human to respond to bot's offer, or human plays first
                break

        # ── Playing ──────────────────────────────────────────────────────────
        elif phase == GamePhase.PLAYING:
            if engine.active_idx != bot_idx:
                break  # Human's turn to play
            play_ids = bot.choose_play(engine)
            engine.play_cards(bot_idx, play_ids)

        # ── Cutting / Forced cut ─────────────────────────────────────────────
        elif phase in (GamePhase.CUTTING, GamePhase.FORCED_CUT):
            if engine.playing_player_idx == bot_idx:
                break  # Human needs to cut or pass
            action, card_ids = bot.choose_cut_or_pass(engine)
            if action == "cut":
                engine.cut_cards(bot_idx, card_ids)
            else:
                engine.pass_cards(bot_idx, card_ids)

        # ── Calculating ──────────────────────────────────────────────────────
        elif phase == GamePhase.CALCULATING:
            if engine.calculator_idx != bot_idx:
                break  # Human has the calculate right
            if bot.choose_calculate(engine):
                engine.calculate(bot_idx)
            else:
                engine.skip_calculate(bot_idx)

        else:
            break  # Unknown phase — don't get stuck
