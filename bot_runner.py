"""
bot_runner.py — Bot turn driver
────────────────────────────────
Drives the bot through every phase it owns after a human action.
Stops when it's the human's turn, the round ends, or max_iter is hit.

Depends on: game_engine.py
Imported by: room_manager.py, server.py
"""

from game_engine import GameEngine, GamePhase


def bot_act_if_needed(engine: GameEngine, max_iter: int = 30):
    bot_idx = next((i for i, p in enumerate(engine.players) if not p.is_human()), None)
    if bot_idx is None:
        return

    bot = engine.players[bot_idx]

    for _ in range(max_iter):
        phase = engine.phase

        if phase in (GamePhase.GAME_OVER, GamePhase.ROUND_OVER, GamePhase.WAITING):
            break

        # ── Pending stake offer — handle at any phase ─────────────────────────
        # Human can raise stakes mid-round (during playing/cutting too).
        # If the human raised, the bot must respond before anything else.
        if engine.stake_offerer_idx is not None and engine.stake_offerer_idx != bot_idx:
            if engine.pending_stake <= 3:
                engine.accept_stake(bot_idx)
            else:
                engine.decline_stake(bot_idx)
            continue  # re-check phase after responding

        # ── Stakes ───────────────────────────────────────────────────────────
        if phase == GamePhase.STAKES:
            if engine.stake_offerer_idx is None and engine.active_idx == bot_idx:
                # Bot goes first — transition to playing and continue loop so
                # bot immediately plays its card in the next iteration
                engine.start_play()
            else:
                break  # Human's turn to act on stakes

        # ── Playing ──────────────────────────────────────────────────────────
        elif phase == GamePhase.PLAYING:
            if engine.active_idx != bot_idx:
                break  # Human's turn
            engine.play_cards(bot_idx, bot.choose_play(engine))

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
                break  # Human has calculate right
            if bot.choose_calculate(engine):
                engine.calculate(bot_idx)
            else:
                engine.skip_calculate(bot_idx)

        else:
            break