"""
bot_runner.py — Bot turn driver
────────────────────────────────
Drives the bot through every phase it owns after a human action.
Stops when it's the human's turn, the round ends, or max_iter is hit.

Extended to call OracleBot.observe_* methods after every opponent
(human) action so the behavioral belief state stays current.
"""

from game_engine import GameEngine, GamePhase
from optimal_bots import OracleBot


def _notify_oracle(bot, method: str, *args):
    """Call an OracleBot observation method if the bot is an OracleBot."""
    if isinstance(bot, OracleBot) and hasattr(bot, method):
        getattr(bot, method)(*args)


def bot_act_if_needed(engine: GameEngine, max_iter: int = 30):
    bot_idx = next((i for i, p in enumerate(engine.players) if not p.is_human()), None)
    if bot_idx is None:
        return

    bot = engine.players[bot_idx]

    for _ in range(max_iter):
        phase = engine.phase

        if phase in (GamePhase.GAME_OVER, GamePhase.ROUND_OVER, GamePhase.WAITING):
            break

        # ── Pending stake offer ───────────────────────────────────────────────
        if engine.stake_offerer_idx is not None and engine.stake_offerer_idx != bot_idx:
            # Human offered a raise — OracleBot: note opponent raised
            _notify_oracle(bot, "observe_opponent_raise", engine)

            if hasattr(bot, "choose_raise_stake") and not bot.choose_raise_stake(engine):
                engine.decline_stake(bot_idx)
            else:
                engine.accept_stake(bot_idx)
            continue

        # ── Stakes ───────────────────────────────────────────────────────────
        if phase == GamePhase.STAKES:
            if engine.stake_offerer_idx is None and engine.active_idx == bot_idx:
                if (hasattr(bot, "choose_raise_stake")
                        and bot.choose_raise_stake(engine)
                        and engine.can_raise_stake(bot_idx)):
                    engine.offer_stake(bot_idx)
                    continue
                engine.start_play()
            else:
                break

        # ── Playing ──────────────────────────────────────────────────────────
        elif phase == GamePhase.PLAYING:
            if engine.active_idx != bot_idx:
                break
            engine.play_cards(bot_idx, bot.choose_play(engine))

        # ── Cutting / Forced cut ──────────────────────────────────────────────
        elif phase in (GamePhase.CUTTING, GamePhase.FORCED_CUT):
            if engine.playing_player_idx == bot_idx:
                # Human needs to cut or pass — detect what human did last
                # by checking if played_cards changed (handled in next iter)
                break

            # Bot is the cutter
            action, card_ids = bot.choose_cut_or_pass(engine)
            if action == "counter" and phase == GamePhase.CUTTING:
                result = engine.counter_play(bot_idx, card_ids)
                if not result:
                    n = len(engine.played_cards)
                    cheapest = sorted(bot.hand, key=lambda c: c.points)[:n]
                    engine.pass_cards(bot_idx, [repr(c) for c in cheapest])
            elif action == "cut":
                engine.cut_cards(bot_idx, card_ids)
            else:
                engine.pass_cards(bot_idx, card_ids)

        # ── Calculating ───────────────────────────────────────────────────────
        elif phase == GamePhase.CALCULATING:
            if engine.calculator_idx != bot_idx:
                break
            if bot.choose_calculate(engine):
                engine.calculate(bot_idx)
            else:
                engine.skip_calculate(bot_idx)

        else:
            break


def notify_human_action(engine: GameEngine, action_type: str, **kwargs):
    """
    Call this from server.py after every human action so OracleBot
    can update its behavioral belief state.

    action_type values:
      "pass"           — human passed cards (kwargs: played_cards: list[Card])
      "raise"          — human offered a stake raise
      "decline"        — human declined bot's raise
      "skip_calculate" — human skipped calculate
      "calculate_win"  — human calculated and won
      "calculate_lose" — human calculated and lost (kwargs: total: int)

    Usage in server.py action routes:
      from bot_runner import notify_human_action
      notify_human_action(engine, "pass", played_cards=engine.played_cards)
    """
    bot_idx = next((i for i, p in enumerate(engine.players) if not p.is_human()), None)
    if bot_idx is None:
        return
    bot = engine.players[bot_idx]
    if not isinstance(bot, OracleBot):
        return

    if action_type == "pass":
        played = kwargs.get("played_cards", [])
        bot.observe_opponent_pass(engine, played)
    elif action_type == "raise":
        bot.observe_opponent_raise(engine)
    elif action_type == "decline":
        bot.observe_opponent_decline(engine)
    elif action_type == "skip_calculate":
        bot.observe_opponent_skip_calculate(engine)
    elif action_type == "calculate_win":
        bot.observe_opponent_calculated_win(engine)
    elif action_type == "calculate_lose":
        total = kwargs.get("total", 0)
        bot.observe_opponent_calculated_lose(engine, total)