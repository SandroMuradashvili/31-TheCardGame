"""
bvb_runner.py — Bot vs Bot step driver
───────────────────────────────────────
Drives BOTH bots one action at a time.
Returns after each individual action so the frontend can animate it.

Depends on: game_engine.py, bot.py
Imported by: server.py
"""

from game_engine import GameEngine, GamePhase


def bvb_step(engine: GameEngine) -> str:
    """
    Execute exactly ONE action by whichever bot is active right now.
    Returns a string describing what happened, e.g. 'play', 'cut', 'pass',
    'calculate', 'skip_calculate', 'start_round', 'done'.
    Returns 'done' if no action was possible (terminal phase).
    """
    phase = engine.phase

    if phase in (GamePhase.GAME_OVER,):
        return 'done'

    if phase == GamePhase.WAITING:
        return 'done'

    if phase == GamePhase.ROUND_OVER:
        engine.start_round()
        return 'start_round'

    # Handle pending stake offer first
    for idx, bot in enumerate(engine.players):
        if (engine.stake_offerer_idx is not None
                and engine.stake_offerer_idx != idx):
            if engine.pending_stake <= 3:
                engine.accept_stake(idx)
                return 'accept_stake'
            else:
                engine.decline_stake(idx)
                return 'decline_stake'

    if phase == GamePhase.STAKES:
        active = engine.active_idx
        bot    = engine.players[active]
        engine.start_play()
        return 'start_play'

    if phase == GamePhase.PLAYING:
        active = engine.active_idx
        bot    = engine.players[active]
        engine.play_cards(active, bot.choose_play(engine))
        return 'play'

    if phase in (GamePhase.CUTTING, GamePhase.FORCED_CUT):
        cutter_idx = 1 - engine.playing_player_idx
        bot        = engine.players[cutter_idx]
        action, card_ids = bot.choose_cut_or_pass(engine)
        if action == 'cut':
            engine.cut_cards(cutter_idx, card_ids)
            return 'cut'
        else:
            engine.pass_cards(cutter_idx, card_ids)
            return 'pass'

    if phase == GamePhase.CALCULATING:
        calc_idx = engine.calculator_idx
        bot      = engine.players[calc_idx]
        if bot.choose_calculate(engine):
            engine.calculate(calc_idx)
            return 'calculate'
        else:
            engine.skip_calculate(calc_idx)
            return 'skip_calculate'

    return 'done'


def bvb_run_full(engine: GameEngine, max_steps: int = 2000) -> list[dict]:
    """
    Run an entire game headlessly, collecting state after every step.
    Used for bulk simulations — no animations, just data.
    Returns list of (action, state_snapshot) dicts.
    """
    history = []
    for _ in range(max_steps):
        if engine.phase == GamePhase.GAME_OVER:
            break
        if engine.phase == GamePhase.ROUND_OVER:
            engine.start_round()
            history.append({'action': 'start_round',
                            'scores': [p.game_score for p in engine.players]})
            continue
        if engine.phase == GamePhase.WAITING:
            break
        action = bvb_step(engine)
        history.append({'action': action,
                        'phase':  engine.phase.value,
                        'scores': [p.game_score for p in engine.players]})
        if action == 'done':
            break
    return history