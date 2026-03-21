"""
bot.py — SimpleBot AI
─────────────────────
Subclass of BotPlayer. Contains all bot decision-making logic.

Depends on: game_engine.py (BotPlayer, GameEngine, Rank)

To write a smarter bot, subclass BotPlayer in game_engine.py:

    class MyBot(BotPlayer):
        def choose_play(self, engine): ...
        def choose_cut_or_pass(self, engine): ...
        def choose_calculate(self, engine): ...
        def choose_raise_stake(self, engine): ...

Then swap it in room_manager.py where SimpleBot is instantiated.
"""

from game_engine import BotPlayer, GameEngine


class SimpleBot(BotPlayer):
    """
    Conservative bot strategy:
    - Never raises stakes
    - Plays highest non-trump card (preserves trump)
    - Cuts with minimum-value valid combo
    - Calculates only when guaranteed 31+
    """

    def choose_play(self, engine: GameEngine) -> list[str]:
        hand  = self.hand
        trump = engine.trump_suit

        # Play all 3 trumps for instant round win
        trumps = [c for c in hand if c.suit == trump]
        if len(trumps) == 3:
            return [repr(c) for c in trumps]

        # Prefer single highest non-trump — keep trump in reserve
        non_trump = [c for c in hand if c.suit != trump]
        if non_trump:
            best = max(non_trump, key=lambda c: c.points)
            return [repr(best)]

        # Only trumps remain — play the cheapest one
        best = min(hand, key=lambda c: c.points)
        return [repr(best)]

    def choose_cut_or_pass(self, engine: GameEngine):
        my_idx = engine.players.index(self)
        valid  = engine.get_valid_cuts(my_idx)

        if valid:
            # Cut with cheapest valid combo to keep strong cards in hand
            def combo_value(combo):
                return sum(
                    next(c for c in self.hand if repr(c) == cid).points
                    for cid in combo
                )
            best = min(valid, key=combo_value)
            return ("cut", best)

        # Cannot cut — pass the n cheapest cards.
        # If the bot has fewer cards than needed (can happen when deck ran low),
        # pass however many we have — engine will error, but at least we don't loop.
        n        = len(engine.played_cards)
        cheapest = sorted(self.hand, key=lambda c: c.points)[:n]
        if len(cheapest) < n:
            # Not enough cards to pass — should not happen in a normal game,
            # but pass what we have to avoid infinite loop in bot_runner
            cheapest = sorted(self.hand, key=lambda c: c.points)
        return ("pass", [repr(c) for c in cheapest])

    def choose_calculate(self, engine: GameEngine) -> bool:
        # Only calculate when the true total is already a guaranteed win
        return self.pile_points >= 31

    def choose_raise_stake(self, engine: GameEngine) -> bool:
        return False