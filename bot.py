"""
bot.py — Bot AI implementations
─────────────────────────────────
Contains all bot classes. Each is a subclass of BotPlayer.

To write a new bot:
    class MyBot(BotPlayer):
        def choose_play(self, engine): ...
        def choose_cut_or_pass(self, engine): ...
        def choose_calculate(self, engine): ...
        def choose_raise_stake(self, engine): ...

Then register it in BOT_REGISTRY at the bottom of this file.
"""

from game_engine import BotPlayer, GameEngine, Rank, Suit, POINT_VALUES


# ─── SimpleBot ────────────────────────────────────────────────────────────────

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
            def combo_value(combo):
                return sum(
                    next(c for c in self.hand if repr(c) == cid).points
                    for cid in combo
                )
            best = min(valid, key=combo_value)
            return ("cut", best)

        n        = len(engine.played_cards)
        cheapest = sorted(self.hand, key=lambda c: c.points)[:n]
        if len(cheapest) < n:
            cheapest = sorted(self.hand, key=lambda c: c.points)
        return ("pass", [repr(c) for c in cheapest])

    def choose_calculate(self, engine: GameEngine) -> bool:
        if len(engine.deck) == 0:
            if any(len(p.hand) < 3 for p in engine.players):
                return True
        known     = self.known_pile_points
        hc        = self.hidden_card_count
        min_total = known + hc * 2
        return min_total >= 31

    def choose_raise_stake(self, engine: GameEngine) -> bool:
        return False


# ─── AggressiveBot ────────────────────────────────────────────────────────────

class AggressiveBot(BotPlayer):
    """
    Aggressive bot strategy:
    - Raises stakes early and often (when holding strong cards)
    - Plays trump cards offensively to force passes
    - Calculates at a lower threshold (takes risks)
    - Cuts with the strongest combo to dominate
    - Plays multiple cards (maliutka attempts) when possible
    """

    def _hand_strength(self, engine: GameEngine) -> float:
        """Score the hand: trump cards and high-point cards are strong."""
        trump = engine.trump_suit
        total = 0
        for c in self.hand:
            val = c.points
            if c.suit == trump:
                val *= 2.0   # trumps are worth double in aggression calc
            total += val
        return total

    def choose_play(self, engine: GameEngine) -> list[str]:
        hand  = self.hand
        trump = engine.trump_suit

        # Always play all 3 trumps for instant win
        trumps = [c for c in hand if c.suit == trump]
        if len(trumps) == 3:
            return [repr(c) for c in trumps]

        # Try to set up a maliutka: 3 non-trump same-suit cards
        from itertools import combinations
        for suit in set(c.suit for c in hand):
            if suit == trump:
                continue
            suit_cards = [c for c in hand if c.suit == suit]
            if len(suit_cards) >= 3:
                # Play highest 3 of this suit to maximise forcing power
                best3 = sorted(suit_cards, key=lambda c: c.points, reverse=True)[:3]
                return [repr(c) for c in best3]

        # Play highest trump to force a difficult cut
        if trumps:
            best_trump = max(trumps, key=lambda c: c.points)
            return [repr(best_trump)]

        # Fallback: play highest card overall
        best = max(hand, key=lambda c: c.points)
        return [repr(best)]

    def choose_cut_or_pass(self, engine: GameEngine):
        my_idx = engine.players.index(self)
        valid  = engine.get_valid_cuts(my_idx)

        if valid:
            # Cut with the strongest combo — dominate, not conserve
            def combo_value(combo):
                return sum(
                    next(c for c in self.hand if repr(c) == cid).points
                    for cid in combo
                )
            best = max(valid, key=combo_value)
            return ("cut", best)

        # Cannot cut — pass cheapest cards
        n        = len(engine.played_cards)
        cheapest = sorted(self.hand, key=lambda c: c.points)[:n]
        if len(cheapest) < n:
            cheapest = sorted(self.hand, key=lambda c: c.points)
        return ("pass", [repr(c) for c in cheapest])

    def choose_calculate(self, engine: GameEngine) -> bool:
        # Must calculate if deck is empty and someone has fewer than 3 cards
        if len(engine.deck) == 0:
            if any(len(p.hand) < 3 for p in engine.players):
                return True

        known     = self.known_pile_points
        hc        = self.hidden_card_count
        # Aggressive: calculate if there's a reasonable chance of winning
        # Uses midpoint estimate rather than minimum
        mid_total = known + hc * 6   # midpoint between J(2) and A(11)
        return mid_total >= 31

    def choose_raise_stake(self, engine: GameEngine) -> bool:
        # Raise if hand is strong and stake is not already high
        if engine.current_stake >= 4:
            return False
        strength = self._hand_strength(engine)
        return strength >= 28   # two high trumps or three high non-trumps


# ─── Bot registry ─────────────────────────────────────────────────────────────
# Maps display name → class. Add new bots here to make them available everywhere.

BOT_REGISTRY: dict[str, type] = {
    "SimpleBot":     SimpleBot,
    "AggressiveBot": AggressiveBot,
}