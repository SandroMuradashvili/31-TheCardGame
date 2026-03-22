"""
bot.py — Bot implementations
─────────────────────────────
Contains all bot classes. Each subclasses BotPlayer from game_engine.py.

To write a new bot, subclass BotPlayer:

    class MyBot(BotPlayer):
        def choose_play(self, engine): ...
        def choose_cut_or_pass(self, engine): ...
        def choose_calculate(self, engine): ...
        def choose_raise_stake(self, engine): ...

Then register it in BOT_REGISTRY at the bottom of this file.
"""

from game_engine import BotPlayer, GameEngine, Rank, Suit
from itertools import combinations


# ─── SimpleBot ────────────────────────────────────────────────────────────────

class SimpleBot(BotPlayer):
    """
    Conservative bot strategy:
    - Never raises stakes
    - Plays highest non-trump card (preserves trump)
    - Cuts with minimum-value valid combo
    - Calculates only when guaranteed 31+
    """

    DISPLAY_NAME = "Simple (Conservative)"

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
    - Raises stakes when hand looks strong
    - Plays trump early to apply pressure
    - Tries counter-plays more often
    - Calculates at a lower threshold (risk-taking)
    - Cuts with highest-value combo to deny opponent points
    """

    DISPLAY_NAME = "Aggressive (Risk-taker)"

    def _hand_strength(self, engine: GameEngine) -> float:
        """Score hand 0–1 based on trump count and high cards."""
        trump = engine.trump_suit
        score = 0
        for c in self.hand:
            pts = c.points
            if c.suit == trump:
                score += pts * 1.5
            else:
                score += pts
        max_possible = 3 * 11 * 1.5  # 3 trump aces
        return score / max_possible

    def choose_play(self, engine: GameEngine) -> list[str]:
        hand  = self.hand
        trump = engine.trump_suit

        # Always play 3 trumps if possible
        trumps = [c for c in hand if c.suit == trump]
        if len(trumps) == 3:
            return [repr(c) for c in trumps]

        # If we have 2+ trumps, play the lowest trump to drain opponent
        if len(trumps) >= 2:
            cheapest_trump = min(trumps, key=lambda c: c.points)
            return [repr(cheapest_trump)]

        # Try to play 3 same-suit non-trump (maliutka attack)
        non_trump = [c for c in hand if c.suit != trump]
        suits = {}
        for c in non_trump:
            suits.setdefault(c.suit, []).append(c)
        for suit, cards in suits.items():
            if len(cards) == 3:
                # Play maliutka — all 3
                return [repr(c) for c in cards]

        # Play highest non-trump to maximize points
        if non_trump:
            best = max(non_trump, key=lambda c: c.points)
            return [repr(best)]

        # Only trump left — play highest to force opponent to use resources
        best = max(hand, key=lambda c: c.points)
        return [repr(best)]

    def choose_cut_or_pass(self, engine: GameEngine):
        my_idx = engine.players.index(self)
        valid  = engine.get_valid_cuts(my_idx)

        if valid:
            # Cut with HIGHEST value combo — deny opponent the points
            def combo_value(combo):
                return sum(
                    next(c for c in self.hand if repr(c) == cid).points
                    for cid in combo
                )
            best = max(valid, key=combo_value)
            return ("cut", best)

        # Check for counter-play opportunity (3 non-trump same suit)
        if engine.phase != GamePhase.FORCED_CUT:
            trump = engine.trump_suit
            non_trump = [c for c in self.hand if c.suit != trump]
            suits = {}
            for c in non_trump:
                suits.setdefault(c.suit, []).append(c)
            for suit, cards in suits.items():
                if len(cards) >= 3:
                    # Use highest 3 of this suit as counter
                    top3 = sorted(cards, key=lambda c: c.points, reverse=True)[:3]
                    return ("counter", [repr(c) for c in top3])

        # Cannot cut — pass cheapest cards
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
        # Aggressive: calculate if there's a reasonable chance (not guaranteed)
        expected  = known + hc * 6   # assume average hidden card value ~6
        return expected >= 31

    def choose_raise_stake(self, engine: GameEngine) -> bool:
        # Raise if hand is above 60% strength
        return self._hand_strength(engine) >= 0.60


# ─── Bot Registry ─────────────────────────────────────────────────────────────
# Add new bot classes here. The key is the bot's identifier used in API calls.

BOT_REGISTRY: dict[str, type] = {
    "simple":     SimpleBot,
    "aggressive": AggressiveBot,
}

def get_bot(bot_id: str, player_id: str, name: str) -> BotPlayer:
    """Instantiate a bot by registry key."""
    cls = BOT_REGISTRY.get(bot_id, SimpleBot)
    return cls(player_id, name)

def list_bots() -> list[dict]:
    """Return bot list for the frontend dropdown."""
    return [
        {"id": k, "name": getattr(v, "DISPLAY_NAME", k)}
        for k, v in BOT_REGISTRY.items()
    ]