"""
Card Probability Tracker for BURA

Maintains a probability distribution over unseen cards.
Updates based on:
  - Cards played (now visible)
  - Passes (opponent couldn't/wouldn't cut → informs hand)
  - Cuts (opponent revealed cards)
"""

from game_engine import Card, SUITS, RANKS, RANK_ORDER, make_deck, _is_valid_cut
from typing import Optional


class CardTracker:
    """
    From the perspective of ONE player (the bot).
    Tracks probability that each unseen card is in opponent's hand vs deck.
    """

    def __init__(self, my_hand: list[Card], trump_suit: str, deck_size: int):
        self.trump_suit = trump_suit
        all_cards = make_deck()

        # Cards I know: my own hand
        self.my_hand = set(my_hand)

        # All cards not in my hand are "unseen" — either in deck or opponent's hand
        self.unseen: set[Card] = set(all_cards) - self.my_hand

        # We know deck_size and opponent has 3 cards
        # Initially: probability card is in opponent hand = 3 / len(unseen)
        self.deck_size = deck_size
        self.opp_hand_size = 3

        # Belief: for each unseen card, P(in opponent hand)
        # Start uniform
        self._beliefs: dict[Card, float] = {}
        self._update_uniform_beliefs()

        # Inference log for display
        self.inference_log: list[str] = []

    def _update_uniform_beliefs(self):
        """Redistribute probability uniformly among unseen cards."""
        n = len(self.unseen)
        if n == 0:
            return
        p_opp = min(self.opp_hand_size / n, 1.0)
        for c in self.unseen:
            self._beliefs[c] = p_opp

    def prob_opponent_has(self, card: Card) -> float:
        """P(opponent holds this card)."""
        if card in self.my_hand:
            return 0.0
        if card not in self.unseen:
            return 0.0
        return self._beliefs.get(card, 0.0)

    def card_seen(self, card: Card):
        """Mark card as now seen/known (played, cut, etc.)"""
        if card in self.unseen:
            self.unseen.discard(card)
            self._beliefs.pop(card, None)
            self._renormalize()

    def opponent_played(self, cards: list[Card]):
        """Opponent revealed these cards by playing/cutting."""
        for c in cards:
            self.opp_hand_size = max(0, self.opp_hand_size - 1)
            self.card_seen(c)
        self.inference_log.append(f"Opponent played {cards} → removed from unseen")

    def i_played(self, cards: list[Card]):
        """I played these cards."""
        for c in cards:
            self.my_hand.discard(c)
        self.inference_log.append(f"I played {cards}")

    def opponent_drew(self, n: int):
        """Opponent drew n cards from deck."""
        self.opp_hand_size = min(3, self.opp_hand_size + n)
        self.deck_size = max(0, self.deck_size - n)
        self._renormalize()

    def i_drew(self, new_cards: list[Card]):
        """I drew these cards."""
        for c in new_cards:
            self.my_hand.add(c)
            self.unseen.discard(c)
            self._beliefs.pop(c, None)
        self._renormalize()

    def opponent_passed(self, n: int, played_cards: list[Card]):
        """
        Opponent passed n cards instead of cutting.
        This means they COULDN'T or CHOSE NOT to cut.

        We can infer: they likely don't have cards that could beat played_cards.
        This is a soft inference (they might have chosen strategically).
        """
        played_suit = played_cards[0].suit
        played_max_rank = max(c.rank_value for c in played_cards)
        played_are_trump = (played_suit == self.trump_suit)

        # Reduce probability that opponent holds trump cards (if play was non-trump)
        # and same-suit higher cards
        downweight_factor = 0.25  # soft — they might be hiding

        for card in list(self.unseen):
            if card not in self._beliefs:
                continue
            could_cut = False
            if played_are_trump:
                if card.suit == self.trump_suit and card.rank_value > played_max_rank:
                    could_cut = True
            else:
                if card.suit == self.trump_suit:
                    could_cut = True
                elif card.suit == played_suit and card.rank_value > played_max_rank:
                    could_cut = True

            if could_cut:
                self._beliefs[card] *= downweight_factor

        self._renormalize()
        self.inference_log.append(
            f"Opponent passed {n} cards vs {played_cards} → downweighted cutting cards"
        )

    def _renormalize(self):
        """Ensure sum of beliefs equals opp_hand_size (expected cards in hand)."""
        total = sum(self._beliefs.values())
        if total == 0:
            self._update_uniform_beliefs()
            return
        scale = min(self.opp_hand_size, len(self.unseen)) / total if total > 0 else 1
        for c in self._beliefs:
            self._beliefs[c] = min(self._beliefs[c] * scale, 1.0)

    # ─────────────────────────────────────────
    #  Key queries for the bot
    # ─────────────────────────────────────────

    def prob_opponent_can_cut(self, play: list[Card]) -> float:
        """
        P(opponent can cut this play) — considering all combinations of
        n cards they might hold that form a valid cut.
        """
        n = len(play)
        if self.opp_hand_size < n:
            return 0.0

        # Sample-based estimate over possible opponent hands
        unseen_list = sorted(self.unseen, key=lambda c: str(c))
        weights = [self._beliefs.get(c, 0.0) for c in unseen_list]
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0

        # For efficiency, check each individual card combination
        from itertools import combinations
        cut_weight = 0.0
        total_combo_weight = 0.0

        for combo in combinations(range(len(unseen_list)), n):
            cards = [unseen_list[i] for i in combo]
            # Joint probability (approximate, ignoring correlation)
            combo_prob = 1.0
            for i in combo:
                combo_prob *= weights[i]

            total_combo_weight += combo_prob
            if _is_valid_cut(cards, play, self.trump_suit):
                cut_weight += combo_prob

        if total_combo_weight == 0:
            return 0.0
        return min(cut_weight / total_combo_weight, 1.0)

    def expected_opponent_score(self) -> float:
        """Rough estimate of points opponent has accumulated."""
        # We can't know for sure, but we track what they've taken
        # This is maintained externally — return 0 as placeholder
        return 0.0

    def prob_opponent_has_trump(self) -> float:
        """P(opponent holds at least one trump card)."""
        trump_cards = [c for c in self.unseen if c.suit == self.trump_suit]
        if not trump_cards:
            return 0.0
        p_none = 1.0
        for c in trump_cards:
            p_none *= (1.0 - self._beliefs.get(c, 0.0))
        return 1.0 - p_none

    def strongest_safe_plays(self, hand: list[Card], safety_threshold: float = 0.35) -> list[list[Card]]:
        """
        Return plays ordered by expected value, filtering out too-risky ones.
        A play is 'safe' if P(opponent can cut) < safety_threshold.
        """
        from itertools import combinations

        by_suit: dict[str, list[Card]] = {}
        for c in hand:
            by_suit.setdefault(c.suit, []).append(c)

        candidates = []
        for suit, cards in by_suit.items():
            for size in range(1, min(len(cards), 3) + 1):
                for combo in combinations(cards, size):
                    play = list(combo)
                    p_cut = self.prob_opponent_can_cut(play)
                    pts = sum(c.points for c in play)
                    # Score: high points + low cut probability
                    ev = pts * (1 - p_cut) - pts * p_cut * 0.5
                    candidates.append((play, p_cut, ev))

        candidates.sort(key=lambda x: -x[2])
        return [(play, p_cut, ev) for play, p_cut, ev in candidates]

    def get_belief_summary(self) -> str:
        """Human-readable summary of what we think opponent holds."""
        top = sorted(self.unseen, key=lambda c: -self._beliefs.get(c, 0))[:5]
        lines = ["Top cards likely in opponent hand:"]
        for c in top:
            lines.append(f"  {c}: {self._beliefs.get(c, 0):.1%}")
        return "\n".join(lines)