"""
BURA Bot — Strong AI player

Decision framework:
1. What to play  → EV over cut probability
2. Cut or pass   → EV based on hand strength + score state
3. Calculate     → Score threshold + risk assessment
4. Stakes        → Game-theoretic bluff/fold logic
"""

from game_engine import (
    Card, GameState, Phase, Move,
    MovePlay, MoveCut, MovePass, MoveCalculate, MoveContinue,
    MoveRaise, MoveDeclineRaise, MoveAcceptRaise,
    legal_moves, score_pile, is_maliutka, is_three_trumps,
    RANK_ORDER, _is_valid_cut
)
from card_tracker import CardTracker
from itertools import combinations
from typing import Optional
import random


class BotReasoning:
    """Stores the bot's reasoning for display."""
    def __init__(self):
        self.move_chosen: Optional[Move] = None
        self.explanation: str = ""
        self.probabilities: dict[str, float] = {}
        self.ev_scores: list[tuple] = []  # (play, ev, p_cut)
        self.stake_reasoning: str = ""

    def __str__(self):
        lines = [f"Decision: {self.move_chosen}"]
        lines.append(f"Reason: {self.explanation}")
        if self.probabilities:
            for k, v in self.probabilities.items():
                lines.append(f"  {k}: {v:.1%}")
        return "\n".join(lines)


class BuraBot:
    """
    Strong Bura bot using:
    - Probability tracking of opponent's hand
    - Expected value calculations for all decisions
    - Game-theoretic stake reasoning
    """

    def __init__(self, player_id: int):
        self.player_id = player_id
        self.tracker: Optional[CardTracker] = None
        self.opponent_score_pile_estimate: int = 0  # track what we think they have
        self.my_known_score: int = 0
        self.reasoning = BotReasoning()

    # ─────────────────────────────────────────
    #  Main entry: choose move
    # ─────────────────────────────────────────

    def choose_move(self, state: GameState) -> tuple[Move, BotReasoning]:
        self.reasoning = BotReasoning()
        p = self.player_id

        # Initialize tracker if needed
        if self.tracker is None:
            deck_size = len(state.deck)
            self.tracker = CardTracker(
                my_hand=state.hands[p][:],
                trump_suit=state.trump_suit,
                deck_size=deck_size
            )

        phase = state.phase

        if phase == Phase.STAKE_OFFER:
            move = self._decide_stake(state)
        elif phase == Phase.PLAY:
            move = self._decide_play(state)
        elif phase == Phase.CUT_OR_PASS:
            move = self._decide_cut_or_pass(state)
        elif phase == Phase.CALCULATE:
            move = self._decide_calculate(state)
        else:
            move = MoveContinue()

        self.reasoning.move_chosen = move
        return move, self.reasoning

    # ─────────────────────────────────────────
    #  Phase 1: Stakes decision
    # ─────────────────────────────────────────

    def _decide_stake(self, state: GameState) -> Move:
        p = self.player_id
        hand = state.hands[p]
        trump = state.trump_suit
        my_score = state.pile_score(p)
        current_stake = state.current_stake

        # Evaluate hand strength
        hand_strength = self._hand_strength(hand, trump, my_score)
        win_probability = self._estimate_win_probability(state)

        self.reasoning.probabilities["win_probability"] = win_probability
        self.reasoning.probabilities["hand_strength"] = hand_strength

        # If someone raised at us (we need to respond)
        if state.stake_last_raised_by == p:
            # This shouldn't happen — if we raised, we wait
            return MoveContinue()

        if state.stake_last_raised_by is not None and state.stake_last_raised_by != p:
            # Opponent raised — decide to accept or decline
            return self._respond_to_raise(state, win_probability, current_stake)

        # Our turn to optionally raise
        if current_stake < 6 and win_probability > 0.72:
            # We're confident — raise
            new_stake = min(current_stake + 1, 6)
            self.reasoning.explanation = (
                f"Raising to {new_stake} — win probability {win_probability:.0%}, "
                f"hand strength {hand_strength:.0%}"
            )
            self.reasoning.stake_reasoning = f"Strong hand (wp={win_probability:.0%}), raising stakes"
            return MoveRaise(new_stake)

        # Bluff raise with moderate hand at low stakes
        if current_stake == 1 and win_probability > 0.55 and random.random() < 0.3:
            self.reasoning.explanation = "Bluff raise with moderate hand"
            self.reasoning.stake_reasoning = "Bluff raise"
            return MoveRaise(2)

        self.reasoning.explanation = f"No raise — win probability {win_probability:.0%}"
        return MoveContinue()

    def _respond_to_raise(self, state: GameState, win_prob: float, stake: int) -> Move:
        """Decide whether to accept or decline an opponent raise."""
        # EV of accepting: win_prob * stake - (1-win_prob) * stake = (2*win_prob-1)*stake
        # EV of declining: lose current stake (stake - 1, since opponent offered stake)
        prev_stake = stake - 1  # what we'd lose by declining

        ev_accept = (2 * win_prob - 1) * stake
        ev_decline = -prev_stake  # we give them the points

        if ev_accept > ev_decline:
            self.reasoning.stake_reasoning = (
                f"Accepting raise to {stake} — EV accept={ev_accept:.2f} > EV decline={ev_decline:.2f}"
            )
            self.reasoning.explanation = f"Accept raise (EV: {ev_accept:.2f} vs {ev_decline:.2f})"
            return MoveAcceptRaise()
        else:
            self.reasoning.stake_reasoning = (
                f"Declining raise to {stake} — bad EV (wp={win_prob:.0%})"
            )
            self.reasoning.explanation = f"Decline raise — too risky (win prob only {win_prob:.0%})"
            return MoveDeclineRaise()

    # ─────────────────────────────────────────
    #  Phase 2: What to play
    # ─────────────────────────────────────────

    def _decide_play(self, state: GameState) -> Move:
        p = self.player_id
        hand = state.hands[p]
        trump = state.trump_suit
        my_score = state.pile_score(p)

        # Always check 3 trumps first
        trump_cards = [c for c in hand if c.suit == trump]
        if len(trump_cards) == 3:
            self.reasoning.explanation = "Playing 3 trumps — AUTO WIN!"
            return MovePlay(trump_cards)

        # Check Maliutka opportunity
        by_suit: dict[str, list[Card]] = {}
        for c in hand:
            by_suit.setdefault(c.suit, []).append(c)

        for suit, cards in by_suit.items():
            if len(cards) == 3 and suit != trump:
                p_cut = self.tracker.prob_opponent_can_cut(cards)
                if p_cut < 0.3:
                    self.reasoning.explanation = (
                        f"Playing Maliutka ({suit}) — P(opponent cuts)={p_cut:.0%}"
                    )
                    return MovePlay(cards)

        # Get all candidate plays with EV
        candidates = self.tracker.strongest_safe_plays(hand)
        self.reasoning.ev_scores = candidates[:5]

        if not candidates:
            # Fallback: play lowest value card
            lowest = min(hand, key=lambda c: c.points)
            self.reasoning.explanation = "No good plays — playing lowest card"
            return MovePlay([lowest])

        # Strategic logic based on game state
        cards_needed = max(0, 31 - my_score)
        deck_remaining = len(state.deck)

        best_play, best_p_cut, best_ev = candidates[0]

        # If we're close to 31, maximize point collection
        if my_score >= 20:
            # Try to collect high-point cards
            high_point_plays = sorted(candidates, key=lambda x: -sum(c.points for c in x[0]))
            for play, p_cut, ev in high_point_plays:
                if p_cut < 0.5:  # acceptable risk when we need points
                    self.reasoning.explanation = (
                        f"Need points ({my_score}/31) — playing {play} "
                        f"({sum(c.points for c in play)}pts, cut risk {p_cut:.0%})"
                    )
                    return MovePlay(play)

        # Early game: play conservatively
        if my_score < 15:
            safe_plays = [(p, pc, ev) for p, pc, ev in candidates if pc < 0.25]
            if safe_plays:
                play, p_cut, ev = safe_plays[0]
                self.reasoning.explanation = (
                    f"Early game — safe play {play} (cut risk {p_cut:.0%})"
                )
                return MovePlay(play)

        # Default: best EV play
        self.reasoning.explanation = (
            f"Best EV play: {best_play} (EV={best_ev:.1f}, cut risk={best_p_cut:.0%})"
        )
        return MovePlay(best_play)

    # ─────────────────────────────────────────
    #  Phase 3: Cut or pass
    # ─────────────────────────────────────────

    def _decide_cut_or_pass(self, state: GameState) -> Move:
        p = self.player_id
        hand = state.hands[p]
        played = state.played_cards
        trump = state.trump_suit
        n = len(played)

        # Find all valid cuts
        valid_cuts = []
        for combo in combinations(hand, n):
            combo = list(combo)
            if _is_valid_cut(combo, played, trump):
                valid_cuts.append(combo)

        if not valid_cuts:
            # Must pass — choose cards to pass wisely
            return self._choose_pass(state, n)

        # We have valid cuts — decide whether to cut
        played_points = sum(c.points for c in played)
        my_score = state.pile_score(p)
        opp_score = state.pile_score(state.opponent(p))

        # Score we'd gain by cutting
        for cut in valid_cuts:
            cut_points = sum(c.points for c in cut)
            total_gained = played_points + cut_points

        # EV of cutting: gain points + take calculate right
        # EV of passing: give points to opponent, stay anonymous

        # Key factors:
        # 1. How many points we gain
        # 2. Whether we want calculate right
        # 3. What we reveal about our hand

        # Prefer cutting if:
        # - We'd gain significant points
        # - We're close to being able to calculate (score + gained >= 31)
        # - Opponent is likely close to calculating

        best_cut = max(valid_cuts, key=lambda cut: self._cut_ev(cut, played, state))
        cut_ev = self._cut_ev(best_cut, played, state)
        pass_ev = self._pass_ev(state, n)

        self.reasoning.probabilities["cut_ev"] = cut_ev
        self.reasoning.probabilities["pass_ev"] = pass_ev

        if cut_ev > pass_ev:
            total_gained = sum(c.points for c in played) + sum(c.points for c in best_cut)
            new_score = my_score + total_gained
            self.reasoning.explanation = (
                f"Cutting — gain {total_gained}pts → score {new_score} "
                f"(EV cut={cut_ev:.1f} > pass={pass_ev:.1f})"
            )
            # Update tracker
            self.tracker.opponent_played(played)
            return MoveCut(best_cut)
        else:
            self.reasoning.explanation = (
                f"Passing — revealing cards not worth it "
                f"(EV cut={cut_ev:.1f} < pass={pass_ev:.1f})"
            )
            return self._choose_pass(state, n)

    def _cut_ev(self, cut: list[Card], played: list[Card], state: GameState) -> float:
        """Expected value of cutting with these cards."""
        p = self.player_id
        my_score = state.pile_score(p)
        points_gained = sum(c.points for c in played) + sum(c.points for c in cut)
        new_score = my_score + points_gained

        # Value of points
        ev = points_gained * 1.0

        # Bonus if this gets us to calculate range
        if new_score >= 31:
            ev += 15  # high bonus for winning position
        elif new_score >= 25:
            ev += 8   # close to winning

        # Penalty for revealing cards (opponent learns our trump/suit)
        if any(c.suit == state.trump_suit for c in cut):
            ev -= 3  # reveal trump cards

        return ev

    def _pass_ev(self, state: GameState, n: int) -> float:
        """Expected value of passing."""
        # Opponent gains points, but we stay anonymous
        played = state.played_cards
        played_points = sum(c.points for c in played)

        # EV = -points_given + anonymity_value
        anonymity_value = 4  # base value of staying hidden
        ev = -played_points * 0.8 + anonymity_value

        # Higher anonymity value if we have strong trump cards to hide
        p = self.player_id
        hand = state.hands[p]
        trump_in_hand = sum(1 for c in hand if c.suit == state.trump_suit)
        ev += trump_in_hand * 2

        return ev

    def _choose_pass(self, state: GameState, n: int) -> MovePass:
        """Choose which n cards to pass — give away lowest value."""
        p = self.player_id
        hand = state.hands[p]
        trump = state.trump_suit

        # Sort by strategic value (low = expendable)
        def card_value(c: Card) -> float:
            v = c.points
            if c.suit == trump:
                v += 20  # keep trump
            # Keep high cards in suits we have multiple of
            suit_count = sum(1 for x in hand if x.suit == c.suit)
            if suit_count >= 2:
                v += 5
            return v

        sorted_hand = sorted(hand, key=card_value)
        pass_cards = sorted_hand[:n]

        self.reasoning.explanation = (
            f"Passing {pass_cards} — keeping valuable cards"
        )
        return MovePass(pass_cards)

    # ─────────────────────────────────────────
    #  Phase 4: Calculate decision
    # ─────────────────────────────────────────

    def _decide_calculate(self, state: GameState) -> Move:
        p = self.player_id
        my_score = state.pile_score(p)
        opp_score = state.pile_score(state.opponent(p))
        deck_remaining = len(state.deck)
        hand = state.hands[p]
        trump = state.trump_suit

        self.reasoning.probabilities["my_score"] = my_score
        self.reasoning.probabilities["opp_score_visible"] = opp_score

        if my_score < 31:
            self.reasoning.explanation = f"Can't calculate — only {my_score}/31 points"
            return MoveContinue()

        # We have 31+ — should we calculate now or wait for more?
        # Risk of waiting: opponent might get calculate right and win
        # Benefit of waiting: if stakes might be raised

        # Check if opponent could also be close to calculating
        opp_threat = opp_score >= 20  # they could have hidden points we don't see

        if my_score >= 31:
            if deck_remaining <= 4 or opp_threat:
                self.reasoning.explanation = (
                    f"Calculating now — {my_score} points ≥ 31! "
                    f"({'deck nearly empty' if deck_remaining <= 4 else 'opponent is a threat'})"
                )
                return MoveCalculate()

            # Have comfortable buffer — calculate unless we're well above 31
            if my_score >= 38:
                self.reasoning.explanation = f"Calculating — strong {my_score} points, well above 31"
                return MoveCalculate()

            # Marginal: calculate unless we can get even more secure
            if my_score >= 31:
                self.reasoning.explanation = f"Calculating — {my_score} ≥ 31, why risk it?"
                return MoveCalculate()

        self.reasoning.explanation = f"Not enough points ({my_score}/31) — continue"
        return MoveContinue()

    # ─────────────────────────────────────────
    #  Hand strength evaluation
    # ─────────────────────────────────────────

    def _hand_strength(self, hand: list[Card], trump: str, my_score: int) -> float:
        """
        0.0 to 1.0 — how strong is our current position.
        """
        # Point value of hand
        hand_pts = sum(c.points for c in hand)
        trump_count = sum(1 for c in hand if c.suit == trump)
        high_cards = sum(1 for c in hand if c.rank_value >= 4)  # A or T

        # Max possible: 3 aces = 33pts
        pt_score = min(hand_pts / 30, 1.0) * 0.4
        trump_score = (trump_count / 3) * 0.3
        high_score = (high_cards / 3) * 0.2
        progress_score = min((my_score + hand_pts) / 40, 1.0) * 0.1

        return pt_score + trump_score + high_score + progress_score

    def _estimate_win_probability(self, state: GameState) -> float:
        """
        Estimate probability of winning this round.
        Combines: current score, hand strength, deck state.
        """
        p = self.player_id
        hand = state.hands[p]
        trump = state.trump_suit
        my_score = state.pile_score(p)
        opp_visible = state.pile_score(state.opponent(p))

        hand_pts = sum(c.points for c in hand)
        my_potential = my_score + hand_pts

        # Can we win outright?
        if my_potential >= 31:
            # High but not certain — depends on opponent
            base = 0.75
        elif my_potential >= 25:
            base = 0.55
        elif my_potential >= 20:
            base = 0.40
        else:
            base = 0.25

        # Trump cards boost
        trump_count = sum(1 for c in hand if c.suit == trump)
        trump_bonus = trump_count * 0.07

        # 3 trumps = auto win
        if trump_count == 3:
            return 0.99

        # Maliutka potential
        by_suit: dict[str, list[Card]] = {}
        for c in hand:
            by_suit.setdefault(c.suit, []).append(c)
        has_maliutka = any(
            len(cards) == 3 and suit != trump
            for suit, cards in by_suit.items()
        )
        maliutka_bonus = 0.08 if has_maliutka else 0

        # Opponent score concern
        opp_penalty = min(opp_visible / 60, 0.2)

        wp = base + trump_bonus + maliutka_bonus - opp_penalty
        return max(0.05, min(0.97, wp))

    def notify_draw(self, new_cards: list[Card], opp_drew: int):
        """Called after drawing to update tracker."""
        if self.tracker:
            self.tracker.i_drew(new_cards)
            self.tracker.opponent_drew(opp_drew)

    def reset_for_new_round(self, hand: list[Card], trump_suit: str, deck_size: int):
        """Reset tracker for a new round."""
        self.tracker = CardTracker(
            my_hand=hand,
            trump_suit=trump_suit,
            deck_size=deck_size
        )
        self.opponent_score_pile_estimate = 0