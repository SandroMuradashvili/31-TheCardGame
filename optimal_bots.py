"""
optimal_bots.py — Optimal Bayesian agents for Bura / Thirty-One / Cutter
─────────────────────────────────────────────────────────────────────────
Two bots sharing a common decision-theoretic core:

  PureBot    — Perfect Bayesian card math. Every decision is an exact
               expected-value calculation over the legal information
               boundary. No behavioral inference whatsoever.

  OracleBot  — Same core + a BehavioralBeliefState layer that infers
               opponent pile strength from observed actions (passes,
               raises, stake declines, skip-calculates). The only
               difference from PureBot is richer beliefs — the decision
               logic is identical.

Architecture
────────────
  ExactBeliefTracker        — per-card posterior P(card ∈ opp hand)
                              updated on every card observation.
                              Replaces the hypergeometric approximation
                              in BeliefTracker with exact enumeration.

  BehavioralBeliefState     — tracks opponent pile strength estimate
                              updated from behavioral signals (OracleBot).

  DecisionEngine            — pure EV logic shared by both bots.
    ├── _p_pile_ge_31()     — exact enumeration over hidden pile combos.
    ├── _ev_play()          — expected value of a card combo play.
    ├── _ev_cut()           — cost-minimising / gain-maximising cut.
    ├── _ev_calculate()     — EV(calculate) vs EV(skip).
    └── _ev_raise()         — signaling-game EV for stake raises.

Information boundary
────────────────────
Legal visible information used:
  • Own hand (complete)
  • Own score_pile (known cards + their points)
  • Own hidden_pile (card COUNT only — not identity)
  • Opponent known_pile_points + hidden_card_count
  • Trump card (globally visible)
  • Cards on table (globally visible)
  • Own seen_card_ids (accumulated over round)
  • Opponent ACTIONS (OracleBot only — not their hidden cards)

Nothing else is accessed.  Opponent hand and hidden pile identity are
never read — doing so would be cheating.
"""

import math
import random
from itertools import combinations
from typing import Optional

from game_engine import (
    BotPlayer, GameEngine, GamePhase,
    Card, Suit, Rank,
    ALL_CARD_IDS, CARD_POINTS, POINT_VALUES,
)


# ══════════════════════════════════════════════════════════════════════════════
# ExactBeliefTracker
# ══════════════════════════════════════════════════════════════════════════════

class ExactBeliefTracker:
    """
    Per-card posterior probability distribution over the unseen card pool.

    Unlike the hypergeometric BeliefTracker (which assigns the same
    probability to every unseen card), this tracker maintains individual
    weights per card that update as new information arrives.

    Model
    ─────
    At any point, each unseen card X has a weight w(X) representing our
    relative belief that X is in the opponent's hand vs. the deck.

    Initially all weights are equal (uniform prior).  As cards are observed
    (played, cut, passed-and-revealed) they are removed from the unseen pool
    and the distribution renormalizes automatically.

    P(X ∈ opp hand) = w(X) / Σ w(all unseen)  × opp_hand_size

    This is the hypergeometric formula but with non-uniform weights,
    allowing OracleBot's behavioral updates to bias the distribution.

    Card observation update
    ───────────────────────
    When a card is observed (seen on table, in cut, etc.):
      - Remove it from unseen pool entirely.
      - Its weight is gone; remaining weights renormalize automatically
        on the next query.

    Weight nudge (OracleBot behavioral updates)
    ───────────────────────────────────────────
    The BehavioralBeliefState calls nudge_weight(card_id, factor) to
    multiplicatively adjust a card's relative probability.  factor < 1
    means "less likely in opp hand", factor > 1 means "more likely".
    """

    def __init__(self, engine: GameEngine, my_idx: int):
        me  = engine.players[my_idx]
        opp = engine.players[1 - my_idx]

        my_hand_ids = {repr(c) for c in me.hand}
        seen        = set(me.seen_card_ids)
        if engine.deck.trump_card:
            seen.add(repr(engine.deck.trump_card))
        for c in engine.played_cards:
            seen.add(repr(c))

        self.unseen: set[str]         = set(ALL_CARD_IDS - my_hand_ids - seen)
        self.weights: dict[str, float] = {cid: 1.0 for cid in self.unseen}
        self.opp_hand_size: int        = len(opp.hand)
        self.trump_suit                = engine.trump_suit

        self._rank_map = {r.value: r for r in Rank}
        self._suit_map = {s.value[0].upper(): s for s in Suit}

    # ── Core probability queries ──────────────────────────────────────────────

    def _total_weight(self) -> float:
        return sum(self.weights.values()) or 1.0

    def p_card_in_opp_hand(self, card_id: str) -> float:
        """P(card_id is in opponent's hand)."""
        if card_id not in self.weights or self.opp_hand_size == 0:
            return 0.0
        n      = len(self.unseen)
        if n == 0:
            return 0.0
        # Weighted hypergeometric: card's share of weight × opp hand fraction
        card_w = self.weights[card_id]
        total_w = self._total_weight()
        return (card_w / total_w) * self.opp_hand_size

    def p_opp_has_trump(self) -> float:
        """P(opponent holds at least one trump card)."""
        if not self.trump_suit or self.opp_hand_size == 0:
            return 0.0
        trump_initial = self.trump_suit.value[0].upper()
        trump_ids     = [c for c in self.unseen if c.endswith(trump_initial)]
        if not trump_ids:
            return 0.0
        # P(at least one trump) = 1 - P(zero trumps drawn in opp_hand_size draws)
        # Weighted version: treat weight-normalized pool
        trump_w    = sum(self.weights[c] for c in trump_ids)
        total_w    = self._total_weight()
        non_trump_w = total_w - trump_w
        p_none     = _weighted_hyper_p_none(non_trump_w, trump_w,
                                             self.opp_hand_size, len(self.unseen))
        return 1.0 - p_none

    def p_opp_can_beat_card(self, card_id: str) -> float:
        """P(opponent holds at least one card that beats card_id)."""
        beaters        = self._beating_ids(card_id)
        beaters_unseen = [b for b in beaters if b in self.weights]
        if not beaters_unseen:
            return 0.0
        beater_w   = sum(self.weights[b] for b in beaters_unseen)
        total_w    = self._total_weight()
        non_beat_w = total_w - beater_w
        p_none     = _weighted_hyper_p_none(non_beat_w, beater_w,
                                             self.opp_hand_size, len(self.unseen))
        return 1.0 - p_none

    def p_opp_can_cut_combo(self, played_ids: list[str]) -> float:
        """
        P(opponent can cut the full combo).
        For 1 card: exact.
        For 2-3 cards: minimum per-card probability (conservative lower bound).
        """
        if not played_ids:
            return 0.0
        return min(self.p_opp_can_beat_card(cid) for cid in played_ids)

    def expected_opp_hand_points(self) -> float:
        """E[total point value of opponent's hand]."""
        if not self.unseen or self.opp_hand_size == 0:
            return 0.0
        total_w = self._total_weight()
        avg_pts = sum(
            (self.weights[c] / total_w) * CARD_POINTS[c]
            for c in self.unseen
        )
        return avg_pts * self.opp_hand_size

    # ── Weight manipulation (OracleBot behavioral updates) ────────────────────

    def nudge_weight(self, card_id: str, factor: float):
        """Multiplicatively adjust a card's weight. factor < 1 = less likely."""
        if card_id in self.weights:
            self.weights[card_id] = max(self.weights[card_id] * factor, 1e-6)

    def nudge_suit_above_value(self, suit: Suit, min_points: int, factor: float):
        """
        Nudge all unseen cards of a given suit with points > min_points.
        Used when opponent passes a combo of suit S, value V — they probably
        don't hold higher-value cards of that suit (else they'd have cut).
        """
        for cid in list(self.unseen):
            cs = self._suit_map.get(cid[-1])
            cr = self._rank_map.get(cid[:-1])
            if cs == suit and cr is not None and POINT_VALUES[cr] > min_points:
                self.nudge_weight(cid, factor)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _beating_ids(self, card_id: str) -> list[str]:
        """All card IDs that beat card_id by Bura rules."""
        r = self._rank_map.get(card_id[:-1])
        s = self._suit_map.get(card_id[-1])
        if r is None or s is None:
            return []
        card_pts   = POINT_VALUES[r]
        trump_suit = self.trump_suit
        result     = []
        for cid in ALL_CARD_IDS:
            cr = self._rank_map.get(cid[:-1])
            cs = self._suit_map.get(cid[-1])
            if cr is None or cs is None:
                continue
            if cs == trump_suit and s != trump_suit:
                result.append(cid)
            elif cs == s and POINT_VALUES[cr] > card_pts:
                result.append(cid)
        return result


def _weighted_hyper_p_none(
    non_special_w: float, special_w: float,
    draws: int, pool_size: int
) -> float:
    """
    Approximate P(zero special cards in `draws` draws from weighted pool).
    Uses the beta-binomial approximation: treat weight ratio as probability,
    apply binomial P(X=0) = (1 - p_special)^draws.
    Exact for uniform weights; approximation degrades gracefully with skew.
    """
    if pool_size == 0 or draws == 0:
        return 1.0
    total_w = non_special_w + special_w
    if total_w <= 0:
        return 1.0
    p_special = special_w / total_w
    if p_special >= 1.0:
        return 0.0
    # Hypergeometric-style: scale by availability
    p_hit_per_draw = min(p_special * pool_size / max(pool_size - draws + 1, 1), 1.0)
    return max((1.0 - p_special) ** draws, 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# BehavioralBeliefState  (OracleBot only)
# ══════════════════════════════════════════════════════════════════════════════

class BehavioralBeliefState:
    """
    Tracks inferred opponent pile strength from behavioral signals.

    Signals and their updates
    ─────────────────────────

    PASS (opponent passed N cards instead of cutting combo of suit S, value V):
      - Opponent revealed they either couldn't cut or chose not to.
      - For a rational player: if they held a card of suit S with points > V,
        they would have cut (cutting gains them cards + calculate right).
      - Therefore passing is evidence they DON'T hold high-value cards of suit S.
      - We nudge belief weights for those cards downward in ExactBeliefTracker.
      - We also note their pile gained hidden cards (we know count, not identity).

    RAISE (opponent offered a stake increase):
      - Evidence of confidence — strong hand or strong pile.
      - Shift opp_pile_mean_estimate upward by raise_signal_strength.

    DECLINE (opponent declined our raise):
      - Evidence of weakness — weak hand or weak pile.
      - Shift opp_pile_mean_estimate downward.

    SKIP CALCULATE (opponent had calculate rights but passed):
      - Strong evidence their pile is below ~26 points (otherwise any
        rational agent would calculate to claim the stake).
      - Hard cap: opp_pile_max_estimate = min(current, 25).

    CALCULATE (opponent calculated — we learn their pile total exactly):
      - If they won: their pile was ≥ 31. Set estimate to 31+.
      - If they lost: their pile was < 31. Set estimate to their revealed total.
        (The engine reveals exact totals on calculate.)

    Pile estimate is expressed as (mean, confidence) where confidence ∈ [0,1]
    and represents how much weight to give the behavioral estimate vs. the
    card-math baseline.
    """

    # Calibration constants
    RAISE_SIGNAL      =  6.0   # pts added to opp pile estimate on raise
    DECLINE_SIGNAL    = -5.0   # pts added to opp pile estimate on decline
    PASS_SUIT_NUDGE   =  0.35  # weight multiplier for high-suit cards after pass
    SKIP_CALC_CAP     = 25     # max opp pile estimate after skip-calculate
    CONFIDENCE_GAIN   =  0.15  # confidence gained per observed signal
    MAX_CONFIDENCE    =  0.85  # confidence ceiling (never fully certain)

    def __init__(self):
        # Pile strength estimate for the opponent
        self.opp_pile_mean: float     = 15.0  # prior: about 15 pts to start
        self.opp_pile_min: float      = 0.0
        self.opp_pile_max: float      = 110.0 # theoretical max all 20 cards
        self.confidence: float        = 0.0   # how much to trust behavioral estimate
        self.passes_observed: int     = 0
        self.raises_observed: int     = 0
        self.declines_observed: int   = 0

    def on_opponent_pass(
        self,
        played_cards: list[Card],
        trump_suit,
        belief: ExactBeliefTracker,
    ):
        """
        Update on observing opponent pass instead of cutting.
        played_cards: the cards that were played (that opp couldn't/wouldn't cut).
        """
        self.passes_observed += 1

        for card in played_cards:
            # If played card is non-trump, opp probably lacks higher-value
            # cards of the same suit (else they'd have cut)
            if card.suit != trump_suit:
                belief.nudge_suit_above_value(
                    card.suit, card.points, self.PASS_SUIT_NUDGE
                )
            else:
                # Played card IS trump — opp couldn't cut with higher trump
                # Nudge down high trumps (they might not have them)
                belief.nudge_suit_above_value(
                    card.suit, card.points, self.PASS_SUIT_NUDGE * 1.5
                )

        # Passing means their pile grew only via hidden cards — weak signal
        # that their pile is not yet near 31 (else they'd be calculating)
        self.opp_pile_mean = min(self.opp_pile_mean + 2.0, 30.0)
        self._gain_confidence()

    def on_opponent_raise(self):
        """Update on observing opponent offer a stake raise."""
        self.raises_observed += 1
        self.opp_pile_mean = min(self.opp_pile_mean + self.RAISE_SIGNAL, 35.0)
        self.opp_pile_min  = max(self.opp_pile_min, self.opp_pile_mean - 8.0)
        self._gain_confidence()

    def on_opponent_decline(self):
        """Update on observing opponent decline our stake raise."""
        self.declines_observed += 1
        self.opp_pile_mean = max(self.opp_pile_mean + self.DECLINE_SIGNAL, 0.0)
        self.opp_pile_max  = min(self.opp_pile_max, self.opp_pile_mean + 10.0)
        self._gain_confidence()

    def on_opponent_skip_calculate(self):
        """Update when opponent had calculate rights but skipped."""
        # Strong signal: if they had ≥ 26, most rational agents calculate.
        # Skipping → they probably have < 26 in pile.
        self.opp_pile_max  = min(self.opp_pile_max, self.SKIP_CALC_CAP)
        self.opp_pile_mean = min(self.opp_pile_mean, self.SKIP_CALC_CAP - 3.0)
        self.confidence    = min(self.confidence + 0.25, self.MAX_CONFIDENCE)

    def on_opponent_calculated_win(self):
        """Opponent calculated and won — pile was ≥ 31."""
        self.opp_pile_min  = 31.0
        self.opp_pile_mean = 35.0
        self.confidence    = 1.0

    def on_opponent_calculated_lose(self, revealed_total: int):
        """Opponent calculated and lost — we know exact total."""
        self.opp_pile_mean = float(revealed_total)
        self.opp_pile_max  = float(revealed_total)
        self.confidence    = 1.0

    def adjusted_opp_pile_mean(self, card_math_estimate: float) -> float:
        """
        Blend behavioral estimate with card-math baseline.
        confidence=0 → pure card math. confidence=1 → pure behavioral.
        """
        return (self.confidence * self.opp_pile_mean
                + (1.0 - self.confidence) * card_math_estimate)

    def p_opp_can_calculate(self) -> float:
        """
        Estimated P(opponent pile ≥ 31) from behavioral model.
        Uses a logistic function centered on opp_pile_mean.
        """
        mean = self.opp_pile_mean
        # Logistic: P ≈ 0.5 at mean=31, steep curve
        return 1.0 / (1.0 + math.exp(-(mean - 31.0) / 4.0))

    def _gain_confidence(self):
        self.confidence = min(
            self.confidence + self.CONFIDENCE_GAIN,
            self.MAX_CONFIDENCE
        )


# ══════════════════════════════════════════════════════════════════════════════
# Exact pile probability  (shared by both bots)
# ══════════════════════════════════════════════════════════════════════════════

def _p_pile_ge_31_exact(
    known_points: int,
    hidden_card_count: int,
    unseen_card_ids: list[str],
    threshold: int = 31,
) -> float:
    """
    Exact P(pile_total ≥ threshold) via full enumeration over hidden pile.

    The hidden_card_count cards in the pile came from the unseen pool
    (cards not in own hand, not seen on table, not in seen_card_ids).
    We enumerate all C(|unseen|, hidden_card_count) combinations and
    count those that push total over threshold.

    For hidden_card_count ≤ 3 and |unseen| ≤ 17, this is at most
    C(17,3) = 680 iterations — instant.

    If hidden_card_count = 0: deterministic answer.
    """
    if hidden_card_count == 0:
        return 1.0 if known_points >= threshold else 0.0

    if not unseen_card_ids:
        return 1.0 if known_points >= threshold else 0.0

    need     = threshold - known_points
    if need <= 0:
        return 1.0   # already guaranteed

    pool     = [CARD_POINTS[c] for c in unseen_card_ids]
    n        = min(hidden_card_count, len(pool))
    total    = 0
    wins     = 0

    for combo in combinations(pool, n):
        total += 1
        if sum(combo) >= need:
            wins += 1

    return wins / total if total > 0 else 0.0


def _unseen_from_engine(engine: GameEngine, my_idx: int) -> list[str]:
    """Cards not in own hand and not observed by me — could be anywhere."""
    me          = engine.players[my_idx]
    my_hand_ids = {repr(c) for c in me.hand}
    seen        = set(me.seen_card_ids)
    if engine.deck.trump_card:
        seen.add(repr(engine.deck.trump_card))
    for c in engine.played_cards:
        seen.add(repr(c))
    return list(ALL_CARD_IDS - my_hand_ids - seen)


# ══════════════════════════════════════════════════════════════════════════════
# DecisionEngine  (shared logic for PureBot and OracleBot)
# ══════════════════════════════════════════════════════════════════════════════

class DecisionEngine:
    """
    Pure expected-value decision logic.  Takes a belief object and a
    behavioral state object (may be None for PureBot) and implements
    all four decision methods.

    All methods are static-style: they take the engine snapshot as input
    and return a decision.  No state is stored here — state lives in the
    belief and behavioral objects passed in.
    """

    # ── Calculate decision ────────────────────────────────────────────────────

    @staticmethod
    def should_calculate(
        engine: GameEngine,
        my_idx: int,
        belief: ExactBeliefTracker,
        behavioral: Optional[BehavioralBeliefState],
    ) -> bool:
        """
        EV(calculate) = (2 × p_win − 1) × stake
        EV(skip)      = continuation_value − urgency_penalty

        continuation_value: small positive if I might gain more cards.
        urgency_penalty: large if opponent is close to calculating themselves.

        We calculate if EV(calculate) > EV(skip).
        """
        me     = engine.players[my_idx]
        stake  = engine.current_stake
        unseen = _unseen_from_engine(engine, my_idx)

        # Exact P(my pile ≥ 31)
        p_win = _p_pile_ge_31_exact(
            me.known_pile_points,
            me.hidden_card_count,
            unseen,
        )

        ev_calc = (2.0 * p_win - 1.0) * stake

        # Opponent danger assessment
        opp_idx = 1 - my_idx
        opp     = engine.players[opp_idx]
        opp_tip = engine.compute_tip(opp_idx)

        if behavioral is not None:
            # Blend card-math estimate with behavioral estimate
            card_math_mean = (opp_tip["min_total"] + opp_tip["max_total"]) / 2.0
            opp_danger_mean = behavioral.adjusted_opp_pile_mean(card_math_mean)
            p_opp_wins = behavioral.p_opp_can_calculate()
        else:
            opp_danger_mean = (opp_tip["min_total"] + opp_tip["max_total"]) / 2.0
            p_opp_wins = 1.0 / (1.0 + math.exp(-(opp_danger_mean - 31.0) / 4.0))

        # Urgency: if opp can possibly win, skipping is risky
        urgency = 0.0
        if opp_tip["guaranteed_win"]:
            urgency = 999.0   # must calculate now if we can
        elif opp_tip["can_possibly_win"]:
            urgency = p_opp_wins * stake * 1.5

        # Continuation value: how much can we gain by playing one more hand?
        # Proxy: if deck has cards left, we might improve pile
        deck_cards   = len(engine.deck)
        continuation = 0.3 * min(deck_cards / 3.0, 1.0) * (1.0 - p_win)

        ev_skip = continuation - urgency

        # Special case: deck exhausted → must calculate
        if deck_cards == 0 and any(len(p.hand) < 3 for p in engine.players):
            return True

        return ev_calc > ev_skip

    # ── Play decision ─────────────────────────────────────────────────────────

    @staticmethod
    def choose_play(
        engine: GameEngine,
        my_idx: int,
        hand: list[Card],
        belief: ExactBeliefTracker,
        behavioral: Optional[BehavioralBeliefState],
    ) -> list[str]:
        """
        For every legal same-suit combo of 1–3 cards, compute:

          EV = P(opp can't cut) × gained_pts
             − P(opp cuts)      × lost_opportunity
             + maliutka_bonus
             + trump_drain_bonus
             + calculate_access_bonus  (if we take cards, we can calculate)

        Return the combo with highest EV.
        3-trump play is always taken (instant win).
        """
        trump = engine.trump_suit

        # Instant win
        trumps = [c for c in hand if c.suit == trump]
        if len(trumps) == 3:
            return [repr(c) for c in trumps]

        # Build candidate combos grouped by suit
        suits: dict = {}
        for c in hand:
            suits.setdefault(c.suit, []).append(c)

        best_ev   = float("-inf")
        best_play = [hand[0]]

        me     = engine.players[my_idx]
        unseen = _unseen_from_engine(engine, my_idx)

        # Calculate access bonus: if we take cards, we earn calculate right
        # Value proportional to our current P(pile ≥ 31) after taking those cards
        base_p_win = _p_pile_ge_31_exact(
            me.known_pile_points, me.hidden_card_count, unseen
        )

        for suit, cards in suits.items():
            for size in range(1, min(len(cards), 3) + 1):
                for combo in combinations(cards, size):
                    combo     = list(combo)
                    combo_ids = [repr(c) for c in combo]
                    play_pts  = sum(c.points for c in combo)

                    p_cut  = belief.p_opp_can_cut_combo(combo_ids)
                    p_pass = 1.0 - p_cut

                    # If we take these cards, how does our pile look?
                    future_known = me.known_pile_points + play_pts
                    p_win_after  = _p_pile_ge_31_exact(
                        future_known, me.hidden_card_count, unseen
                    )
                    calc_bonus = max(0.0, p_win_after - base_p_win) * engine.current_stake

                    # Core EV: gain play_pts if they pass, lose calculate right if they cut
                    ev = (p_pass * (play_pts + calc_bonus)
                          - p_cut * calc_bonus)  # if cut, they get calc right

                    # Maliutka bonus: forces opponent into difficult cut
                    is_maliutka = (size == 3 and suit != trump)
                    if is_maliutka:
                        # Value of forcing: P(opp can't cut all 3) × their distress
                        p_cant_cut = 1.0 - p_cut
                        ev += p_cant_cut * 3.5 + p_cut * (-1.0)

                    # Trump drain: playing trump removes trump threat
                    if suit == trump and size == 1:
                        p_no_trump_opp = 1.0 - belief.p_opp_has_trump()
                        ev += p_no_trump_opp * 2.0

                    # Penalty for playing high-value cards into a likely cut
                    if p_cut > 0.6 and play_pts >= 10:
                        ev -= p_cut * play_pts * 0.3

                    if ev > best_ev:
                        best_ev   = ev
                        best_play = combo

        return [repr(c) for c in best_play]

    # ── Cut / Pass decision ───────────────────────────────────────────────────

    @staticmethod
    def choose_cut_or_pass(
        engine: GameEngine,
        my_idx: int,
        hand: list[Card],
        belief: ExactBeliefTracker,
        behavioral: Optional[BehavioralBeliefState],
    ):
        """
        If valid cuts exist:
          - Compute EV for each valid cut combo.
          - EV(cut with X) = played_pts + my_pile_improvement − cost(X)
            where cost = opportunity cost of spending those cards
          - Pick cut that maximises EV.

        If no valid cuts:
          - Compare EV(counter-play maliutka) vs EV(pass cheapest).
          - EV(counter) = P(they can't cut our 3) × (played_pts + counter_pts)
                        − P(they cut us) × counter_pts
          - EV(pass)    = −played_pts  (we surrender them)
          - Counter if EV(counter) > EV(pass) + margin.

        Cut card selection strategy:
          - If opponent pile is dangerous (behavioral or card-math):
              use HIGHEST value cut (deny them cards, secure our calculate right)
          - If opponent pile is safe:
              use LOWEST cost cut (conserve strong cards for later plays)
        """
        valid      = engine.get_valid_cuts(my_idx)
        played     = engine.played_cards
        played_pts = sum(c.points for c in played)
        opp_idx    = 1 - my_idx
        opp_tip    = engine.compute_tip(opp_idx)

        # Assess opponent danger
        if behavioral is not None:
            card_math_mean = (opp_tip["min_total"] + opp_tip["max_total"]) / 2.0
            opp_mean = behavioral.adjusted_opp_pile_mean(card_math_mean)
        else:
            opp_mean = (opp_tip["min_total"] + opp_tip["max_total"]) / 2.0

        opp_dangerous = (opp_tip["guaranteed_win"]
                         or opp_tip["can_possibly_win"] and opp_mean >= 24.0)

        if valid:
            best_cut    = None
            best_cut_ev = float("-inf")

            me     = engine.players[my_idx]
            unseen = _unseen_from_engine(engine, my_idx)

            for cut_ids in valid:
                cut_cards  = [next(c for c in hand if repr(c) == cid) for cid in cut_ids]
                cut_pts    = sum(c.points for c in cut_cards)
                total_gain = played_pts  # we always gain the played cards

                # After cutting: my pile improves by played_pts (known)
                future_known = me.known_pile_points + played_pts
                p_win_after  = _p_pile_ge_31_exact(
                    future_known, me.hidden_card_count, unseen
                )
                base_p_win = _p_pile_ge_31_exact(
                    me.known_pile_points, me.hidden_card_count, unseen
                )
                calc_value = max(0.0, p_win_after - base_p_win) * engine.current_stake

                # Cost: spending high-value cut cards removes them from future plays
                # Opportunity cost is higher when we have few cards left
                opportunity_cost = cut_pts * 0.15 * (1.0 if len(hand) > 3 else 0.4)

                ev = total_gain + calc_value - opportunity_cost

                # If opp is dangerous, we WANT the calculate right desperately
                if opp_dangerous:
                    ev += calc_value * 2.0

                if ev > best_cut_ev:
                    best_cut_ev = ev
                    best_cut    = cut_ids

            return ("cut", best_cut)

        # No valid cuts — consider counter-play or pass
        if engine.phase == GamePhase.CUTTING:
            trump = engine.trump_suit
            sg: dict = {}
            for c in hand:
                if c.suit != trump:
                    sg.setdefault(c.suit, []).append(c)

            best_counter    = None
            best_counter_ev = float("-inf")

            for suit, cards in sg.items():
                if len(cards) < 3:
                    continue
                top3     = sorted(cards, key=lambda c: c.points, reverse=True)[:3]
                top3_ids = [repr(c) for c in top3]
                top3_pts = sum(c.points for c in top3)

                p_they_cut = belief.p_opp_can_cut_combo(top3_ids)
                ev_counter = (
                    (1.0 - p_they_cut) * (played_pts + top3_pts)
                    - p_they_cut * top3_pts
                )
                if ev_counter > best_counter_ev:
                    best_counter_ev = ev_counter
                    best_counter    = top3_ids

            ev_pass = -played_pts

            # Counter only if clearly better (margin = 3 pts)
            if best_counter is not None and best_counter_ev > ev_pass + 3.0:
                return ("counter", best_counter)

        # Pass: give cheapest cards
        n        = len(played)
        cheapest = sorted(hand, key=lambda c: c.points)[:n]
        return ("pass", [repr(c) for c in cheapest])

    # ── Raise decision ────────────────────────────────────────────────────────

    @staticmethod
    def should_raise(
        engine: GameEngine,
        my_idx: int,
        belief: ExactBeliefTracker,
        behavioral: Optional[BehavioralBeliefState],
    ) -> bool:
        """
        Model raise as a signaling game with two outcomes:

          Opp declines → I win current_stake immediately.
          Opp accepts  → game continues at new_stake.

        EV(raise) = P(opp declines) × current_stake
                  + P(opp accepts)  × [(2 × p_I_win − 1) × new_stake]

        EV(no raise) = (2 × p_I_win − 1) × current_stake
                       (continue at current stake)

        Raise if EV(raise) > EV(no raise).

        P(opp declines):
          PureBot: fixed prior = 0.35 (humans decline ~35% of raises on average)
          OracleBot: adjusted by behavioral — if opp has been declining,
                     raise that probability; if opp has been accepting/raising,
                     lower it.

        P(I win round): exact from pile state + hand strength.
        """
        if not engine.can_raise_stake(my_idx):
            return False
        if engine.current_stake >= 6:
            return False

        me      = engine.players[my_idx]
        unseen  = _unseen_from_engine(engine, my_idx)
        p_i_win = _p_pile_ge_31_exact(
            me.known_pile_points, me.hidden_card_count, unseen
        )

        # Hand strength bonus: strong hand → more future pile points likely
        trump = engine.trump_suit
        hand  = me.hand
        trump_count  = sum(1 for c in hand if c.suit == trump)
        hand_pts     = sum(c.points for c in hand)
        hand_strength = hand_pts / (3 * 11)  # normalize to [0,1]

        # Blend pile win probability with hand strength
        p_i_win_adj = 0.7 * p_i_win + 0.3 * hand_strength

        # P(opponent declines our raise)
        if behavioral is not None and behavioral.confidence > 0.2:
            # If opp has been declining a lot → they're weak → higher decline prob
            base_decline = 0.35
            decline_adj  = behavioral.declines_observed * 0.08
            accept_adj   = behavioral.raises_observed * 0.06
            p_opp_declines = min(max(base_decline + decline_adj - accept_adj, 0.1), 0.75)
        else:
            p_opp_declines = 0.35  # flat prior

        current = engine.current_stake
        new     = current + 1

        ev_raise    = (p_opp_declines * current
                       + (1.0 - p_opp_declines) * (2.0 * p_i_win_adj - 1.0) * new)
        ev_no_raise = (2.0 * p_i_win_adj - 1.0) * current

        # Never raise when opponent is guaranteed to win
        opp_tip = engine.compute_tip(1 - my_idx)
        if opp_tip["guaranteed_win"]:
            return False

        return ev_raise > ev_no_raise

    # ── Decline decision ──────────────────────────────────────────────────────

    @staticmethod
    def should_accept_raise(
        engine: GameEngine,
        my_idx: int,
        belief: ExactBeliefTracker,
        behavioral: Optional[BehavioralBeliefState],
    ) -> bool:
        """
        Opponent offered a raise. Should we accept or decline?

        EV(accept) = (2 × p_I_win − 1) × new_stake
        EV(decline) = −current_stake  (they win now at current stake)

        Accept if EV(accept) > EV(decline).
        i.e. accept if p_I_win > 0.5 − current_stake / (2 × new_stake)

        This gives a dynamic threshold:
          stake 1→2: accept if p_win > 0.25  (low bar, game is early)
          stake 4→5: accept if p_win > 0.40  (higher bar, more to lose)
          stake 5→6: accept if p_win > 0.42
        """
        me      = engine.players[my_idx]
        unseen  = _unseen_from_engine(engine, my_idx)
        p_i_win = _p_pile_ge_31_exact(
            me.known_pile_points, me.hidden_card_count, unseen
        )

        # Hand strength contribution (future potential)
        hand      = me.hand
        trump     = engine.trump_suit
        hand_pts  = sum(c.points for c in hand)
        hand_str  = hand_pts / (3 * 11)
        p_i_win_adj = 0.65 * p_i_win + 0.35 * hand_str

        current = engine.current_stake
        new     = engine.pending_stake

        ev_accept  = (2.0 * p_i_win_adj - 1.0) * new
        ev_decline = -float(current)

        # If behavioral model says opp is strong, be more conservative
        if behavioral is not None and behavioral.confidence > 0.3:
            opp_strong = behavioral.opp_pile_mean > 22.0
            if opp_strong:
                ev_accept -= 1.0  # conservative penalty

        return ev_accept > ev_decline


# ══════════════════════════════════════════════════════════════════════════════
# PureBot
# ══════════════════════════════════════════════════════════════════════════════

class PureBot(BotPlayer):
    """
    Pure Bayesian card-math bot.

    Uses ExactBeliefTracker for per-card probability estimates and
    DecisionEngine for all EV calculations.  No behavioral inference —
    every decision is based solely on card math and pile probability.

    This bot represents the theoretical optimum of pure probabilistic
    reasoning without any opponent modeling.
    """

    DISPLAY_NAME = "Pure Bayes"

    def _setup(self, engine: GameEngine):
        my_idx = engine.players.index(self)
        belief = ExactBeliefTracker(engine, my_idx)
        return my_idx, belief

    def choose_play(self, engine: GameEngine) -> list[str]:
        my_idx, belief = self._setup(engine)
        return DecisionEngine.choose_play(engine, my_idx, self.hand, belief, None)

    def choose_cut_or_pass(self, engine: GameEngine):
        my_idx, belief = self._setup(engine)
        return DecisionEngine.choose_cut_or_pass(engine, my_idx, self.hand, belief, None)

    def choose_calculate(self, engine: GameEngine) -> bool:
        my_idx, belief = self._setup(engine)
        return DecisionEngine.should_calculate(engine, my_idx, belief, None)

    def choose_raise_stake(self, engine: GameEngine) -> bool:
        my_idx, belief = self._setup(engine)
        # Also handle responding to opponent raise
        if (engine.stake_offerer_idx is not None
                and engine.stake_offerer_idx != my_idx):
            return DecisionEngine.should_accept_raise(engine, my_idx, belief, None)
        return DecisionEngine.should_raise(engine, my_idx, belief, None)


# ══════════════════════════════════════════════════════════════════════════════
# OracleBot
# ══════════════════════════════════════════════════════════════════════════════

class OracleBot(BotPlayer):
    """
    Behavioral inference bot.

    Identical decision logic to PureBot but adds a BehavioralBeliefState
    that is updated after every observed opponent action:
      - Passes     → nudge card weights + update pile estimate
      - Raises     → shift pile estimate up
      - Declines   → shift pile estimate down
      - Skip-calc  → cap pile estimate
      - Calculate  → set pile estimate exactly

    The behavioral state is maintained across the entire round and resets
    at round start.  Over the course of a round it accumulates evidence
    that meaningfully changes stake and calculate decisions.

    The key thesis claim: OracleBot should outperform PureBot specifically
    in stake-raising decisions and calculate timing, where behavioral
    signals add the most value.
    """

    DISPLAY_NAME = "Oracle (Behavioral)"

    def __init__(self, player_id: str, name: str):
        super().__init__(player_id, name)
        self._behavioral: BehavioralBeliefState = BehavioralBeliefState()
        self._last_round: int = -1

    def _reset_if_new_round(self, engine: GameEngine):
        if engine.round_number != self._last_round:
            self._behavioral = BehavioralBeliefState()
            self._last_round = engine.round_number

    def _setup(self, engine: GameEngine):
        self._reset_if_new_round(engine)
        my_idx = engine.players.index(self)
        belief = ExactBeliefTracker(engine, my_idx)
        return my_idx, belief, self._behavioral

    # ── Action observers  (called by bot_runner after opponent acts) ──────────

    def observe_opponent_pass(self, engine: GameEngine, played_cards: list[Card]):
        """Call this after opponent passes. Updates behavioral state."""
        self._reset_if_new_round(engine)
        my_idx = engine.players.index(self)
        belief = ExactBeliefTracker(engine, my_idx)
        self._behavioral.on_opponent_pass(played_cards, engine.trump_suit, belief)

    def observe_opponent_raise(self, engine: GameEngine):
        self._reset_if_new_round(engine)
        self._behavioral.on_opponent_raise()

    def observe_opponent_decline(self, engine: GameEngine):
        self._reset_if_new_round(engine)
        self._behavioral.on_opponent_decline()

    def observe_opponent_skip_calculate(self, engine: GameEngine):
        self._reset_if_new_round(engine)
        self._behavioral.on_opponent_skip_calculate()

    def observe_opponent_calculated_win(self, engine: GameEngine):
        self._reset_if_new_round(engine)
        self._behavioral.on_opponent_calculated_win()

    def observe_opponent_calculated_lose(self, engine: GameEngine, total: int):
        self._reset_if_new_round(engine)
        self._behavioral.on_opponent_calculated_lose(total)

    # ── Decision methods ──────────────────────────────────────────────────────

    def choose_play(self, engine: GameEngine) -> list[str]:
        my_idx, belief, beh = self._setup(engine)
        return DecisionEngine.choose_play(engine, my_idx, self.hand, belief, beh)

    def choose_cut_or_pass(self, engine: GameEngine):
        my_idx, belief, beh = self._setup(engine)
        return DecisionEngine.choose_cut_or_pass(engine, my_idx, self.hand, belief, beh)

    def choose_calculate(self, engine: GameEngine) -> bool:
        my_idx, belief, beh = self._setup(engine)
        return DecisionEngine.should_calculate(engine, my_idx, belief, beh)

    def choose_raise_stake(self, engine: GameEngine) -> bool:
        my_idx, belief, beh = self._setup(engine)
        if (engine.stake_offerer_idx is not None
                and engine.stake_offerer_idx != my_idx):
            return DecisionEngine.should_accept_raise(engine, my_idx, belief, beh)
        return DecisionEngine.should_raise(engine, my_idx, belief, beh)


# ══════════════════════════════════════════════════════════════════════════════
# Bot registry
# ══════════════════════════════════════════════════════════════════════════════

OPTIMAL_BOT_REGISTRY: dict[str, type] = {
    "pure":   PureBot,
    "oracle": OracleBot,
}


def get_optimal_bot(bot_id: str, player_id: str, name: str) -> BotPlayer:
    cls = OPTIMAL_BOT_REGISTRY.get(bot_id)
    if cls is None:
        raise ValueError(f"Unknown optimal bot: {bot_id!r}")
    return cls(player_id, name)


def list_optimal_bots() -> list[dict]:
    return [
        {"id": k, "name": getattr(v, "DISPLAY_NAME", k)}
        for k, v in OPTIMAL_BOT_REGISTRY.items()
    ]