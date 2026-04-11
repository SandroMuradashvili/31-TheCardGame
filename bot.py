"""
bot.py — Bot implementations
─────────────────────────────
Three bots, increasing sophistication:

  SimpleBot     — cautious, rule-based, fully informed
  AggressiveBot — pressure-oriented, risk-taking, fully informed
  EVBot         — probability math + belief tracking, EV-maximising

All bots use only legally visible information (own hand, own pile, opponent
known pile stats, deck size, trump card, own seen_card_ids, table cards).
"""

import math
import random
from itertools import combinations
from game_engine import (
    BotPlayer, GameEngine, Rank, Suit, GamePhase,
    ALL_CARD_IDS, CARD_POINTS, POINT_VALUES,
)


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _tip(engine: GameEngine, idx: int) -> dict:
    return engine.compute_tip(idx)

def _opp_idx(engine: GameEngine, me: "BotPlayer") -> int:
    return 1 - engine.players.index(me)

def _must_calculate(engine: GameEngine) -> bool:
    return len(engine.deck) == 0 and any(len(p.hand) < 3 for p in engine.players)

def _combo_points(hand, combo_ids: list[str]) -> int:
    return sum(
        next(c for c in hand if repr(c) == cid).points
        for cid in combo_ids
    )

def _opp_danger(engine: GameEngine, opp_idx: int) -> str:
    """
    'safe'     — opponent cannot reach 31 even best case
    'possible' — opponent might reach 31 depending on hidden cards
    'danger'   — opponent is guaranteed 31+ right now
    """
    tip = _tip(engine, opp_idx)
    if tip["guaranteed_win"]:   return "danger"
    if tip["can_possibly_win"]: return "possible"
    return "safe"


# ══════════════════════════════════════════════════════════════════════════════
# SimpleBot
# ══════════════════════════════════════════════════════════════════════════════

class SimpleBot(BotPlayer):
    """
    Cautious, percentage-based. Fully informed, risk-averse.

    Play:   Highest non-trump; cheapest trump only if opponent guaranteed to win.
    Cut:    Min-cost when opponent safe; max-value when opponent close.
    Calc:   Only when guaranteed, or racing a dangerous opponent.
    Raise:  Never.
    """

    DISPLAY_NAME = "Simple"

    def choose_play(self, engine: GameEngine) -> list[str]:
        hand   = self.hand
        trump  = engine.trump_suit
        danger = _opp_danger(engine, _opp_idx(engine, self))

        trumps    = [c for c in hand if c.suit == trump]
        non_trump = [c for c in hand if c.suit != trump]

        if len(trumps) == 3:
            return [repr(c) for c in trumps]
        if danger == "danger" and trumps:
            return [repr(min(trumps, key=lambda c: c.points))]
        if non_trump:
            return [repr(max(non_trump, key=lambda c: c.points))]
        return [repr(min(hand, key=lambda c: c.points))]

    def choose_cut_or_pass(self, engine: GameEngine):
        my_idx = engine.players.index(self)
        opp    = _opp_idx(engine, self)
        valid  = engine.get_valid_cuts(my_idx)
        danger = _opp_danger(engine, opp)

        if valid:
            key  = max if danger in ("danger", "possible") else min
            best = key(valid, key=lambda c: _combo_points(self.hand, c))
            return ("cut", best)

        n        = len(engine.played_cards)
        cheapest = sorted(self.hand, key=lambda c: c.points)[:n]
        return ("pass", [repr(c) for c in cheapest])

    def choose_calculate(self, engine: GameEngine) -> bool:
        my_idx = engine.players.index(self)
        opp    = _opp_idx(engine, self)

        if _must_calculate(engine): return True

        my_tip = _tip(engine, my_idx)
        if not my_tip["can_possibly_win"]:    return False
        if my_tip["guaranteed_win"]:          return True
        if _opp_danger(engine, opp) == "danger": return True
        return False

    def choose_raise_stake(self, engine: GameEngine) -> bool:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# AggressiveBot
# ══════════════════════════════════════════════════════════════════════════════

class AggressiveBot(BotPlayer):
    """
    Pressure-oriented, risk-taking. Fully informed, draws bold conclusions.

    Play:   Hunts maliutka; drains opponent trump; plays highest non-trump.
    Cut:    Always max-value; counter-plays opportunistically.
    Calc:   Sliding threshold (26/29/31) based on opponent danger level.
    Raise:  When strong hand or opponent is weak; never when opponent winning.
    """

    DISPLAY_NAME = "Aggressive"

    def _hand_strength(self, engine: GameEngine) -> float:
        trump = engine.trump_suit
        score = sum(c.points * (1.5 if c.suit == trump else 1.0) for c in self.hand)
        return score / (3 * 11 * 1.5)

    def choose_play(self, engine: GameEngine) -> list[str]:
        hand   = self.hand
        trump  = engine.trump_suit
        danger = _opp_danger(engine, _opp_idx(engine, self))

        trumps    = [c for c in hand if c.suit == trump]
        non_trump = [c for c in hand if c.suit != trump]

        if len(trumps) == 3:
            return [repr(c) for c in trumps]

        suits: dict = {}
        for c in non_trump:
            suits.setdefault(c.suit, []).append(c)
        for suit, cards in suits.items():
            if len(cards) == 3:
                threshold = 0.60 if danger == "safe" else 0.90
                if random.random() < threshold:
                    return [repr(c) for c in cards]

        if len(trumps) >= 2:
            return [repr(min(trumps, key=lambda c: c.points))]
        if non_trump:
            return [repr(max(non_trump, key=lambda c: c.points))]
        return [repr(max(hand, key=lambda c: c.points))]

    def choose_cut_or_pass(self, engine: GameEngine):
        my_idx = engine.players.index(self)
        opp    = _opp_idx(engine, self)
        valid  = engine.get_valid_cuts(my_idx)
        danger = _opp_danger(engine, opp)

        if valid:
            best = max(valid, key=lambda c: _combo_points(self.hand, c))
            return ("cut", best)

        if engine.phase == GamePhase.CUTTING:
            trump = engine.trump_suit
            sg: dict = {}
            for c in self.hand:
                if c.suit != trump:
                    sg.setdefault(c.suit, []).append(c)
            for suit, cards in sg.items():
                if len(cards) >= 3:
                    threshold = 0.40 if danger == "safe" else 0.70
                    if random.random() < threshold:
                        top3 = sorted(cards, key=lambda c: c.points, reverse=True)[:3]
                        return ("counter", [repr(c) for c in top3])

        n        = len(engine.played_cards)
        cheapest = sorted(self.hand, key=lambda c: c.points)[:n]
        return ("pass", [repr(c) for c in cheapest])

    def choose_calculate(self, engine: GameEngine) -> bool:
        my_idx = engine.players.index(self)
        opp    = _opp_idx(engine, self)

        if _must_calculate(engine): return True

        my_tip = _tip(engine, my_idx)
        if not my_tip["can_possibly_win"]: return False

        expected  = (my_tip["min_total"] + my_tip["max_total"]) / 2
        threshold = {"danger": 26, "possible": 29, "safe": 31}[_opp_danger(engine, opp)]
        return expected >= threshold

    def choose_raise_stake(self, engine: GameEngine) -> bool:
        opp    = _opp_idx(engine, self)
        danger = _opp_danger(engine, opp)
        if danger == "danger": return False

        strength = self._hand_strength(engine)
        if strength >= 0.55: return True

        my_tip = _tip(engine, engine.players.index(self))
        if danger == "safe" and my_tip["can_possibly_win"] and strength >= 0.35:
            return True
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Belief Tracker
# ══════════════════════════════════════════════════════════════════════════════

class BeliefTracker:
    """
    Probability distribution over cards the opponent holds.

    The 20-card deck is fully partitioned at any point:
      known to me  = my hand + my seen_card_ids + table cards + trump card
      unseen       = everything else

    Unseen cards are split between opponent hand and remaining deck.
    Under a uniform (maximum-entropy) prior:

      P(card X in opponent hand) = opp_hand_size / |unseen|

    This is the hypergeometric model — exact for uniform priors, updates
    automatically as seen_card_ids grows with each observed card.

    We also expose:
      p_opp_has_trump()              — P(opponent holds at least one trump)
      p_opp_can_beat_card(card_id)   — P(opponent can beat a single card)
      p_opp_can_cut_combo(ids)       — P(opponent can cut the whole combo)
      expected_opp_hand_points()     — E[points in opponent hand]
    """

    def __init__(self, engine: GameEngine, my_idx: int):
        me  = engine.players[my_idx]
        opp = engine.players[1 - my_idx]

        my_hand_ids: set[str] = {repr(c) for c in me.hand}

        # Everything this player has personally observed
        seen: set[str] = set(me.seen_card_ids)
        # Trump card is globally visible
        if engine.deck.trump_card:
            seen.add(repr(engine.deck.trump_card))
        # Cards on table are globally visible
        for c in engine.played_cards:
            seen.add(repr(c))

        # Unseen = all 20 cards minus my hand minus everything observed
        self.unseen: set[str] = ALL_CARD_IDS - seen - my_hand_ids
        # Remove table cards from unseen (they're visible, not in hand/deck)
        for c in engine.played_cards:
            self.unseen.discard(repr(c))

        self.opp_hand_size: int  = len(opp.hand)
        self.unseen_count:  int  = len(self.unseen)
        self.trump_suit           = engine.trump_suit
        self._rank_map            = {r.value: r for r in Rank}
        self._suit_map            = {s.value[0].upper(): s for s in Suit}

    def p_card_in_opp_hand(self, card_id: str) -> float:
        """P(card_id is in opponent's hand)."""
        if card_id not in self.unseen or self.unseen_count == 0:
            return 0.0
        return self.opp_hand_size / self.unseen_count

    def p_opp_has_trump(self) -> float:
        """P(opponent holds at least one trump)."""
        if not self.trump_suit or self.unseen_count == 0 or self.opp_hand_size == 0:
            return 0.0
        trump_initial = self.trump_suit.value[0].upper()
        trump_unseen  = sum(1 for c in self.unseen if c.endswith(trump_initial))
        non_trump     = self.unseen_count - trump_unseen
        p_none        = _hyper_p_none(non_trump, trump_unseen, self.opp_hand_size)
        return 1.0 - p_none

    def p_opp_can_beat_card(self, card_id: str) -> float:
        """P(opponent has at least one card that beats card_id)."""
        beaters        = self._beating_ids(card_id)
        beaters_unseen = sum(1 for c in beaters if c in self.unseen)
        if beaters_unseen == 0 or self.unseen_count == 0:
            return 0.0
        non_beaters = self.unseen_count - beaters_unseen
        p_none      = _hyper_p_none(non_beaters, beaters_unseen, self.opp_hand_size)
        return 1.0 - p_none

    def p_opp_can_cut_combo(self, played_ids: list[str]) -> float:
        """
        P(opponent can cut the full played combo).
        For 1 card: exact. For 2–3 cards: min over individual probabilities
        (lower bound — conservative, avoids over-estimating cut risk).
        """
        if not played_ids:
            return 0.0
        if len(played_ids) == 1:
            return self.p_opp_can_beat_card(played_ids[0])
        return min(self.p_opp_can_beat_card(cid) for cid in played_ids)

    def expected_opp_hand_points(self) -> float:
        """E[total points in opponent's hand]."""
        if self.unseen_count == 0:
            return 0.0
        avg = sum(CARD_POINTS[c] for c in self.unseen) / self.unseen_count
        return avg * self.opp_hand_size

    def _beating_ids(self, card_id: str) -> list[str]:
        """All card ids that beat card_id by the game's rules."""
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
                result.append(cid)   # trump beats non-trump
            elif cs == s and POINT_VALUES[cr] > card_pts:
                result.append(cid)   # same suit, higher rank
        return result


def _hyper_p_none(non_special: int, special: int, draws: int) -> float:
    """
    P(drawing `draws` cards from (non_special + special) without hitting
    any special card) = C(non_special, draws) / C(total, draws).
    Computed in log-space to avoid overflow.
    """
    total = non_special + special
    if draws > non_special or draws > total:
        return 0.0
    lp = (math.lgamma(non_special + 1) - math.lgamma(draws + 1)
          - math.lgamma(non_special - draws + 1)
          - math.lgamma(total + 1) + math.lgamma(draws + 1)
          + math.lgamma(total - draws + 1))
    return math.exp(lp)


# ══════════════════════════════════════════════════════════════════════════════
# EVBot
# ══════════════════════════════════════════════════════════════════════════════

class EVBot(BotPlayer):
    """
    Expected Value bot with Belief Tracking.

    Uses the BeliefTracker to maintain a probability distribution over the
    opponent's hand, then calculates expected value for every legal option
    and picks the highest-EV move.

    ── Play ─────────────────────────────────────────────────────────────────
    Enumerates all same-suit groups of 1–3 cards. For each:

      EV = P(opp can't cut) × play_pts
         - P(opp can cut)   × play_pts
         + maliutka_bonus   (forces forced_cut, limits opponent options)
         + trump_drain_bonus (opponent less likely to have trump)

    3-trump play → instant win, always taken.

    ── Cut / Pass ───────────────────────────────────────────────────────────
    If a valid cut exists:
      EV(cut) = played_pts (we always gain the played cards)
      Tie-break: min-cost when opponent safe, max-value when opponent close.

    Counter-play EV vs pass EV:
      EV(counter) = P(they can't cut our 3) × (played_pts + counter_pts)
                  - P(they cut us)          × counter_pts
      EV(pass)    = −played_pts
      Counter if EV(counter) > EV(pass) + margin.

    ── Calculate ────────────────────────────────────────────────────────────
    P(own_total >= 31) estimated via Monte Carlo (400 samples) over the
    unseen card pool for own hidden pile cards.

      EV(calculate) = (2 × p_win − 1) × stake
      EV(skip)      = small positive + urgency_penalty

    Calculate if EV(calculate) > EV(skip).

    ── Raise ────────────────────────────────────────────────────────────────
    Estimates P(win_round) from own pile Monte Carlo and opponent pile range.
    Raises if p_win >= threshold (0.58 safe / 0.65 possible).
    Never raises when opponent is in 'danger'.
    """

    DISPLAY_NAME = "EV (Belief Tracker)"
    _MONTE_CARLO_SAMPLES = 400

    # ── Belief + MC helpers ───────────────────────────────────────────────────

    def _belief(self, engine: GameEngine) -> BeliefTracker:
        return BeliefTracker(engine, engine.players.index(self))

    def _p_own_win(self, engine: GameEngine) -> float:
        """P(my true pile total >= 31) via Monte Carlo over unseen card pool."""
        my_idx = engine.players.index(self)
        me     = engine.players[my_idx]
        known  = me.known_pile_points
        hc     = me.hidden_card_count

        if hc == 0:
            return 1.0 if known >= 31 else 0.0

        unseen_list = list(self._belief(engine).unseen)
        if not unseen_list:
            return 1.0 if known >= 31 else 0.0

        wins = 0
        for _ in range(self._MONTE_CARLO_SAMPLES):
            sample      = random.sample(unseen_list, min(hc, len(unseen_list)))
            total       = known + sum(CARD_POINTS[c] for c in sample)
            if total >= 31:
                wins += 1
        return wins / self._MONTE_CARLO_SAMPLES

    def _p_win_round(self, engine: GameEngine) -> float:
        """Rough P(I win this round) combining own MC win chance and opponent range."""
        my_idx  = engine.players.index(self)
        opp_idx = _opp_idx(engine, self)
        opp_tip = _tip(engine, opp_idx)
        opp_mid = (opp_tip["min_total"] + opp_tip["max_total"]) / 2
        p_opp_wins = 1 / (1 + math.exp(-(opp_mid - 31) / 4))
        p_i_win    = self._p_own_win(engine)
        return p_i_win * (1 - p_opp_wins)

    # ── Play ─────────────────────────────────────────────────────────────────

    def choose_play(self, engine: GameEngine) -> list[str]:
        hand   = self.hand
        trump  = engine.trump_suit
        belief = self._belief(engine)

        # Instant win
        trumps = [c for c in hand if c.suit == trump]
        if len(trumps) == 3:
            return [repr(c) for c in trumps]

        # Build all candidate plays: every same-suit group of 1–3 cards
        suits: dict = {}
        for c in hand:
            suits.setdefault(c.suit, []).append(c)

        best_ev   = float('-inf')
        best_play = [hand[0]]   # fallback

        for suit, cards in suits.items():
            for size in range(1, min(len(cards), 3) + 1):
                for combo in combinations(cards, size):
                    combo       = list(combo)
                    combo_ids   = [repr(c) for c in combo]
                    play_pts    = sum(c.points for c in combo)
                    is_maliutka = (
                        size == 3
                        and suit != trump
                    )
                    is_trump_play = suit == trump

                    p_cut = belief.p_opp_can_cut_combo(combo_ids)
                    p_pass = 1.0 - p_cut

                    ev = p_pass * play_pts - p_cut * play_pts

                    # Maliutka: forces forced_cut, limits opponent response options
                    if is_maliutka:
                        ev += 4.0

                    # Playing trump when opponent unlikely to have trump
                    # drains the suit that threatens our piles
                    if is_trump_play and size == 1:
                        p_no_trump = 1.0 - belief.p_opp_has_trump()
                        ev += p_no_trump * 2.0

                    if ev > best_ev:
                        best_ev   = ev
                        best_play = combo

        return [repr(c) for c in best_play]

    # ── Cut / Pass ────────────────────────────────────────────────────────────

    def choose_cut_or_pass(self, engine: GameEngine):
        my_idx     = engine.players.index(self)
        opp_idx    = _opp_idx(engine, self)
        valid      = engine.get_valid_cuts(my_idx)
        played     = engine.played_cards
        played_pts = sum(c.points for c in played)
        danger     = _opp_danger(engine, opp_idx)

        # Cut: net gain is always played_pts (we win the played cards for free)
        # Tie-break on cut card cost based on opponent danger
        if valid:
            key  = max if danger in ("danger", "possible") else min
            best = key(valid, key=lambda c: _combo_points(self.hand, c))
            return ("cut", best)

        # Counter-play (only in normal cutting phase)
        if engine.phase == GamePhase.CUTTING:
            belief = self._belief(engine)
            trump  = engine.trump_suit
            sg: dict = {}
            for c in self.hand:
                if c.suit != trump:
                    sg.setdefault(c.suit, []).append(c)

            best_counter_ev = float('-inf')
            best_counter    = None

            for suit, cards in sg.items():
                if len(cards) < 3:
                    continue
                top3     = sorted(cards, key=lambda c: c.points, reverse=True)[:3]
                top3_ids = [repr(c) for c in top3]
                top3_pts = sum(c.points for c in top3)

                p_they_cut = belief.p_opp_can_cut_combo(top3_ids)
                ev_counter = (
                    (1 - p_they_cut) * (played_pts + top3_pts)
                    - p_they_cut * top3_pts
                )
                if ev_counter > best_counter_ev:
                    best_counter_ev = ev_counter
                    best_counter    = top3_ids

            # EV(pass) = we lose played_pts (opponent wins them)
            ev_pass = -played_pts
            # Use a margin of 3 to avoid reckless counters
            if best_counter is not None and best_counter_ev > ev_pass + 3.0:
                return ("counter", best_counter)

        # Pass: give away cheapest cards
        n        = len(played)
        cheapest = sorted(self.hand, key=lambda c: c.points)[:n]
        return ("pass", [repr(c) for c in cheapest])

    # ── Calculate ────────────────────────────────────────────────────────────

    def choose_calculate(self, engine: GameEngine) -> bool:
        my_idx  = engine.players.index(self)
        opp_idx = _opp_idx(engine, self)
        stake   = engine.current_stake

        if _must_calculate(engine):
            return True

        my_tip = _tip(engine, my_idx)
        if not my_tip["can_possibly_win"]:
            return False

        p_win  = self._p_own_win(engine)
        danger = _opp_danger(engine, opp_idx)

        # EV(calculate) = (2p - 1) × stake
        ev_calc = (2 * p_win - 1) * stake

        # EV(skip): small continuation value, discounted by urgency
        urgency = {"danger": 2.5, "possible": 0.8, "safe": 0.0}[danger]
        ev_skip = max(0.0, (1 - p_win) * 1.5 - urgency)

        return ev_calc > ev_skip

    # ── Raise ─────────────────────────────────────────────────────────────────

    def choose_raise_stake(self, engine: GameEngine) -> bool:
        opp_idx = _opp_idx(engine, self)
        danger  = _opp_danger(engine, opp_idx)

        # Never inflate a stake the opponent is about to win
        if danger == "danger":
            return False

        p_win     = self._p_win_round(engine)
        threshold = 0.58 if danger == "safe" else 0.65
        return p_win >= threshold


# ══════════════════════════════════════════════════════════════════════════════
# Bot Registry
# ══════════════════════════════════════════════════════════════════════════════

BOT_REGISTRY: dict[str, type] = {
    "simple":     SimpleBot,
    "aggressive": AggressiveBot,
    "ev":         EVBot,
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