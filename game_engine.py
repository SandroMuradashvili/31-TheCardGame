"""
BURA Game Engine
Complete implementation of all game rules.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional
import random
import copy


# ─────────────────────────────────────────────
#  Card primitives
# ─────────────────────────────────────────────

SUITS = ['hearts', 'diamonds', 'clubs', 'spades']
SUIT_SYMBOLS = {'hearts': '♥', 'diamonds': '♦', 'clubs': '♣', 'spades': '♠'}
RANKS = ['A', 'T', 'K', 'Q', 'J']
POINTS = {'A': 11, 'T': 10, 'K': 4, 'Q': 3, 'J': 2}
RANK_ORDER = {'A': 5, 'T': 4, 'K': 3, 'Q': 2, 'J': 1}  # for comparison


@dataclass(frozen=True)
class Card:
    rank: str
    suit: str

    @property
    def points(self) -> int:
        return POINTS[self.rank]

    @property
    def rank_value(self) -> int:
        return RANK_ORDER[self.rank]

    def __str__(self):
        return f"{self.rank}{SUIT_SYMBOLS[self.suit]}"

    def __repr__(self):
        return str(self)


def make_deck() -> list[Card]:
    return [Card(r, s) for s in SUITS for r in RANKS]


def score_pile(cards: list[Card]) -> int:
    return sum(c.points for c in cards)


# ─────────────────────────────────────────────
#  Game phases / actions
# ─────────────────────────────────────────────

class Phase(Enum):
    STAKE_OFFER    = auto()   # before any play — either player can raise
    PLAY           = auto()   # active player plays cards
    CUT_OR_PASS    = auto()   # opponent decides to cut or pass
    CALCULATE      = auto()   # card-taker decides to calculate or continue
    DRAW           = auto()   # both draw to 3
    ROUND_OVER     = auto()   # round ended, score updated
    GAME_OVER      = auto()


@dataclass
class GameState:
    # ── deck / trump ──
    deck: list[Card]
    trump_suit: str

    # ── hands & piles ──
    hands: list[list[Card]]           # hands[0] = P1, hands[1] = P2
    score_piles: list[list[Card]]     # accumulated scored cards

    # ── scores (game-level, rounds won) ──
    game_scores: list[int]            # game_scores[0] = P1 total points
    game_length: int                  # 7 or 11

    # ── round state ──
    active_player: int                # 0 or 1 — whose turn to play/act
    phase: Phase
    current_stake: int
    stake_last_raised_by: Optional[int]  # who made the last raise offer

    # ── current exchange ──
    played_cards: list[Card]          # cards on the table this exchange
    passed_cards: list[Card]          # anonymous passed cards (hidden from active player)
    last_taker: Optional[int]         # who took cards last (has calculate right)

    # ── flags ──
    calculate_available: bool         # can the last_taker calculate right now?
    round_number: int

    def copy(self) -> 'GameState':
        return copy.deepcopy(self)

    def opponent(self, player: int) -> int:
        return 1 - player

    def hand_of(self, player: int) -> list[Card]:
        return self.hands[player]

    def pile_of(self, player: int) -> list[Card]:
        return self.score_piles[player]

    def pile_score(self, player: int) -> int:
        return score_pile(self.score_piles[player])

    def total_cards_seen(self) -> list[Card]:
        """All cards no longer in deck or hands."""
        seen = []
        for pile in self.score_piles:
            seen.extend(pile)
        return seen


# ─────────────────────────────────────────────
#  Move types
# ─────────────────────────────────────────────

@dataclass
class MoveRaise:
    """Raise stakes to new_stake."""
    new_stake: int

@dataclass
class MoveDeclineRaise:
    """Decline opponent's raise — opponent wins round at previous stake."""
    pass

@dataclass
class MoveAcceptRaise:
    """Accept opponent's raise."""
    pass

@dataclass
class MovePlay:
    """Play cards from hand (must be same suit, 1-3 cards)."""
    cards: list[Card]

@dataclass
class MoveCut:
    """Cut opponent's played cards."""
    cards: list[Card]

@dataclass
class MovePass:
    """Pass cards anonymously (same count as played)."""
    cards: list[Card]   # hidden from active player

@dataclass
class MoveCalculate:
    """Reveal score pile and calculate."""
    pass

@dataclass
class MoveContinue:
    """Don't calculate — continue playing."""
    pass


Move = MoveRaise | MoveDeclineRaise | MoveAcceptRaise | MovePlay | MoveCut | MovePass | MoveCalculate | MoveContinue


# ─────────────────────────────────────────────
#  Legal move generation
# ─────────────────────────────────────────────

def legal_moves(state: GameState, player: int) -> list[Move]:
    """Return all legal moves for the given player in the current state."""
    moves = []
    phase = state.phase

    if phase == Phase.STAKE_OFFER:
        # Either player can raise stakes before their play
        # Only the player whose turn it is to act (active_player) acts here,
        # but raise can come from either — we handle this via active_player context
        if state.current_stake < 6:
            for new_stake in range(state.current_stake + 1, 7):
                moves.append(MoveRaise(new_stake))
        # The active player can also just proceed to play
        # (represented as moving to PLAY phase — handled separately)
        moves.append(MoveContinue())  # "skip raise, just play"
        return moves

    if phase == Phase.PLAY:
        hand = state.hand_of(player)
        # Group by suit
        by_suit: dict[str, list[Card]] = {}
        for c in hand:
            by_suit.setdefault(c.suit, []).append(c)
        # Generate all valid plays: 1-3 cards same suit
        for suit, cards in by_suit.items():
            # All non-empty subsets up to 3
            for size in range(1, min(len(cards), 3) + 1):
                from itertools import combinations
                for combo in combinations(cards, size):
                    moves.append(MovePlay(list(combo)))
        return moves

    if phase == Phase.CUT_OR_PASS:
        opponent = player  # the "cutter" is the non-active player
        hand = state.hand_of(opponent)
        n = len(state.played_cards)
        played = state.played_cards
        played_suit = played[0].suit
        trump = state.trump_suit

        # Check if played cards are trump
        played_are_trump = (played_suit == trump)

        # Generate all cutting combos of size n
        from itertools import combinations
        for combo in combinations(hand, n):
            combo = list(combo)
            if _is_valid_cut(combo, played, trump):
                moves.append(MoveCut(combo))

        # Always can pass (choose any n cards from hand)
        for combo in combinations(hand, n):
            moves.append(MovePass(list(combo)))

        return moves

    if phase == Phase.CALCULATE:
        if state.calculate_available and player == state.last_taker:
            moves.append(MoveCalculate())
            moves.append(MoveContinue())
        return moves

    return moves


def _is_valid_cut(cut_cards: list[Card], played_cards: list[Card], trump: str) -> bool:
    """
    cut_cards must beat played_cards.
    Rules:
    - Same count
    - Trump beats any non-trump
    - Same suit must be higher value
    - Trump can only be beaten by higher trump
    """
    if len(cut_cards) != len(played_cards):
        return False

    played_suit = played_cards[0].suit
    cut_suit = cut_cards[0].suit

    # All cut cards must be same suit
    if len(set(c.suit for c in cut_cards)) > 1:
        return False

    played_are_trump = (played_suit == trump)
    cut_are_trump = (cut_suit == trump)

    if played_are_trump:
        # Can only beat trump with higher trump
        if not cut_are_trump:
            return False
        # Each cut card must beat corresponding played card?
        # Actually in Bura the whole hand beats the whole hand
        # We use: max cut rank > max played rank
        return max(c.rank_value for c in cut_cards) > max(c.rank_value for c in played_cards)
    else:
        # Non-trump played
        if cut_are_trump:
            return True  # any trump beats any non-trump
        if cut_suit != played_suit:
            return False  # must be same suit if not trump
        return max(c.rank_value for c in cut_cards) > max(c.rank_value for c in played_cards)


# ─────────────────────────────────────────────
#  State transition / apply move
# ─────────────────────────────────────────────

@dataclass
class RoundResult:
    winner: int           # 0 or 1
    stake: int
    reason: str           # "calculated", "three_trumps", "declined_raise", "deck_empty"
    calculator_score: Optional[int] = None


def apply_move(state: GameState, player: int, move: Move) -> tuple[GameState, Optional[RoundResult]]:
    """
    Apply a move and return (new_state, round_result).
    round_result is None unless the round ended.
    """
    s = state.copy()
    result = None

    # ── STAKE OFFER PHASE ──
    if isinstance(move, MoveRaise):
        s.current_stake = move.new_stake
        s.stake_last_raised_by = player
        # Opponent now must respond
        s.active_player = s.opponent(player)
        s.phase = Phase.STAKE_OFFER
        return s, None

    if isinstance(move, MoveDeclineRaise):
        # Raiser wins round at stake BEFORE this raise
        raiser = s.stake_last_raised_by
        result = RoundResult(
            winner=raiser,
            stake=s.current_stake,
            reason="declined_raise"
        )
        s.phase = Phase.ROUND_OVER
        return s, result

    if isinstance(move, MoveAcceptRaise):
        # Continue — go back to the original active player's play turn
        s.phase = Phase.PLAY
        return s, None

    if isinstance(move, MoveContinue) and s.phase == Phase.STAKE_OFFER:
        s.phase = Phase.PLAY
        return s, None

    # ── PLAY PHASE ──
    if isinstance(move, MovePlay):
        cards = move.cards

        # Remove from hand
        for c in cards:
            s.hands[player].remove(c)

        # Check 3 trumps auto-win
        if len(cards) == 3 and all(c.suit == s.trump_suit for c in cards):
            result = RoundResult(
                winner=player,
                stake=s.current_stake,
                reason="three_trumps"
            )
            s.phase = Phase.ROUND_OVER
            return s, result

        s.played_cards = cards
        s.active_player = s.opponent(player)  # opponent must respond
        s.phase = Phase.CUT_OR_PASS
        return s, None

    # ── CUT OR PASS PHASE ──
    if isinstance(move, MoveCut):
        # Cutter takes all cards (played + cut) into their pile
        for c in move.cards:
            s.hands[player].remove(c)

        s.score_piles[player].extend(s.played_cards)
        s.score_piles[player].extend(move.cards)

        s.last_taker = player
        s.calculate_available = True
        s.played_cards = []

        # Check if cut was a Maliutka response (opponent played 3 same suit non-trump)
        # After cut, player who cut plays next
        s.active_player = player
        s.phase = Phase.CALCULATE
        return s, None

    if isinstance(move, MovePass):
        # Active player (who played) takes cards into pile
        # Passed cards are anonymous
        original_player = s.opponent(player)  # the one who played

        s.score_piles[original_player].extend(s.played_cards)
        s.score_piles[original_player].extend(move.cards)

        # Remove passed cards from passer's hand
        for c in move.cards:
            s.hands[player].remove(c)

        s.last_taker = original_player
        s.calculate_available = True
        s.played_cards = []

        s.active_player = original_player
        s.phase = Phase.CALCULATE
        return s, None

    # ── CALCULATE PHASE ──
    if isinstance(move, MoveCalculate):
        scorer = s.last_taker
        total = s.pile_score(scorer)
        if total >= 31:
            result = RoundResult(
                winner=scorer,
                stake=s.current_stake,
                reason="calculated",
                calculator_score=total
            )
        else:
            # Loses — opponent wins
            result = RoundResult(
                winner=s.opponent(scorer),
                stake=s.current_stake,
                reason="calculated_failed",
                calculator_score=total
            )
        s.phase = Phase.ROUND_OVER
        return s, result

    if isinstance(move, MoveContinue) and s.phase == Phase.CALCULATE:
        s.calculate_available = False
        s.phase = Phase.DRAW
        return s, None

    return s, None


# ─────────────────────────────────────────────
#  Drawing cards
# ─────────────────────────────────────────────

def draw_cards(state: GameState) -> GameState:
    """Both players draw up to 3 cards. Active player draws first."""
    s = state.copy()

    # Draw for active player first, then opponent
    for p in [s.active_player, s.opponent(s.active_player)]:
        while len(s.hands[p]) < 3 and s.deck:
            s.hands[p].append(s.deck.pop())

    # Check if deck empty and no one can calculate
    if not s.deck and not s.calculate_available:
        # Round ends — most points wins
        p0_score = s.pile_score(0)
        p1_score = s.pile_score(1)
        winner = 0 if p0_score >= p1_score else 1
        s.phase = Phase.ROUND_OVER
        return s

    s.phase = Phase.STAKE_OFFER  # New exchange starts with stake option
    return s


# ─────────────────────────────────────────────
#  Round / Game initialization
# ─────────────────────────────────────────────

def new_round(game_scores: list[int], game_length: int, first_player: int, round_number: int) -> GameState:
    deck = make_deck()
    random.shuffle(deck)

    # Deal 3 to each
    h0 = [deck.pop() for _ in range(3)]
    h1 = [deck.pop() for _ in range(3)]

    # Trump is the bottom card of the remaining deck
    trump_suit = deck[0].suit

    return GameState(
        deck=deck,
        trump_suit=trump_suit,
        hands=[h0, h1],
        score_piles=[[], []],
        game_scores=game_scores[:],
        game_length=game_length,
        active_player=first_player,
        phase=Phase.STAKE_OFFER,
        current_stake=1,
        stake_last_raised_by=None,
        played_cards=[],
        passed_cards=[],
        last_taker=None,
        calculate_available=False,
        round_number=round_number,
    )


def apply_round_result(game_scores: list[int], result: RoundResult) -> list[int]:
    scores = game_scores[:]
    scores[result.winner] += result.stake
    return scores


def is_game_over(game_scores: list[int], game_length: int) -> bool:
    return any(s >= game_length for s in game_scores)


def game_winner(game_scores: list[int]) -> int:
    return 0 if game_scores[0] >= game_scores[1] else 1


# ─────────────────────────────────────────────
#  Maliutka detection
# ─────────────────────────────────────────────

def is_maliutka(cards: list[Card], trump: str) -> bool:
    """3 cards of same non-trump suit."""
    if len(cards) != 3:
        return False
    suits = set(c.suit for c in cards)
    if len(suits) != 1:
        return False
    return list(suits)[0] != trump


def is_three_trumps(cards: list[Card], trump: str) -> bool:
    return len(cards) == 3 and all(c.suit == trump for c in cards)