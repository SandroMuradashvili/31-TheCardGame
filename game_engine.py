"""
BURA (Thirty-One / Cutter) Game Engine  v2
Pure game logic — no Flask, no I/O.
"""

import random
from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional
from itertools import combinations


# ─────────────────────────────────────────────
# ENUMS & CONSTANTS
# ─────────────────────────────────────────────

class Suit(str, Enum):
    HEARTS   = "hearts"
    DIAMONDS = "diamonds"
    CLUBS    = "clubs"
    SPADES   = "spades"

class Rank(str, Enum):
    ACE   = "A"
    TEN   = "T"
    KING  = "K"
    QUEEN = "Q"
    JACK  = "J"

POINT_VALUES = {
    Rank.ACE:   11,
    Rank.TEN:   10,
    Rank.KING:  4,
    Rank.QUEEN: 3,
    Rank.JACK:  2,
}

RANK_ORDER = [Rank.JACK, Rank.QUEEN, Rank.KING, Rank.TEN, Rank.ACE]

class GamePhase(str, Enum):
    WAITING      = "waiting"
    STAKES       = "stakes"
    PLAYING      = "playing"
    CUTTING      = "cutting"
    FORCED_CUT   = "forced_cut"
    CALCULATING  = "calculating"
    ROUND_OVER   = "round_over"
    GAME_OVER    = "game_over"

class RoundEndReason(str, Enum):
    CALCULATED_WIN  = "calculated_win"
    CALCULATED_LOSE = "calculated_lose"
    THREE_TRUMPS    = "three_trumps"
    STAKE_DECLINED  = "stake_declined"
    DECK_EXHAUSTED  = "deck_exhausted"


# ─────────────────────────────────────────────
# CARD
# ─────────────────────────────────────────────

class Card:
    def __init__(self, suit: Suit, rank: Rank):
        self.suit = suit
        self.rank = rank

    @property
    def points(self) -> int:
        return POINT_VALUES[self.rank]

    def __repr__(self):
        return f"{self.rank.value}{self.suit.value[0].upper()}"

    def to_dict(self) -> dict:
        return {"suit": self.suit.value, "rank": self.rank.value, "points": self.points, "id": repr(self)}

    def __eq__(self, other):
        return isinstance(other, Card) and self.suit == other.suit and self.rank == other.rank

    def __hash__(self):
        return hash((self.suit, self.rank))


# ─────────────────────────────────────────────
# DECK
# ─────────────────────────────────────────────

class Deck:
    def __init__(self):
        self.cards: list[Card] = []
        self.trump_card: Optional[Card] = None
        self.cards = [Card(suit, rank) for suit in Suit for rank in Rank]

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self, n: int) -> list[Card]:
        dealt = self.cards[:n]
        self.cards = self.cards[n:]
        return dealt

    def reveal_trump(self):
        if self.cards:
            self.trump_card = self.cards[-1]
        return self.trump_card

    @property
    def trump_suit(self) -> Optional[Suit]:
        return self.trump_card.suit if self.trump_card else None

    def __len__(self):
        return len(self.cards)

    def to_dict(self, debug=False) -> dict:
        return {
            "size": len(self.cards),
            "trump_card": self.trump_card.to_dict() if self.trump_card else None,
            "trump_suit": self.trump_suit.value if self.trump_suit else None,
            "cards": [c.to_dict() for c in self.cards] if debug else [],
        }


# ─────────────────────────────────────────────
# PLAYER
# ─────────────────────────────────────────────

class Player(ABC):
    def __init__(self, player_id: str, name: str):
        self.player_id = player_id
        self.name = name
        self.hand: list[Card] = []
        self.score_pile: list[Card] = []   # Cards whose faces are known to player
        self.hidden_pile: list[Card] = []  # Cards passed by opponent — content unknown
        self.game_score: int = 0

    @property
    def known_pile_points(self) -> int:
        return sum(c.points for c in self.score_pile)

    @property
    def hidden_card_count(self) -> int:
        return len(self.hidden_pile)

    @property
    def pile_points(self) -> int:
        """True total — used only internally / debug / at calculate reveal."""
        return sum(c.points for c in self.score_pile) + sum(c.points for c in self.hidden_pile)

    @property
    def pile_count(self) -> int:
        return len(self.score_pile) + len(self.hidden_pile)

    def draw_to_three(self, deck: Deck):
        need = 3 - len(self.hand)
        if need > 0 and len(deck) > 0:
            self.hand.extend(deck.deal(min(need, len(deck))))

    def add_to_pile(self, cards: list[Card], hidden: bool = False):
        if hidden:
            self.hidden_pile.extend(cards)
        else:
            self.score_pile.extend(cards)

    def remove_from_hand(self, cards: list[Card]):
        for c in cards:
            self.hand.remove(c)

    def reset_round(self):
        self.hand = []
        self.score_pile = []
        self.hidden_pile = []

    def to_dict(self, reveal_hand=False, debug=False) -> dict:
        return {
            "id": self.player_id,
            "name": self.name,
            "hand": [c.to_dict() for c in self.hand] if reveal_hand else [{"hidden": True} for _ in self.hand],
            "hand_count": len(self.hand),
            # What the owner is allowed to know (no cheating):
            "known_pile_points": self.known_pile_points,
            "hidden_card_count": self.hidden_card_count,
            "hidden_min_points": self.hidden_card_count * POINT_VALUES[Rank.JACK],
            "hidden_max_points": self.hidden_card_count * POINT_VALUES[Rank.ACE],
            "pile_count": self.pile_count,
            # True total only in debug / at calculate time:
            "pile_points": self.pile_points if debug else None,
            "pile_cards": [c.to_dict() for c in self.score_pile] if debug else [],
            "hidden_pile_cards": [c.to_dict() for c in self.hidden_pile] if debug else [],
            "game_score": self.game_score,
            "type": self.__class__.__name__,
        }

    @abstractmethod
    def is_human(self) -> bool:
        pass


class HumanPlayer(Player):
    def is_human(self) -> bool:
        return True


class BotPlayer(Player):
    def is_human(self) -> bool:
        return False

    def choose_play(self, engine) -> list[str]:
        raise NotImplementedError

    def choose_cut_or_pass(self, engine):
        raise NotImplementedError

    def choose_calculate(self, engine) -> bool:
        raise NotImplementedError

    def choose_raise_stake(self, engine) -> bool:
        """Return True to raise stake by 1."""
        raise NotImplementedError


# ─────────────────────────────────────────────
# MOVE LOG
# ─────────────────────────────────────────────

class MoveRecord:
    def __init__(self, move_type: str, player_id: str, data: dict):
        self.move_type = move_type
        self.player_id = player_id
        self.data = data

    def to_dict(self):
        return {"type": self.move_type, "player": self.player_id, "data": self.data}


# ─────────────────────────────────────────────
# GAME ENGINE
# ─────────────────────────────────────────────

class GameEngine:
    """
    STAKE RULES (corrected vs v1):
    - Stake starts at 1. Max is 6.
    - Either player can raise by exactly 1 at any point during STAKES phase.
    - Once player A raises, player B must respond: accept / decline / raise-further.
    - Player A cannot raise again until player B has acted.
    - Playing cards directly (without pressing "raise") = implicitly no raise wanted.
    - play_cards() auto-transitions from STAKES → PLAYING (accepting any pending stake).
    """

    def __init__(self, player1: Player, player2: Player, target_score: int = 7):
        self.players: list[Player] = [player1, player2]
        self.target_score = target_score
        self.deck = Deck()
        self.phase = GamePhase.WAITING
        self.current_stake = 1
        self.pending_stake = 1
        self.stake_offerer_idx: Optional[int] = None
        self.active_idx: int = 0
        self.calculator_idx: Optional[int] = None
        self.played_cards: list[Card] = []
        self.playing_player_idx: Optional[int] = None
        self.last_round_winner_idx: Optional[int] = None
        self.round_number: int = 0
        self.move_history: list[MoveRecord] = []
        self.round_end_reason: Optional[RoundEndReason] = None
        self.round_winner_idx: Optional[int] = None
        self.is_maliutka: bool = False
        self.error: Optional[str] = None

    @property
    def active_player(self) -> Player:
        return self.players[self.active_idx]

    @property
    def opponent_idx(self) -> int:
        return 1 - self.active_idx

    @property
    def trump_suit(self) -> Optional[Suit]:
        return self.deck.trump_suit

    def _log(self, move_type: str, player_id: str, data: dict):
        self.move_history.append(MoveRecord(move_type, player_id, data))

    def _cards_from_ids(self, player: Player, card_ids: list[str]) -> list[Card]:
        result = []
        for cid in card_ids:
            found = next((c for c in player.hand if repr(c) == cid), None)
            if found is None:
                return []
            result.append(found)
        return result

    # ── validation ────────────────────────────

    def _validate_play(self, cards: list[Card]) -> Optional[str]:
        if not cards:
            return "Must play at least 1 card."
        if len(cards) > 3:
            return "Cannot play more than 3 cards."
        if len({c.suit for c in cards}) > 1:
            return "All played cards must be the same suit."
        return None

    def _can_cut(self, played: list[Card], cut_attempt: list[Card]) -> bool:
        if len(cut_attempt) != len(played):
            return False
        if len({c.suit for c in cut_attempt}) > 1:
            return False
        played_suit = played[0].suit
        cut_suit = cut_attempt[0].suit
        played_is_trump = played_suit == self.trump_suit
        cut_is_trump = cut_suit == self.trump_suit
        played_total = sum(c.points for c in played)
        cut_total = sum(c.points for c in cut_attempt)
        if played_is_trump:
            return cut_is_trump and cut_total > played_total
        if cut_suit == played_suit:
            return cut_total > played_total
        return cut_is_trump  # trump beats non-trump

    def _is_three_trumps(self, cards: list[Card]) -> bool:
        return len(cards) == 3 and self.trump_suit is not None and all(c.suit == self.trump_suit for c in cards)

    def _is_maliutka(self, cards: list[Card]) -> bool:
        if len(cards) != 3:
            return False
        suits = {c.suit for c in cards}
        return len(suits) == 1 and list(suits)[0] != self.trump_suit

    # ── round lifecycle ───────────────────────

    def start_round(self):
        self.round_number += 1
        self.current_stake = 1
        self.pending_stake = 1
        self.stake_offerer_idx = None
        self.played_cards = []
        self.playing_player_idx = None
        self.calculator_idx = None
        self.round_end_reason = None
        self.round_winner_idx = None
        self.is_maliutka = False
        self.error = None

        for p in self.players:
            p.reset_round()

        self.deck = Deck()
        self.deck.shuffle()
        for p in self.players:
            p.hand = self.deck.deal(3)
        self.deck.reveal_trump()

        self.active_idx = self.last_round_winner_idx if self.last_round_winner_idx is not None else random.randint(0, 1)
        self.phase = GamePhase.STAKES
        self._log("round_start", "system", {
            "round": self.round_number,
            "trump": self.deck.trump_suit.value if self.deck.trump_suit else None,
            "first_player": self.active_player.player_id,
        })

    # ── STAKES ───────────────────────────────

    def can_raise_stake(self, player_idx: int) -> bool:
        """True if this player is currently allowed to raise the stake."""
        if self.phase != GamePhase.STAKES:
            return False
        if self.current_stake >= 6:
            return False
        # Can't raise if you just raised (must wait for opponent to respond)
        if self.stake_offerer_idx is not None and self.stake_offerer_idx == player_idx:
            return False
        # Can't raise beyond 6
        base = self.pending_stake if self.stake_offerer_idx is not None else self.current_stake
        return base + 1 <= 6

    def offer_stake(self, player_idx: int) -> bool:
        """Raise stake by exactly 1. Returns True on success."""
        self.error = None
        if not self.can_raise_stake(player_idx):
            self.error = "You cannot raise the stake right now."
            return False
        base = self.pending_stake if self.stake_offerer_idx is not None else self.current_stake
        self.pending_stake = base + 1
        self.stake_offerer_idx = player_idx
        self._log("stake_offer", self.players[player_idx].player_id, {"stake": self.pending_stake})
        return True

    def accept_stake(self, player_idx: int) -> bool:
        self.error = None
        if self.stake_offerer_idx is None:
            self.error = "No stake offer pending."
            return False
        if player_idx == self.stake_offerer_idx:
            self.error = "Cannot accept your own offer."
            return False
        self.current_stake = self.pending_stake
        self.stake_offerer_idx = None
        self._log("stake_accept", self.players[player_idx].player_id, {"stake": self.current_stake})
        return True

    def decline_stake(self, player_idx: int) -> bool:
        """Offerer wins at the stake BEFORE their offer."""
        self.error = None
        if self.stake_offerer_idx is None:
            self.error = "No stake offer pending."
            return False
        if player_idx == self.stake_offerer_idx:
            self.error = "Cannot decline your own offer."
            return False
        winner_idx = self.stake_offerer_idx
        win_stake = self.current_stake  # previous stake, not pending
        self._log("stake_decline", self.players[player_idx].player_id, {
            "declined_stake": self.pending_stake,
            "win_stake": win_stake,
            "winner": self.players[winner_idx].player_id
        })
        self._end_round(winner_idx, win_stake, RoundEndReason.STAKE_DECLINED)
        return True

    def start_play(self) -> bool:
        """Transition STAKES → PLAYING. Silently accepts pending stake if any."""
        self.error = None
        if self.phase == GamePhase.PLAYING:
            return True
        if self.phase != GamePhase.STAKES:
            self.error = "Not in stakes phase."
            return False
        if self.stake_offerer_idx is not None:
            self.current_stake = self.pending_stake
            self.stake_offerer_idx = None
        self.phase = GamePhase.PLAYING
        self._log("play_start", "system", {"stake": self.current_stake})
        return True

    # ── PLAYING ──────────────────────────────

    def play_cards(self, player_idx: int, card_ids: list[str]) -> bool:
        """Play 1-3 same-suit cards. Auto-starts play if still in STAKES."""
        self.error = None
        if self.phase == GamePhase.STAKES:
            self.start_play()
        if self.phase != GamePhase.PLAYING:
            self.error = "Not in playing phase."
            return False
        if player_idx != self.active_idx:
            self.error = "Not your turn to play."
            return False

        player = self.players[player_idx]
        cards = self._cards_from_ids(player, card_ids)
        if not cards:
            self.error = "Invalid card selection."
            return False
        err = self._validate_play(cards)
        if err:
            self.error = err
            return False

        if self._is_three_trumps(cards):
            player.remove_from_hand(cards)
            player.add_to_pile(cards)
            self._log("three_trumps", player.player_id, {"cards": [repr(c) for c in cards]})
            self._end_round(player_idx, self.current_stake, RoundEndReason.THREE_TRUMPS)
            return True

        self.is_maliutka = self._is_maliutka(cards)
        player.remove_from_hand(cards)
        self.played_cards = cards
        self.playing_player_idx = player_idx
        self._log("play", player.player_id, {
            "cards": [repr(c) for c in cards],
            "is_maliutka": self.is_maliutka
        })
        self.phase = GamePhase.FORCED_CUT if self.is_maliutka else GamePhase.CUTTING
        return True

    # ── CUTTING ──────────────────────────────

    def cut_cards(self, player_idx: int, card_ids: list[str]) -> bool:
        self.error = None
        if self.phase not in (GamePhase.CUTTING, GamePhase.FORCED_CUT):
            self.error = "Not in cutting phase."
            return False
        if player_idx == self.playing_player_idx:
            self.error = "Cannot cut your own cards."
            return False
        cutter = self.players[player_idx]
        cut_cards = self._cards_from_ids(cutter, card_ids)
        if not cut_cards:
            self.error = "Invalid card selection."
            return False
        if not self._can_cut(self.played_cards, cut_cards):
            self.error = "These cards cannot beat the played cards."
            return False

        cutter.remove_from_hand(cut_cards)
        cutter.add_to_pile(self.played_cards + cut_cards, hidden=False)
        self.calculator_idx = player_idx
        self.active_idx = player_idx
        self._log("cut", cutter.player_id, {
            "cut_cards": [repr(c) for c in cut_cards],
            "played_cards": [repr(c) for c in self.played_cards],
        })
        self.played_cards = []
        self.is_maliutka = False
        self.phase = GamePhase.CALCULATING
        return True

    def pass_cards(self, player_idx: int, card_ids: list[str]) -> bool:
        self.error = None
        if self.phase not in (GamePhase.CUTTING, GamePhase.FORCED_CUT):
            self.error = "Not in cutting phase."
            return False
        if player_idx == self.playing_player_idx:
            self.error = "Cannot pass your own cards."
            return False
        passer = self.players[player_idx]
        pass_list = self._cards_from_ids(passer, card_ids)
        if not pass_list:
            self.error = "Invalid card selection."
            return False
        if len(pass_list) != len(self.played_cards):
            self.error = f"Must pass exactly {len(self.played_cards)} card(s)."
            return False

        passer.remove_from_hand(pass_list)
        playing_player = self.players[self.playing_player_idx]
        playing_player.add_to_pile(self.played_cards, hidden=False)  # own cards back — known
        playing_player.add_to_pile(pass_list, hidden=True)            # opponent's — hidden
        self.calculator_idx = self.playing_player_idx
        self.active_idx = self.playing_player_idx
        self._log("pass", passer.player_id, {
            "passed_count": len(pass_list),
        })
        self.played_cards = []
        self.is_maliutka = False
        self.phase = GamePhase.CALCULATING
        return True

    # ── CALCULATING ──────────────────────────

    def calculate(self, player_idx: int) -> bool:
        self.error = None
        if self.phase != GamePhase.CALCULATING:
            self.error = "Not in calculating phase."
            return False
        if player_idx != self.calculator_idx:
            self.error = "You don't have the right to calculate."
            return False
        total = self.players[player_idx].pile_points
        self._log("calculate", self.players[player_idx].player_id, {
            "total": total,
            "win": total >= 31,
        })
        if total >= 31:
            self._end_round(player_idx, self.current_stake, RoundEndReason.CALCULATED_WIN)
        else:
            self._end_round(1 - player_idx, self.current_stake, RoundEndReason.CALCULATED_LOSE)
        return True

    def skip_calculate(self, player_idx: int) -> bool:
        self.error = None
        if self.phase != GamePhase.CALCULATING:
            self.error = "Not in calculating phase."
            return False
        if player_idx != self.calculator_idx:
            self.error = "You don't have the right to calculate."
            return False
        self._log("skip_calculate", self.players[player_idx].player_id, {})
        self._draw_phase()
        return True

    # ── DRAW ─────────────────────────────────

    def _draw_phase(self):
        self.players[self.active_idx].draw_to_three(self.deck)
        self.players[1 - self.active_idx].draw_to_three(self.deck)
        self._log("draw", "system", {"deck_remaining": len(self.deck)})
        if len(self.deck) == 0 and (len(self.players[0].hand) == 0 or len(self.players[1].hand) == 0):
            self._resolve_deck_exhausted()
        else:
            self.phase = GamePhase.PLAYING

    def _resolve_deck_exhausted(self):
        p0, p1 = self.players[0].pile_points, self.players[1].pile_points
        self._log("deck_exhausted", "system", {"p0": p0, "p1": p1})
        self._end_round(0 if p0 >= p1 else 1, self.current_stake, RoundEndReason.DECK_EXHAUSTED)

    # ── ROUND END ────────────────────────────

    def _end_round(self, winner_idx: int, stake: int, reason: RoundEndReason):
        self.round_winner_idx = winner_idx
        self.round_end_reason = reason
        self.last_round_winner_idx = winner_idx
        self.players[winner_idx].game_score += stake
        self._log("round_end", "system", {
            "winner": self.players[winner_idx].player_id,
            "stake": stake,
            "reason": reason.value,
            "scores": {p.player_id: p.game_score for p in self.players}
        })
        if self.players[winner_idx].game_score >= self.target_score:
            self.phase = GamePhase.GAME_OVER
        else:
            self.phase = GamePhase.ROUND_OVER

    # ── HELPERS ──────────────────────────────

    def get_valid_cuts(self, player_idx: int) -> list[list[str]]:
        player = self.players[player_idx]
        n = len(self.played_cards)
        valid = []
        for combo in combinations(player.hand, n):
            combo_list = list(combo)
            if len({c.suit for c in combo_list}) == 1 and self._can_cut(self.played_cards, combo_list):
                valid.append([repr(c) for c in combo_list])
        return valid

    def compute_tip(self, player_idx: int) -> dict:
        """Min/max score estimate for player — what they can know about their pile."""
        p = self.players[player_idx]
        known = p.known_pile_points
        hc = p.hidden_card_count
        min_total = known + hc * POINT_VALUES[Rank.JACK]
        max_total = known + hc * POINT_VALUES[Rank.ACE]
        return {
            "known_points": known,
            "hidden_count": hc,
            "min_total": min_total,
            "max_total": max_total,
            "can_possibly_win": max_total >= 31,
            "guaranteed_win": min_total >= 31,
        }

    # ── STATE SNAPSHOT ────────────────────────

    def get_state(self, perspective_idx: Optional[int] = None, debug: bool = False) -> dict:
        player_states = []
        for i, p in enumerate(self.players):
            reveal_hand = debug or (perspective_idx is None) or (i == perspective_idx)
            player_states.append(p.to_dict(reveal_hand=reveal_hand, debug=debug))

        valid_cuts = []
        if self.phase in (GamePhase.CUTTING, GamePhase.FORCED_CUT) and self.playing_player_idx is not None:
            valid_cuts = self.get_valid_cuts(1 - self.playing_player_idx)

        tip = None
        if perspective_idx is not None and self.phase == GamePhase.CALCULATING:
            if self.calculator_idx == perspective_idx:
                tip = self.compute_tip(perspective_idx)

        return {
            "phase": self.phase.value,
            "round_number": self.round_number,
            "active_player_idx": self.active_idx,
            "active_player_id": self.active_player.player_id,
            "calculator_idx": self.calculator_idx,
            "calculator_id": self.players[self.calculator_idx].player_id if self.calculator_idx is not None else None,
            "playing_player_idx": self.playing_player_idx,
            "current_stake": self.current_stake,
            "pending_stake": self.pending_stake,
            "stake_offerer_idx": self.stake_offerer_idx,
            "stake_offerer_id": self.players[self.stake_offerer_idx].player_id if self.stake_offerer_idx is not None else None,
            "can_raise_stake": [self.can_raise_stake(i) for i in range(2)],
            "played_cards": [c.to_dict() for c in self.played_cards],
            "is_maliutka": self.is_maliutka,
            "deck": self.deck.to_dict(debug=debug),
            "trump_suit": self.trump_suit.value if self.trump_suit else None,
            "trump_card": self.deck.trump_card.to_dict() if self.deck.trump_card else None,
            "players": player_states,
            "target_score": self.target_score,
            "round_winner_idx": self.round_winner_idx,
            "round_end_reason": self.round_end_reason.value if self.round_end_reason else None,
            "move_history": [m.to_dict() for m in self.move_history[-30:]],
            "error": self.error,
            "valid_cuts": valid_cuts,
            "tip": tip,
        }