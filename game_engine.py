"""
BURA (Thirty-One / Cutter) Game Engine
Pure game logic — no Flask, no I/O. Easily adaptable for AI bots, tournaments, multiplayer.
"""

import random
from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional
import copy


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

# Rank order for cutting (higher index = higher value)
RANK_ORDER = [Rank.JACK, Rank.QUEEN, Rank.KING, Rank.TEN, Rank.ACE]

class GamePhase(str, Enum):
    WAITING         = "waiting"          # Between rounds or at start
    STAKES          = "stakes"           # Stake negotiation phase
    PLAYING         = "playing"          # Active player plays cards
    CUTTING         = "cutting"          # Opponent decides to cut or pass
    FORCED_CUT      = "forced_cut"       # Maliutka — opponent MUST try to cut
    CALCULATING     = "calculating"      # Right-to-calculate moment
    ROUND_OVER      = "round_over"       # Round just ended, show result
    GAME_OVER       = "game_over"        # Game complete

class RoundEndReason(str, Enum):
    CALCULATED_WIN      = "calculated_win"
    CALCULATED_LOSE     = "calculated_lose"
    THREE_TRUMPS        = "three_trumps"
    STAKE_DECLINED      = "stake_declined"
    DECK_EXHAUSTED      = "deck_exhausted"
    MALIUTKA_CUT        = "maliutka_cut"
    MALIUTKA_PASS       = "maliutka_pass"


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

    @property
    def rank_index(self) -> int:
        return RANK_ORDER.index(self.rank)

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
        self._build()

    def _build(self):
        self.cards = [Card(suit, rank) for suit in Suit for rank in Rank]

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self, n: int) -> list[Card]:
        dealt = self.cards[:n]
        self.cards = self.cards[n:]
        return dealt

    def reveal_trump(self):
        """Place the bottom card face up; it stays in deck until drawn naturally."""
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
# PLAYER (abstract — supports Human & Bot)
# ─────────────────────────────────────────────

class Player(ABC):
    def __init__(self, player_id: str, name: str):
        self.player_id = player_id
        self.name = name
        self.hand: list[Card] = []
        self.score_pile: list[Card] = []      # Cards they've won
        self.hidden_pile: list[Card] = []     # Cards passed to them (unknown origin)
        self.game_score: int = 0              # Running total toward 7/11

    @property
    def pile_points(self) -> int:
        return sum(c.points for c in self.score_pile) + sum(c.points for c in self.hidden_pile)

    def draw_to_three(self, deck: Deck):
        need = 3 - len(self.hand)
        if need > 0 and len(deck) > 0:
            drawn = deck.deal(min(need, len(deck)))
            self.hand.extend(drawn)

    def add_to_pile(self, cards: list[Card], hidden: bool = False):
        if hidden:
            self.hidden_pile.extend(cards)
        else:
            self.score_pile.extend(cards)

    def remove_from_hand(self, cards: list[Card]) -> bool:
        for card in cards:
            if card not in self.hand:
                return False
        for card in cards:
            self.hand.remove(card)
        return True

    def reset_round(self):
        self.hand = []
        self.score_pile = []
        self.hidden_pile = []

    def to_dict(self, reveal_hand=False, reveal_pile=True) -> dict:
        return {
            "id": self.player_id,
            "name": self.name,
            "hand": [c.to_dict() for c in self.hand] if reveal_hand else [{"hidden": True} for _ in self.hand],
            "hand_count": len(self.hand),
            "pile_points": self.pile_points if reveal_pile else None,
            "pile_count": len(self.score_pile) + len(self.hidden_pile),
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
    """Stub — subclass and override choose_* methods for AI implementations."""
    def is_human(self) -> bool:
        return False

    def choose_play(self, game_state: dict) -> list[Card]:
        raise NotImplementedError("Bot subclass must implement choose_play")

    def choose_cut_or_pass(self, played_cards: list[Card], game_state: dict):
        raise NotImplementedError("Bot subclass must implement choose_cut_or_pass")

    def choose_calculate(self, game_state: dict) -> bool:
        raise NotImplementedError("Bot subclass must implement choose_calculate")

    def choose_stake(self, current_stake: int, game_state: dict) -> Optional[int]:
        raise NotImplementedError("Bot subclass must implement choose_stake")


# ─────────────────────────────────────────────
# MOVE HISTORY
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
    Pure game logic. No I/O. Thread-safe when used with a single game per instance.

    State machine:
        WAITING → STAKES → PLAYING → CUTTING/FORCED_CUT → CALCULATING → PLAYING … → ROUND_OVER → WAITING
                                                                                              ↓
                                                                                         GAME_OVER
    """

    def __init__(self, player1: Player, player2: Player, target_score: int = 7):
        self.players: list[Player] = [player1, player2]
        self.target_score = target_score
        self.deck = Deck()
        self.phase = GamePhase.WAITING
        self.current_stake = 1
        self.pending_stake = 1           # Stake being offered but not accepted yet
        self.stake_offerer_idx: Optional[int] = None
        self.active_idx: int = 0         # Index of active player
        self.calculator_idx: Optional[int] = None  # Who has right to calculate
        self.played_cards: list[Card] = []  # Cards on the "table"
        self.playing_player_idx: Optional[int] = None  # Who played current cards
        self.last_round_winner_idx: Optional[int] = None
        self.round_number: int = 0
        self.move_history: list[MoveRecord] = []
        self.round_end_reason: Optional[RoundEndReason] = None
        self.round_winner_idx: Optional[int] = None
        self.is_maliutka: bool = False
        self.error: Optional[str] = None  # Last validation error

    # ── helpers ──────────────────────────────

    @property
    def active_player(self) -> Player:
        return self.players[self.active_idx]

    @property
    def opponent_idx(self) -> int:
        return 1 - self.active_idx

    @property
    def opponent(self) -> Player:
        return self.players[self.opponent_idx]

    @property
    def trump_suit(self) -> Optional[Suit]:
        return self.deck.trump_suit

    def _log(self, move_type: str, player_id: str, data: dict):
        self.move_history.append(MoveRecord(move_type, player_id, data))

    def _cards_from_ids(self, player: Player, card_ids: list[str]) -> list[Card]:
        """Resolve card ids like 'AH' back to Card objects from player's hand."""
        result = []
        for cid in card_ids:
            found = next((c for c in player.hand if repr(c) == cid), None)
            if found is None:
                return []
            result.append(found)
        return result

    # ── validation helpers ────────────────────

    def _validate_play(self, cards: list[Card]) -> Optional[str]:
        if not cards:
            return "Must play at least 1 card."
        if len(cards) > 3:
            return "Cannot play more than 3 cards."
        suits = {c.suit for c in cards}
        if len(suits) > 1:
            return "All played cards must be the same suit."
        return None

    def _can_cut(self, played: list[Card], cut_attempt: list[Card]) -> bool:
        """
        Returns True if cut_attempt beats played.
        Rules:
          - Must play same number of cards
          - If played are all non-trump: cut must be same suit with strictly higher TOTAL value
            OR all trump cards (any value beats non-trump)
          - If played are all trump: cut must be all trump AND strictly higher total value
        """
        if len(cut_attempt) != len(played):
            return False

        played_suit = played[0].suit
        cut_suit = cut_attempt[0].suit

        # All cut cards must be same suit
        if len({c.suit for c in cut_attempt}) > 1:
            return False

        played_is_trump = played_suit == self.trump_suit
        cut_is_trump = cut_suit == self.trump_suit

        played_total = sum(c.points for c in played)
        cut_total = sum(c.points for c in cut_attempt)

        if played_is_trump:
            # Only higher trump beats trump
            return cut_is_trump and cut_total > played_total
        else:
            # Same suit beats with higher value
            if cut_suit == played_suit:
                return cut_total > played_total
            # Trump beats any non-trump
            if cut_is_trump:
                return True
            return False

    def _is_three_trumps(self, cards: list[Card]) -> bool:
        return (
            len(cards) == 3
            and self.trump_suit is not None
            and all(c.suit == self.trump_suit for c in cards)
        )

    def _is_maliutka(self, cards: list[Card]) -> bool:
        """3 same-suit non-trump cards."""
        if len(cards) != 3:
            return False
        suits = {c.suit for c in cards}
        if len(suits) != 1:
            return False
        return list(suits)[0] != self.trump_suit

    # ── round lifecycle ───────────────────────

    def start_round(self):
        """Initialize a new round. Call after WAITING phase."""
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

        # Deal 3 cards each
        for p in self.players:
            p.hand = self.deck.deal(3)

        self.deck.reveal_trump()

        # Active player: winner of last round goes first; first round is random
        if self.last_round_winner_idx is not None:
            self.active_idx = self.last_round_winner_idx
        else:
            self.active_idx = random.randint(0, 1)

        self.phase = GamePhase.STAKES
        self._log("round_start", "system", {
            "round": self.round_number,
            "trump": self.deck.trump_suit.value if self.deck.trump_suit else None,
            "first_player": self.active_player.player_id,
        })

    # ── STAKES PHASE ─────────────────────────

    def offer_stake(self, player_idx: int, new_stake: int) -> bool:
        """
        Either player can offer to raise the stake before any cards are played.
        Returns True on success.
        """
        self.error = None
        if self.phase != GamePhase.STAKES:
            self.error = "Cannot raise stakes now."
            return False
        if new_stake <= self.current_stake or new_stake > 6:
            self.error = f"Stake must be between {self.current_stake + 1} and 6."
            return False
        if self.stake_offerer_idx is not None and self.stake_offerer_idx == player_idx:
            self.error = "Cannot raise your own pending offer."
            return False

        self.pending_stake = new_stake
        self.stake_offerer_idx = player_idx
        self._log("stake_offer", self.players[player_idx].player_id, {"stake": new_stake})
        return True

    def accept_stake(self, player_idx: int) -> bool:
        """Accepting player agrees to play at the new stake."""
        self.error = None
        if self.phase != GamePhase.STAKES:
            self.error = "No pending stake to accept."
            return False
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
        """Declining player forfeits the round; offerer wins at previous stake."""
        self.error = None
        if self.phase != GamePhase.STAKES:
            self.error = "No pending stake to decline."
            return False
        if self.stake_offerer_idx is None:
            self.error = "No stake offer pending."
            return False
        if player_idx == self.stake_offerer_idx:
            self.error = "Cannot decline your own offer."
            return False

        winner_idx = self.stake_offerer_idx
        # Win at PREVIOUS (current before the offer) stake
        win_stake = self.current_stake
        self._log("stake_decline", self.players[player_idx].player_id, {
            "stake": win_stake,
            "winner": self.players[winner_idx].player_id
        })
        self._end_round(winner_idx, win_stake, RoundEndReason.STAKE_DECLINED)
        return True

    def start_play(self) -> bool:
        """Transition from STAKES to PLAYING (skip stake negotiation)."""
        self.error = None
        if self.phase != GamePhase.STAKES:
            self.error = "Not in stakes phase."
            return False
        if self.stake_offerer_idx is not None:
            self.error = "Pending stake must be resolved first."
            return False
        self.phase = GamePhase.PLAYING
        self._log("play_start", "system", {"stake": self.current_stake})
        return True

    # ── PLAYING PHASE ────────────────────────

    def play_cards(self, player_idx: int, card_ids: list[str]) -> bool:
        """Active player plays 1-3 cards of same suit."""
        self.error = None
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

        # Check three trumps — instant win
        if self._is_three_trumps(cards):
            player.remove_from_hand(cards)
            player.add_to_pile(cards)
            self._log("three_trumps", player.player_id, {"cards": [repr(c) for c in cards]})
            self._end_round(player_idx, self.current_stake, RoundEndReason.THREE_TRUMPS)
            return True

        # Check Maliutka
        self.is_maliutka = self._is_maliutka(cards)

        player.remove_from_hand(cards)
        self.played_cards = cards
        self.playing_player_idx = player_idx

        self._log("play", player.player_id, {
            "cards": [repr(c) for c in cards],
            "is_maliutka": self.is_maliutka
        })

        if self.is_maliutka:
            self.phase = GamePhase.FORCED_CUT
        else:
            self.phase = GamePhase.CUTTING
        return True

    # ── CUTTING PHASE ────────────────────────

    def cut_cards(self, player_idx: int, card_ids: list[str]) -> bool:
        """
        Opponent attempts to cut.
        In CUTTING phase: optional.
        In FORCED_CUT phase: mandatory attempt (but may fail).
        """
        self.error = None
        if self.phase not in (GamePhase.CUTTING, GamePhase.FORCED_CUT):
            self.error = "Not in cutting phase."
            return False
        if player_idx == self.playing_player_idx:
            self.error = "You cannot cut your own cards."
            return False

        cutter = self.players[player_idx]
        cut_cards = self._cards_from_ids(cutter, card_ids)
        if not cut_cards:
            self.error = "Invalid card selection."
            return False

        if not self._can_cut(self.played_cards, cut_cards):
            self.error = "These cards cannot beat the played cards."
            return False

        # Successful cut
        cutter.remove_from_hand(cut_cards)
        all_cards = self.played_cards + cut_cards
        cutter.add_to_pile(all_cards)
        self.calculator_idx = player_idx
        self.active_idx = player_idx  # Cutter goes first next turn

        self._log("cut", cutter.player_id, {
            "cut_cards": [repr(c) for c in cut_cards],
            "played_cards": [repr(c) for c in self.played_cards],
            "is_maliutka": self.is_maliutka
        })

        self.played_cards = []
        self.is_maliutka = False
        self.phase = GamePhase.CALCULATING
        return True

    def pass_cards(self, player_idx: int, card_ids: list[str]) -> bool:
        """
        Opponent passes cards (cannot or chooses not to cut).
        Not allowed during FORCED_CUT if a valid cut exists — but we leave enforcement
        to the UI/bot; the engine accepts it and the player loses those cards.
        Actually: in Maliutka the opponent MUST attempt to cut. If they physically
        cannot cut (no valid combination), they pass. Engine allows pass always —
        UI should only show pass when no valid cut exists during FORCED_CUT.
        """
        self.error = None
        if self.phase not in (GamePhase.CUTTING, GamePhase.FORCED_CUT):
            self.error = "Not in cutting phase."
            return False
        if player_idx == self.playing_player_idx:
            self.error = "You cannot pass your own cards."
            return False

        passer = self.players[player_idx]
        pass_cards_list = self._cards_from_ids(passer, card_ids)
        if not pass_cards_list:
            self.error = "Invalid card selection."
            return False
        if len(pass_cards_list) != len(self.played_cards):
            self.error = f"Must pass exactly {len(self.played_cards)} card(s)."
            return False

        passer.remove_from_hand(pass_cards_list)

        # Played cards go to playing_player's pile (known), passed cards go hidden
        playing_player = self.players[self.playing_player_idx]
        playing_player.add_to_pile(self.played_cards, hidden=False)
        playing_player.add_to_pile(pass_cards_list, hidden=True)
        self.calculator_idx = self.playing_player_idx
        self.active_idx = self.playing_player_idx  # Original player keeps going

        self._log("pass", passer.player_id, {
            "passed_count": len(pass_cards_list),
            "played_cards": [repr(c) for c in self.played_cards],
        })

        self.played_cards = []
        self.is_maliutka = False
        self.phase = GamePhase.CALCULATING
        return True

    # ── CALCULATING PHASE ────────────────────

    def calculate(self, player_idx: int) -> bool:
        """Player with right to calculate reveals pile."""
        self.error = None
        if self.phase != GamePhase.CALCULATING:
            self.error = "Not in calculating phase."
            return False
        if player_idx != self.calculator_idx:
            self.error = "You don't have the right to calculate."
            return False

        calculator = self.players[player_idx]
        total = calculator.pile_points

        self._log("calculate", calculator.player_id, {
            "total": total,
            "win": total >= 31,
            "pile": [repr(c) for c in calculator.score_pile + calculator.hidden_pile]
        })

        if total >= 31:
            self._end_round(player_idx, self.current_stake, RoundEndReason.CALCULATED_WIN)
        else:
            loser_idx = player_idx
            winner_idx = 1 - loser_idx
            self._end_round(winner_idx, self.current_stake, RoundEndReason.CALCULATED_LOSE)
        return True

    def skip_calculate(self, player_idx: int) -> bool:
        """Pass on calculating — proceed to draw."""
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

    # ── DRAW PHASE ───────────────────────────

    def _draw_phase(self):
        """Both players draw back to 3 cards. Check for deck exhaustion."""
        # Active player draws first (convention)
        self.players[self.active_idx].draw_to_three(self.deck)
        self.players[1 - self.active_idx].draw_to_three(self.deck)

        self._log("draw", "system", {
            "deck_remaining": len(self.deck),
            "p0_hand": len(self.players[0].hand),
            "p1_hand": len(self.players[1].hand),
        })

        # If deck exhausted and both have cards, keep playing
        if len(self.deck) == 0 and (len(self.players[0].hand) == 0 or len(self.players[1].hand) == 0):
            # Deck gone and someone has no cards — resolve by points
            self._resolve_deck_exhausted()
            return

        self.phase = GamePhase.PLAYING

    def _resolve_deck_exhausted(self):
        p0_pts = self.players[0].pile_points
        p1_pts = self.players[1].pile_points
        if p0_pts >= p1_pts:
            winner_idx = 0
        else:
            winner_idx = 1
        self._log("deck_exhausted", "system", {"p0": p0_pts, "p1": p1_pts})
        self._end_round(winner_idx, self.current_stake, RoundEndReason.DECK_EXHAUSTED)

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

        # Check game over
        if self.players[winner_idx].game_score >= self.target_score:
            self.phase = GamePhase.GAME_OVER
            self._log("game_over", "system", {
                "winner": self.players[winner_idx].player_id,
                "scores": {p.player_id: p.game_score for p in self.players}
            })
        else:
            self.phase = GamePhase.ROUND_OVER

    # ── VALID CUTS HELPER ─────────────────────

    def get_valid_cuts(self, player_idx: int) -> list[list[str]]:
        """
        Return all valid cut combinations from player's hand.
        Used by UI to show which combos are legal, and by AI bots.
        """
        player = self.players[player_idx]
        n = len(self.played_cards)
        from itertools import combinations
        valid = []
        for combo in combinations(player.hand, n):
            combo_list = list(combo)
            suits = {c.suit for c in combo_list}
            if len(suits) == 1 and self._can_cut(self.played_cards, combo_list):
                valid.append([repr(c) for c in combo_list])
        return valid

    # ── STATE SNAPSHOT ────────────────────────

    def get_state(self, perspective_idx: Optional[int] = None, debug: bool = False) -> dict:
        """
        Return a JSON-serializable game state.
        perspective_idx: if set, hide opponent's hand (for human player views).
        debug: reveal everything.
        """
        states = []
        for i, p in enumerate(self.players):
            reveal_hand = debug or (perspective_idx is None) or (i == perspective_idx)
            states.append(p.to_dict(reveal_hand=reveal_hand, reveal_pile=True))

        return {
            "phase": self.phase.value,
            "round_number": self.round_number,
            "active_player_idx": self.active_idx,
            "active_player_id": self.active_player.player_id,
            "opponent_idx": self.opponent_idx,
            "calculator_idx": self.calculator_idx,
            "calculator_id": self.players[self.calculator_idx].player_id if self.calculator_idx is not None else None,
            "playing_player_idx": self.playing_player_idx,
            "current_stake": self.current_stake,
            "pending_stake": self.pending_stake,
            "stake_offerer_idx": self.stake_offerer_idx,
            "stake_offerer_id": self.players[self.stake_offerer_idx].player_id if self.stake_offerer_idx is not None else None,
            "played_cards": [c.to_dict() for c in self.played_cards],
            "is_maliutka": self.is_maliutka,
            "deck": self.deck.to_dict(debug=debug),
            "trump_suit": self.trump_suit.value if self.trump_suit else None,
            "trump_card": self.deck.trump_card.to_dict() if self.deck.trump_card else None,
            "players": states,
            "target_score": self.target_score,
            "round_winner_idx": self.round_winner_idx,
            "round_end_reason": self.round_end_reason.value if self.round_end_reason else None,
            "move_history": [m.to_dict() for m in self.move_history[-20:]],  # last 20 moves
            "error": self.error,
            "valid_cuts": self.get_valid_cuts(1 - self.active_idx) if self.phase in (GamePhase.CUTTING, GamePhase.FORCED_CUT) else [],
        }