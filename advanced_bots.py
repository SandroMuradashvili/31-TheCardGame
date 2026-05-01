"""
advanced_bots.py — Advanced AI bots for Bura / Thirty-One / Cutter
────────────────────────────────────────────────────────────────────
Three new bots of increasing sophistication:

  PIMCBot          — Perfect Information Monte Carlo
  ISMCTSBot        — Information Set MCTS
  ISMCTSBeliefBot  — ISMCTS + Belief-weighted determinization + EV rollouts

All bots respect the legal information boundary:
  - Own hand (fully known)
  - Own score pile known-points + hidden-card-count
  - Opponent's known_pile_points + hidden_card_count (but not identity)
  - Cards on table (globally visible)
  - Trump card (globally visible)
  - own seen_card_ids (cards observed through hands / table)

None of these bots peek at opponent hand, opponent hidden pile content,
or remaining deck order — that would be cheating.

Design notes
────────────
PIMC
  For each decision, sample N "worlds" by dealing unseen cards randomly to
  opponent and deck. In each world, run a full deterministic rollout using
  SimpleBot logic for both sides. Average the win/loss result across worlds.
  Pick the action with the best average reward.

ISMCTS
  A single MCTS tree built across multiple determinizations. Each node
  tracks (visits, wins) across all worlds in which it was available. UCB1
  is used for selection. Rollouts use AggressiveBot logic (stronger signal
  than random). This correctly handles imperfect information: a node may
  not be available in every determinization, and UCB1 is only applied
  when a node is reachable.

ISMCTS + Belief EV (hybrid)
  Same as ISMCTS but:
  1. Worlds are sampled with probability weights from BeliefTracker
     (opponent is more likely to hold high-probability cards).
  2. Rollouts use EVBot decision logic — the strongest available heuristic.
  This gives the tree search informed starting positions AND informed
  rollout trajectories, making it the strongest of the three.

Reward signal
  +1  if the bot wins the round
  -1  if the opponent wins the round
  (stake-neutral for the tree; stake is handled by the real game engine)
"""

import copy
import math
import random
from collections import defaultdict
from itertools import combinations

from game_engine import (
    BotPlayer, GameEngine, GamePhase,
    Card, Suit, Rank,
    ALL_CARD_IDS,
)
from bot import (
    SimpleBot, AggressiveBot, EVBot,
    BeliefTracker,
)

from optimal_bots import OPTIMAL_BOT_REGISTRY
# merge into _all_bots dict


# ══════════════════════════════════════════════════════════════════════════════
# Configuration constants — tune these for speed vs. strength trade-off
# ══════════════════════════════════════════════════════════════════════════════

PIMC_WORLDS            = 75      # worlds sampled per PIMC decision
ISMCTS_ITERATIONS      = 1000    # MCTS iterations per decision
ISMCTS_UCB_C           = 1.4     # UCB1 exploration constant (√2 ≈ 1.41)
ROLLOUT_MAX_STEPS      = 60      # safety cap on rollout depth (prevents infinite loops)


# ══════════════════════════════════════════════════════════════════════════════
# Engine cloning
# ══════════════════════════════════════════════════════════════════════════════

def _clone_engine(engine: GameEngine) -> GameEngine:
    """
    Deep-copy the entire GameEngine.  This is the only safe way to branch
    the state for simulation — GameEngine has no native clone() method.

    copy.deepcopy handles all nested objects (Deck, Card, Player, etc.)
    correctly because none of them hold external resources (sockets, file
    handles, etc.).  The copy is completely independent of the original.
    """
    return copy.deepcopy(engine)


# ══════════════════════════════════════════════════════════════════════════════
# World sampling  (determinization)
# ══════════════════════════════════════════════════════════════════════════════

def _unseen_cards(engine: GameEngine, observer_idx: int) -> list[str]:
    """
    Return card IDs that are genuinely unknown to observer_idx:
    ALL_CARD_IDS  minus  observer's hand  minus  observer's seen_card_ids.

    These cards could be in the opponent's hand OR in the deck.
    """
    me = engine.players[observer_idx]
    my_hand_ids = {repr(c) for c in me.hand}
    seen        = set(me.seen_card_ids)  # includes trump + table cards already
    return list(ALL_CARD_IDS - my_hand_ids - seen)


def _deal_world_uniform(engine: GameEngine, my_idx: int) -> GameEngine:
    """
    Uniform determinization: randomly distribute unseen cards to opponent
    hand and remaining deck, preserving the correct hand-size and deck-size.
    Returns an independent cloned engine with a plausible complete world.
    """
    sim   = _clone_engine(engine)
    opp   = sim.players[1 - my_idx]
    unseen = _unseen_cards(engine, my_idx)
    random.shuffle(unseen)

    opp_need  = len(opp.hand)          # how many cards opponent is short (0–3)
    deck_need = len(sim.deck.cards)    # cards still in deck

    # Total unseen must equal opp_hand_size + deck_size (sanity check)
    total_need = opp_need + deck_need
    if len(unseen) < total_need:
        # Edge case: slight over-counting from seen — pad with duplicates of
        # lowest-value unseen card to avoid crashing.
        pad = (total_need - len(unseen))
        unseen += unseen[:pad] if unseen else ["JC"] * pad

    # Build new card objects from IDs
    rank_map = {r.value: r for r in Rank}
    suit_map = {s.value[0].upper(): s for s in Suit}

    def make_card(cid: str) -> Card:
        r = rank_map[cid[:-1]]
        s = suit_map[cid[-1]]
        return Card(s, r)

    sampled    = [make_card(cid) for cid in unseen[:total_need]]
    opp.hand   = sampled[:opp_need]
    sim.deck.cards = sampled[opp_need:opp_need + deck_need]

    return sim


def _deal_world_belief_weighted(
    engine: GameEngine, my_idx: int, belief: BeliefTracker
) -> GameEngine:
    """
    Belief-weighted determinization: sample opponent hand proportional to
    BeliefTracker probabilities.  Cards the opponent is more likely to hold
    (per hypergeometric model) appear in more sampled worlds.

    Algorithm:
      1. Compute P(card in opp hand) for every unseen card via BeliefTracker.
      2. Sample opponent hand WITHOUT replacement, weighted by those probs.
      3. Fill deck with remaining unseen cards (shuffled).
    """
    unseen = _unseen_cards(engine, my_idx)
    if not unseen:
        return _deal_world_uniform(engine, my_idx)

    opp_need  = len(engine.players[1 - my_idx].hand)
    deck_need = len(engine.deck.cards)

    # Weights for sampling opponent hand
    weights = [max(belief.p_card_in_opp_hand(cid), 1e-6) for cid in unseen]

    # Weighted sample without replacement for opponent hand
    opp_ids: list[str] = []
    remaining_unseen = list(unseen)
    remaining_weights = list(weights)

    for _ in range(min(opp_need, len(remaining_unseen))):
        total_w = sum(remaining_weights)
        r = random.uniform(0, total_w)
        cumulative = 0.0
        chosen_i = 0
        for i, w in enumerate(remaining_weights):
            cumulative += w
            if cumulative >= r:
                chosen_i = i
                break
        opp_ids.append(remaining_unseen.pop(chosen_i))
        remaining_weights.pop(chosen_i)

    # Remaining go to deck
    deck_ids = remaining_unseen[:deck_need]
    random.shuffle(deck_ids)

    # Build world
    sim = _clone_engine(engine)
    rank_map = {r.value: r for r in Rank}
    suit_map = {s.value[0].upper(): s for s in Suit}

    def make_card(cid: str) -> Card:
        r = rank_map[cid[:-1]]
        s = suit_map[cid[-1]]
        return Card(s, r)

    opp = sim.players[1 - my_idx]
    opp.hand       = [make_card(cid) for cid in opp_ids]
    sim.deck.cards = [make_card(cid) for cid in deck_ids]

    return sim


# ══════════════════════════════════════════════════════════════════════════════
# Rollout policies (used in both PIMC and ISMCTS)
# ══════════════════════════════════════════════════════════════════════════════

def _apply_bot_decision(sim: GameEngine, bot_cls_map: dict) -> bool:
    """
    Apply one logical game action using bot_cls_map[player_idx] as the
    decision-maker for the active side.  Returns True if an action was
    successfully applied, False if terminal or stuck.

    Handles all phases: STAKES, PLAYING, CUTTING, FORCED_CUT, CALCULATING.
    """
    phase = sim.phase

    if phase in (GamePhase.ROUND_OVER, GamePhase.GAME_OVER):
        return False

    if phase == GamePhase.STAKES:
        # Active player decides whether to raise
        actor_idx = sim.active_idx
        bot = bot_cls_map.get(actor_idx)
        if bot and bot.choose_raise_stake(sim):
            sim.offer_stake(actor_idx)
            # Opponent immediately responds
            opp_idx = 1 - actor_idx
            opp_bot = bot_cls_map.get(opp_idx)
            if opp_bot and opp_bot.choose_raise_stake(sim):
                # Counter-raise or accept
                sim.accept_stake(opp_idx)
            else:
                sim.decline_stake(opp_idx)
        else:
            sim.start_play()
        return True

    if phase == GamePhase.PLAYING:
        actor_idx = sim.active_idx
        bot = bot_cls_map.get(actor_idx)
        if bot is None:
            return False
        card_ids = bot.choose_play(sim)
        return sim.play_cards(actor_idx, card_ids)

    if phase in (GamePhase.CUTTING, GamePhase.FORCED_CUT):
        # The cutter is the NON-playing player
        cutter_idx = 1 - sim.playing_player_idx
        bot = bot_cls_map.get(cutter_idx)
        if bot is None:
            return False
        result = bot.choose_cut_or_pass(sim)
        action, cards = result[0], result[1]
        if action == "cut":
            ok = sim.cut_cards(cutter_idx, cards)
            if not ok:
                # Fallback: pass cheapest cards
                n = len(sim.played_cards)
                cheapest = sorted(sim.players[cutter_idx].hand, key=lambda c: c.points)[:n]
                sim.pass_cards(cutter_idx, [repr(c) for c in cheapest])
        elif action == "counter":
            ok = sim.counter_play(cutter_idx, cards)
            if not ok:
                n = len(sim.played_cards)
                cheapest = sorted(sim.players[cutter_idx].hand, key=lambda c: c.points)[:n]
                sim.pass_cards(cutter_idx, [repr(c) for c in cheapest])
        else:  # pass
            sim.pass_cards(cutter_idx, cards)
        return True

    if phase == GamePhase.CALCULATING:
        actor_idx = sim.calculator_idx
        if actor_idx is None:
            return False
        bot = bot_cls_map.get(actor_idx)
        if bot and bot.choose_calculate(sim):
            sim.calculate(actor_idx)
        else:
            sim.skip_calculate(actor_idx)
        return True

    return False


def _rollout(sim: GameEngine, my_idx: int, bot_cls_map: dict) -> float:
    """
    Run a full rollout from the current sim state.
    Returns +1.0 if my_idx wins the round, -1.0 if they lose.
    Bots in bot_cls_map drive both sides.
    """
    steps = 0
    while sim.phase not in (GamePhase.ROUND_OVER, GamePhase.GAME_OVER):
        if steps >= ROLLOUT_MAX_STEPS:
            break
        ok = _apply_bot_decision(sim, bot_cls_map)
        if not ok:
            break
        steps += 1

    winner = sim.round_winner_idx
    if winner is None:
        # Round didn't finish cleanly — fallback: compare pile points
        p0 = sim.players[0].pile_points
        p1 = sim.players[1].pile_points
        winner = 0 if p0 >= p1 else 1

    return 1.0 if winner == my_idx else -1.0


def _make_simple_bots(engine: GameEngine) -> dict:
    """Build a {idx: BotInstance} map using SimpleBot for both players."""
    bots = {}
    for i, p in enumerate(engine.players):
        b = SimpleBot(p.player_id, p.name)
        b.hand = p.hand  # point at the simulation player's hand
        bots[i] = b
    return bots


def _make_aggressive_bots(engine: GameEngine) -> dict:
    bots = {}
    for i, p in enumerate(engine.players):
        b = AggressiveBot(p.player_id, p.name)
        b.hand = p.hand
        bots[i] = b
    return bots


def _make_ev_bots(engine: GameEngine) -> dict:
    bots = {}
    for i, p in enumerate(engine.players):
        b = EVBot(p.player_id, p.name)
        b.hand = p.hand
        bots[i] = b
    return bots


def _sync_bot_hands(sim: GameEngine, bot_map: dict):
    """Keep bot hand references pointing at the simulation player hands."""
    for i, p in enumerate(sim.players):
        if i in bot_map:
            bot_map[i].hand = p.hand


# ══════════════════════════════════════════════════════════════════════════════
# Action enumeration  (what moves can the active player make?)
# ══════════════════════════════════════════════════════════════════════════════

def _enumerate_play_actions(engine: GameEngine, player_idx: int) -> list[tuple]:
    """
    Return all legal PLAYING actions as (action_type, card_ids) tuples.
    Same-suit groups of 1–3 cards.
    """
    player = engine.players[player_idx]
    trump  = engine.trump_suit
    suits: dict = {}
    for c in player.hand:
        suits.setdefault(c.suit, []).append(c)

    actions = []
    for suit, cards in suits.items():
        for size in range(1, min(len(cards), 3) + 1):
            for combo in combinations(cards, size):
                actions.append(("play", [repr(c) for c in combo]))
    return actions


def _enumerate_cut_actions(engine: GameEngine, player_idx: int) -> list[tuple]:
    """
    Return all legal CUTTING actions: valid cuts + pass + optional counter.
    """
    valid_cuts = engine.get_valid_cuts(player_idx)
    actions = [("cut", ids) for ids in valid_cuts]

    # Pass: choose cheapest N cards (only one pass option enumerated — the
    # choice of which cards to pass is secondary to cut/pass decision)
    n = len(engine.played_cards)
    player = engine.players[player_idx]
    cheapest = sorted(player.hand, key=lambda c: c.points)[:n]
    actions.append(("pass", [repr(c) for c in cheapest]))

    # Counter (only in CUTTING phase, not FORCED_CUT)
    if engine.phase == GamePhase.CUTTING:
        trump = engine.trump_suit
        sg: dict = {}
        for c in player.hand:
            if c.suit != trump:
                sg.setdefault(c.suit, []).append(c)
        for suit, cards in sg.items():
            if len(cards) >= 3:
                top3 = sorted(cards, key=lambda c: c.points, reverse=True)[:3]
                actions.append(("counter", [repr(c) for c in top3]))

    return actions


def _enumerate_calculate_actions(engine: GameEngine) -> list[tuple]:
    return [("calculate", []), ("skip", [])]


def _enumerate_stake_actions(engine: GameEngine, player_idx: int) -> list[tuple]:
    actions = [("play", [])]  # just start playing (no raise)
    if engine.can_raise_stake(player_idx):
        actions.append(("raise", []))
    return actions


def _enumerate_actions(engine: GameEngine, my_idx: int) -> list[tuple]:
    """Enumerate all legal actions for my_idx given current engine phase."""
    phase = engine.phase
    if phase == GamePhase.STAKES:
        return _enumerate_stake_actions(engine, my_idx)
    if phase == GamePhase.PLAYING:
        return _enumerate_play_actions(engine, my_idx)
    if phase in (GamePhase.CUTTING, GamePhase.FORCED_CUT):
        return _enumerate_cut_actions(engine, my_idx)
    if phase == GamePhase.CALCULATING:
        return _enumerate_calculate_actions(engine)
    return []


def _apply_action(sim: GameEngine, my_idx: int, action: tuple) -> bool:
    """Apply a specific (action_type, card_ids) to the simulation."""
    act_type, card_ids = action[0], action[1]

    if act_type == "play":
        if sim.phase == GamePhase.STAKES:
            sim.start_play()
            return True
        return sim.play_cards(my_idx, card_ids)

    if act_type == "raise":
        ok = sim.offer_stake(my_idx)
        if ok:
            # Opponent auto-responds with SimpleBot logic
            opp_idx = 1 - my_idx
            opp_b = SimpleBot(sim.players[opp_idx].player_id, "opp")
            opp_b.hand = sim.players[opp_idx].hand
            if opp_b.choose_raise_stake(sim):
                sim.accept_stake(opp_idx)
            else:
                sim.decline_stake(opp_idx)
        return ok

    if act_type == "cut":
        ok = sim.cut_cards(my_idx, card_ids)
        if not ok:
            n = len(sim.played_cards)
            cheapest = sorted(sim.players[my_idx].hand, key=lambda c: c.points)[:n]
            sim.pass_cards(my_idx, [repr(c) for c in cheapest])
        return True

    if act_type == "counter":
        ok = sim.counter_play(my_idx, card_ids)
        if not ok:
            n = len(sim.played_cards)
            cheapest = sorted(sim.players[my_idx].hand, key=lambda c: c.points)[:n]
            sim.pass_cards(my_idx, [repr(c) for c in cheapest])
        return True

    if act_type == "pass":
        return sim.pass_cards(my_idx, card_ids)

    if act_type == "calculate":
        return sim.calculate(my_idx)

    if act_type == "skip":
        return sim.skip_calculate(my_idx)

    return False


# ══════════════════════════════════════════════════════════════════════════════
# PIMC Bot
# ══════════════════════════════════════════════════════════════════════════════

class PIMCBot(BotPlayer):
    """
    Perfect Information Monte Carlo bot.

    For every decision point:
      1. Sample PIMC_WORLDS random complete world states (fill opponent hand
         and deck from unseen card pool).
      2. For each world, try every legal action for the current player.
      3. After taking that action, run a full rollout using SimpleBot for
         both sides.
      4. Score each action by its average reward across all worlds.
      5. Return the highest-scoring action.

    Raise / Calculate decisions follow the same simulation framework.

    Complexity: O(PIMC_WORLDS × |actions| × rollout_depth)
    With PIMC_WORLDS=75 and typical ~10 rollout steps, this is fast enough
    for real-time play.
    """

    DISPLAY_NAME = "PIMC"
    NUM_WORLDS   = PIMC_WORLDS

    def _my_idx(self, engine: GameEngine) -> int:
        return engine.players.index(self)

    def _evaluate_actions(
        self, engine: GameEngine, actions: list[tuple]
    ) -> dict:
        """
        Returns {action: avg_reward} for all actions using PIMC.
        action is a tuple (type, card_ids).
        """
        my_idx  = self._my_idx(engine)
        totals  = defaultdict(float)
        counts  = defaultdict(int)

        for _ in range(self.NUM_WORLDS):
            world = _deal_world_uniform(engine, my_idx)

            for action in actions:
                sim = _clone_engine(world)
                ok  = _apply_action(sim, my_idx, action)
                if not ok:
                    continue

                # Build rollout bots — simple for speed
                bots = {}
                for i, p in enumerate(sim.players):
                    b = SimpleBot(p.player_id, p.name)
                    b.hand = p.hand
                    bots[i] = b

                reward = _rollout(sim, my_idx, bots)
                totals[action] += reward
                counts[action] += 1

        return {
            a: (totals[a] / counts[a]) if counts[a] > 0 else 0.0
            for a in actions
        }

    def _best_action(self, engine: GameEngine, actions: list[tuple]) -> tuple:
        if not actions:
            return ("play", [])
        scores = self._evaluate_actions(engine, actions)
        return max(actions, key=lambda a: scores.get(a, -999))

    # ── Public interface ──────────────────────────────────────────────────────

    def choose_play(self, engine: GameEngine) -> list[str]:
        my_idx  = self._my_idx(engine)
        actions = _enumerate_play_actions(engine, my_idx)
        best    = self._best_action(engine, actions)
        return best[1]

    def choose_cut_or_pass(self, engine: GameEngine):
        my_idx  = self._my_idx(engine)
        actions = _enumerate_cut_actions(engine, my_idx)
        best    = self._best_action(engine, actions)
        act_type, card_ids = best
        if act_type == "cut":
            return ("cut", card_ids)
        if act_type == "counter":
            return ("counter", card_ids)
        # pass — need to pass exactly len(played_cards) cards
        n = len(engine.played_cards)
        cheapest = sorted(self.hand, key=lambda c: c.points)[:n]
        return ("pass", [repr(c) for c in cheapest])

    def choose_calculate(self, engine: GameEngine) -> bool:
        my_idx  = self._my_idx(engine)
        actions = _enumerate_calculate_actions(engine)
        best    = self._best_action(engine, actions)
        return best[0] == "calculate"

    def choose_raise_stake(self, engine: GameEngine) -> bool:
        my_idx  = self._my_idx(engine)
        if not engine.can_raise_stake(my_idx):
            return False
        actions = _enumerate_stake_actions(engine, my_idx)
        best    = self._best_action(engine, actions)
        return best[0] == "raise"


# ══════════════════════════════════════════════════════════════════════════════
# ISMCTS Node
# ══════════════════════════════════════════════════════════════════════════════

class ISMCTSNode:
    """
    A node in the Information Set MCTS tree.

    In ISMCTS, the tree is built over the information set (what the current
    player knows), not over a single determinized world.  Each node stores:

      action      — the action taken TO reach this node (None for root)
      parent      — parent node
      children    — child nodes
      visits      — total times this node was visited across all worlds
      wins        — total reward accumulated (can be fractional)
      avail       — number of worlds in which this node was AVAILABLE
                    (denominator for the availability heuristic)

    UCB1 with availability:
      UCB1 = wins/visits + C × √(ln(avail) / visits)

    The availability term (avail instead of parent.visits) prevents the
    algorithm from favouring nodes that happen to appear in many worlds.
    """

    __slots__ = ("action", "parent", "children", "visits", "wins", "avail")

    def __init__(self, action=None, parent=None):
        self.action   = action    # (type, card_ids) tuple or None
        self.parent   = parent
        self.children: list["ISMCTSNode"] = []
        self.visits   = 0
        self.wins     = 0.0
        self.avail    = 0         # times this node was available for selection

    def ucb1(self, c: float = ISMCTS_UCB_C) -> float:
        if self.visits == 0:
            return float("inf")
        exploitation = self.wins / self.visits
        exploration  = c * math.sqrt(math.log(max(self.avail, 1)) / self.visits)
        return exploitation + exploration

    def best_child(self, c: float = ISMCTS_UCB_C) -> "ISMCTSNode":
        return max(self.children, key=lambda n: n.ucb1(c))

    def most_visited_child(self) -> "ISMCTSNode":
        return max(self.children, key=lambda n: n.visits)

    def find_or_create_child(self, action: tuple) -> "ISMCTSNode":
        for ch in self.children:
            if ch.action == action:
                return ch
        child = ISMCTSNode(action=action, parent=self)
        self.children.append(child)
        return child


# ══════════════════════════════════════════════════════════════════════════════
# ISMCTS Bot
# ══════════════════════════════════════════════════════════════════════════════

class ISMCTSBot(BotPlayer):
    """
    Information Set Monte Carlo Tree Search bot.

    The single tree is re-used across multiple determinizations within one
    decision call.  Each iteration:

      1. Sample a world (uniform determinization).
      2. SELECT: walk the tree using UCB1, updating avail counts.
         Stop when we reach a node whose children don't cover all legal
         actions in THIS world (expansion needed), or a terminal node.
      3. EXPAND: add one new child for an unexplored action.
      4. ROLLOUT: play out the rest of the round with AggressiveBot.
      5. BACKPROPAGATE: update visits and wins along the path.

    After ISMCTS_ITERATIONS, return the child of root with most visits
    (most robust action).

    Rollout policy: AggressiveBot — better signal than random play.

    Raise / Calculate: same tree approach, fewer iterations (fast).
    """

    DISPLAY_NAME    = "ISMCTS"
    NUM_ITERATIONS  = ISMCTS_ITERATIONS
    ROLLOUT_POLICY  = "aggressive"  # subclasses can override

    def _my_idx(self, engine: GameEngine) -> int:
        return engine.players.index(self)

    def _build_rollout_bots(self, sim: GameEngine) -> dict:
        bots = {}
        for i, p in enumerate(sim.players):
            if self.ROLLOUT_POLICY == "ev":
                b = EVBot(p.player_id, p.name)
            else:
                b = AggressiveBot(p.player_id, p.name)
            b.hand = p.hand
            bots[i] = b
        return bots

    def _run_ismcts(
        self,
        engine: GameEngine,
        my_idx: int,
        n_iterations: int,
        world_sampler=None,
    ) -> ISMCTSNode:
        """
        Run ISMCTS for n_iterations and return the root node.
        world_sampler(engine, my_idx) -> GameEngine  (defaults to uniform)
        """
        if world_sampler is None:
            world_sampler = _deal_world_uniform

        root = ISMCTSNode()

        for _ in range(n_iterations):
            # ── 1. Determinize ────────────────────────────────────────────
            world = world_sampler(engine, my_idx)

            # ── 2. Selection ──────────────────────────────────────────────
            node    = root
            sim     = _clone_engine(world)
            path    = [node]

            while True:
                if sim.phase in (GamePhase.ROUND_OVER, GamePhase.GAME_OVER):
                    break

                # Determine whose turn it is in the sim
                actor = _sim_actor_idx(sim, my_idx)
                if actor != my_idx:
                    # Opponent move — just apply opponent bot decision
                    bots = self._build_rollout_bots(sim)
                    _sync_bot_hands(sim, bots)
                    ok = _apply_bot_decision(sim, bots)
                    if not ok:
                        break
                    continue

                legal = _enumerate_actions(sim, my_idx)
                if not legal:
                    break

                # Mark all legal children as available
                legal_set = set(_action_key(a) for a in legal)
                for ch in node.children:
                    if _action_key(ch.action) in legal_set:
                        ch.avail += 1

                # Check if any legal action is unexplored
                explored_keys = {_action_key(ch.action) for ch in node.children}
                unexplored = [a for a in legal if _action_key(a) not in explored_keys]

                if unexplored:
                    # ── 3. Expand ─────────────────────────────────────────
                    action = random.choice(unexplored)
                    child  = node.find_or_create_child(action)
                    child.avail += 1
                    ok = _apply_action(sim, my_idx, action)
                    if not ok:
                        break
                    node = child
                    path.append(node)
                    break  # go straight to rollout
                else:
                    # All expanded: UCB1 selection among available children
                    available_children = [
                        ch for ch in node.children
                        if _action_key(ch.action) in legal_set
                    ]
                    if not available_children:
                        break
                    chosen = max(available_children, key=lambda n: n.ucb1())
                    ok = _apply_action(sim, my_idx, chosen.action)
                    if not ok:
                        break
                    node = chosen
                    path.append(node)

            # ── 4. Rollout ────────────────────────────────────────────────
            bots = self._build_rollout_bots(sim)
            _sync_bot_hands(sim, bots)
            reward = _rollout(sim, my_idx, bots)

            # ── 5. Backpropagate ──────────────────────────────────────────
            for n in path:
                n.visits += 1
                n.wins   += (reward + 1) / 2  # normalize to [0, 1]

        return root

    def _best_root_action(self, root: ISMCTSNode) -> tuple:
        if not root.children:
            return ("play", [])
        best = root.most_visited_child()
        return best.action

    # ── Public interface ──────────────────────────────────────────────────────

    def choose_play(self, engine: GameEngine) -> list[str]:
        my_idx = self._my_idx(engine)
        root   = self._run_ismcts(engine, my_idx, self.NUM_ITERATIONS)
        action = self._best_root_action(root)
        return action[1] if action[1] else [repr(self.hand[0])]

    def choose_cut_or_pass(self, engine: GameEngine):
        my_idx = self._my_idx(engine)
        root   = self._run_ismcts(engine, my_idx, self.NUM_ITERATIONS)
        action = self._best_root_action(root)
        act_type, card_ids = action

        if act_type == "cut":
            return ("cut", card_ids)
        if act_type == "counter":
            return ("counter", card_ids)
        n = len(engine.played_cards)
        cheapest = sorted(self.hand, key=lambda c: c.points)[:n]
        return ("pass", [repr(c) for c in cheapest])

    def choose_calculate(self, engine: GameEngine) -> bool:
        my_idx = self._my_idx(engine)
        root   = self._run_ismcts(engine, my_idx, self.NUM_ITERATIONS // 4)
        action = self._best_root_action(root)
        return action[0] == "calculate"

    def choose_raise_stake(self, engine: GameEngine) -> bool:
        my_idx = self._my_idx(engine)
        if not engine.can_raise_stake(my_idx):
            return False
        root   = self._run_ismcts(engine, my_idx, self.NUM_ITERATIONS // 4)
        action = self._best_root_action(root)
        return action[0] == "raise"


# ══════════════════════════════════════════════════════════════════════════════
# ISMCTS + Belief EV Bot (hybrid — the strongest)
# ══════════════════════════════════════════════════════════════════════════════

class ISMCTSBeliefBot(ISMCTSBot):
    """
    ISMCTS with two enhancements over plain ISMCTSBot:

    1. Belief-weighted determinization
       Instead of uniform random worlds, opponent hands are sampled weighted
       by BeliefTracker probabilities.  Cards more likely to be in the
       opponent's hand (per hypergeometric model) appear more often.
       This concentrates search effort on plausible worlds, reducing wasted
       rollouts on impossible states.

    2. EV-guided rollout policy
       Rollouts use EVBot instead of AggressiveBot.  EVBot uses expected-
       value calculations and its own BeliefTracker during rollout, giving
       a much stronger signal from each simulation.

    Together these make the MCTS tree explore smarter (better worlds) and
    evaluate deeper (better rollouts), which is the principled way to
    combine Bayesian belief tracking with tree search.

    This is the strongest bot in the project and the one expected to perform
    best against all opponents.
    """

    DISPLAY_NAME   = "ISMCTS + Belief EV"
    ROLLOUT_POLICY = "ev"          # use EVBot for rollouts (inherited by _build_rollout_bots)

    def _world_sampler(self, engine: GameEngine, my_idx: int) -> GameEngine:
        belief = BeliefTracker(engine, my_idx)
        return _deal_world_belief_weighted(engine, my_idx, belief)

    def choose_play(self, engine: GameEngine) -> list[str]:
        my_idx = self._my_idx(engine)
        root   = self._run_ismcts(
            engine, my_idx, self.NUM_ITERATIONS,
            world_sampler=self._world_sampler
        )
        action = self._best_root_action(root)
        return action[1] if action[1] else [repr(self.hand[0])]

    def choose_cut_or_pass(self, engine: GameEngine):
        my_idx = self._my_idx(engine)
        root   = self._run_ismcts(
            engine, my_idx, self.NUM_ITERATIONS,
            world_sampler=self._world_sampler
        )
        action = self._best_root_action(root)
        act_type, card_ids = action

        if act_type == "cut":
            return ("cut", card_ids)
        if act_type == "counter":
            return ("counter", card_ids)
        n = len(engine.played_cards)
        cheapest = sorted(self.hand, key=lambda c: c.points)[:n]
        return ("pass", [repr(c) for c in cheapest])

    def choose_calculate(self, engine: GameEngine) -> bool:
        my_idx = self._my_idx(engine)
        root   = self._run_ismcts(
            engine, my_idx, self.NUM_ITERATIONS // 4,
            world_sampler=self._world_sampler
        )
        action = self._best_root_action(root)
        return action[0] == "calculate"

    def choose_raise_stake(self, engine: GameEngine) -> bool:
        my_idx = self._my_idx(engine)
        if not engine.can_raise_stake(my_idx):
            return False
        root   = self._run_ismcts(
            engine, my_idx, self.NUM_ITERATIONS // 4,
            world_sampler=self._world_sampler
        )
        action = self._best_root_action(root)
        return action[0] == "raise"


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _action_key(action: tuple) -> tuple:
    """Hashable key for an action (handles list card_ids)."""
    if action is None:
        return None
    act_type, card_ids = action
    return (act_type, tuple(sorted(card_ids)))


def _sim_actor_idx(sim: GameEngine, my_idx: int) -> int:
    """
    Determine whose turn it is to make the primary decision in the sim.
    Returns my_idx if it's our turn, opp_idx otherwise.
    """
    phase = sim.phase
    if phase == GamePhase.PLAYING:
        return sim.active_idx
    if phase in (GamePhase.CUTTING, GamePhase.FORCED_CUT):
        return 1 - sim.playing_player_idx  # cutter's turn
    if phase == GamePhase.CALCULATING:
        return sim.calculator_idx if sim.calculator_idx is not None else sim.active_idx
    if phase == GamePhase.STAKES:
        return sim.active_idx
    return sim.active_idx


# ══════════════════════════════════════════════════════════════════════════════
# Bot Registry extension
# ══════════════════════════════════════════════════════════════════════════════

ADVANCED_BOT_REGISTRY: dict[str, type] = {
    "pimc":          PIMCBot,
    "ismcts":        ISMCTSBot,
    "ismcts_belief": ISMCTSBeliefBot,
}


def get_advanced_bot(bot_id: str, player_id: str, name: str) -> BotPlayer:
    """Instantiate an advanced bot by registry key."""
    cls = ADVANCED_BOT_REGISTRY.get(bot_id)
    if cls is None:
        raise ValueError(f"Unknown advanced bot: {bot_id!r}")
    return cls(player_id, name)


def list_advanced_bots() -> list[dict]:
    """Return bot list for the frontend dropdown."""
    return [
        {"id": k, "name": getattr(v, "DISPLAY_NAME", k)}
        for k, v in ADVANCED_BOT_REGISTRY.items()
    ]


def get_any_bot(bot_id: str, player_id: str, name: str) -> BotPlayer:
    """Instantiate a basic, advanced, or optimal bot."""
    # 1. Check Advanced Bots
    if bot_id in ADVANCED_BOT_REGISTRY:
        return get_advanced_bot(bot_id, player_id, name)

    # 2. Check Optimal Bots
    try:
        from optimal_bots import OPTIMAL_BOT_REGISTRY, get_optimal_bot
        if bot_id in OPTIMAL_BOT_REGISTRY:
            return get_optimal_bot(bot_id, player_id, name)
    except ImportError:
        pass

    # 3. Fallback to Basic Bots
    from bot import get_bot
    return get_bot(bot_id, player_id, name)


def list_all_bots() -> list[dict]:
    """Return a combined list of all basic, advanced, and optimal bots for the UI."""
    from bot import list_bots
    bots = list_bots() + list_advanced_bots()

    try:
        from optimal_bots import list_optimal_bots
        bots += list_optimal_bots()
    except ImportError:
        pass

    return bots