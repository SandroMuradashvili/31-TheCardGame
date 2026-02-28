"""
BURA Game Server
Flask API that drives the game engine and serves the UI.
"""

from flask import Flask, jsonify, request, send_from_directory, make_response
import json
import os

from game_engine import (
    GameState, Phase, Card, RoundResult,
    new_round, apply_move, draw_cards, apply_round_result,
    is_game_over, game_winner, legal_moves,
    MovePlay, MoveCut, MovePass, MoveCalculate, MoveContinue,
    MoveRaise, MoveDeclineRaise, MoveAcceptRaise,
    SUIT_SYMBOLS, is_maliutka, is_three_trumps, score_pile
)
from bot import BuraBot

app = Flask(__name__)

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route('/api/<path:path>', methods=['OPTIONS'])
def options_handler(path):
    return '', 204

# Global game state (single session for now)
game_state: GameState = None
bot = BuraBot(player_id=1)  # Bot is always P2
game_log: list[str] = []
last_bot_reasoning: str = ""
round_results: list[dict] = []
first_player = 0


def card_to_dict(c: Card) -> dict:
    return {
        "rank": c.rank,
        "suit": c.suit,
        "symbol": SUIT_SYMBOLS[c.suit],
        "points": c.points,
        "id": f"{c.rank}_{c.suit}"
    }


def state_to_dict(state: GameState, reveal_bot_hand: bool = False) -> dict:
    """Convert game state to JSON-serializable dict."""
    # Bot hand is hidden unless debugging
    bot_hand = state.hands[1]
    human_hand = state.hands[0]

    return {
        "phase": state.phase.name,
        "trump_suit": state.trump_suit,
        "trump_symbol": SUIT_SYMBOLS[state.trump_suit],
        "human_hand": [card_to_dict(c) for c in human_hand],
        "bot_hand_count": len(bot_hand),
        "bot_hand": [card_to_dict(c) for c in bot_hand] if reveal_bot_hand else [],
        "human_score_pile": score_pile(state.score_piles[0]),
        "bot_score_pile": score_pile(state.score_piles[1]),
        "human_pile_cards": [card_to_dict(c) for c in state.score_piles[0]],
        "bot_pile_cards": [card_to_dict(c) for c in state.score_piles[1]],
        "game_scores": state.game_scores,
        "game_length": state.game_length,
        "current_stake": state.current_stake,
        "active_player": state.active_player,
        "played_cards": [card_to_dict(c) for c in state.played_cards],
        "deck_remaining": len(state.deck),
        "last_taker": state.last_taker,
        "calculate_available": state.calculate_available,
        "round_number": state.round_number,
        "stake_last_raised_by": state.stake_last_raised_by,
        "human_is_active": state.active_player == 0,
        "game_over": is_game_over(state.game_scores, state.game_length),
        "winner": game_winner(state.game_scores) if is_game_over(state.game_scores, state.game_length) else None,
    }


def log(msg: str):
    game_log.append(msg)
    print(msg)


def run_bot_turn(state: GameState) -> GameState:
    """Run bot's turn(s) until it's human's turn or round ends."""
    global last_bot_reasoning, game_state

    max_iterations = 20
    i = 0

    while i < max_iterations:
        i += 1
        s = state

        # Is it the game over?
        if is_game_over(s.game_scores, s.game_length):
            return s

        # Is it human's turn?
        if s.phase == Phase.ROUND_OVER:
            return s

        if s.phase == Phase.DRAW:
            state = draw_cards(s)
            log("Cards drawn")
            continue

        # Check if bot needs to act
        bot_should_act = False

        if s.phase == Phase.STAKE_OFFER:
            # Either player can act — if human hasn't raised yet and it's not human's raise to respond to
            if s.stake_last_raised_by == 0:
                # Human raised — bot must respond
                bot_should_act = True
            elif s.active_player == 1:
                # Bot's turn to play, they can optionally raise first
                bot_should_act = True
            else:
                # Human's turn — stop
                return state

        elif s.phase == Phase.PLAY:
            if s.active_player == 1:
                bot_should_act = True
            else:
                return state

        elif s.phase == Phase.CUT_OR_PASS:
            if s.active_player == 0:
                # Human played, bot must respond
                bot_should_act = True
            else:
                return state

        elif s.phase == Phase.CALCULATE:
            if s.last_taker == 1:
                bot_should_act = True
            else:
                return state

        if not bot_should_act:
            return state

        # Bot acts
        move, reasoning = bot.choose_move(s)
        last_bot_reasoning = reasoning.explanation

        move_name = type(move).__name__
        log(f"Bot: {move_name} — {reasoning.explanation}")

        new_state, result = apply_move(s, 1, move)

        if result:
            new_state = handle_round_result(new_state, result)

        state = new_state

        # After bot plays in STAKE_OFFER, transition to PLAY for human
        if isinstance(move, MoveContinue) and s.phase == Phase.STAKE_OFFER and s.active_player == 1:
            # Bot decided not to raise — now it's bot's PLAY turn if active, or human's
            continue

    return state


def handle_round_result(state: GameState, result: RoundResult) -> GameState:
    """Apply round result to game scores and start new round."""
    global round_results, bot, first_player

    winner_name = "You" if result.winner == 0 else "Bot"
    round_results.append({
        "winner": result.winner,
        "winner_name": winner_name,
        "stake": result.stake,
        "reason": result.reason,
        "score": result.calculator_score,
        "new_scores": None  # filled below
    })

    new_scores = apply_round_result(state.game_scores, result)

    log(f"Round over — {winner_name} wins {result.stake} point(s)! Reason: {result.reason}")
    log(f"Game scores: P1={new_scores[0]}, P2={new_scores[1]}")

    if is_game_over(new_scores, state.game_length):
        state.game_scores = new_scores
        state.phase = Phase.GAME_OVER
        round_results[-1]["new_scores"] = new_scores
        return state

    # New round
    first_player = result.winner  # winner goes first
    new_state = new_round(
        game_scores=new_scores,
        game_length=state.game_length,
        first_player=first_player,
        round_number=state.round_number + 1
    )
    round_results[-1]["new_scores"] = new_scores

    # Reset bot tracker
    bot.reset_for_new_round(
        hand=new_state.hands[1],
        trump_suit=new_state.trump_suit,
        deck_size=len(new_state.deck)
    )

    return new_state


# ─────────────────────────────────────────────
#  API Routes
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/api/new_game', methods=['POST'])
def new_game():
    global game_state, bot, game_log, round_results, first_player
    data = request.json or {}
    game_length = data.get('game_length', 7)
    first_player = 0  # random.randint(0, 1)

    game_log = []
    round_results = []
    bot = BuraBot(player_id=1)

    game_state = new_round(
        game_scores=[0, 0],
        game_length=game_length,
        first_player=first_player,
        round_number=1
    )

    bot.reset_for_new_round(
        hand=game_state.hands[1],
        trump_suit=game_state.trump_suit,
        deck_size=len(game_state.deck)
    )

    log(f"New game started! Game to {game_length}. Trump: {game_state.trump_suit}")

    # If bot goes first, run bot's turn
    if first_player == 1:
        game_state = run_bot_turn(game_state)

    return jsonify({
        "state": state_to_dict(game_state),
        "log": game_log[-10:],
        "bot_reasoning": last_bot_reasoning,
        "round_results": round_results
    })


@app.route('/api/state', methods=['GET'])
def get_state():
    if game_state is None:
        return jsonify({"error": "No game in progress"}), 400
    return jsonify({
        "state": state_to_dict(game_state),
        "log": game_log[-10:],
        "bot_reasoning": last_bot_reasoning,
        "round_results": round_results
    })


@app.route('/api/move', methods=['POST'])
def make_move():
    global game_state, last_bot_reasoning

    if game_state is None:
        return jsonify({"error": "No game in progress"}), 400

    data = request.json
    move_type = data.get('type')

    # Parse move
    move = None
    try:
        if move_type == 'play':
            card_ids = data.get('cards', [])
            cards = []
            for cid in card_ids:
                rank, suit = cid.split('_')
                cards.append(Card(rank, suit))
            move = MovePlay(cards)
            log(f"Human plays: {cards}")

        elif move_type == 'cut':
            card_ids = data.get('cards', [])
            cards = []
            for cid in card_ids:
                rank, suit = cid.split('_')
                cards.append(Card(rank, suit))
            move = MoveCut(cards)
            log(f"Human cuts: {cards}")

        elif move_type == 'pass':
            card_ids = data.get('cards', [])
            cards = []
            for cid in card_ids:
                rank, suit = cid.split('_')
                cards.append(Card(rank, suit))
            move = MovePass(cards)
            log(f"Human passes: {[str(c) for c in cards]}")

        elif move_type == 'calculate':
            move = MoveCalculate()
            log("Human calculates!")

        elif move_type == 'continue':
            move = MoveContinue()
            log("Human continues")

        elif move_type == 'raise':
            new_stake = data.get('new_stake')
            move = MoveRaise(new_stake)
            log(f"Human raises to {new_stake}")

        elif move_type == 'decline_raise':
            move = MoveDeclineRaise()
            log("Human declines raise")

        elif move_type == 'accept_raise':
            move = MoveAcceptRaise()
            log("Human accepts raise")

        else:
            return jsonify({"error": f"Unknown move type: {move_type}"}), 400

    except Exception as e:
        return jsonify({"error": f"Move parse error: {str(e)}"}), 400

    # Apply human move
    new_state, result = apply_move(game_state, 0, move)

    if result:
        new_state = handle_round_result(new_state, result)

    game_state = new_state

    # Run bot's response
    if not is_game_over(game_state.game_scores, game_state.game_length):
        if game_state.phase != Phase.GAME_OVER:
            game_state = run_bot_turn(game_state)

    return jsonify({
        "state": state_to_dict(game_state),
        "log": game_log[-15:],
        "bot_reasoning": last_bot_reasoning,
        "round_results": round_results[-3:]
    })


@app.route('/api/log', methods=['GET'])
def get_log():
    return jsonify({"log": game_log, "round_results": round_results})


if __name__ == '__main__':
    app.run(debug=True, port=5000)