# BURA — Architecture & File Map

## Python backend

```
server.py         Flask app entry point. HTTP routes only — no game logic here.
                  Imports: room_manager, bot_runner, game_engine (for GamePhase)

room_manager.py   Room class and in-memory rooms dict.
                  Knows how to create rooms and call start_game().
                  Imports: game_engine, bot

bot.py            SimpleBot — all AI decision-making.
                  Subclass BotPlayer here to write a smarter AI.
                  Imports: game_engine (BotPlayer, GameEngine)

bot_runner.py     bot_act_if_needed() — drives the bot through consecutive turns.
                  Separated from bot.py so the turn loop is easy to read/modify.
                  Imports: game_engine (GameEngine, GamePhase)

game_engine.py    Pure game logic. No Flask, no I/O.
                  Card, Deck, Player, GameEngine, all enums and rules.
                  No imports from this project — safe to test in isolation.
```

### Where to look when fixing backend things

| Problem                              | File             |
|--------------------------------------|------------------|
| A game rule is wrong                 | game_engine.py   |
| Bot makes a bad decision             | bot.py           |
| Bot takes too many / too few turns   | bot_runner.py    |
| An API route is broken               | server.py        |
| Room creation / join logic           | room_manager.py  |

---

## JavaScript frontend

Load order in index.html (each file depends only on files above it):

```
js/state.js           Global State object + SUIT_SYM / color_cls constants.
                      No dependencies.

js/utils.js           toast(), api(), logAction(), ROOM() helper, sleep().
                      Depends on: state.js (State.roomId)

js/animations.js      flyCard(), animateCutSequence(), animatePassSequence().
                      Depends on: (DOM only)

js/cards.js           buildCard(), buildTrumpCard(), toggleCard(), clearSel(), buzzCard().
                      Depends on: state.js, utils.js, animations.js

js/render.js          render(), renderZone(), renderTable(), renderTrump(), showRoundModal().
                      Depends on: state.js, utils.js, animations.js, cards.js

js/actions_render.js  renderActions() — action panel HTML.
                      showPassUI(), cancelPass() — pass mode.
                      Depends on: state.js, cards.js, render.js

js/stake_actions.js   doRaiseStake(), doAcceptStake(), doDeclineStake().
                      Depends on: state.js, utils.js, render.js

js/game_actions.js    doPlayCards(), doCutCards(), doPassCards(), doCalculate(), doSkipCalc().
                      refreshState(), doStartRound().
                      Depends on: state.js, utils.js, cards.js, animations.js, render.js

js/lobby.js           switchTab(), startHvB(), createRoom(), joinRoom(), enterGame().
                      startPolling(), stopPolling(), toggleDebug(), keyboard shortcuts.
                      Depends on: everything above.
```

### Where to look when fixing frontend things

| Problem                                    | File                 |
|--------------------------------------------|----------------------|
| Card selection, suit locking, buzz effect  | js/cards.js          |
| Card movement animations                   | js/animations.js     |
| Hand / table / trump rendering             | js/render.js         |
| Action panel buttons and messages          | js/actions_render.js |
| Raise / accept / decline stake buttons     | js/stake_actions.js  |
| Play, cut, pass, calculate API calls       | js/game_actions.js   |
| Lobby, room creation, multiplayer polling  | js/lobby.js          |
| All visual styles                          | css/style.css        |

---

## Data flow (HvB — human vs bot)

```
Human clicks Play
  └─ doPlayCards()           game_actions.js
       ├─ POST /play_only    → engine.play_cards() only (no bot yet)
       ├─ render()           shows human cards sliding to table
       └─ setTimeout → _handleBotResponse()
            ├─ POST /bot_respond  → bot_act_if_needed() → engine.*
            └─ animateCutSequence() or animatePassSequence() or render()
```

## Data flow (HvH — human vs human)

```
Human clicks Play
  └─ doPlayCards()      game_actions.js
       ├─ POST /play    → engine.play_cards()
       └─ render()

Polling loop (lobby.js → startPolling())
  └─ every 1.2s: GET /state → if state changed → render()
```

## Game Rules (BURA / Thirty-One / Cutter)
**AI CONTEXT:** Use these exact rules to validate gameplay logic. If the code contradicts these rules, the code is bugged.

*   **Setup:** 20 cards (A, T, K, Q, J in 4 suits). A=11, T=10, K=4, Q=3, J=2. 3 cards dealt each. Trump suit revealed at bottom. Game to 7 or 11 points.
*   **Stakes:** Either player may raise (1 to 6). Declining a raise instantly awards the raiser the previous stake amount.
*   **Turn/Playing:** Active player plays 1-3 cards of the SAME SUIT.
*   **Cutting:** Defender must play the EXACT same number of cards. Must be same suit but higher value, OR a trump card. Trump beats non-trump.
*   **Passing:** If unable to cut, defender passes the exact same number of cards face down. All cards go to the active player's score pile.
*   **Maliutka:** Playing 3 same-suit *non-trump* cards. Forces the opponent to cut all 3.
*   **Bura (3 Trumps):** Playing 3 *trump* cards instantly wins the round at any time, regardless of phase or turn priority.
*   **Calculating:** Only the player who last took cards can calculate. If their pile is ≥ 31, they win. If < 31, they lose.