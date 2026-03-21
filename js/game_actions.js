'use strict';
// ─────────────────────────────────────────────────────────────────────────────
// game_actions.js — Card play, cut, pass, calculate API calls
// ─────────────────────────────────────────────────────────────────────────────
// Does NOT handle stakes — see stake_actions.js for raise/accept/decline.
//
// Depends on: state.js, utils.js, cards.js, animations.js, render.js
// Loaded by:  index.html (after all the above)
// ─────────────────────────────────────────────────────────────────────────────


// ─── State refresh ───────────────────────────────────────────────────────────

window.refreshState = async function(withRender = true) {
  if (!State.roomId) return;
  const debug = State.debugMode ? '&debug=true' : '';
  const data  = await api(ROOM(`state?perspective=${State.myPlayerIdx}${debug}`));
  if (data?.state) {
    State.prevState = State.gameState;
    State.gameState = data.state;
    if (withRender) render();
  }
};


// ─── Round lifecycle ──────────────────────────────────────────────────────────

window.doStartRound = async function() {
  // Reset all transient UI state for the new round
  document.getElementById('round-modal').classList.remove('visible');
  State.modalShown     = false;
  State.selectedCards  = [];
  State.passModeActive = false;
  State.showTip        = false;
  State.prevHandKeys   = ['', ''];
  State.prevPlayedKey  = '';
  State.tableCards     = [];
  document.getElementById('played-row').innerHTML = '';
  document.getElementById('tip-popup-container').innerHTML = '';
  document.getElementById('tip-icon-btn')?.classList.remove('active');

  const data = await api(ROOM('start_round'), { player_idx: State.myPlayerIdx });
  if (data?.state) { State.gameState = data.state; render(); }
};


// ─── Playing cards ────────────────────────────────────────────────────────────
//
// HvH: single API call, no animation split needed.
// HvB: two-step sequence —
//   1. play_only  → human's cards go to table (animate)
//   2. bot_respond → bot acts (animate cut/pass/etc.)

window.doPlayCards = async function() {
  if (State.selectedCards.length === 0) return;
  const cards = [...State.selectedCards];
  clearSel();
  State.passModeActive = false;

  // ── HvH path ────────────────────────────────────────────────
  if (State.gameMode !== 'hvb') {
    const data = await api(ROOM('play'), { player_idx: State.myPlayerIdx, cards });
    if (!data?.state) return;
    State.gameState = data.state;
    logAction(`<span class="le-type">PLAY</span> You played ${cards.join(' ')}`);
    render();
    return;
  }

  // ── HvB step 1: play human cards, get intermediate state ────
  const step1 = await api(ROOM('play_only'), { player_idx: State.myPlayerIdx, cards });
  if (!step1?.state) return;

  logAction(`<span class="le-type">PLAY</span> You played ${cards.join(' ')}`);

  // Force re-render so the cards slide onto the table
  State.prevHandKeys  = ['', ''];
  State.prevPlayedKey = '';
  State.gameState     = step1.state;
  render();

  // ── HvB step 2: after the play animation, trigger bot ───────
  const playAnimDuration = 400 + cards.length * 70;
  setTimeout(_handleBotResponse, playAnimDuration);
};


// Called after the human's play animation finishes — fetches and animates bot's response.
async function _handleBotResponse() {
  const step2 = await api(ROOM('bot_respond'), { player_idx: State.myPlayerIdx });
  if (!step2?.state) return;

  const botState  = step2.state;
  const prevState = State.gameState;
  const botIdx    = 1 - State.myPlayerIdx;

  // Find the bot's last action in move history
  const lastBotMove = [...(botState.move_history || [])].reverse().find(m =>
    m.player !== 'system' && m.player !== State.gameState?.players[State.myPlayerIdx]?.id
  );
  const botAction = lastBotMove?.type;

  if (botAction === 'cut') {
    _animateBotCut(lastBotMove, botState, prevState, botIdx);
  } else if (botAction === 'pass') {
    _animateBotPass(lastBotMove, botState, prevState, botIdx);
  } else {
    // Round end, three_trumps, etc. — just render the new state
    _forceRender(botState);
  }
}

function _animateBotCut(move, botState, prevState, botIdx) {
  const playedEls  = Array.from(document.querySelectorAll('#played-row .table-card'));
  const row        = document.getElementById('played-row');
  const cutCardIds = move.data?.cut_cards || [];
  const botHand    = prevState.players[botIdx]?.hand || [];

  const cutEls = cutCardIds.map(cid => {
    const cardData = botHand.find(c => c.id === cid);
    const wrapper  = document.createElement('div');
    const isTrump  = botState.trump_suit && cardData?.suit === botState.trump_suit;
    wrapper.innerHTML = buildCard(cardData || { hidden: true }, false, false, false, isTrump);
    const el = wrapper.firstElementChild;
    el.classList.add('no-hover', 'cut-card-anim');
    row.appendChild(el);
    return el;
  });

  logAction(`<span class="le-type">CUT</span> ${botState.players[botIdx].name} cut`);
  _forceReRenderKeys();

  animateCutSequence(cutEls, playedEls, botIdx, () => _forceRender(botState));
}

function _animateBotPass(move, botState, prevState, botIdx) {
  const playedEls = Array.from(document.querySelectorAll('#played-row .table-card'));
  const passCount = move.data?.passed_count || prevState.played_cards?.length || 1;
  const takerIdx  = State.myPlayerIdx;

  logAction(`<span class="le-type">PASS</span> ${botState.players[botIdx].name} passed`);
  _forceReRenderKeys();

  animatePassSequence(passCount, botIdx, takerIdx, playedEls, () => _forceRender(botState));
}


// ─── Cutting ──────────────────────────────────────────────────────────────────

window.doCutCards = async function() {
  if (State.selectedCards.length === 0) return;
  const cards     = [...State.selectedCards];
  const cutterIdx = State.myPlayerIdx;

  // Snapshot played-card DOM nodes before the API call clears them
  const playedEls = Array.from(document.querySelectorAll('#played-row .table-card'));
  clearSel();
  State.passModeActive = false;

  const data = await api(ROOM('cut'), { player_idx: cutterIdx, cards });
  if (!data?.state) return;

  logAction(`<span class="le-type">CUT</span> You cut with ${cards.join(' ')}`);

  // Build DOM elements for the cut cards so we can animate them onto the table
  const row      = document.getElementById('played-row');
  const prevHand = State.gameState.players[cutterIdx].hand || [];

  const cutEls = cards.map(cardId => {
    const cardData = prevHand.find(c => c.id === cardId);
    const wrapper  = document.createElement('div');
    const isTrump  = data.state.trump_suit && cardData?.suit === data.state.trump_suit;
    wrapper.innerHTML = buildCard(cardData || { hidden: true }, false, false, false, isTrump);
    const el = wrapper.firstElementChild;
    el.classList.add('no-hover', 'cut-card-anim');
    row.appendChild(el);
    return el;
  });

  _forceReRenderKeys();
  animateCutSequence(cutEls, playedEls, cutterIdx, () => _forceRender(data.state));
};


// ─── Passing ──────────────────────────────────────────────────────────────────

// Called from pass-mode confirm button (cards pre-selected in pass mode)
window.doPassCards = async function() {
  if (State.selectedCards.length !== State.passCount) return;
  await _executePass([...State.selectedCards]);
};

// Called from the "Pass N" button when cards are already selected in normal mode
window.doPassDirect = async function(n) {
  const cards = [...State.selectedCards];
  if (cards.length !== n) { toast(`Select exactly ${n} card(s) to pass`, 'error'); return; }
  clearSel();
  await _executePass(cards);
};

// Called from the Pass button when 0 cards are selected (or wrong count).
// Automatically picks the N cheapest non-trump cards to pass, so the player
// never has to go through a second selection step just to pass.
window.doPassAuto = async function(n) {
  const s = State.gameState;
  if (!s) return;

  // If player already has exactly n cards selected, just use those
  if (State.selectedCards.length === n) {
    await doPassDirect(n);
    return;
  }

  // Auto-pick: cheapest non-trump cards first, fall back to trump if needed
  const hand      = s.players[State.myPlayerIdx].hand || [];
  const trump     = s.trump_suit;
  const nonTrump  = hand.filter(c => c.suit !== trump).sort((a, b) => a.points - b.points);
  const trumpCards = hand.filter(c => c.suit === trump).sort((a, b) => a.points - b.points);
  const sorted    = [...nonTrump, ...trumpCards];
  const picked    = sorted.slice(0, n).map(c => c.id);

  if (picked.length < n) {
    toast(`Not enough cards to pass`, 'error');
    return;
  }

  clearSel();
  await _executePass(picked);
};

async function _executePass(cards) {
  State.passModeActive = false;
  const playedCount = State.gameState?.played_cards?.length || cards.length;
  const takerIdx    = State.gameState?.playing_player_idx ?? (1 - State.myPlayerIdx);
  const passerIdx   = State.myPlayerIdx;

  // Snapshot played-card elements before the state changes
  const playedEls = Array.from(document.querySelectorAll('#played-row .table-card'));

  const data = await api(ROOM('pass'), { player_idx: passerIdx, cards });
  if (!data?.state) return;

  logAction(`<span class="le-type">PASS</span> You passed ${playedCount} card${playedCount > 1 ? 's' : ''}`);

  _forceReRenderKeys();
  animatePassSequence(playedCount, passerIdx, takerIdx, playedEls,
    () => _forceRender(data.state));
}


// ─── Calculating ──────────────────────────────────────────────────────────────

window.doCalculate = async function() {
  const data = await api(ROOM('calculate'), { player_idx: State.myPlayerIdx });
  if (!data?.state) return;
  State.gameState = data.state;
  State.showTip   = false;
  render();
};

window.doSkipCalc = async function() {
  clearSel();
  _forceReRenderKeys();
  const data = await api(ROOM('skip_calculate'), { player_idx: State.myPlayerIdx });
  if (!data?.state) return;
  State.gameState = data.state;
  State.showTip   = false;
  render();
};


// ─── Internal helpers ─────────────────────────────────────────────────────────

/** Force a full re-render by invalidating the hand and table cache keys. */
function _forceReRenderKeys() {
  State.prevHandKeys  = ['', ''];
  State.prevPlayedKey = '';
}

/** Apply a new game state and re-render from scratch. */
function _forceRender(newState) {
  _forceReRenderKeys();
  State.gameState = newState;
  render();
}