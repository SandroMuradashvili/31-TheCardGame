'use strict';
// ─────────────────────────────────────────────────────────────────────────────
// game_actions.js — Play, cut, counter, pass, calculate API calls
// ─────────────────────────────────────────────────────────────────────────────
// Stake actions (raise/accept/decline) are in stake_actions.js.
// Depends on: state.js, utils.js, cards.js, animations.js, render.js
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


// ─── Playing ──────────────────────────────────────────────────────────────────
// HvH: single API call.
// HvB: two-step — play_only (animate human), then bot_respond (animate bot).

window.doPlayCards = async function() {
  if (State.selectedCards.length === 0) return;
  const cards = [...State.selectedCards];
  clearSel();
  State.passModeActive = false;

  if (State.gameMode !== 'hvb') {
    const data = await api(ROOM('play'), { player_idx: State.myPlayerIdx, cards });
    if (!data?.state) return;
    State.gameState = data.state;
    logAction(`<span class="le-type">PLAY</span> You played ${cards.join(' ')}`);
    render();
    return;
  }

  const step1 = await api(ROOM('play_only'), { player_idx: State.myPlayerIdx, cards });
  if (!step1?.state) return;
  logAction(`<span class="le-type">PLAY</span> You played ${cards.join(' ')}`);
  _forceReRenderKeys();
  State.gameState = step1.state;
  render();
  setTimeout(_handleBotResponse, 400 + cards.length * 70);
};


// ─── Bot response (HvB only) ──────────────────────────────────────────────────

async function _handleBotResponse() {
  const data = await api(ROOM('bot_respond'), { player_idx: State.myPlayerIdx });
  if (!data?.state) return;

  const botState = data.state;
  const botIdx   = 1 - State.myPlayerIdx;
  const lastBotMove = [...(botState.move_history || [])].reverse().find(m =>
    m.player !== 'system' && m.player !== State.gameState?.players[State.myPlayerIdx]?.id
  );
  const botAction = lastBotMove?.type;

  if (botAction === 'cut') {
    const playedEls  = Array.from(document.querySelectorAll('#played-row .table-card'));
    const row        = document.getElementById('played-row');
    const cutCardIds = lastBotMove.data?.cut_cards || [];
    const botHand    = State.gameState.players[botIdx]?.hand || [];
    const cutEls = cutCardIds.map(cid => {
      const cardData = botHand.find(c => c.id === cid);
      const wrapper  = document.createElement('div');
      wrapper.innerHTML = buildCard(cardData || {hidden:true}, false, false, false,
        botState.trump_suit && cardData?.suit === botState.trump_suit);
      const el = wrapper.firstElementChild;
      el.classList.add('no-hover', 'cut-card-anim');
      row.appendChild(el);
      return el;
    });
    logAction(`<span class="le-type">CUT</span> ${botState.players[botIdx].name} cut`);
    _forceReRenderKeys();
    animateCutSequence(cutEls, playedEls, botIdx, () => _forceRender(botState));

  } else if (botAction === 'pass') {
    const playedEls = Array.from(document.querySelectorAll('#played-row .table-card'));
    const passCount = lastBotMove.data?.passed_count || State.gameState.played_cards?.length || 1;
    logAction(`<span class="le-type">PASS</span> ${botState.players[botIdx].name} passed`);
    _forceReRenderKeys();
    animatePassSequence(passCount, botIdx, State.myPlayerIdx, playedEls, () => _forceRender(botState));

  } else {
    _forceRender(botState);
  }
}


// ─── Cutting ──────────────────────────────────────────────────────────────────

window.doCutCards = async function() {
  if (State.selectedCards.length === 0) return;
  const cards     = [...State.selectedCards];
  const cutterIdx = State.myPlayerIdx;
  const playedEls = Array.from(document.querySelectorAll('#played-row .table-card'));
  clearSel();
  State.passModeActive = false;

  const data = await api(ROOM('cut'), { player_idx: cutterIdx, cards });
  if (!data?.state) return;
  logAction(`<span class="le-type">CUT</span> You cut with ${cards.join(' ')}`);

  const row      = document.getElementById('played-row');
  const prevHand = State.gameState.players[cutterIdx].hand || [];
  const cutEls = cards.map(cardId => {
    const cardData = prevHand.find(c => c.id === cardId);
    const wrapper  = document.createElement('div');
    wrapper.innerHTML = buildCard(cardData || {hidden:true}, false, false, false,
      data.state.trump_suit && cardData?.suit === data.state.trump_suit);
    const el = wrapper.firstElementChild;
    el.classList.add('no-hover', 'cut-card-anim');
    row.appendChild(el);
    return el;
  });

  _forceReRenderKeys();
  animateCutSequence(cutEls, playedEls, cutterIdx, () => _forceRender(data.state));
};


// ─── Counter-play ─────────────────────────────────────────────────────────────
// During cutting phase, play 3 non-trump same-suit cards to flip the attack.
// The opponent's played cards go back to them; your 3 become the new table cards.

window.doCounterPlay = async function() {
  if (!isCounterSelection()) return;
  const cards     = [...State.selectedCards];
  const cutterIdx = State.myPlayerIdx;
  const playedEls = Array.from(document.querySelectorAll('#played-row .table-card'));
  clearSel();
  State.passModeActive = false;

  const data = await api(ROOM('counter_play'), { player_idx: cutterIdx, cards });
  if (!data?.state) return;
  logAction(`<span class="le-type">COUNTER</span> You counter-played ${cards.join(' ')}`);

  // Remove original played cards from DOM
  playedEls.forEach(el => { try { el.parentNode?.removeChild(el); } catch(_){} });

  // Add your counter cards and animate them to the table
  const row      = document.getElementById('played-row');
  const prevHand = State.gameState.players[cutterIdx].hand || [];
  const counterEls = cards.map(cardId => {
    const cardData = prevHand.find(c => c.id === cardId);
    const wrapper  = document.createElement('div');
    wrapper.innerHTML = buildCard(cardData || {hidden:true}, false, false, false,
      data.state.trump_suit && cardData?.suit === data.state.trump_suit);
    const el = wrapper.firstElementChild;
    el.classList.add('no-hover', 'table-card');
    row.appendChild(el);
    return el;
  });

  _forceReRenderKeys();
  animatePlayToTable(counterEls, cutterIdx, () => {
    if (State.gameMode === 'hvb') {
      setTimeout(_handleBotResponse, 200);
    } else {
      _forceRender(data.state);
    }
  });
};


// ─── Passing ──────────────────────────────────────────────────────────────────

// Auto-pass: if cards are selected uses them, otherwise picks cheapest non-trump
window.doPassAuto = async function(n) {
  const s = State.gameState;
  if (!s) return;

  let cards;
  if (State.selectedCards.length === n) {
    cards = [...State.selectedCards];
  } else {
    const hand     = s.players[State.myPlayerIdx].hand || [];
    const sorted   = [...hand.filter(c => c.suit !== s.trump_suit).sort((a,b) => a.points - b.points),
                      ...hand.filter(c => c.suit === s.trump_suit).sort((a,b) => a.points - b.points)];
    cards = sorted.slice(0, n).map(c => c.id);
    if (cards.length < n) { toast('Not enough cards to pass', 'error'); return; }
  }

  clearSel();
  await _executePass(cards);
};

// Used by pass-mode confirm button
window.doPassCards = async function() {
  if (State.selectedCards.length !== State.passCount) return;
  await _executePass([...State.selectedCards]);
};

async function _executePass(cards) {
  State.passModeActive  = false;
  const playedCount     = State.gameState?.played_cards?.length || cards.length;
  const takerIdx        = State.gameState?.playing_player_idx ?? (1 - State.myPlayerIdx);
  const passerIdx       = State.myPlayerIdx;
  const playedEls       = Array.from(document.querySelectorAll('#played-row .table-card'));

  const data = await api(ROOM('pass'), { player_idx: passerIdx, cards });
  if (!data?.state) return;
  logAction(`<span class="le-type">PASS</span> You passed ${playedCount} card${playedCount>1?'s':''}`);
  _forceReRenderKeys();
  animatePassSequence(playedCount, passerIdx, takerIdx, playedEls, () => _forceRender(data.state));
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


// ─── Helpers ──────────────────────────────────────────────────────────────────

function _forceReRenderKeys() {
  State.prevHandKeys  = ['', ''];
  State.prevPlayedKey = '';
}

function _forceRender(newState) {
  _forceReRenderKeys();
  State.gameState = newState;
  render();
}