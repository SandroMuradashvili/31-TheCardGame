'use strict';
// ─────────────────────────────────────────────────────────────────────────────
// game_actions.js — Play, cut, counter, pass, calculate API calls
// ─────────────────────────────────────────────────────────────────────────────

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

  if (State.gameMode === 'hvh') {
    // In HvH, only the host (player 0) starts the round.
    // Player 1 resumes polling and will see the new round state automatically.
    if (State.myPlayerIdx === 0) {
      const data = await api(ROOM('start_round'), { player_idx: State.myPlayerIdx });
      if (data?.state) { State.gameState = data.state; render(); }
    } else {
      // Guest: just wait — polling will pick up the new round once host starts it.
      // Show a brief waiting message.
      document.getElementById('action-msg').textContent = 'Waiting for host to start next round…';
      document.getElementById('action-btns').innerHTML  = '';
    }
    return;
  }

  // HvB or single-player
  const data = await api(ROOM('start_round'), { player_idx: State.myPlayerIdx });
  if (data?.state) { State.gameState = data.state; render(); }
};


// ─── Playing ──────────────────────────────────────────────────────────────────

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
  const s0 = State.gameState;
  if (s0) {
    State.prevHandKeys[0] = JSON.stringify((s0.players[0].hand || []).map((c,i) => c.id || `h${i}`));
    State.prevHandKeys[1] = JSON.stringify((s0.players[1].hand || []).map((c,i) => c.id || `h${i}`));
  }
  State.prevPlayedKey = '';
  State.gameState = step1.state;
  render();

  // BUG FIX: If playing BURA instantly ended the round, do NOT ask the bot to respond!
  if (step1.state.phase === 'round_over' || step1.state.phase === 'game_over') return;

  setTimeout(_handleBotResponse, 400 + cards.length * 70);
};


// ─── Bot response (HvB only) ──────────────────────────────────────────────────

async function _handleBotResponse() {
  const data = await api(ROOM('bot_respond'), { player_idx: State.myPlayerIdx });
  if (!data?.state) return;

  const botState = data.state;
  const botIdx   = 1 - State.myPlayerIdx;
  const botId    = State.gameState?.players[1 - State.myPlayerIdx]?.id;
  const lastBotMove = [...(botState.move_history || [])].reverse().find(m =>
    m.player === botId && (m.type === 'cut' || m.type === 'pass' || m.type === 'counter_play')
  );
  const botAction = lastBotMove?.type;

  if (botAction === 'cut') {
    const playedEls  = Array.from(document.querySelectorAll('#played-row .table-card'));
    const row        = document.getElementById('played-row');
    const cutCardIds = lastBotMove.data?.cut_cards || [];

    _removeCardsFromHandDOM(botIdx, cutCardIds);
    _snapshotHandAfterAction(botIdx, cutCardIds);

    const cutEls = cutCardIds.map(cid => {
      const rankMap = { A:'A', T:'T', K:'K', Q:'Q', J:'J' };
      const suitMap = { H:'hearts', D:'diamonds', C:'clubs', S:'spades' };
      const rank = rankMap[cid[0]];
      const suit = suitMap[cid[cid.length - 1]];
      const cardData = rank && suit ? { id: cid, rank, suit } : null;
      const isTrump  = cardData && botState.trump_suit && cardData.suit === botState.trump_suit;
      const wrapper  = document.createElement('div');
      wrapper.innerHTML = buildCard(cardData || {hidden:true}, false, false, false, isTrump);
      const el = wrapper.firstElementChild;
      el.classList.add('no-hover', 'cut-card-anim');
      row.appendChild(el);
      return el;
    });
    logAction(`<span class="le-type">CUT</span> ${botState.players[botIdx].name} cut`);
    _clearTableKey();
    animateCutSequence(cutEls, playedEls, botIdx, () => _forceRender(botState));

  } else if (botAction === 'pass') {
    const playedEls = Array.from(document.querySelectorAll('#played-row .table-card'));
    const passCount = lastBotMove.data?.passed_count || State.gameState.played_cards?.length || 1;

    _removeNCardsFromHandDOM(botIdx, passCount);
    _snapshotHandAfterRemoval(botIdx, passCount);

    logAction(`<span class="le-type">PASS</span> ${botState.players[botIdx].name} passed`);
    _clearTableKey();
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

  _removeCardsFromHandDOM(cutterIdx, cards);

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

  _clearTableKey();
  animateCutSequence(cutEls, playedEls, cutterIdx, () => _forceRender(data.state));
};


// ─── Counter-play ─────────────────────────────────────────────────────────────

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

  playedEls.forEach(el => { try { el.parentNode?.removeChild(el); } catch(_){} });

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

  _clearTableKey();
  animatePlayToTable(counterEls, cutterIdx, () => {
    if (State.gameMode === 'hvb') {
      setTimeout(_handleBotResponse, 200);
    } else {
      _forceRender(data.state);
    }
  });
};


// ─── Passing ──────────────────────────────────────────────────────────────────

window.doPassAuto = async function(n) {
  if (State.selectedCards.length !== n) {
    toast(`Select exactly ${n} card${n > 1 ? 's' : ''} to pass`, 'error');
    return;
  }
  const cards = [...State.selectedCards];
  clearSel();
  await _executePass(cards);
};

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

  _removeCardsFromHandDOM(passerIdx, cards);
  _clearTableKey();
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
  const s = State.gameState;
  if (s) {
    State.prevHandKeys[0] = JSON.stringify((s.players[0].hand || []).map((c,i) => c.id || `h${i}`));
    State.prevHandKeys[1] = JSON.stringify((s.players[1].hand || []).map((c,i) => c.id || `h${i}`));
  }
  State.prevPlayedKey = '';
  const data = await api(ROOM('skip_calculate'), { player_idx: State.myPlayerIdx });
  if (!data?.state) return;
  State.gameState = data.state;
  State.showTip   = false;
  render();
};


// ─── DOM helpers ──────────────────────────────────────────────────────────────

function _clearTableKey() {
  State.prevPlayedKey = '';
}

function _removeNCardsFromHandDOM(playerIdx, n) {
  const container = document.getElementById(`hand-${playerIdx}`);
  if (!container) return;
  const cards = Array.from(container.querySelectorAll('.card'));
  cards.slice(0, n).forEach(el => el.remove());
  const remaining = Math.max(0, cards.length - n);
  State.prevHandKeys[playerIdx] = JSON.stringify(Array.from({length: remaining}, (_,i) => `h${i}`));
}

function _snapshotHandAfterAction(playerIdx, removedIds) {
  const s = State.gameState;
  if (!s) return;
  const remaining = (s.players[playerIdx].hand || [])
    .filter(c => !removedIds.includes(c.id));
  State.prevHandKeys[playerIdx] = JSON.stringify(remaining.map((c,i) => c.id || `h${i}`));
}

function _snapshotHandAfterRemoval(playerIdx, n) {
  const s = State.gameState;
  if (!s) return;
  const hand = s.players[playerIdx].hand || [];
  const remaining = hand.slice(n);
  State.prevHandKeys[playerIdx] = JSON.stringify(remaining.map((c,i) => c.id || `h${i}`));
}

function _removeCardsFromHandDOM(playerIdx, cardIds) {
  const container = document.getElementById(`hand-${playerIdx}`);
  if (!container) return;
  cardIds.forEach(id => {
    const el = container.querySelector(`.card[data-id="${id}"]`);
    if (el) el.remove();
  });
  const s = State.gameState;
  if (s) {
    const remaining = (s.players[playerIdx].hand || [])
      .filter(c => !cardIds.includes(c.id));
    State.prevHandKeys[playerIdx] = JSON.stringify(remaining.map((c,i) => c.id || `h${i}`));
  }
}

function _forceRender(newState) {
  _clearTableKey();
  State.gameState = newState;
  render();
}