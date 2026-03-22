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
  // Snapshot current keys so remaining hand cards don't re-animate from deck
  const s0 = State.gameState;
  if (s0) {
    State.prevHandKeys[0] = JSON.stringify((s0.players[0].hand || []).map((c,i) => c.id || `h${i}`));
    State.prevHandKeys[1] = JSON.stringify((s0.players[1].hand || []).map((c,i) => c.id || `h${i}`));
  }
  State.prevPlayedKey = '';
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
  // Find the bot's first significant action (cut or pass) — not calculate/draw/skip
  // which happen after and would mask the cut/pass we need to animate
  const myId = State.gameState?.players[State.myPlayerIdx]?.id;
  const botId = State.gameState?.players[1 - State.myPlayerIdx]?.id;
  const lastBotMove = [...(botState.move_history || [])].reverse().find(m =>
    m.player === botId && (m.type === 'cut' || m.type === 'pass' || m.type === 'counter_play')
  );
  const botAction = lastBotMove?.type;

  if (botAction === 'cut') {
    const playedEls  = Array.from(document.querySelectorAll('#played-row .table-card'));
    const row        = document.getElementById('played-row');
    const cutCardIds = lastBotMove.data?.cut_cards || [];

    // Remove cut cards from bot's hand DOM immediately (same as human cut)
    _removeCardsFromHandDOM(botIdx, cutCardIds);

    // Snapshot bot hand keys after removal so only drawn cards animate later
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

    // Remove passed cards from bot's hand DOM immediately (same as human pass)
    // We don't know which cards the bot passed (they're hidden) so just remove
    // the first N face-down card elements from the bot's hand
    _removeNCardsFromHandDOM(botIdx, passCount);

    // Snapshot bot hand keys so only drawn cards animate after the sweep
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

  // Immediately remove the cut cards from the hand DOM
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

// Pass using currently selected cards — requires exactly n cards selected
window.doPassAuto = async function(n) {
  if (State.selectedCards.length !== n) {
    toast(`Select exactly ${n} card${n > 1 ? 's' : ''} to pass`, 'error');
    return;
  }
  const cards = [...State.selectedCards];
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

  // Immediately remove the passed cards from the hand DOM so the player
  // doesn't see them sitting there while the animation plays.
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
  // Snapshot current hand keys BEFORE the API call so renderHandCards
  // can diff old vs new — only the drawn card(s) will animate from deck.
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


// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Clear the table key so renderTable re-renders, but KEEP hand keys intact.
 *  This means renderHandCards will correctly diff old vs new hands and only
 *  animate genuinely new cards (drawn from deck), not cards the player already had. */
function _clearTableKey() {
  State.prevPlayedKey = '';
}

/** Remove N face-down cards from the bot's hand DOM (pass — we don't know which ones). */
function _removeNCardsFromHandDOM(playerIdx, n) {
  const container = document.getElementById(`hand-${playerIdx}`);
  if (!container) return;
  const cards = Array.from(container.querySelectorAll('.card'));
  cards.slice(0, n).forEach(el => el.remove());
  // Update prevHandKeys to reflect n fewer cards
  const remaining = Math.max(0, cards.length - n);
  // Use hidden placeholders since we don't know bot's card IDs
  State.prevHandKeys[playerIdx] = JSON.stringify(Array.from({length: remaining}, (_,i) => `h${i}`));
}

/** Snapshot hand keys after cut cards are removed, so only drawn cards animate. */
function _snapshotHandAfterAction(playerIdx, removedIds) {
  const s = State.gameState;
  if (!s) return;
  const remaining = (s.players[playerIdx].hand || [])
    .filter(c => !removedIds.includes(c.id));
  State.prevHandKeys[playerIdx] = JSON.stringify(remaining.map((c,i) => c.id || `h${i}`));
}

/** Snapshot hand keys after N cards removed (bot pass — IDs unknown). */
function _snapshotHandAfterRemoval(playerIdx, n) {
  const s = State.gameState;
  if (!s) return;
  const hand = s.players[playerIdx].hand || [];
  // Keep remaining cards (skip first n since those were passed)
  const remaining = hand.slice(n);
  State.prevHandKeys[playerIdx] = JSON.stringify(remaining.map((c,i) => c.id || `h${i}`));
}


/** Instantly remove specific cards from the player's hand DOM.
 *  Called before animations so the cards disappear immediately on action,
 *  rather than sitting in the hand while the animation plays.
 *  Also updates prevHandKeys so renderHandCards won't re-animate remaining cards. */
function _removeCardsFromHandDOM(playerIdx, cardIds) {
  const container = document.getElementById(`hand-${playerIdx}`);
  if (!container) return;

  cardIds.forEach(id => {
    const el = container.querySelector(`.card[data-id="${id}"]`);
    if (el) el.remove();
  });

  // Update prevHandKeys to reflect the hand minus the removed cards
  // so the next render() call doesn't re-animate remaining cards from deck
  const s = State.gameState;
  if (s) {
    const remaining = (s.players[playerIdx].hand || [])
      .filter(c => !cardIds.includes(c.id));
    State.prevHandKeys[playerIdx] = JSON.stringify(remaining.map((c,i) => c.id || `h${i}`));
  }
}

/** Snapshot current hand keys so the next render knows what cards existed before. */
function _snapshotHandKeys() {
  State.players?.forEach?.((p, i) => {});  // no-op, keys already set by last render
  // Keys are already up to date — just clear the table so it re-renders
  _clearTableKey();
}

/** Apply new state and render. Hand keys are intentionally NOT wiped —
 *  renderHandCards will diff and only animate cards that are genuinely new. */
function _forceRender(newState) {
  _clearTableKey();
  State.gameState = newState;
  render();
}

/** Used only when we need to force a full hand re-render (e.g. debug toggle). */
function _forceReRenderKeys() {
  State.prevHandKeys  = ['', ''];
  State.prevPlayedKey = '';
}