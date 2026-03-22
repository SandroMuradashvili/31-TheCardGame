'use strict';
// ─────────────────────────────────────────────────────────────────────────────
// bvb_ui.js — BvB Spectator UI
// ─────────────────────────────────────────────────────────────────────────────
// Completely separate from the HvB render pipeline.
// Owns #bvb-ui. Never touches #game-ui or #lobby.
//
// Depends on: state.js (State), utils.js (api, toast, logAction)
// ─────────────────────────────────────────────────────────────────────────────

// ─── BvB State ────────────────────────────────────────────────────────────────
const BvB = {
  roomId:      null,
  state:       null,
  prevState:   null,
  playing:     false,
  speed:       500,      // ms between auto steps; 0 = manual
  timer:       null,
  stepping:    false,    // prevent overlapping step calls
  actionLog:   [],
  lastAction:  '',
};

const SUIT_SYMS = { hearts: '♥', diamonds: '♦', clubs: '♣', spades: '♠' };
const SUIT_CLS  = { hearts: 'bvb-red', diamonds: 'bvb-red', clubs: 'bvb-blk', spades: 'bvb-blk' };

function bvbSym(suit) { return SUIT_SYMS[suit] || suit; }
function bvbCls(suit) { return SUIT_CLS[suit] || ''; }


// ─── Entry / exit ─────────────────────────────────────────────────────────────

window.enterBvB = async function() {
  const botA   = document.getElementById('bvb-bot-a').value;
  const botB   = document.getElementById('bvb-bot-b').value;
  const target = document.getElementById('bvb-target').value;
  const speed  = document.getElementById('bvb-speed').value;

  BvB.speed    = parseInt(speed, 10);
  BvB.playing  = false;
  BvB.stepping = false;
  BvB.actionLog = [];

  const data = await api('/api/create_bvb', { bot_a: botA, bot_b: botB, target_score: target });
  if (!data) return;

  BvB.roomId = data.room_id;
  BvB.state  = data.state;

  document.getElementById('lobby').style.display   = 'none';
  document.getElementById('game-ui').style.display = 'none';
  document.getElementById('bvb-ui').style.display  = 'block';

  // Update static labels that don't come from state
  const targetDisplay = document.getElementById('bvb-target-val');
  if (targetDisplay) targetDisplay.textContent = target;
  const nameA = document.getElementById('bvb-hand-name-0');
  const nameB = document.getElementById('bvb-hand-name-1');
  if (nameA) nameA.textContent = botA;
  if (nameB) nameB.textContent = botB;
  document.getElementById('bvb-pile-name-0').textContent = `${botA} — Pile`;
  document.getElementById('bvb-pile-name-1').textContent = `${botB} — Pile`;

  bvbRender();

  if (BvB.speed === 0) {
    bvbSetStatus('Manual mode — press Step to advance one action.');
  } else {
    bvbStartAuto();
  }
};

window.exitBvB = function() {
  bvbStopAuto();
  BvB.roomId = null;
  BvB.state  = null;
  document.getElementById('bvb-ui').style.display  = 'none';
  document.getElementById('lobby').style.display   = 'flex';
};


// ─── Step ─────────────────────────────────────────────────────────────────────

window.bvbManualStep = async function() {
  if (BvB.stepping) return;
  await _doStep();
};

async function _doStep() {
  if (!BvB.roomId || BvB.stepping) return;
  BvB.stepping = true;
  try {
    const data = await api(`/api/room/${BvB.roomId}/bvb_step`, {});
    if (!data) return;
    BvB.prevState = BvB.state;
    BvB.state     = data.state;
    BvB.lastAction = data.action;
    _logAction(data.action, data.state);
    bvbRender();

    if (data.state.phase === 'game_over') {
      bvbStopAuto();
      bvbSetStatus(`Game over! Check scores above.`);
    }
  } finally {
    BvB.stepping = false;
  }
}

function _logAction(action, s) {
  if (!action || action === 'done') return;
  const labels = {
    play:           '▶ Play',
    cut:            '✂ Cut',
    pass:           '→ Pass',
    counter_play:   '⚡ Counter',
    calculate:      '🧮 Calculate',
    skip_calculate: '↷ Skip calc',
    start_round:    '🃏 New round',
    start_play:     '▶ Start play',
    accept_stake:   '✓ Accept stake',
    decline_stake:  '✗ Decline stake',
  };
  const label = labels[action] || action;
  BvB.actionLog.unshift(`${label}`);
  if (BvB.actionLog.length > 40) BvB.actionLog.pop();
}


// ─── Auto play ────────────────────────────────────────────────────────────────

window.bvbStartAuto = function() {
  bvbStopAuto();
  if (!BvB.roomId) return;
  BvB.playing = true;
  bvbSetStatus(`Auto-playing at ${BvB.speed}ms/step…`);
  _autoTick();
};

function _autoTick() {
  if (!BvB.playing || !BvB.roomId) return;
  if (BvB.state?.phase === 'game_over') { bvbStopAuto(); return; }
  _doStep().then(() => {
    if (BvB.playing) BvB.timer = setTimeout(_autoTick, BvB.speed);
  });
}

window.bvbStopAuto = function() {
  BvB.playing = false;
  if (BvB.timer) { clearTimeout(BvB.timer); BvB.timer = null; }
  bvbSetStatus('Paused.');
};

window.bvbChangeSpeed = function(ms) {
  BvB.speed = ms;
  if (BvB.playing) { bvbStopAuto(); bvbStartAuto(); }
};

function bvbSetStatus(msg) {
  const el = document.getElementById('bvb-status');
  if (el) el.textContent = msg;
}


// ─── Main render ──────────────────────────────────────────────────────────────

window.bvbRender = function() {
  const s = BvB.state;
  if (!s) return;

  _renderBvBScoreBar(s);
  _renderBvBTrump(s);
  _renderBvBDeck(s);
  _renderBvBTable(s);
  _renderBvBHand(0, s);
  _renderBvBHand(1, s);
  _renderBvBPiles(s);
  _renderBvBLog();
  _renderBvBControls(s);
};


// ─── Score bar ────────────────────────────────────────────────────────────────

function _renderBvBScoreBar(s) {
  const p0 = s.players[0];
  const p1 = s.players[1];

  document.getElementById('bvb-name-0').textContent  = p0.name;
  document.getElementById('bvb-name-1').textContent  = p1.name;
  document.getElementById('bvb-score-0').textContent = p0.game_score;
  document.getElementById('bvb-score-1').textContent = p1.game_score;
  document.getElementById('bvb-target').textContent  = s.target_score;
  document.getElementById('bvb-round-num').textContent = s.round_number;
  document.getElementById('bvb-stake-num').textContent = s.current_stake;
  document.getElementById('bvb-phase-txt').textContent = s.phase.replace('_', ' ').toUpperCase();

  // Active player highlight
  document.getElementById('bvb-player-0').classList.toggle('bvb-active-player', s.active_player_idx === 0 && s.phase === 'playing');
  document.getElementById('bvb-player-1').classList.toggle('bvb-active-player', s.active_player_idx === 1 && s.phase === 'playing');
}


// ─── Trump + deck ─────────────────────────────────────────────────────────────

function _renderBvBTrump(s) {
  const tc = s.trump_card;
  const ts = s.trump_suit;
  const el = document.getElementById('bvb-trump-display');
  if (!el) return;
  if (tc) {
    el.innerHTML = `<span class="${bvbCls(ts)}">${tc.rank}${bvbSym(ts)}</span>`;
    el.title = `Trump: ${tc.rank} of ${ts}`;
  } else {
    el.textContent = '—';
  }
}

function _renderBvBDeck(s) {
  const el = document.getElementById('bvb-deck-list');
  if (!el) return;
  const cards = s.deck?.cards || [];
  const trump = s.trump_suit;
  document.getElementById('bvb-deck-count').textContent = `${s.deck?.size ?? 0} cards`;

  if (cards.length === 0) {
    el.innerHTML = '<span style="color:var(--bvb-dim)">Deck empty</span>';
    return;
  }

  // Group by suit
  const bySuit = { hearts: [], diamonds: [], clubs: [], spades: [] };
  cards.forEach(c => { if (bySuit[c.suit]) bySuit[c.suit].push(c.rank); });

  el.innerHTML = Object.entries(bySuit).map(([suit, ranks]) => {
    if (!ranks.length) return '';
    const isTrump = suit === trump;
    const sym     = bvbSym(suit);
    const cls     = bvbCls(suit);
    const pips    = ranks.join(' ');
    return `<span class="${cls}${isTrump ? ' bvb-trump-suit' : ''}" title="${suit}">${sym} ${pips}</span>`;
  }).filter(Boolean).join('<span class="bvb-suit-sep"> | </span>');
}


// ─── Table ────────────────────────────────────────────────────────────────────

function _renderBvBTable(s) {
  const el = document.getElementById('bvb-table-cards');
  if (!el) return;
  const played = s.played_cards || [];
  const trump  = s.trump_suit;

  if (!played.length) {
    const phaseMsg = {
      stakes:      'Stakes phase',
      playing:     'Waiting for play…',
      calculating: 'Calculating…',
      round_over:  'Round over',
      game_over:   'Game over',
    };
    el.innerHTML = `<span class="bvb-table-empty">${phaseMsg[s.phase] || s.phase}</span>`;
    return;
  }

  el.innerHTML = played.map(c => {
    const isTrump = c.suit === trump;
    const cls = `bvb-table-card ${bvbCls(c.suit)} ${isTrump ? 'bvb-trump-card' : ''}`;
    return `<span class="${cls}">${c.rank}${bvbSym(c.suit)}</span>`;
  }).join('');

  if (s.is_maliutka) {
    el.insertAdjacentHTML('beforeend', '<span class="bvb-maliutka-tag">⚡ MALIUTKA</span>');
  }
}


// ─── Hands ────────────────────────────────────────────────────────────────────

function _renderBvBHand(idx, s) {
  const el = document.getElementById(`bvb-hand-${idx}`);
  if (!el) return;
  const player = s.players[idx];
  const hand   = player.hand || [];
  const trump  = s.trump_suit;

  if (!hand.length) {
    el.innerHTML = '<span class="bvb-no-cards">No cards</span>';
    return;
  }

  el.innerHTML = hand.map(c => {
    if (c.hidden) return `<span class="bvb-hand-card bvb-hidden">?</span>`;
    const isTrump = c.suit === trump;
    const cls = `bvb-hand-card ${bvbCls(c.suit)} ${isTrump ? 'bvb-trump-card' : ''}`;
    return `<span class="${cls}" title="${c.rank} of ${c.suit} (${c.points}pts)">${c.rank}${bvbSym(c.suit)}</span>`;
  }).join('');
}


// ─── Piles ────────────────────────────────────────────────────────────────────

function _renderBvBPiles(s) {
  [0, 1].forEach(idx => {
    const p   = s.players[idx];
    const el  = document.getElementById(`bvb-pile-${idx}`);
    if (!el) return;
    const known  = p.known_pile_points;
    const hidden = p.hidden_card_count;
    const total  = p.pile_points ?? (known + hidden * 6); // debug gives true total
    const hmin   = p.hidden_min_points ?? hidden * 2;
    const hmax   = p.hidden_max_points ?? hidden * 11;

    el.innerHTML = `
      <span class="bvb-pile-known">${known}pts known</span>
      ${hidden > 0
        ? `<span class="bvb-pile-hidden">+${hidden} hidden (${hmin}–${hmax})</span>`
        : ''}
      <span class="bvb-pile-true">= ${total}pts true</span>
      <span class="bvb-pile-count">${p.pile_count} cards</span>
    `;

    // 31+ threshold indicator
    const bar = document.getElementById(`bvb-pile-bar-${idx}`);
    if (bar) {
      const pct = Math.min(100, Math.round(total / 31 * 100));
      bar.style.width = `${pct}%`;
      bar.className = `bvb-pile-bar-fill ${total >= 31 ? 'bvb-bar-win' : total >= 20 ? 'bvb-bar-close' : ''}`;
    }
  });
}


// ─── Action log ───────────────────────────────────────────────────────────────

function _renderBvBLog() {
  const el = document.getElementById('bvb-action-log');
  if (!el) return;
  el.innerHTML = BvB.actionLog.slice(0, 20).map((entry, i) =>
    `<div class="bvb-log-entry ${i === 0 ? 'bvb-log-latest' : ''}">${entry}</div>`
  ).join('');
}


// ─── Controls ─────────────────────────────────────────────────────────────────

function _renderBvBControls(s) {
  const stepBtn  = document.getElementById('bvb-btn-step');
  const playBtn  = document.getElementById('bvb-btn-play');
  const pauseBtn = document.getElementById('bvb-btn-pause');
  const done     = s.phase === 'game_over';

  if (stepBtn)  stepBtn.disabled  = done || BvB.playing;
  if (playBtn)  playBtn.disabled  = done || BvB.playing || BvB.speed === 0;
  if (pauseBtn) pauseBtn.disabled = !BvB.playing;

  // Highlight active speed button
  document.querySelectorAll('.bvb-speed-btn').forEach(btn => {
    btn.classList.toggle('bvb-speed-active', parseInt(btn.dataset.speed) === BvB.speed);
  });
}


// ─── Init: populate bot dropdowns ─────────────────────────────────────────────

window.bvbInitLobby = async function() {
  const data = await api('/api/bots/list');
  if (!data?.bots) return;
  const bots = data.bots;
  ['bvb-lobby-bot-a', 'bvb-lobby-bot-b'].forEach((id, i) => {
    const sel = document.getElementById(id);
    if (!sel) return;
    sel.innerHTML = bots.map((b, j) =>
      `<option value="${b}" ${j === i % bots.length ? 'selected' : ''}>${b}</option>`
    ).join('');
  });
};