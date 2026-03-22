'use strict';
// ─────────────────────────────────────────────────────────────────────────────
// lobby.js — Lobby UI, room creation/joining, HvH polling, keyboard shortcuts
// ─────────────────────────────────────────────────────────────────────────────
// Depends on: state.js, utils.js, render.js, game_actions.js
// Loaded by:  index.html (last — depends on everything else)
// ─────────────────────────────────────────────────────────────────────────────


// ─── Tab switching ────────────────────────────────────────────────────────────

window.switchTab = function(tab) {
  ['hvb', 'create', 'join', 'bvb'].forEach(t => {
    document.getElementById(`tab-${t}`).classList.toggle('active', t === tab);
    document.getElementById(`tab-${t}-content`).style.display = t === tab ? '' : 'none';
  });
};


// ─── Room creation / joining ──────────────────────────────────────────────────

window.startHvB = async function() {
  const name   = document.getElementById('lb-name-hvb').value.trim() || 'Player';
  const target = document.getElementById('lb-target-hvb').value;
  State.gameMode    = 'hvb';
  State.myPlayerIdx = 0;
  const data = await api('/api/create_room', { player_name: name, mode: 'hvb', target_score: target });
  if (!data) return;
  State.roomId = data.room_id;
  enterGame(data);
};

window.createRoom = async function() {
  const name   = document.getElementById('lb-name-host').value.trim() || 'Host';
  const target = document.getElementById('lb-target-host').value;
  State.gameMode    = 'hvh';
  State.myPlayerIdx = 0;
  const data = await api('/api/create_room', { player_name: name, mode: 'hvh', target_score: target });
  if (!data) return;
  State.roomId = data.room_id;

  document.getElementById('room-code-display').textContent    = data.room_id;
  document.getElementById('room-link-display').innerHTML =
    `Share: <a href="${data.join_link}" target="_blank">${data.join_link}</a>`;
  document.getElementById('room-waiting').style.display       = 'block';
  document.querySelector('#tab-create-content .btn').disabled = true;

  // Poll until the guest joins
  const waitPoll = setInterval(async () => {
    const st = await api(`/api/room/${State.roomId}/status`);
    if (st && st.is_full) {
      clearInterval(waitPoll);
      const stData = await api(ROOM(`state?perspective=${State.myPlayerIdx}`));
      if (stData && stData.state) {
        State.gameState = stData.state;
        enterGame({ room_id: State.roomId });
      }
    }
  }, 1200);
};

window.joinRoom = async function() {
  const name = document.getElementById('lb-name-guest').value.trim() || 'Guest';
  const code = document.getElementById('lb-room-code').value.toUpperCase().trim();
  if (!code) { toast('Enter a room code', 'error'); return; }
  State.gameMode    = 'hvh';
  State.myPlayerIdx = 1;
  const data = await api('/api/join_room', { player_name: name, room_id: code });
  if (!data) return;
  State.roomId = data.room_id;
  enterGame(data);
};


// ─── Enter / leave game ───────────────────────────────────────────────────────

window.enterGame = function(data) {
  document.getElementById('lobby').style.display   = 'none';
  document.getElementById('game-ui').style.display = 'block';
  State.selectedCards  = [];
  State.passModeActive = false;
  State.modalShown     = false;
  document.getElementById('action-log').innerHTML  = '';

  if (State.gameMode === 'hvb') {
    refreshState();
  } else {
    startRoundAfterJoin();
    startPolling();
  }
};

window.startRoundAfterJoin = async function() {
  const data = await api(ROOM('start_round'), { player_idx: State.myPlayerIdx });
  if (data && data.state) { State.gameState = data.state; render(); }
};

window.showLobby = function() {
  stopPolling();
  document.getElementById('lobby').style.display   = 'flex';
  document.getElementById('game-ui').style.display = 'none';
  document.getElementById('room-waiting').style.display = 'none';
  const createBtn = document.querySelector('#tab-create-content .btn');
  if (createBtn) createBtn.disabled = false;
  State.roomId        = null;
  State.gameState     = null;
  State.selectedCards = [];
  State.modalShown    = false;
  State.bvbRunning    = false;
};


// ─── HvH polling ─────────────────────────────────────────────────────────────
// Polls for opponent's move every 1.2s when it's not our turn.

window.startPolling = function() {
  stopPolling();
  State.pollTimer = setInterval(async () => {
    if (!State.roomId || !State.gameState) return;
    const phase        = State.gameState?.phase;
    const { myPlayerIdx } = State;
    const s            = State.gameState;

    const isMyTurn =
      (phase === 'playing'    && s.active_player_idx    === myPlayerIdx)
   || (phase === 'stakes'     && s.stake_offerer_idx    !== myPlayerIdx)
   || ((phase === 'cutting' || phase === 'forced_cut')  && s.playing_player_idx === myPlayerIdx)
   || (phase === 'calculating' && s.calculator_idx      === myPlayerIdx);

    if (isMyTurn) return;
    if (phase === 'round_over' || phase === 'game_over') return;

    await refreshState(false);
  }, 1200);
};

window.stopPolling = function() {
  if (State.pollTimer) { clearInterval(State.pollTimer); State.pollTimer = null; }
};


// ─── Bot vs Bot ──────────────────────────────────────────────────────────────

window.startBvB = async function() {
  const target = document.getElementById('lb-target-bvb').value;
  const speed  = document.getElementById('lb-bvb-speed').value;

  const data = await api('/api/create_bvb', { target_score: target });
  if (!data) return;

  State.roomId      = data.room_id;
  State.gameMode    = 'bvb';
  State.myPlayerIdx = 0;   // spectator — view from p0 perspective, but debug=true shows all
  State.bvbSpeed    = speed;
  State.bvbRunning  = false;
  State.bvbGameState = data.state;

  document.getElementById('lobby').style.display   = 'none';
  document.getElementById('game-ui').style.display = 'block';
  State.gameState = data.state;
  render();

  // Show the BvB control bar
  _bvbShowControls();

  if (speed === 'instant') {
    await _bvbRunInstant();
  } else if (speed !== '0') {
    _bvbStartAuto();
  }
  // speed === '0' means manual — user clicks Step button
};

function _bvbShowControls() {
  const panel = document.getElementById('action-panel');
  const ms    = State.bvbSpeed;
  panel.innerHTML = `
    <div class="action-title">Bot vs Bot</div>
    <div class="action-msg" id="action-msg" style="color:var(--text-dim);font-size:.85rem;">
      ${ms === '0' ? 'Manual mode — click Step to advance.' : ms === 'instant' ? 'Running instantly…' : `Auto-stepping every ${ms}ms`}
    </div>
    <div class="btn-row" id="action-btns">
      ${ms === '0' ? `<button class="btn btn-gold" onclick="bvbStep()">Step →</button>` : ''}
      ${ms !== 'instant' ? `<button class="btn btn-ghost btn-sm" onclick="bvbStop()">Stop</button>` : ''}
      <button class="btn btn-ghost btn-sm" onclick="showLobby()">◀ Back</button>
    </div>
  `;
}

window.bvbStep = async function() {
  if (!State.roomId) return;
  const data = await api(ROOM('bvb_step'), {});
  if (!data) return;
  State.prevHandKeys  = ['', ''];
  State.prevPlayedKey = '';
  State.gameState     = data.state;
  render();

  const msgEl = document.getElementById('action-msg');
  if (msgEl) msgEl.textContent = `Last action: ${data.action}`;

  if (data.state.phase === 'game_over') {
    const w = data.state.players.find(p => p.game_score >= data.state.target_score);
    if (msgEl) msgEl.textContent = `Game over — ${w?.name || '?'} wins!`;
  }
};

window.bvbStop = function() {
  State.bvbRunning = false;
  const msgEl = document.getElementById('action-msg');
  if (msgEl) msgEl.textContent = 'Stopped. Click Step to continue.';
  const btnsEl = document.getElementById('action-btns');
  if (btnsEl) btnsEl.innerHTML = `
    <button class="btn btn-gold" onclick="bvbStep()">Step →</button>
    <button class="btn btn-ghost btn-sm" onclick="showLobby()">◀ Back</button>
  `;
};

function _bvbStartAuto() {
  State.bvbRunning = true;
  const delay = parseInt(State.bvbSpeed, 10);
  const tick = async () => {
    if (!State.bvbRunning || !State.roomId) return;
    if (State.gameState?.phase === 'game_over') { State.bvbRunning = false; return; }
    await bvbStep();
    if (State.bvbRunning) setTimeout(tick, delay);
  };
  setTimeout(tick, delay);
}

async function _bvbRunInstant() {
  const data = await api(ROOM('bvb_run'));
  if (!data) return;
  State.prevHandKeys  = ['', ''];
  State.prevPlayedKey = '';
  State.gameState     = data.state;
  render();
  const msgEl = document.getElementById('action-msg');
  if (msgEl) {
    const w = data.state.players.find(p => p.game_score >= data.state.target_score);
    msgEl.textContent = `Done — ${data.history.length} actions. ${w?.name || '?'} wins!`;
  }
  const btnsEl = document.getElementById('action-btns');
  if (btnsEl) btnsEl.innerHTML = `<button class="btn btn-ghost btn-sm" onclick="showLobby()">◀ Back</button>`;
}


// ─── URL-based join ───────────────────────────────────────────────────────────
// Handles links like http://192.168.1.x:5000/join/ABCDEF

window.checkUrlJoin = function() {
  const m = window.location.pathname.match(/^\/join\/([A-Z]{6})$/i);
  if (m) {
    const code = m[1].toUpperCase();
    switchTab('join');
    document.getElementById('lb-room-code').value = code;
  }
};


// ─── Debug toggles ────────────────────────────────────────────────────────────

window.toggleDebug = function() {
  State.debugMode = !State.debugMode;
  document.getElementById('debug-panel').classList.toggle('visible', State.debugMode);
  document.getElementById('debug-btn').textContent = State.debugMode ? '⚙ Debug ON' : '⚙ Debug';
  State.prevHandKeys = ['', ''];
  if (State.gameState) render();
  if (State.debugMode) refreshState(true);
};




// ─── Keyboard shortcuts ───────────────────────────────────────────────────────

document.addEventListener('keydown', e => {
  if (!State.gameState) return;
  const s = State.gameState;

  if (e.key === 'Enter') {
    if (s.phase === 'playing' && s.active_player_idx === State.myPlayerIdx && State.selectedCards.length)
      doPlayCards();
    if ((s.phase === 'cutting' || s.phase === 'forced_cut') &&
        State.selectedCards.length === (s.played_cards?.length || 0))
      doCutCards();
    if (State.passModeActive && State.selectedCards.length === State.passCount)
      doPassCards();
  }

  if (e.key === 'Escape') {
    clearSel();
    if (State.passModeActive) cancelPass();
  }
});