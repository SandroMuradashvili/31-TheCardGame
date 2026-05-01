'use strict';
window.switchTab = function(tab) {
  ['hvb', 'create', 'join', 'bvb'].forEach(t => {
    document.getElementById(`tab-${t}`).classList.toggle('active', t === tab);
    document.getElementById(`tab-${t}-content`).style.display = t === tab ? '' : 'none';
  });
};

window.startHvB = async function() {
  const name   = document.getElementById('lb-name-hvb').value.trim() || 'Player';
  const target = document.getElementById('lb-target-hvb').value;
  const botId  = document.getElementById('lb-bot-hvb').value;   // ← add this
  State.gameMode    = 'hvb';
  State.myPlayerIdx = 0;
  const data = await api('/api/create_room', { player_name: name, mode: 'hvb', target_score: target, bot_id: botId });
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

window.enterGame = function(data) {
  document.getElementById('lobby').style.display   = 'none';
  document.getElementById('game-ui').style.display = 'block';
  State.selectedCards  = [];
  State.passModeActive = false;
  State.modalShown     = false;
  State.prevHandKeys   = ['', ''];
  State.prevPlayedKey  = '';
  document.getElementById('action-log').innerHTML  = '';
  if (State.gameMode === 'hvb') {
    refreshState();
  } else {
    // In HvH only the host (player 0) calls start_round.
    // The guest fetches the already-started state instead.
    if (State.myPlayerIdx === 0) {
      startRoundAfterJoin();
    } else {
      fetchAndRender();
    }
    startPolling();
  }
};

window.startRoundAfterJoin = async function() {
  const data = await api(ROOM('start_round'), { player_idx: State.myPlayerIdx });
  if (data && data.state) { State.gameState = data.state; render(); }
};

window.fetchAndRender = async function() {
  const data = await api(ROOM(`state?perspective=${State.myPlayerIdx}`));
  if (data && data.state) { State.gameState = data.state; render(); }
};

window.showLobby = function() {
  stopPolling();
  document.getElementById('lobby').style.display   = 'flex';
  document.getElementById('game-ui').style.display = 'none';
  document.getElementById('bvb-ui').style.display  = 'none';
  document.getElementById('room-waiting').style.display = 'none';
  const createBtn = document.querySelector('#tab-create-content .btn');
  if (createBtn) createBtn.disabled = false;
  State.roomId        = null;
  State.gameState     = null;
  State.selectedCards = [];
  State.modalShown    = false;
};

window.startPolling = function() {
  stopPolling();
  State.pollTimer = setInterval(async () => {
    if (!State.roomId || !State.gameState) return;
    const phase           = State.gameState?.phase;
    const { myPlayerIdx } = State;
    const s               = State.gameState;

    // Stop polling only when it's definitively MY turn to act with no pending responses.
    // IMPORTANT: never stop polling during stake negotiations — counter-raises can flip
    // stake_offerer_idx at any time and we must see that change.
    const isMyTurn =
       // My turn to play cards — but not while a stake offer is pending
       (phase === 'playing'      && s.active_player_idx    === myPlayerIdx && s.stake_offerer_idx === null)
       // Stakes with no offer pending and it's my turn — I need to act, not poll
    || (phase === 'stakes'       && s.stake_offerer_idx    === null
                                 && s.active_player_idx    === myPlayerIdx)
       // My turn to cut/pass (I am NOT the one who played)
    || ((phase === 'cutting' || phase === 'forced_cut') && s.playing_player_idx !== myPlayerIdx && s.stake_offerer_idx === null)
       // I have the calculate right
    || (phase === 'calculating'  && s.calculator_idx       === myPlayerIdx);

    if (isMyTurn) return;
    // Keep polling during round_over so guest sees when host starts next round.
    // Only stop polling during game_over (truly terminal).
    if (phase === 'game_over') return;
    // Block polling while a cut/pass animation is playing on this screen
    if (State.animating) return;

    const data = await api(ROOM(`state?perspective=${myPlayerIdx}`));
    if (!data?.state) return;

    // Full fingerprint — detects phase, table, hands, piles, scores, turn
    const fp = st => [
      st.phase,
      st.played_cards?.map(c => c.id).join(',') || '',
      st.players[0].hand_count, st.players[1].hand_count,
      st.players[0].pile_count, st.players[1].pile_count,
      st.players[0].game_score, st.players[1].game_score,
      st.active_player_idx, st.calculator_idx,
      st.stake_offerer_idx, st.current_stake,
    ].join('|');

    if (fp(data.state) !== fp(s)) {
      const oldTableKey = s.played_cards?.map(c => c.id).join(',') || '';
      const newTableKey = data.state.played_cards?.map(c => c.id).join(',') || '';
      const tableCleared = oldTableKey !== '' && newTableKey === '';

      if (tableCleared) {
        State.animating = true;
        const row      = document.getElementById('played-row');
        const newState = data.state;
        const history  = newState.move_history || [];

        // Find the last meaningful move — cut, pass, or counter_play
        const lastMove = [...history].reverse().find(m =>
          m.type === 'cut' || m.type === 'pass' || m.type === 'counter_play'
        );

        const takerPlayerIdx = newState.active_player_idx ?? State.myPlayerIdx;
        const takerDomSlot   = takerPlayerIdx === State.myPlayerIdx ? 0 : 1;
        const dy             = takerDomSlot === 0 ? 200 : -200;

        // Helper: parse card id string like "AH" → {rank, suit}
        const parseCardId = cid => {
          const rankMap = { A:'A', T:'T', K:'K', Q:'Q', J:'J' };
          const suitMap = { H:'hearts', D:'diamonds', C:'clubs', S:'spades' };
          const rank = rankMap[cid[0]];
          const suit = suitMap[cid[cid.length - 1]];
          return (rank && suit) ? { id: cid, rank, suit } : null;
        };
        const findCard = cid => {
          for (const p of (s.players || [])) {
            const found = (p.hand || []).find(c => c.id === cid);
            if (found) return found;
          }
          return parseCardId(cid);
        };

        // Fly-in duration — will be set when cut/pass cards are added
        let flyInDuration = 0;

        if (lastMove?.type === 'cut' || lastMove?.type === 'counter_play') {
          const cutCardIds      = lastMove.data?.cut_cards || lastMove.data?.cards || [];
          const cutterPlayerIdx = newState.players.findIndex(p => p.id === lastMove.player);
          const cutterDomSlot   = cutterPlayerIdx === State.myPlayerIdx ? 0 : 1;
          const fromRect        = getHandRect(cutterDomSlot);
          const newCutEls       = [];

          cutCardIds.forEach(cid => {
            const cardData = findCard(cid);
            const isTrump  = newState.trump_suit && cardData?.suit === newState.trump_suit;
            const wrapper  = document.createElement('div');
            wrapper.innerHTML = buildCard(cardData || { hidden: true }, false, false, false, isTrump);
            const el = wrapper.firstElementChild;
            el.classList.add('no-hover', 'cut-card-anim');
            row.appendChild(el);
            newCutEls.push(el);
          });

          newCutEls.forEach((el, i) => setTimeout(() => flyCard(el, fromRect, 260), i * 70));
          flyInDuration = newCutEls.length * 70 + 260;

        } else if (lastMove?.type === 'pass') {
          const passCount       = lastMove.data?.passed_count || (s.played_cards?.length || 1);
          const passerPlayerIdx = newState.players.findIndex(p => p.id === lastMove.player);
          const passerDomSlot   = passerPlayerIdx === State.myPlayerIdx ? 0 : 1;
          const fromRect        = getHandRect(passerDomSlot);
          const newPassEls      = [];

          for (let i = 0; i < passCount; i++) {
            const el = document.createElement('div');
            el.className = 'card face-down no-hover pass-ghost';
            row.appendChild(el);
            newPassEls.push(el);
          }

          newPassEls.forEach((el, i) => setTimeout(() => flyCard(el, fromRect, 260), i * 70));
          flyInDuration = newPassEls.length * 70 + 260;
        }

        // Collect all cards now on the table (originals + newly added)
        const allEls = Array.from(row.querySelectorAll('.table-card, .cut-card-anim, .pass-ghost'));

        // Wait for fly-in to land, pause so all cards are visible together, then sweep
        setTimeout(() => {
          allEls.forEach((el, i) => {
            setTimeout(() => {
              el.style.transition = 'transform 300ms ease-in, opacity 250ms ease-in';
              el.style.transform  = `translateY(${dy}px)`;
              el.style.opacity    = '0';
            }, i * 35);
          });

          const sweepDuration = allEls.length * 35 + 360;
          setTimeout(() => {
            allEls.forEach(el => { try { el.parentNode?.removeChild(el); } catch(_){} });
            State.animating     = false;
            State.prevPlayedKey = '';
            State.prevState     = s;
            State.gameState     = newState;
            render();
          }, sweepDuration);
        }, flyInDuration + 300);

        return;
      }

      // No table animation needed — just update state normally
      if (oldTableKey !== newTableKey) {
        State.prevPlayedKey = '';
      }
      State.prevState = State.gameState;
      State.gameState = data.state;
      render();
    }
  }, 1000);
};

window.stopPolling = function() {
  if (State.pollTimer) { clearInterval(State.pollTimer); State.pollTimer = null; }
};

window.toggleDebug = function() {
  State.debugMode = !State.debugMode;
  document.getElementById('debug-panel').classList.toggle('visible', State.debugMode);
  document.getElementById('debug-btn').textContent = State.debugMode ? '⚙ Debug ON' : '⚙ Debug';
  State.prevHandKeys = ['', ''];
  if (State.gameState) render();
  if (State.debugMode) refreshState(true);
};

window.checkUrlJoin = function() {
  const m = window.location.pathname.match(/^\/join\/([A-Z]{6})$/i);
  if (m) {
    const code = m[1].toUpperCase();
    switchTab('join');
    document.getElementById('lb-room-code').value = code;
  }
};

// Find the document.addEventListener('keydown', e => { block at the very bottom
document.addEventListener('keydown', e => {
  if (!State.gameState) return;
  const s = State.gameState;
  if (e.key === 'Enter') {
    if (s.phase === 'playing' && s.active_player_idx === State.myPlayerIdx && State.selectedCards.length)
      doPlayCards();

    // Allow enter for normal cuts OR if a Bura is selected
    if ((s.phase === 'cutting' || s.phase === 'forced_cut') &&
        (State.selectedCards.length === (s.played_cards?.length || 0) || isBuraSelection()))
      doCutCards();

    if (State.passModeActive && State.selectedCards.length === State.passCount)
      doPassCards();
  }
  if (e.key === 'Escape') {
    clearSel();
    if (State.passModeActive) cancelPass();
  }
});