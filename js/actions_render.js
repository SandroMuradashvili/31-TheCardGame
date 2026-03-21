'use strict';
// ─────────────────────────────────────────────────────────────────────────────
// actions_render.js — Action panel HTML + pass mode UI
// ─────────────────────────────────────────────────────────────────────────────
// Builds the "Your Action" panel shown below the table.
// Also handles the special pass-mode flow (select N cards to pass anonymously).
//
// Depends on: state.js, cards.js (canSelectCards, clearSel), render.js (renderZone)
// Loaded by:  index.html (after render.js)
//
// Note: doRaiseStake / doAcceptStake / doDeclineStake live in stake_actions.js
//       doPlayCards / doCutCards / doPassCards live in game_actions.js
// ─────────────────────────────────────────────────────────────────────────────

window.renderActions = function(s) {
  const msgEl  = document.getElementById('action-msg');
  const btnsEl = document.getElementById('action-btns');
  msgEl.innerHTML  = '';
  btnsEl.innerHTML = '';

  const { myPlayerIdx } = State;
  const me   = s.players[myPlayerIdx];
  const them = s.players[1 - myPlayerIdx];

  const iAmActive  = s.active_player_idx  === myPlayerIdx;
  const iAmPlaying = s.playing_player_idx === myPlayerIdx;
  const iAmCalc    = s.calculator_idx     === myPlayerIdx;

  // Stake offer state
  const hasPending     = s.stake_offerer_idx !== null && s.stake_offerer_idx !== undefined;
  const pendingFromMe  = hasPending && s.stake_offerer_idx === myPlayerIdx;
  const pendingFromOpp = hasPending && s.stake_offerer_idx !== myPlayerIdx;
  const canRaise       = !hasPending && s.current_stake < 6 && s.phase === 'stakes';

  // ── Waiting ──────────────────────────────────────────────────────────────
  if (s.phase === 'waiting') {
    msgEl.textContent = 'Ready to start.';
    btnsEl.innerHTML  = `<button class="btn btn-gold" onclick="doStartRound()">Deal Cards</button>`;
    return;
  }

  // ── Game over ─────────────────────────────────────────────────────────────
  if (s.phase === 'game_over') {
    const wIdx = s.round_winner_idx ?? (s.players[0].game_score >= s.target_score ? 0 : 1);
    msgEl.innerHTML  = `<span class="hl">🏆 ${s.players[wIdx].name} wins the game!</span>`;
    btnsEl.innerHTML = `<button class="btn btn-gold" onclick="showLobby()">Play Again</button>`;
    return;
  }

  // ── Round over ────────────────────────────────────────────────────────────
  if (s.phase === 'round_over') {
    msgEl.textContent = 'Round over.';
    btnsEl.innerHTML  = `<button class="btn btn-gold" onclick="doStartRound()">Next Round →</button>`;
    return;
  }

  // ── Stakes ────────────────────────────────────────────────────────────────
  if (s.phase === 'stakes') {
    if (pendingFromOpp) {
      const newS       = s.pending_stake;
      const canCounter = newS < 6;
      msgEl.innerHTML  = `<span class="hl">${them.name}</span> raises stake to <span class="hl">${newS}</span>`;
      btnsEl.innerHTML = `
        <button class="btn btn-green" onclick="doAcceptStake()">Accept (${newS} pt)</button>
        <button class="btn btn-red"   onclick="doDeclineStake()">Decline (give ${s.current_stake} pt)</button>
        ${canCounter ? `<button class="btn btn-ghost btn-sm" onclick="doRaiseStake()">Counter → ${newS + 1}</button>` : ''}
      `;
    } else if (pendingFromMe) {
      msgEl.innerHTML = `Offered stake <span class="hl">${s.pending_stake}</span>. Waiting for ${them.name}…`;
    } else {
      const stakeInfo = `<span class="stake-info">Stake <span class="sv">${s.current_stake}</span></span>`;
      msgEl.innerHTML = `${stakeInfo} Select cards to play, or raise the stake.`;
      if (canRaise) {
        btnsEl.innerHTML = `<button class="btn btn-ghost btn-sm" onclick="doRaiseStake()">⬆ Raise to ${s.current_stake + 1}</button>`;
      }
      // Play button
      const playBtn = document.createElement('button');
      playBtn.className   = 'btn btn-gold';
      playBtn.id          = 'btn-play';
      playBtn.disabled    = State.selectedCards.length === 0;
      playBtn.textContent = State.selectedCards.length > 0
        ? `Play ${State.selectedCards.length}` : 'Play cards…';
      playBtn.onclick = doPlayCards;
      btnsEl.appendChild(playBtn);
      if (State.selectedCards.length > 0) {
        const clrBtn = document.createElement('button');
        clrBtn.className   = 'btn btn-ghost btn-sm';
        clrBtn.textContent = 'Clear';
        clrBtn.onclick     = clearSel;
        btnsEl.appendChild(clrBtn);
      }
    }
    return;
  }

  // ── Playing — my turn ─────────────────────────────────────────────────────
  if (s.phase === 'playing' && iAmActive) {
    const sel      = State.selectedCards.length;
    const myHand   = me.hand || [];
    const selCards = State.selectedCards
      .map(id => myHand.find(c => c.id === id)).filter(Boolean);
    const isMaliutkaReady = sel === 3
      && selCards.every(c => c.suit === selCards[0].suit)
      && selCards[0].suit !== s.trump_suit;
    const hint = isMaliutkaReady
      ? `<span class="warn"> ⚡ MALIUTKA — forces opponent to cut!</span>`
      : `<em style="color:var(--text-dim);font-size:.85em;"> 3 non-trump same suit = Maliutka</em>`;
    msgEl.innerHTML  = `Your turn — select 1–3 cards of the same suit.${sel > 0 ? hint : ''}`;
    btnsEl.innerHTML = `
      <button class="btn btn-gold" id="btn-play" onclick="doPlayCards()" ${sel === 0 ? 'disabled' : ''}>
        ${sel > 0 ? `Play ${sel} card${sel > 1 ? 's' : ''}` : 'Play cards…'}
      </button>
      ${sel > 0 ? `<button class="btn btn-ghost btn-sm" onclick="clearSel()">Clear</button>` : ''}
    `;
    return;
  }

  // ── Playing — opponent's turn ─────────────────────────────────────────────
  if (s.phase === 'playing' && !iAmActive) {
    msgEl.innerHTML = `<span class="hl">${them.name}</span> is playing…`;
    return;
  }

  // ── Cutting — I must respond ──────────────────────────────────────────────
  if ((s.phase === 'cutting' || s.phase === 'forced_cut') && !iAmPlaying) {
    const n         = s.played_cards?.length || 0;
    const validCuts = s.valid_cuts || [];
    const isForced  = s.phase === 'forced_cut';
    const sel       = State.selectedCards.length;
    const canCutNow = sel === n;

    if (isForced) {
      msgEl.innerHTML =
        `<span class="warn">⚡ MALIUTKA!</span> ${them.name} played ${n} non-trump — you must beat ${n > 1 ? 'them' : 'it'} with ${n} higher card${n > 1 ? 's' : ''} of the same suit or trump, or pass ${n} card${n > 1 ? 's' : ''} back.`;
    } else {
      msgEl.innerHTML =
        `${them.name} played ${n} card${n > 1 ? 's' : ''}. <span class="hl">To cut:</span> select ${n} higher same-suit card${n > 1 ? 's' : ''} or any trump${n > 1 ? 's' : ''}, then click Cut. <span style="color:var(--text-dim)">Or pass ${n} card${n > 1 ? 's' : ''} back.</span>`;
    }

    btnsEl.innerHTML = `
      <button class="btn btn-gold" id="btn-cut" onclick="doCutCards()" ${!canCutNow ? 'disabled' : ''}>
        ${sel > 0 ? `Cut with ${sel}` : 'Cut…'}
      </button>
      ${!isForced
        ? `<button class="btn btn-ghost" onclick="${canCutNow ? `doPassDirect(${n})` : `doPassAuto(${n})`}">
             ${canCutNow ? `Pass selected` : `Pass ${n}`}
           </button>`
        : (validCuts.length === 0
            ? `<button class="btn btn-ghost" onclick="doPassAuto(${n})">Pass ${n}</button>`
            : '')
      }
      ${sel > 0 ? `<button class="btn btn-ghost btn-sm" onclick="clearSel()">Clear</button>` : ''}
    `;

    if (validCuts.length > 0) {
      const hint = validCuts.slice(0, 2).map(c => `[${c.join(' ')}]`).join('  ');
      btnsEl.insertAdjacentHTML('beforeend',
        `<div style="width:100%;font-size:.72rem;color:var(--text-dim);margin-top:4px;">
           Valid cuts: <span style="color:var(--gold)">${hint}</span>
         </div>`);
    }
    return;
  }

  // ── Cutting — I played, waiting ───────────────────────────────────────────
  if ((s.phase === 'cutting' || s.phase === 'forced_cut') && iAmPlaying) {
    const label = s.phase === 'forced_cut' ? '⚡ Maliutka! ' : '';
    msgEl.innerHTML = `${label}Waiting for <span class="hl">${them.name}</span> to cut or pass…`;
    return;
  }

  // ── Calculating — my right ────────────────────────────────────────────────
  if (s.phase === 'calculating' && iAmCalc) {
    const known = me.known_pile_points;
    const hc    = me.hidden_card_count;
    msgEl.innerHTML =
      `You may calculate! Known: <span class="hl">${known}pts</span> + ${hc} hidden (${me.hidden_min_points}–${me.hidden_max_points}). <em style="color:var(--text-dim);font-size:.85em;">💡 for estimate.</em>`;
    btnsEl.innerHTML = `
      <button class="btn btn-gold"  onclick="doCalculate()">Calculate</button>
      <button class="btn btn-ghost" onclick="doSkipCalc()">Keep playing →</button>
    `;
    return;
  }

  // ── Calculating — opponent's right ────────────────────────────────────────
  if (s.phase === 'calculating' && !iAmCalc) {
    msgEl.innerHTML = `<span class="hl">${them.name}</span> is deciding whether to calculate…`;
    return;
  }
};


// ─── Pass mode ────────────────────────────────────────────────────────────────
// Activated when player clicks "Pass N" without pre-selecting cards.
// Lets them pick exactly N cards to pass face-down.

window.showPassUI = function(n) {
  State.passModeActive = true;
  State.passCount      = n;
  clearSel();

  const msgEl  = document.getElementById('action-msg');
  const btnsEl = document.getElementById('action-btns');
  msgEl.innerHTML  = `Select <span class="hl">${n}</span> card${n > 1 ? 's' : ''} to pass anonymously.`;
  btnsEl.innerHTML = `
    <button class="btn btn-red" id="btn-pass-confirm" onclick="doPassCards()" disabled>
      Pass ${n} card${n > 1 ? 's' : ''}
    </button>
    <button class="btn btn-ghost btn-sm" onclick="cancelPass()">Cancel</button>
  `;

  // Force hand re-render so cards become clickable in pass mode
  State.prevHandKeys[State.myPlayerIdx] = '';
  renderZone(State.myPlayerIdx, State.gameState);
};

window.cancelPass = function() {
  State.passModeActive = false;
  clearSel();
  renderActions(State.gameState);
  State.prevHandKeys[State.myPlayerIdx] = '';
  renderZone(State.myPlayerIdx, State.gameState);
};