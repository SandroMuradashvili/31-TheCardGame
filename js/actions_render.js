'use strict';
// ─────────────────────────────────────────────────────────────────────────────
// actions_render.js — Action panel HTML + pass mode UI
// ─────────────────────────────────────────────────────────────────────────────
// Depends on: state.js, cards.js, render.js (renderZone)
// Stake API calls  → stake_actions.js
// Card API calls   → game_actions.js
// ─────────────────────────────────────────────────────────────────────────────

window.renderActions = function(s) {
  const msgEl  = document.getElementById('action-msg');
  const btnsEl = document.getElementById('action-btns');
  msgEl.innerHTML  = '';
  btnsEl.innerHTML = '';

  const { myPlayerIdx } = State;
  const me   = s.players[myPlayerIdx];
  const them = s.players[1 - myPlayerIdx];
  const sel  = State.selectedCards.length;

  const iAmActive  = s.active_player_idx  === myPlayerIdx;
  const iAmPlaying = s.playing_player_idx === myPlayerIdx;
  const iAmCalc    = s.calculator_idx     === myPlayerIdx;

  const pendingFromMe  = s.stake_offerer_idx === myPlayerIdx;
  const pendingFromOpp = s.stake_offerer_idx != null && !pendingFromMe;
  const canRaise       = (s.can_raise_stake || [])[myPlayerIdx] === true;


  // ══════════════════════════════════════════════════════════════════════════
  // STAKE INTERCEPTS — these take over the whole panel when stake negotiation
  // is happening. Same behaviour regardless of which game phase we're in.
  // ══════════════════════════════════════════════════════════════════════════

  // Opponent raised — I must respond before anything else
  if (pendingFromOpp) {
    const newS = s.pending_stake;
    msgEl.innerHTML = `<span class="hl">${them.name}</span> raises the stake to <span class="hl">${newS}</span>. Accept, decline, or counter.`;
    btnsEl.innerHTML = `
      <button class="btn btn-green" onclick="doAcceptStake()">Accept (${newS} pt)</button>
      <button class="btn btn-red"   onclick="doDeclineStake()">Decline — give ${s.current_stake} pt</button>
      ${newS < 6 ? `<button class="btn btn-ghost btn-sm" onclick="doRaiseStake()">Counter → ${newS + 1}</button>` : ''}
    `;
    return;
  }

  // I raised — waiting for opponent's response
  if (pendingFromMe) {
    msgEl.innerHTML = `You offered to raise to <span class="hl">${s.pending_stake}</span>. Waiting for ${them.name}…`;
    // No buttons — must wait
    return;
  }


  // ══════════════════════════════════════════════════════════════════════════
  // TERMINAL / TRANSITION PHASES
  // ══════════════════════════════════════════════════════════════════════════

  if (s.phase === 'waiting') {
    msgEl.textContent = 'Ready to start.';
    btnsEl.innerHTML  = `<button class="btn btn-gold" onclick="doStartRound()">Deal Cards</button>`;
    return;
  }

  if (s.phase === 'game_over') {
    const wIdx = s.round_winner_idx ?? (s.players[0].game_score >= s.target_score ? 0 : 1);
    msgEl.innerHTML  = `<span class="hl">🏆 ${s.players[wIdx].name} wins the game!</span>`;
    btnsEl.innerHTML = `<button class="btn btn-gold" onclick="showLobby()">Play Again</button>`;
    return;
  }

  if (s.phase === 'round_over') {
    msgEl.textContent = 'Round over.';
    btnsEl.innerHTML  = `<button class="btn btn-gold" onclick="doStartRound()">Next Round →</button>`;
    return;
  }


  // ══════════════════════════════════════════════════════════════════════════
  // ACTIVE PHASES — raise button always shown as last button when available
  // ══════════════════════════════════════════════════════════════════════════

  // ── Stakes (before first card is played) ──────────────────────────────────
  if (s.phase === 'stakes') {
    msgEl.innerHTML = `<span class="stake-info">Stake <span class="sv">${s.current_stake}</span></span> Select cards to play — or raise the stake.`;
    _renderPlayAndRaise(btnsEl, sel, canRaise, s);
    return;
  }

  // ── Playing — my turn ─────────────────────────────────────────────────────
  if (s.phase === 'playing' && iAmActive) {
    const selCards   = State.selectedCards.map(id => (me.hand||[]).find(c => c.id === id)).filter(Boolean);
    const isMaliutka = sel === 3
      && selCards.every(c => c.suit === selCards[0].suit)
      && selCards[0].suit !== s.trump_suit;

    msgEl.innerHTML = `Your turn — select 1–3 cards of the same suit.`
      + (isMaliutka ? `<span class="warn"> ⚡ MALIUTKA — forces opponent to cut all 3!</span>` : '');

    _renderPlayAndRaise(btnsEl, sel, canRaise, s);
    return;
  }

  // ── Playing — opponent's turn ─────────────────────────────────────────────
  if (s.phase === 'playing' && !iAmActive) {
    msgEl.innerHTML = `<span class="hl">${them.name}</span> is playing…`;
    if (canRaise) _renderRaiseOnly(btnsEl, s);
    return;
  }

  // ── Cutting — I must respond ──────────────────────────────────────────────
  if ((s.phase === 'cutting' || s.phase === 'forced_cut') && !iAmPlaying) {
    const n        = s.played_cards?.length || 0;
    const isForced = s.phase === 'forced_cut';
    const canCut   = sel === n;
    const canCounter = isCounterSelection();

    if (isForced) {
      msgEl.innerHTML = `<span class="warn">⚡ MALIUTKA!</span> ${them.name} played ${n} — cut with ${n} higher same-suit or trump, or pass.`;
    } else {
      msgEl.innerHTML = `${them.name} played ${n} card${n > 1 ? 's' : ''} — cut, counter with 3 non-trump same-suit, or pass.`;
    }

    btnsEl.innerHTML = `
      <button class="btn btn-gold"  id="btn-cut" onclick="doCutCards()" ${!canCut ? 'disabled' : ''}>
        ${sel > 0 ? `Cut with ${sel}` : 'Cut…'}
      </button>
      ${!isForced ? `
        <button class="btn btn-green" id="btn-counter" onclick="doCounterPlay()" ${!canCounter ? 'disabled' : ''}>
          ⚡ Counter (3)
        </button>` : ''}
      <button class="btn btn-ghost" id="btn-pass" onclick="doPassAuto(${n})" ${sel !== n ? 'disabled' : ''}>
        ${sel === n ? `Pass selected` : `Pass (select ${n})`}
      </button>
      ${sel > 0 ? `<button class="btn btn-ghost btn-sm" onclick="clearSel()">Clear</button>` : ''}
    `;

    if ((s.valid_cuts || []).length > 0) {
      const hint = s.valid_cuts.slice(0, 2).map(c => `[${c.join(' ')}]`).join('  ');
      btnsEl.insertAdjacentHTML('beforeend',
        `<div style="width:100%;font-size:.72rem;color:var(--text-dim);margin-top:4px;">
           Valid cuts: <span style="color:var(--gold)">${hint}</span>
         </div>`);
    }

    if (canRaise) _renderRaiseOnly(btnsEl, s);
    return;
  }

  // ── Cutting — I played, waiting ───────────────────────────────────────────
  if ((s.phase === 'cutting' || s.phase === 'forced_cut') && iAmPlaying) {
    const label = s.phase === 'forced_cut' ? '⚡ Maliutka! ' : '';
    msgEl.innerHTML = `${label}Waiting for <span class="hl">${them.name}</span> to respond…`;
    if (canRaise) _renderRaiseOnly(btnsEl, s);
    return;
  }

  // ── Calculating — my right ────────────────────────────────────────────────
  if (s.phase === 'calculating' && iAmCalc) {
    const known       = me.known_pile_points;
    const hc          = me.hidden_card_count;
    // Must calculate if deck is empty AND someone has fewer than 3 cards
    // (if both have 3, play can still continue even with empty deck)
    const mustCalc = s.deck.size === 0
      && (s.players[0].hand_count < 3 || s.players[1].hand_count < 3);
    msgEl.innerHTML = `You may calculate! Known: <span class="hl">${known} pts</span> + ${hc} hidden (${me.hidden_min_points}–${me.hidden_max_points}). <em style="color:var(--text-dim);font-size:.85em;">💡 for estimate.</em>`
      + (mustCalc ? `<span class="warn" style="display:block;margin-top:4px;font-size:.85em;">⚠ Deck empty — you must calculate now.</span>` : '');
    btnsEl.innerHTML = `
      <button class="btn btn-gold" onclick="doCalculate()">Calculate</button>
      ${!mustCalc ? `<button class="btn btn-ghost" onclick="doSkipCalc()">Keep playing →</button>` : ''}
    `;
    return;
  }

  // ── Calculating — opponent's right ────────────────────────────────────────
  if (s.phase === 'calculating' && !iAmCalc) {
    msgEl.innerHTML = `<span class="hl">${them.name}</span> is deciding whether to calculate…`;
    return;
  }
};


// ─── Button helpers ───────────────────────────────────────────────────────────
// All stake buttons are always in the same position: after the primary action
// buttons. Raise is always the LAST button so it never shifts other buttons.

/** Play button + optional Clear, then Raise as last button. */
function _renderPlayAndRaise(btnsEl, sel, canRaise, s) {
  const playBtn = document.createElement('button');
  playBtn.className   = 'btn btn-gold';
  playBtn.id          = 'btn-play';
  playBtn.disabled    = sel === 0;
  playBtn.textContent = sel > 0 ? `Play ${sel} card${sel > 1 ? 's' : ''}` : 'Play cards…';
  playBtn.onclick     = doPlayCards;
  btnsEl.appendChild(playBtn);

  if (sel > 0) {
    const clrBtn = document.createElement('button');
    clrBtn.className   = 'btn btn-ghost btn-sm';
    clrBtn.textContent = 'Clear';
    clrBtn.onclick     = clearSel;
    btnsEl.appendChild(clrBtn);
  }

  if (canRaise) _renderRaiseOnly(btnsEl, s);
}

/** Just the raise button — appended last so position never jumps. */
function _renderRaiseOnly(btnsEl, s) {
  const btn = document.createElement('button');
  btn.className   = 'btn btn-ghost btn-sm';
  btn.textContent = `⬆ Raise to ${s.current_stake + 1}`;
  btn.onclick     = doRaiseStake;
  btnsEl.appendChild(btn);
}


// ─── Pass mode ────────────────────────────────────────────────────────────────

window.showPassUI = function(n) {
  State.passModeActive = true;
  State.passCount      = n;
  clearSel();
  document.getElementById('action-msg').innerHTML  = `Select <span class="hl">${n}</span> card${n > 1 ? 's' : ''} to pass anonymously.`;
  document.getElementById('action-btns').innerHTML = `
    <button class="btn btn-red" id="btn-pass-confirm" onclick="doPassCards()" disabled>Pass ${n} card${n > 1 ? 's' : ''}</button>
    <button class="btn btn-ghost btn-sm" onclick="cancelPass()">Cancel</button>
  `;
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