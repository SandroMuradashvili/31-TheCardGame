'use strict';
// ─────────────────────────────────────────────────────────────────────────────
// actions_render.js — Action panel HTML + pass mode UI
// ─────────────────────────────────────────────────────────────────────────────
// Depends on: state.js, cards.js, render.js (renderZone)
// stake actions: doRaiseStake / doAcceptStake / doDeclineStake → stake_actions.js
// card actions:  doPlayCards / doCutCards / doPassCards → game_actions.js
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

  const hasPending     = s.stake_offerer_idx != null;
  const pendingFromMe  = hasPending && s.stake_offerer_idx === myPlayerIdx;
  const pendingFromOpp = hasPending && s.stake_offerer_idx !== myPlayerIdx;
  // Can raise any time during an active round, as long as no pending offer from me
  const canRaise       = (s.can_raise_stake || [])[myPlayerIdx] === true;

  // ── Opponent raised mid-round — show respond prompt before anything else ────
  // This can happen during playing or cutting phases.
  if (pendingFromOpp && s.phase !== 'stakes') {
    const newS = s.pending_stake;
    msgEl.innerHTML = `<span class="hl">${them.name}</span> raises the stake to <span class="hl">${newS}</span>! Accept or decline before continuing.`;
    btnsEl.innerHTML = `
      <button class="btn btn-green" onclick="doAcceptStake()">Accept (${newS} pt)</button>
      <button class="btn btn-red"   onclick="doDeclineStake()">Decline (give ${s.current_stake} pt)</button>
      ${newS < 6 ? `<button class="btn btn-ghost btn-sm" onclick="doRaiseStake()">Counter → ${newS+1}</button>` : ''}
    `;
    return;
  }

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
      const newS = s.pending_stake;
      msgEl.innerHTML  = `<span class="hl">${them.name}</span> raises stake to <span class="hl">${newS}</span>`;
      btnsEl.innerHTML = `
        <button class="btn btn-green" onclick="doAcceptStake()">Accept (${newS} pt)</button>
        <button class="btn btn-red"   onclick="doDeclineStake()">Decline (give ${s.current_stake} pt)</button>
        ${newS < 6 ? `<button class="btn btn-ghost btn-sm" onclick="doRaiseStake()">Counter → ${newS+1}</button>` : ''}
      `;
    } else if (pendingFromMe) {
      msgEl.innerHTML = `Offered stake <span class="hl">${s.pending_stake}</span>. Waiting for ${them.name}…`;
    } else {
      msgEl.innerHTML = `<span class="stake-info">Stake <span class="sv">${s.current_stake}</span></span> Select cards to play, or raise the stake.`;
      const playBtn = document.createElement('button');
      playBtn.className   = 'btn btn-gold';
      playBtn.id          = 'btn-play';
      playBtn.disabled    = sel === 0;
      playBtn.textContent = sel > 0 ? `Play ${sel}` : 'Play cards…';
      playBtn.onclick     = doPlayCards;
      if (canRaise) {
        const raiseBtn = document.createElement('button');
        raiseBtn.className   = 'btn btn-ghost btn-sm';
        raiseBtn.textContent = `⬆ Raise to ${s.current_stake + 1}`;
        raiseBtn.onclick     = doRaiseStake;
        btnsEl.appendChild(raiseBtn);
      }
      btnsEl.appendChild(playBtn);
      if (sel > 0) {
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
    const selCards        = State.selectedCards.map(id => (me.hand||[]).find(c => c.id === id)).filter(Boolean);
    const isMaliutka      = sel === 3
      && selCards.every(c => c.suit === selCards[0].suit)
      && selCards[0].suit !== s.trump_suit;
    const hint = isMaliutka
      ? `<span class="warn"> ⚡ MALIUTKA — forces opponent to cut all 3!</span>`
      : `<em style="color:var(--text-dim);font-size:.85em;"> Tip: 3 non-trump same suit = Maliutka</em>`;
    msgEl.innerHTML  = `Your turn — select 1–3 cards of the same suit.${sel > 0 ? hint : ''}`;
    btnsEl.innerHTML = `
      <button class="btn btn-gold" id="btn-play" onclick="doPlayCards()" ${sel===0?'disabled':''}>
        ${sel > 0 ? `Play ${sel} card${sel>1?'s':''}` : 'Play cards…'}
      </button>
      ${sel > 0 ? `<button class="btn btn-ghost btn-sm" onclick="clearSel()">Clear</button>` : ''}
    `;
    _appendStakeBtns(btnsEl, s, them);
    return;
  }

  // ── Playing — opponent's turn ─────────────────────────────────────────────
  if (s.phase === 'playing' && !iAmActive) {
    msgEl.innerHTML = `<span class="hl">${them.name}</span> is playing…`;
    return;
  }

  // ── Cutting — I must respond ──────────────────────────────────────────────
  if ((s.phase === 'cutting' || s.phase === 'forced_cut') && !iAmPlaying) {
    const n        = s.played_cards?.length || 0;
    const isForced = s.phase === 'forced_cut';
    const canCut   = sel === n;
    const canCounter = isCounterSelection();  // 3 non-trump same-suit (only in cutting)

    if (isForced) {
      msgEl.innerHTML = `<span class="warn">⚡ MALIUTKA!</span> ${them.name} played ${n} — beat ${n>1?'them':'it'} with ${n} higher same-suit card${n>1?'s':''} or trump, or pass.`;
    } else {
      msgEl.innerHTML = `${them.name} played ${n} card${n>1?'s':''}.
        <span class="hl">Cut</span> with ${n} higher same-suit card${n>1?'s':''} or trump.
        <span class="hl">Counter</span> with 3 non-trump same-suit cards to flip it back.
        Or <span class="hl">Pass</span>.`;
    }

    btnsEl.innerHTML = `
      <button class="btn btn-gold" id="btn-cut" onclick="doCutCards()" ${!canCut?'disabled':''}>
        ${sel > 0 ? `Cut with ${sel}` : 'Cut…'}
      </button>
      ${!isForced ? `
        <button class="btn btn-green" id="btn-counter" onclick="doCounterPlay()" ${!canCounter?'disabled':''}>
          ⚡ Counter (3)
        </button>` : ''}
      <button class="btn btn-ghost" onclick="doPassAuto(${n})">Pass</button>
      ${sel > 0 ? `<button class="btn btn-ghost btn-sm" onclick="clearSel()">Clear</button>` : ''}
    `;

    if ((s.valid_cuts||[]).length > 0) {
      const hint = s.valid_cuts.slice(0,2).map(c=>`[${c.join(' ')}]`).join('  ');
      btnsEl.insertAdjacentHTML('beforeend',
        `<div style="width:100%;font-size:.72rem;color:var(--text-dim);margin-top:4px;">
           Valid cuts: <span style="color:var(--gold)">${hint}</span>
         </div>`);
    }
    _appendStakeBtns(btnsEl, s, them);
    return;
  }

  // ── Cutting — I played, waiting ───────────────────────────────────────────
  if ((s.phase === 'cutting' || s.phase === 'forced_cut') && iAmPlaying) {
    const label = s.phase === 'forced_cut' ? '⚡ Maliutka! ' : '';
    msgEl.innerHTML = `${label}Waiting for <span class="hl">${them.name}</span> to respond…`;
    return;
  }

  // ── Calculating — my right ────────────────────────────────────────────────
  if (s.phase === 'calculating' && iAmCalc) {
    const known = me.known_pile_points;
    const hc    = me.hidden_card_count;
    msgEl.innerHTML = `You may calculate! Known: <span class="hl">${known}pts</span> + ${hc} hidden (${me.hidden_min_points}–${me.hidden_max_points}). <em style="color:var(--text-dim);font-size:.85em;">💡 for estimate.</em>`;
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


// ─── Stake raise banner ──────────────────────────────────────────────────────
// Appended to the button row whenever raising is possible mid-round.

function _appendStakeBtns(btnsEl, s, them) {
  const hasPending     = s.stake_offerer_idx != null;
  const pendingFromOpp = hasPending && s.stake_offerer_idx !== State.myPlayerIdx;
  const canRaise       = (s.can_raise_stake || [])[State.myPlayerIdx] === true;

  if (pendingFromOpp) {
    // Opponent raised mid-round — show respond buttons
    const newS = s.pending_stake;
    const div  = document.createElement('div');
    div.style.cssText = 'width:100%;margin-top:8px;padding-top:8px;border-top:1px solid rgba(200,168,75,0.2);display:flex;gap:6px;flex-wrap:wrap;align-items:center;';
    div.innerHTML = `
      <span style="font-size:.75rem;color:var(--gold);">${them.name} raises to ${newS}</span>
      <button class="btn btn-green btn-sm" onclick="doAcceptStake()">Accept (${newS}pt)</button>
      <button class="btn btn-red btn-sm"   onclick="doDeclineStake()">Decline</button>
      ${newS < 6 ? `<button class="btn btn-ghost btn-sm" onclick="doRaiseStake()">Counter → ${newS+1}</button>` : ''}
    `;
    btnsEl.appendChild(div);
  } else if (canRaise) {
    const btn = document.createElement('button');
    btn.className   = 'btn btn-ghost btn-sm';
    btn.textContent = `⬆ Raise to ${s.current_stake + 1}`;
    btn.onclick     = doRaiseStake;
    btn.style.marginLeft = 'auto';
    btnsEl.appendChild(btn);
  }
}


// ─── Pass mode ────────────────────────────────────────────────────────────────

window.showPassUI = function(n) {
  State.passModeActive = true;
  State.passCount      = n;
  clearSel();
  document.getElementById('action-msg').innerHTML  = `Select <span class="hl">${n}</span> card${n>1?'s':''} to pass anonymously.`;
  document.getElementById('action-btns').innerHTML = `
    <button class="btn btn-red" id="btn-pass-confirm" onclick="doPassCards()" disabled>Pass ${n} card${n>1?'s':''}</button>
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