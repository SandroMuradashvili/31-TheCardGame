'use strict';
// render.js — DOM rendering

window.render = function() {
  const s = State.gameState;
  if (!s) return;

  document.getElementById('sn0').textContent        = s.players[0].name;
  document.getElementById('sn1').textContent        = s.players[1].name;
  updateScore(0, s.players[0].game_score);
  updateScore(1, s.players[1].game_score);
  document.getElementById('sv-target').textContent  = s.target_score;
  document.getElementById('round-num').textContent  = s.round_number;
  document.getElementById('stake-num').textContent  = s.current_stake;

  renderTrump(s);

  // Always: bottom DOM slot (zone-0/hand-0) = my cards, top slot = opponent
  renderZone(State.myPlayerIdx,         s, 0);
  renderZone(1 - State.myPlayerIdx,     s, 1);

  renderTable(s);
  renderActions(s);

  if (State.debugMode) { renderDebug(s); renderDevTools(); }

  if ((s.phase === 'round_over' || s.phase === 'game_over') && !State.modalShown) {
    State.modalShown = true;
    setTimeout(() => showRoundModal(s), 600);
  }
  if (s.phase !== 'round_over' && s.phase !== 'game_over') {
    State.modalShown = false;
    document.getElementById('round-modal').classList.remove('visible');
  }
};

// ─── Score ────────────────────────────────────────────────────────────────────
window.updateScore = function(idx, val) {
  const el = document.getElementById(`sv${idx}`);
  el.textContent = val;
  if (val !== State.prevScores[idx]) {
    State.prevScores[idx] = val;
    el.classList.add('bump');
    setTimeout(() => el.classList.remove('bump'), 350);
  }
};

// ─── Trump strip ──────────────────────────────────────────────────────────────
window.renderTrump = function(s) {
  const suit = s.trump_suit;
  const card = s.trump_card;
  const sym  = suit ? suit_sym(suit) : '—';
  const cls  = suit ? color_cls(suit) : '';

  document.getElementById('trump-icon').innerHTML =
    suit ? `<span class="${cls}">${sym}</span>` : '—';
  document.getElementById('trump-card-text').textContent =
    card ? `${card.rank}${sym}` : '—';
  document.getElementById('deck-count-text').textContent   = `${s.deck.size} left`;
  document.getElementById('deck-count-widget').textContent = s.deck.size;
  const depleted = s.deck.size === 0;
  document.getElementById('deck-stack').style.opacity = depleted ? '0.3' : '1';

  const tc = document.getElementById('trump-card-in-deck');
  tc.innerHTML = card ? buildTrumpCard(card) : '';
  tc.style.opacity = depleted ? '0.3' : '1';
};

// ─── Player zones ─────────────────────────────────────────────────────────────
// playerIdx = which player's data (0 or 1 in the server state array)
// domSlot   = which DOM zone to render into (0 = bottom/me, 1 = top/opponent)
window.renderZone = function(playerIdx, s, domSlot) {
  if (domSlot === undefined) domSlot = playerIdx;

  const player   = s.players[playerIdx];
  const isMe     = playerIdx === State.myPlayerIdx;
  const isActive = s.active_player_idx === playerIdx && s.phase === 'playing';

  document.getElementById(`zone-name-${domSlot}`).textContent = player.name;
  document.getElementById(`zone-${domSlot}`).classList.toggle('active-turn', isActive);

  const pill = document.getElementById(`active-pill-${domSlot}`);
  pill.innerHTML = isActive ? '<span class="active-pill">YOUR TURN</span>' : '';

  const pi = document.getElementById(`pile-info-${domSlot}`);
  let pileText = '';
  if (isMe || State.debugMode) {
    const known = player.known_pile_points;
    const hc    = player.hidden_card_count;
    pileText = `<span class="kp">${known}pts known</span>`;
    if (hc > 0)
      pileText += ` + <span class="hid">${hc} hidden (${player.hidden_min_points}–${player.hidden_max_points})</span>`;
    pileText += ` (${player.pile_count} cards)`;
    if (State.debugMode && player.pile_points !== null)
      pileText += ` <span style="color:var(--gold)">[TRUE: ${player.pile_points}]</span>`;
  } else {
    pileText = `${player.pile_count} cards`;
  }

  // domSlot 0 = my zone (bottom) — preserve the tip button
  if (domSlot === 0) {
    const tipBtn = pi.querySelector('.tip-icon-btn');
    pi.innerHTML = pileText;
    if (tipBtn) pi.appendChild(tipBtn);
  } else {
    pi.innerHTML = pileText;
  }

  const container   = document.getElementById(`hand-${domSlot}`);
  const revealCards = State.debugMode || isMe;
  const canSelect   = isMe && canSelectCards(s);
  // Track hand keys by playerIdx (not domSlot) so diffs are against correct player
  renderHandCards(container, player.hand, canSelect, playerIdx, s, revealCards);
};

// ─── Hand rendering ───────────────────────────────────────────────────────────
window.renderHandCards = function(container, hand, canSelect, playerIdx, s, revealCards) {
  const makeKey = (cards) => JSON.stringify(cards.map((c, i) => c.id || `h${i}`));
  const newKey  = makeKey(hand);
  const oldKey  = State.prevHandKeys[playerIdx] || '[]';
  // Also track whether cards were selectable last render — if it changed, force rebuild
  // so no-hover class and click listeners are correctly updated
  const selectKey = `${newKey}|${canSelect}`;
  const oldSelectKey = State.prevHandSelectKeys?.[playerIdx] || '';
  if (!State.prevHandSelectKeys) State.prevHandSelectKeys = ['', ''];

  if (newKey === oldKey && selectKey === oldSelectKey && !State.debugMode) {
    container.querySelectorAll('.card[data-id]').forEach(el =>
      el.classList.toggle('selected', State.selectedCards.includes(el.dataset.id))
    );
    return;
  }

  const oldIds = JSON.parse(oldKey);
  // Check wasEmpty BEFORE clearing the container.
  // If we clear first and then check children.length, it's always 0 → wasEmpty=true
  // → all cards animate from deck even though the player already had them.
  const wasEmpty = oldIds.length === 0 || container.children.length === 0;
  const deckRect = getDeckRect();

  container.innerHTML = '';

  if (!hand || hand.length === 0) {
    container.innerHTML =
      `<span style="color:var(--text-dim);font-size:.8rem;font-style:italic;align-self:center;">No cards</span>`;
    State.prevHandKeys[playerIdx] = newKey;
    return;
  }

  const newEls = [];
  hand.forEach((card, i) => {
    const displayCard = revealCards ? card : { hidden: true };
    const isSel       = card.id && State.selectedCards.includes(card.id);
    const isTrump     = s.trump_suit && card.suit === s.trump_suit;
    const wrapper     = document.createElement('div');
    wrapper.innerHTML = buildCard(displayCard, isSel, canSelect && revealCards, false, isTrump);
    const el = wrapper.firstElementChild;
    container.appendChild(el);

    const cardKey = card.id || `h${i}`;
    // Only animate cards that are genuinely new (not in previous hand)
    if (wasEmpty || !oldIds.includes(cardKey)) newEls.push({ el, i });

    if (card.id && canSelect && revealCards)
      el.addEventListener('click', () => toggleCard(card.id));
  });

  // Hide new cards first so they don't flash at final position
  newEls.forEach(({ el }) => { el.style.opacity = '0'; });
  // Stagger deal animation only for genuinely new cards
  newEls.forEach(({ el, i }) => setTimeout(() => flyCard(el, deckRect, 260), i * 80));

  State.prevHandKeys[playerIdx]       = newKey;
  State.prevHandSelectKeys[playerIdx] = selectKey;
};

// ─── Table ────────────────────────────────────────────────────────────────────
window.renderTable = function(s) {
  const row   = document.getElementById('played-row');
  const label = document.getElementById('table-label');

  if (s.played_cards && s.played_cards.length > 0) {
    label.textContent = s.is_maliutka ? '⚡ MALIUTKA — must cut!' : 'On the table';
    label.className   = s.is_maliutka ? 'zone-label maliutka' : 'zone-label';

    const newKey = s.played_cards.map(c => c.id).join(',');
    if (newKey === State.prevPlayedKey) return;
    State.prevPlayedKey = newKey;

    row.querySelectorAll('.pass-ghost').forEach(el => el.remove());
    row.innerHTML = '';

    const fromPlayerIdx = s.playing_player_idx ?? State.myPlayerIdx;
    // Convert server player index → DOM slot (slot 0 = bottom = me, slot 1 = top = opponent)
    const fromDomSlot = fromPlayerIdx === State.myPlayerIdx ? 0 : 1;
    s.played_cards.forEach(card => {
      const isTrump = s.trump_suit && card.suit === s.trump_suit;
      const wrapper = document.createElement('div');
      wrapper.innerHTML = buildCard(card, false, false, false, isTrump);
      const el = wrapper.firstElementChild;
      el.classList.add('no-hover', 'table-card');
      el.dataset.cardId = card.id;
      row.appendChild(el);
    });

    State.tableCards = s.played_cards.map((c, i) => ({ card: c, el: row.children[i] }));
    animatePlayToTable(row.querySelectorAll('.table-card'), fromDomSlot, null);

  } else {
    const phaseLabels = {
      waiting:     'Waiting…',
      stakes:      'Stake negotiation — raise or play',
      playing:     'Select cards to play',
      cutting:     'Cut or pass?',
      forced_cut:  '⚡ MALIUTKA — must cut!',
      calculating: 'Calculate?',
      round_over:  'Round over',
      game_over:   'Game over',
    };
    label.textContent = phaseLabels[s.phase] || '';
    label.className   = s.phase === 'forced_cut' ? 'zone-label maliutka' : 'zone-label';

    if (State.prevPlayedKey !== '') {
      State.prevPlayedKey = '';
      // Immediately remove any lingering table cards (HvH observer side)
      row.querySelectorAll('.table-card, .pass-ghost, .cut-card-anim').forEach(el => el.remove());
    }
  }
};

// ─── Debug panel ──────────────────────────────────────────────────────────────
window.renderDebug = function(s) {
  document.getElementById('dbg-phase').textContent  = s.phase;
  document.getElementById('dbg-stake').textContent  = `${s.current_stake}→${s.pending_stake}`;
  document.getElementById('dbg-active').textContent = s.active_player_id;
  document.getElementById('dbg-calc').textContent   = s.calculator_id || '—';

  const fmt = hand => (hand || []).map(c => {
    if (c.hidden) return `<span class="dcard">?</span>`;
    const t = s.trump_suit && c.suit === s.trump_suit ? 'trump' : '';
    return `<span class="dcard ${t}">${c.rank}${suit_sym(c.suit)}</span>`;
  }).join('');

  document.getElementById('dbg-p1h').innerHTML   = fmt(s.players[0].hand);
  document.getElementById('dbg-p2h').innerHTML   = fmt(s.players[1].hand);
  document.getElementById('dbg-p1p').textContent = `${s.players[0].pile_count}c true:${s.players[0].pile_points ?? '?'}`;
  document.getElementById('dbg-p2p').textContent = `${s.players[1].pile_count}c true:${s.players[1].pile_points ?? '?'}`;

  const top5 = (s.deck.cards || []).slice(0, 5).map(c =>
    `<span class="dcard ${c.suit === s.trump_suit ? 'trump' : ''}">${c.rank}${suit_sym(c.suit)}</span>`
  ).join('');
  document.getElementById('dbg-deck').innerHTML  = top5;
  document.getElementById('dbg-log').textContent =
    (s.move_history || []).slice(-6).reverse().map(m =>
      `[${m.type}] ${m.player}: ${JSON.stringify(m.data).slice(0, 70)}`
    ).join('\n');
};

// ─── Round / game-over modal ──────────────────────────────────────────────────
window.showRoundModal = function(s) {
  const isGameOver = s.phase === 'game_over';
  const wIdx   = s.round_winner_idx ?? 0;
  const iWon   = wIdx === State.myPlayerIdx;
  const winner = s.players[wIdx];

  const title  = document.getElementById('modal-title');
  const sub    = document.getElementById('modal-sub');
  const scores = document.getElementById('modal-scores');
  const btn    = document.getElementById('modal-btn');

  const closeBtn = document.getElementById('modal-btn-close');

   if (isGameOver) {
    title.textContent = `🏆 ${winner.name} Wins!`;
    title.className   = `modal-title ${iWon ? 'win' : 'lose'}`;
    sub.textContent   = `${winner.name} reached ${s.target_score} points. Game over!`;
    btn.textContent   = 'Play Again';
    btn.disabled      = false;
    btn.onclick       = showLobby;

    // 2. Show the Close button and make it hide the modal when clicked
    if (closeBtn) {
      closeBtn.style.display = 'block';
      closeBtn.onclick = () => document.getElementById('round-modal').classList.remove('visible');
    }
  } else {
    title.textContent = iWon ? '✓ You Win!' : `${winner.name} Wins`;
    title.className   = `modal-title ${iWon ? 'win' : 'lose'}`;

     if (closeBtn) closeBtn.style.display = 'none';

    const reasons = {
      calculated_win:  iWon ? 'You calculated 31+ and won!'           : `${winner.name} calculated 31+.`,
      calculated_lose: iWon ? `${s.players[1-wIdx].name} counted but fell short.` : 'You counted but had less than 31.',
      three_trumps:    iWon ? 'You played 3 trumps — instant win!'    : `${winner.name} played 3 trumps!`,
      stake_declined:  iWon ? `${s.players[1-wIdx].name} declined your raise.` : 'You declined the raise.',
      deck_exhausted:  `Deck exhausted — ${winner.name} had more points.`,
    };
    sub.textContent = (reasons[s.round_end_reason] || s.round_end_reason)
                    + `  (Stake: ${s.current_stake})`;

    if (State.gameMode === 'hvh' && State.myPlayerIdx !== 0) {
      btn.textContent = 'Got it — waiting for host…';
      btn.disabled    = false;
      btn.onclick     = () => document.getElementById('round-modal').classList.remove('visible');
    } else {
      btn.textContent = 'Next Round →';
      btn.disabled    = false;
      btn.onclick     = doStartRound;
    }
  }

  scores.innerHTML = s.players.map(p => `
    <div class="ms-entry">
      <div class="ms-name">${p.name}</div>
      <div class="ms-val">${p.game_score}</div>
    </div>`).join('<div style="display:flex;align-items:center;color:var(--text-dim)">vs</div>');

  document.getElementById('round-modal').classList.add('visible');
};

// ─── Score tip popup ──────────────────────────────────────────────────────────
window.toggleTip = function(evt) {
  if (evt) evt.stopPropagation();
  State.showTip = !State.showTip;
  const iconBtn   = document.getElementById('tip-icon-btn');
  const container = document.getElementById('tip-popup-container');

  if (State.showTip) {
    const s   = State.gameState;
    const me  = s?.players[State.myPlayerIdx];
    const opp = s?.players[1 - State.myPlayerIdx];
    if (me && opp) {
      iconBtn?.classList.add('active');
      const tip = s.tip || {
        known_points:     me.known_pile_points,
        hidden_count:     me.hidden_card_count,
        min_total:        me.known_pile_points + me.hidden_card_count * 2,
        max_total:        me.known_pile_points + me.hidden_card_count * 11,
        can_possibly_win: (me.known_pile_points + me.hidden_card_count * 11) >= 31,
        guaranteed_win:   (me.known_pile_points + me.hidden_card_count * 2)  >= 31,
      };
      const oppTip = {
        known_points: opp.known_pile_points,
        hidden_count: opp.hidden_card_count,
        min_total:    opp.known_pile_points + opp.hidden_min_points,
        max_total:    opp.known_pile_points + opp.hidden_max_points,
        pile_count:   opp.pile_count,
        name:         opp.name,
      };
      renderTip(tip, oppTip, container);
    }
    setTimeout(() => document.addEventListener('click', closeTipOnClickOutside, { once: true }), 50);
  } else {
    iconBtn?.classList.remove('active');
    container.innerHTML = '';
  }
};

window.closeTipOnClickOutside = function() {
  State.showTip = false;
  document.getElementById('tip-icon-btn')?.classList.remove('active');
  const c = document.getElementById('tip-popup-container');
  if (c) c.innerHTML = '';
};

window.renderTip = function(tip, oppTip, container) {
  const sure     = tip.guaranteed_win;
  const possible = tip.can_possibly_win;

  // Opponent pile section
  const oppCanWin  = oppTip.max_total >= 31;
  const oppSureDanger = oppTip.min_total >= 31;
  const oppHidden  = oppTip.hidden_count > 0
    ? ` + <span class="tl">${oppTip.hidden_count} hidden (${oppTip.min_total - oppTip.known_points}–${oppTip.max_total - oppTip.known_points})</span>`
    : '';
  const oppStatus = oppSureDanger
    ? `<span class="tx">⚠ They can definitely calculate — danger!</span>`
    : oppCanWin
      ? `<span class="tl">⚠ They might reach 31 with hidden cards.</span>`
      : `<span class="tw">✓ They cannot reach 31 even best case.</span>`;

  container.innerHTML = `
    <div class="tip-popup">
      <strong style="color:var(--gold);font-size:.85rem;">Your Pile</strong><br>
      Known: <span class="tl">${tip.known_points} pts</span><br>
      Hidden: <span class="tl">${tip.hidden_count} cards</span>
        → <span class="tl">${tip.min_total}–${tip.max_total}</span><br>
      ${sure     ? `<span class="tw">✓ Definitely 31+ — safe to calculate!</span>`
      : possible ? `<span class="tl">⚠ Possible win if hidden cards cooperate.</span>`
                 : `<span class="tx">✗ Cannot reach 31 even in best case.</span>`}
      <hr style="border:none;border-top:1px solid rgba(255,255,255,0.1);margin:7px 0;">
      <strong style="color:var(--gold);font-size:.85rem;">${oppTip.name}'s Pile</strong><br>
      Known: <span class="tl">${oppTip.known_points} pts</span>${oppHidden}<br>
      Range: <span class="tl">${oppTip.min_total}–${oppTip.max_total}</span>
        (${oppTip.pile_count} cards)<br>
      ${oppStatus}
    </div>`;
};