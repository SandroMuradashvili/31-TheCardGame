'use strict';
// ─────────────────────────────────────────────────────────────────────────────
// cards.js — Card HTML builder, selection logic, buzz effect
// ─────────────────────────────────────────────────────────────────────────────

// ─── Card HTML ────────────────────────────────────────────────────────────────

window.buildCard = function(card, selected = false, selectable = true, mini = false, isTrump = false) {
  if (!card || card.hidden) return `<div class="card face-down no-hover"></div>`;
  const sym      = suit_sym(card.suit);
  const cc       = color_cls(card.suit);
  const selCls   = selected   ? 'selected'  : '';
  const trumpCls = isTrump    ? 'trump-glow': '';
  const hoverCls = selectable ? ''          : 'no-hover';
  const miniStyle = mini ? 'style="width:56px;height:78px;"' : '';
  return `<div class="card ${selCls} ${trumpCls} ${hoverCls}" data-id="${card.id}" ${miniStyle}>
    <div class="cc top"><span class="rank ${cc}">${card.rank}</span><span class="sym ${cc}">${sym}</span></div>
    <div class="card-mid ${cc}">${sym}</div>
    <div class="cc bot"><span class="rank ${cc}">${card.rank}</span><span class="sym ${cc}">${sym}</span></div>
  </div>`;
};

window.buildTrumpCard = function(card) {
  if (!card) return '';
  const sym = suit_sym(card.suit);
  const cc  = color_cls(card.suit);
  return `<div class="card trump-glow no-hover" style="width:72px;height:100px;flex-shrink:0;">
    <div class="cc top"><span class="rank ${cc}">${card.rank}</span><span class="sym ${cc}">${sym}</span></div>
    <div class="card-mid ${cc}">${sym}</div>
    <div class="cc bot"><span class="rank ${cc}">${card.rank}</span><span class="sym ${cc}">${sym}</span></div>
  </div>`;
};

// ─── Selection permission ─────────────────────────────────────────────────────

window.canSelectCards = function(s) {
  if (!s) return false;
  const { myPlayerIdx } = State;
  const phase = s.phase;

  // Never allow selection while a stake offer is pending (either direction)
  if (s.stake_offerer_idx != null) return false;

  // Stakes phase: only the active player selects cards
  if (phase === 'stakes') return s.active_player_idx === myPlayerIdx;

  // Playing: only on my turn
  if (phase === 'playing') return s.active_player_idx === myPlayerIdx;

  // Cutting/forced-cut: only the defender (not the one who played)
  if (phase === 'cutting' || phase === 'forced_cut')
    return s.playing_player_idx !== myPlayerIdx;

  // Pass mode overrides everything
  if (State.passModeActive) return true;

  return false;
};

window.getSelectedSuit = function() {
  const s = State.gameState;
  if (!s || State.selectedCards.length === 0) return null;
  const firstCard = s.players[State.myPlayerIdx].hand.find(c => c.id === State.selectedCards[0]);
  return firstCard ? firstCard.suit : null;
};

// Returns true if current selection is a valid counter-play (3 non-trump same suit)
window.isCounterSelection = function() {
  const s = State.gameState;
  if (!s || s.phase !== 'cutting') return false;
  if (State.selectedCards.length !== 3) return false;
  const hand     = s.players[State.myPlayerIdx].hand || [];
  const selCards = State.selectedCards.map(id => hand.find(c => c.id === id)).filter(Boolean);
  if (selCards.length !== 3) return false;
  const suit = selCards[0].suit;
  return selCards.every(c => c.suit === suit) && suit !== s.trump_suit;
};

// ─── Card selection ───────────────────────────────────────────────────────────

window.toggleCard = function(cardId) {
  const s = State.gameState;
  if (!canSelectCards(s)) return;

  // Deselect if already selected
  const idx = State.selectedCards.indexOf(cardId);
  if (idx >= 0) {
    State.selectedCards.splice(idx, 1);
    _refreshHighlights();
    updateActionButtons();
    return;
  }

  // ── Pass mode — any cards, any suit, exactly passCount ───────────────────
  if (State.passModeActive) {
    if (State.selectedCards.length >= State.passCount) State.selectedCards.shift();
    State.selectedCards.push(cardId);

  // ── Cutting / forced-cut ─────────────────────────────────────────────────
  } else if (s.phase === 'cutting' || s.phase === 'forced_cut') {
    const maxCut = s.played_cards?.length || 1;
    // In normal cutting: allow up to 3 cards (for counter-play option)
    // In forced-cut: allow up to n cards (exact count for cut or pass)
    const max = s.phase === 'cutting' ? 3 : maxCut;

    if (State.selectedCards.length >= max) {
      buzzCard(cardId);
      toast('Clear your selection first', 'error', 2000);
      return;
    }

    // Suit lock only applies when we have 2+ cards already selected AND
    // all selected so far are the same suit (i.e. player is building a cut/counter).
    // If mixed suits are already selected, allow adding more (player is building a pass).
    const hand = s.players[State.myPlayerIdx].hand || [];
    if (State.selectedCards.length > 0) {
      const selCards  = State.selectedCards.map(id => hand.find(c => c.id === id)).filter(Boolean);
      const allSame   = selCards.every(c => c.suit === selCards[0].suit);
      const newCard   = hand.find(c => c.id === cardId);
      const sameAsAll = newCard && newCard.suit === selCards[0].suit;

      // Only enforce suit lock if:
      // - all current selections are same suit AND
      // - we're not yet at the pass count (still possibly building a cut/counter)
      // Once player deliberately picks a different suit, they're passing — allow it
      if (allSame && State.selectedCards.length < maxCut && !sameAsAll) {
        // They're switching suits — that means they want to pass with mixed cards.
        // Allow it (pass validates count only, not suit).
      }
      // No restriction — let them pick any card
    }
    State.selectedCards.push(cardId);

  // ── Passing (doPassAuto path — inline pass without pass mode) ─────────────
  // When phase is cutting and we are selecting cards to pass inline,
  // we need exactly n cards but any suit is allowed.
  // However, doPassAuto is triggered directly so we don't reach here for pass.
  // The cutting block above handles the suit lock — but if somehow we need
  // to relax this for pass, it's handled by passModeActive above.

  // ── Playing / stakes — up to 3 same-suit cards ───────────────────────────
  } else {
    const lockedSuit = getSelectedSuit();
    if (lockedSuit) {
      const card = s.players[State.myPlayerIdx].hand.find(c => c.id === cardId);
      if (card && card.suit !== lockedSuit) { buzzCard(cardId); return; }
    }
    if (State.selectedCards.length >= 3) { buzzCard(cardId); return; }
    State.selectedCards.push(cardId);
  }

  _refreshHighlights();
  updateActionButtons();
};

window.clearSel = function() {
  State.selectedCards = [];
  document.querySelectorAll('.card.selected').forEach(el => el.classList.remove('selected'));
  updateActionButtons();
};

window.buzzCard = function(cardId) {
  const el = document.querySelector(`.card[data-id="${cardId}"]`);
  if (!el) return;
  el.classList.remove('buzz');
  void el.offsetWidth;
  el.classList.add('buzz');
  setTimeout(() => el.classList.remove('buzz'), 500);
};

function _refreshHighlights() {
  document.querySelectorAll('.card[data-id]').forEach(el =>
    el.classList.toggle('selected', State.selectedCards.includes(el.dataset.id))
  );
}

window.updateActionButtons = function() {
  const s = State.gameState;
  if (!s) return;
  const sel = State.selectedCards.length;

  const playBtn = document.getElementById('btn-play');
  if (playBtn) {
    playBtn.disabled    = sel === 0;
    playBtn.textContent = sel > 0 ? `Play ${sel} card${sel > 1 ? 's' : ''}` : 'Play cards…';
  }

 const cutBtn = document.getElementById('btn-cut');
if (cutBtn) {
    const n = s.played_cards?.length || 0;
    cutBtn.disabled    = sel !== n;
    cutBtn.textContent = sel > 0 ? `Cut with ${sel}` : 'Cut…';
}

  const counterBtn = document.getElementById('btn-counter');
  if (counterBtn) counterBtn.disabled = !isCounterSelection();

  const passBtn = document.getElementById('btn-pass');
  if (passBtn) {
    const n = State.gameState?.played_cards?.length || 0;
    passBtn.disabled    = sel !== n;
    passBtn.textContent = sel === n ? `Pass selected` : `Pass (select ${n})`;
  }

  const passConfBtn = document.getElementById('btn-pass-confirm');
  if (passConfBtn) passConfBtn.disabled = sel !== State.passCount;
};