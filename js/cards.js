'use strict';
// ─────────────────────────────────────────────────────────────────────────────
// cards.js — Card HTML builder, selection logic, buzz effect
// ─────────────────────────────────────────────────────────────────────────────
// Depends on: state.js, utils.js (toast), animations.js (flyCard, getDeckRect)
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

  // If opponent raised mid-round and we haven't responded, lock card selection
  const pendingFromOpp = s.stake_offerer_idx != null
                      && s.stake_offerer_idx !== myPlayerIdx;
  if (pendingFromOpp) return false;

  // If we raised and are waiting for response, lock card selection
  const pendingFromMe = s.stake_offerer_idx != null
                     && s.stake_offerer_idx === myPlayerIdx;
  if (pendingFromMe) return false;

  if (phase === 'stakes')   return true;

  if (phase === 'playing' && s.active_player_idx === myPlayerIdx) return true;

  if ((phase === 'cutting' || phase === 'forced_cut')
      && s.playing_player_idx !== myPlayerIdx) return true;

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

  // ── Pass mode ─────────────────────────────────────────────────────────────
  if (State.passModeActive) {
    if (State.selectedCards.length >= State.passCount) State.selectedCards.shift();
    State.selectedCards.push(cardId);

  // ── Cutting / forced-cut ──────────────────────────────────────────────────
  // In normal cutting: allow up to 3 cards of the same suit so the player can
  // build a counter-play (3 non-trump same suit) as well as a standard cut.
  // In forced-cut (maliutka): only allow exact count, no counter option.
  } else if (s.phase === 'cutting' || s.phase === 'forced_cut') {
    const maxCut = s.played_cards?.length || 1;
    const max    = s.phase === 'cutting' ? 3 : maxCut;

    if (State.selectedCards.length >= max) {
      buzzCard(cardId);
      toast('Clear your selection first, or click Cut / Counter / Pass', 'error', 2500);
      return;
    }

    // All selected cards must be the same suit
    const lockedSuit = getSelectedSuit();
    if (lockedSuit) {
      const card = s.players[State.myPlayerIdx].hand.find(c => c.id === cardId);
      if (card && card.suit !== lockedSuit) {
        buzzCard(cardId);
        toast('All selected cards must be the same suit', 'error', 2000);
        return;
      }
    }
    State.selectedCards.push(cardId);

  // ── Playing / stakes: up to 3 same-suit cards ─────────────────────────────
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
    const n            = s.played_cards?.length || 0;
    cutBtn.disabled    = sel !== n;
    cutBtn.textContent = sel > 0 ? `Cut with ${sel}` : 'Cut…';
  }

  const counterBtn = document.getElementById('btn-counter');
  if (counterBtn) counterBtn.disabled = !isCounterSelection();

  const passConfBtn = document.getElementById('btn-pass-confirm');
  if (passConfBtn) passConfBtn.disabled = sel !== State.passCount;
};