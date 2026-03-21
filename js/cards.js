'use strict';
// ─────────────────────────────────────────────────────────────────────────────
// cards.js — Card HTML builder, selection logic, buzz effect
// ─────────────────────────────────────────────────────────────────────────────
// Depends on: state.js (State, suit_sym, color_cls), utils.js (toast),
//             animations.js (flyCard, getDeckRect)
// Loaded by:  index.html (after animations.js)
// ─────────────────────────────────────────────────────────────────────────────


// ─── Card HTML ────────────────────────────────────────────────────────────────

window.buildCard = function(card, selected = false, selectable = true, mini = false, isTrump = false) {
  if (!card || card.hidden) {
    return `<div class="card face-down no-hover"></div>`;
  }
  const sym      = suit_sym(card.suit);
  const cc       = color_cls(card.suit);
  const selCls   = selected   ? 'selected'   : '';
  const trumpCls = isTrump    ? 'trump-glow' : '';
  const hoverCls = selectable ? ''           : 'no-hover';
  const miniStyle = mini ? 'style="width:56px;height:78px;"' : '';

  return `<div class="card ${selCls} ${trumpCls} ${hoverCls}" data-id="${card.id}" ${miniStyle}>
    <div class="cc top"><span class="rank ${cc}">${card.rank}</span><span class="sym ${cc}">${sym}</span></div>
    <div class="card-mid ${cc}">${sym}</div>
    <div class="cc bot"><span class="rank ${cc}">${card.rank}</span><span class="sym ${cc}">${sym}</span></div>
  </div>`;
};

/** Trump card rendered sideways inside the deck widget. */
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

/** Returns true if the current game phase allows the local player to select cards. */
window.canSelectCards = function(s) {
  if (!s) return false;
  const { myPlayerIdx } = State;
  const phase = s.phase;

  // Stakes: selectable unless I've already raised and am waiting for opponent
  if (phase === 'stakes') {
    const pendingFromMe = s.stake_offerer_idx !== null
                       && s.stake_offerer_idx !== undefined
                       && s.stake_offerer_idx === myPlayerIdx;
    return !pendingFromMe;
  }

  if (phase === 'playing' && s.active_player_idx === myPlayerIdx) return true;

  if ((phase === 'cutting' || phase === 'forced_cut')
      && s.playing_player_idx !== myPlayerIdx) return true;

  if (State.passModeActive) return true;

  return false;
};

/** Returns the suit locked in by the first selected card, or null. */
window.getSelectedSuit = function() {
  const s = State.gameState;
  if (!s || State.selectedCards.length === 0) return null;
  const me        = s.players[State.myPlayerIdx];
  const firstCard = me.hand.find(c => c.id === State.selectedCards[0]);
  return firstCard ? firstCard.suit : null;
};


// ─── Toggle a card in/out of the selection ───────────────────────────────────

window.toggleCard = function(cardId) {
  const s = State.gameState;
  if (!canSelectCards(s)) return;

  const idx = State.selectedCards.indexOf(cardId);
  if (idx >= 0) {
    // Deselect
    State.selectedCards.splice(idx, 1);
  } else {

    // ── Pass mode: cycle oldest out if at limit ──────────────────────────────
    if (State.passModeActive) {
      if (State.selectedCards.length >= State.passCount) State.selectedCards.shift();
      State.selectedCards.push(cardId);

    // ── Cutting phase: select exactly as many cards as were played ─────────────
    // Cut cards must all be the same suit, but trump always beats any non-trump
    // regardless of the suit already selected — so we only lock suit when neither
    // the locked suit nor the new card is trump.
    } else if (s.phase === 'cutting' || s.phase === 'forced_cut') {
      const maxCut = s.played_cards?.length || 1;
      if (State.selectedCards.length >= maxCut) {
        // Already have enough selected — buzz to say "hit Cut, don't click more"
        buzzCard(cardId);
        toast(`Already selected ${maxCut} card${maxCut > 1 ? 's' : ''} — click Cut, or Clear to change`, 'error', 2500);
        return;
      }
      // Suit lock — but trump can always be added regardless of locked suit,
      // and non-trump can be added if no suit is locked yet.
      const lockedSuit = getSelectedSuit();
      if (lockedSuit) {
        const card       = s.players[State.myPlayerIdx].hand.find(c => c.id === cardId);
        const cardIsTrump = card && s.trump_suit && card.suit === s.trump_suit;
        const lockIsTrump = s.trump_suit && lockedSuit === s.trump_suit;
        // Only reject if neither card nor current selection is trump AND suits differ
        if (!cardIsTrump && !lockIsTrump && card && card.suit !== lockedSuit) {
          buzzCard(cardId);
          toast('Cut cards must all be the same suit', 'error', 2000);
          return;
        }
        // If mixing trump with non-trump, clear and restart with just this card
        // (trump cuts are single-suit by definition — all trump)
        if ((cardIsTrump && !lockIsTrump) || (!cardIsTrump && lockIsTrump)) {
          State.selectedCards = [];
        }
      }
      State.selectedCards.push(cardId);

    // ── Playing / stakes: up to 3 same-suit cards ───────────────────────────
    } else {
      const lockedSuit = getSelectedSuit();
      if (lockedSuit) {
        const card = s.players[State.myPlayerIdx].hand.find(c => c.id === cardId);
        if (card && card.suit !== lockedSuit) { buzzCard(cardId); return; }
      }
      if (State.selectedCards.length >= 3) { buzzCard(cardId); return; }
      State.selectedCards.push(cardId);
    }
  }

  // Refresh selection highlights across all rendered cards
  document.querySelectorAll('.card[data-id]').forEach(el =>
    el.classList.toggle('selected', State.selectedCards.includes(el.dataset.id))
  );
  updateActionButtons();
};

window.clearSel = function() {
  State.selectedCards = [];
  document.querySelectorAll('.card.selected').forEach(el => el.classList.remove('selected'));
  updateActionButtons();
};

/** Shake a card to signal an illegal selection. */
window.buzzCard = function(cardId) {
  const el = document.querySelector(`.card[data-id="${cardId}"]`);
  if (!el) return;
  el.classList.remove('buzz');
  void el.offsetWidth; // force reflow to restart animation
  el.classList.add('buzz');
  setTimeout(() => el.classList.remove('buzz'), 500);
};

/** Update Play / Cut / Pass-confirm buttons to reflect current selection count. */
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
    const n             = s.played_cards?.length || 0;
    cutBtn.disabled     = sel !== n;
    cutBtn.textContent  = sel > 0 ? `Cut with ${sel}` : 'Cut…';
  }

  const passConfBtn = document.getElementById('btn-pass-confirm');
  if (passConfBtn) {
    passConfBtn.disabled = sel !== State.passCount;
  }
};