'use strict';
// ─────────────────────────────────────────────────────────────────────────────
// animations.js — Card movement animations
// ─────────────────────────────────────────────────────────────────────────────
// All animation helpers. No game logic lives here — only DOM transitions.
//
// Depends on: (DOM only — no other JS files)
// Loaded by:  index.html (after utils.js)
// ─────────────────────────────────────────────────────────────────────────────


// ─── Core: slide an element from a source rect to its current DOM position ───

window.flyCard = function(el, fromRect, durationMs) {
  if (!el || !fromRect) return;
  const dur    = durationMs || 280;
  const toRect = el.getBoundingClientRect();
  if (!toRect.width) return;

  const dx = fromRect.left - toRect.left;
  const dy = fromRect.top  - toRect.top;

  el.style.transition = 'none';
  el.style.transform  = `translate(${dx}px,${dy}px)`;
  el.style.opacity    = '0';
  el.style.zIndex     = '30';

  // Double rAF ensures the browser paints the start position before animating.
  // Make the card visible here — it starts at the deck position, not the final spot.
  requestAnimationFrame(() => requestAnimationFrame(() => {
    el.style.opacity    = '1';
    el.style.transition = `transform ${dur}ms cubic-bezier(0.25,0.46,0.45,0.94)`;
    el.style.transform  = '';
    setTimeout(() => { el.style.transition = ''; el.style.zIndex = ''; }, dur + 20);
  }));
};


// ─── Rect helpers ─────────────────────────────────────────────────────────────

window.getDeckRect = function() {
  const el = document.getElementById('deck-stack');
  return el ? el.getBoundingClientRect()
            : { left: window.innerWidth - 80, top: window.innerHeight / 2, width: 64, height: 90 };
};

window.getHandRect = function(playerIdx) {
  const el = document.getElementById(`hand-${playerIdx}`);
  return el ? el.getBoundingClientRect()
            : { left: window.innerWidth / 2, top: window.innerHeight / 2, width: 78, height: 110 };
};


// ─── Phase 1: Human plays cards to the table ──────────────────────────────────
// Cards are already inserted in #played-row; slide them from the player's hand.
// onLanded is called after the last card finishes flying.

window.animatePlayToTable = function(cardEls, fromPlayerIdx, onLanded) {
  const arr      = Array.from(cardEls);
  const fromRect = getHandRect(fromPlayerIdx);
  const dur      = 260;
  arr.forEach((el, i) => {
    setTimeout(() => flyCard(el, fromRect, dur), i * 70);
  });
  const total = arr.length * 70 + dur + 30;
  setTimeout(() => { if (onLanded) onLanded(); }, total);
};


// ─── Phase 2a: Bot cuts — cut cards fly in then all sweep away ────────────────

/**
 * @param {HTMLElement[]} cutEls    - cut card elements already appended to #played-row
 * @param {HTMLElement[]} playedEls - original played card elements still in row
 * @param {number}        cutterIdx - player who took the cards (determines sweep direction)
 * @param {Function}      onDone    - called after sweep finishes
 */
window.animateCutSequence = function(cutEls, playedEls, cutterIdx, onDone) {
  const fromRect = getHandRect(cutterIdx);
  const dur      = 260;

  // Cut cards fly from cutter's hand into the row beside the played cards
  cutEls.forEach((el, i) => {
    setTimeout(() => flyCard(el, fromRect, dur), i * 70);
  });

  // Pause after landing so the player can clearly see both cards on the table,
  // then a brief flash/highlight before sweeping them away
  const landTime = cutEls.length * 70 + dur + 60;
  setTimeout(() => {
    // Highlight all cards briefly — gold border flash to signal a successful cut
    const all = [...Array.from(playedEls), ...cutEls];
    all.forEach(el => {
      if (!el) return;
      el.style.transition  = 'box-shadow 0.15s ease';
      el.style.boxShadow   = '0 0 0 2px #c8a84b, 0 0 12px rgba(200,168,75,0.6)';
    });
    // Hold the highlight for 450ms so it registers, then sweep
    setTimeout(() => {
      all.forEach(el => {
        if (!el) return;
        el.style.boxShadow = '';
      });
      setTimeout(() => _sweepToSide(all, cutterIdx, onDone), 80);
    }, 450);
  }, landTime);
};


// ─── Phase 2b: Bot passes — face-down cards fly in then all sweep away ────────

/**
 * @param {number}        passCount  - how many cards were passed
 * @param {number}        passerIdx  - player who passed
 * @param {number}        takerIdx   - player who takes all (sweep direction)
 * @param {HTMLElement[]} playedEls  - original played card elements
 * @param {Function}      onDone
 */
window.animatePassSequence = function(passCount, passerIdx, takerIdx, playedEls, onDone) {
  const row      = document.getElementById('played-row');
  const fromRect = getHandRect(passerIdx);
  const dur      = 260;

  // Append face-down ghost cards representing the passed cards
  const passEls = [];
  for (let i = 0; i < passCount; i++) {
    const el = document.createElement('div');
    el.className = 'card face-down no-hover pass-ghost';
    row.appendChild(el);
    passEls.push(el);
  }

  // Fly face-down cards from passer's hand to their row position
  passEls.forEach((el, i) => {
    setTimeout(() => flyCard(el, fromRect, dur), i * 70);
  });

  // After landing, sweep all cards toward the taker
  const landTime = passEls.length * 70 + dur + 80;
  setTimeout(() => {
    const all = [...Array.from(playedEls), ...passEls];
    _sweepToSide(all, takerIdx, onDone);
  }, landTime);
};


// ─── Internal: sweep cards off-screen toward a player's side ─────────────────
// player 0 = bottom (sweep down), player 1 = top (sweep up)

function _sweepToSide(els, playerIdx, onDone) {
  const dy = playerIdx === 0 ? 200 : -200;

  els.forEach((el, i) => {
    if (!el || !el.parentNode) return;
    setTimeout(() => {
      el.style.transition = 'transform 300ms ease-in, opacity 250ms ease-in';
      el.style.transform  = `translateY(${dy}px)`;
      el.style.opacity    = '0';
    }, i * 35);
  });

  const total = els.length * 35 + 350;
  setTimeout(() => {
    els.forEach(el => {
      try { if (el.parentNode) el.parentNode.removeChild(el); } catch (_) {}
    });
    if (onDone) onDone();
  }, total);
}