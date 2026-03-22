'use strict';
// ─────────────────────────────────────────────────────────────────────────────
// animations.js — Card movement animations
// ─────────────────────────────────────────────────────────────────────────────
// Depends on: (DOM only)
// ─────────────────────────────────────────────────────────────────────────────


// ─── Core: fly a card from fromRect to its current DOM position ───────────────

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
            : { left: window.innerWidth - 80, top: window.innerHeight / 2 };
};

window.getHandRect = function(playerIdx) {
  const el = document.getElementById(`hand-${playerIdx}`);
  return el ? el.getBoundingClientRect()
            : { left: window.innerWidth / 2, top: window.innerHeight / 2 };
};


// ─── Play cards to table ──────────────────────────────────────────────────────
// Cards already in #played-row slide from the player's hand to their position.

window.animatePlayToTable = function(cardEls, fromPlayerIdx, onLanded) {
  const arr      = Array.from(cardEls);
  const fromRect = getHandRect(fromPlayerIdx);
  arr.forEach((el, i) => setTimeout(() => flyCard(el, fromRect, 260), i * 70));
  const total = arr.length * 70 + 260 + 30;
  setTimeout(() => { if (onLanded) onLanded(); }, total);
};


// ─── Shared: incoming cards fly to table, pause, then all sweep to taker ─────
// Used for both cut and pass — same visual rhythm.
//
// incomingEls  — new cards flying in from incomingFrom (already appended to row)
// playedEls    — cards already on the table
// incomingFrom — player index the incoming cards fly FROM
// takerIdx     — player index everything sweeps TOWARD
// onDone       — called after sweep finishes

function _animateIncomingThenSweep(incomingEls, playedEls, incomingFrom, takerIdx, onDone) {
  const fromRect = getHandRect(incomingFrom);
  const dur      = 260;

  // Wait one rAF to guarantee the browser has laid out the incoming elements
  // before we read their getBoundingClientRect in flyCard
  requestAnimationFrame(() => {
    incomingEls.forEach((el, i) => {
      setTimeout(() => flyCard(el, fromRect, dur), i * 70);
    });

    // Pause after last card lands so both sets are visible together
    const landTime = incomingEls.length * 70 + dur + 300;
    setTimeout(() => {
      const all = [...Array.from(playedEls), ...incomingEls];
      _sweepToSide(all, takerIdx, onDone);
    }, landTime);
  });
}


// ─── Cut sequence ─────────────────────────────────────────────────────────────
// cutEls already appended to #played-row by game_actions.js

window.animateCutSequence = function(cutEls, playedEls, cutterIdx, onDone) {
  _animateIncomingThenSweep(cutEls, playedEls, cutterIdx, cutterIdx, onDone);
};


// ─── Pass sequence ────────────────────────────────────────────────────────────
// Creates face-down ghost cards for the passed cards, same flow as cut.

window.animatePassSequence = function(passCount, passerIdx, takerIdx, playedEls, onDone) {
  const row      = document.getElementById('played-row');
  const passEls  = [];
  for (let i = 0; i < passCount; i++) {
    const el = document.createElement('div');
    el.className = 'card face-down no-hover pass-ghost';
    row.appendChild(el);
    passEls.push(el);
  }
  _animateIncomingThenSweep(passEls, playedEls, passerIdx, takerIdx, onDone);
};


// ─── Sweep cards off-screen toward a player ───────────────────────────────────
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