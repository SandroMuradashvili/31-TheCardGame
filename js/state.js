'use strict';
// ─────────────────────────────────────────────────────────────────────────────
// state.js — Global State object + shared constants
// ─────────────────────────────────────────────────────────────────────────────
// No dependencies. Must be loaded FIRST in index.html.
// Every other JS file reads/writes window.State.
// ─────────────────────────────────────────────────────────────────────────────

window.State = {
  roomId:        null,
  myPlayerIdx:   0,
  gameMode:      'hvb',
  gameState:     null,
  prevState:     null,
  selectedCards: [],
  passModeActive: false,
  passCount:     0,
  debugMode:     false,
  pollTimer:     null,
  showTip:       false,
  modalShown:    false,
  prevScores:    [0, 0],
  prevHandKeys:  ['', ''],   // JSON strings of card id arrays per player
  prevPlayedKey: '',          // comma-joined played card ids

  // BvB viewer state
  bvbRunning:   false,
  bvbSpeed:     '500',

  // Animation state — tracks what's on the table so we can animate removal
  tableCards: [],            // [{card, el}] currently rendered on table
  animating:  false,         // true while a sweep animation is running
};

// ─── Card display helpers ─────────────────────────────────────────────────────

window.SUIT_SYM  = { hearts: '♥', diamonds: '♦', clubs: '♣', spades: '♠' };
window.suit_sym  = s => SUIT_SYM[s] || s;
window.is_red    = s => s === 'hearts' || s === 'diamonds';
window.color_cls = s => is_red(s) ? 'red' : 'blk';