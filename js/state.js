'use strict';
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
  prevHandKeys:  ['', ''],
  prevHandSelectKeys: ['', ''],
  prevPlayedKey: '',
  bvbRunning:   false,
  bvbSpeed:     '500',
  tableCards: [],
  animating:  false,
};
window.SUIT_SYM  = { hearts: '♥', diamonds: '♦', clubs: '♣', spades: '♠' };
window.suit_sym  = s => SUIT_SYM[s] || s;
window.is_red    = s => s === 'hearts' || s === 'diamonds';
window.color_cls = s => is_red(s) ? 'red' : 'blk';