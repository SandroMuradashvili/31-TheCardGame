'use strict';
// ─────────────────────────────────────────────────────────────────────────────
// stake_actions.js — Stake negotiation API calls
// ─────────────────────────────────────────────────────────────────────────────
// Handles: raise, accept, decline
//
// Depends on: state.js (State), utils.js (api, ROOM, logAction), render.js (render)
// Loaded by:  index.html (after utils.js and render.js)
// ─────────────────────────────────────────────────────────────────────────────

window.doRaiseStake = async function() {
  const data = await api(ROOM('offer_stake'), { player_idx: State.myPlayerIdx });
  if (!data?.state) return;
  State.gameState = data.state;
  logAction(`<span class="le-type">RAISE</span> You offered → ${data.state.pending_stake}`);
  render();
};

window.doAcceptStake = async function() {
  const data = await api(ROOM('accept_stake'), { player_idx: State.myPlayerIdx });
  if (!data?.state) return;
  State.gameState = data.state;
  logAction(`<span class="le-type">ACCEPT</span> Stake now ${data.state.current_stake}`);
  render();
};

window.doDeclineStake = async function() {
  const data = await api(ROOM('decline_stake'), { player_idx: State.myPlayerIdx });
  if (!data?.state) return;
  State.gameState = data.state;
  logAction(`<span class="le-type">DECLINE</span> You declined`);
  render();
};
