'use strict';
// ─────────────────────────────────────────────────────────────────────────────
// utils.js — Shared UI and network utilities
// ─────────────────────────────────────────────────────────────────────────────
// Depends on: state.js (State.roomId)
// Loaded by:  index.html (after state.js)
// ─────────────────────────────────────────────────────────────────────────────

/** Show a floating notification. type: 'info' | 'error' | 'win' */
window.toast = function(msg, type = 'info', dur = 3000) {
  const el = document.createElement('div');
  el.className   = `toast ${type}`;
  el.textContent = msg;
  document.getElementById('toasts').appendChild(el);
  setTimeout(() => el.remove(), dur);
};

/** Show/hide the status banner below the trump strip. */
window.setStatus = function(msg, type = 'info') {
  const el = document.getElementById('status-banner');
  el.textContent = msg;
  el.className   = msg ? `visible ${type}` : '';
};

/** Append a line to the scrolling action log. */
window.logAction = function(text) {
  const log   = document.getElementById('action-log');
  const entry = document.createElement('div');
  entry.className = 'log-entry';
  entry.innerHTML = text;
  log.appendChild(entry);
  log.scrollTop = log.scrollHeight;
  // Keep the log from growing unbounded
  while (log.children.length > 20) log.removeChild(log.firstChild);
};

/** Fetch wrapper. POST when body provided, GET otherwise. Returns parsed JSON or null on error. */
window.api = async function(path, body = null) {
  const opts = { headers: { 'Content-Type': 'application/json' } };
  if (body !== null) { opts.method = 'POST'; opts.body = JSON.stringify(body); }
  else opts.method = 'GET';
  try {
    const res  = await fetch(path, opts);
    const data = await res.json();
    if (!data.success) { toast(data.error || 'Error', 'error'); return null; }
    return data;
  } catch (e) {
    toast('Network error', 'error');
    return null;
  }
};

/** Build a room-scoped API path. e.g. ROOM('play') → '/api/room/ABCDEF/play' */
window.ROOM = p => `/api/room/${State.roomId}/${p}`;

window.sleep = ms => new Promise(r => setTimeout(r, ms));
