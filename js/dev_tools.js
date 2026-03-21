'use strict';
// ─────────────────────────────────────────────────────────────────────────────
// dev_tools.js — Developer hand editor (visual card picker)
// ─────────────────────────────────────────────────────────────────────────────
// Shows a 20-card grid. Click a target (You / Opponent / Trump), then click
// cards to assign them. Hit "Deal these cards" to reset the round with those
// hands — works at ANY point, not just the start of a round.
//
// Depends on: state.js, utils.js (api, ROOM, toast), render.js (render)
// ─────────────────────────────────────────────────────────────────────────────

const DEV_RANKS = ['A', 'T', 'K', 'Q', 'J'];
const DEV_SUITS = [
  { id: 'H', sym: '♥', color: '#e74c3c' },
  { id: 'D', sym: '♦', color: '#e74c3c' },
  { id: 'C', sym: '♣', color: '#b0afa8' },
  { id: 'S', sym: '♠', color: '#b0afa8' },
];
const DEV_TARGET_COLORS = { p0: '#2e7d52', p1: '#7a3a8a', trump: '#c8a84b' };
const DEV_MAX = { p0: 3, p1: 3, trump: 1 };

// card id (e.g. "AH") → 'p0' | 'p1' | 'trump'
let _devAssign = {};
let _devTarget = 'p0';


// ─── Called from render.js whenever debug panel refreshes ────────────────────

window.renderDevTools = function() {
  const el = document.getElementById('dev-hand-editor');
  if (!el || !State.gameState) { if (el) el.innerHTML = ''; return; }

  const s       = State.gameState;
  const myName  = s.players[State.myPlayerIdx].name;
  const oppName = s.players[1 - State.myPlayerIdx].name;

  el.innerHTML = `<div id="dev-wrap" style="margin-top:10px;border-top:1px solid #2a4a2a;padding-top:10px;">
    <div style="color:#e8c55a;font-size:.7rem;font-weight:600;letter-spacing:.1em;margin-bottom:8px;">🃏 DEAL SPECIFIC HANDS</div>

    <div id="dev-targets" style="display:flex;gap:5px;flex-wrap:wrap;margin-bottom:8px;align-items:center;">
      <span style="font-size:.62rem;color:#4a7a4a;margin-right:2px;">Assign to:</span>
    </div>

    <div id="dev-preview" style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:8px;min-height:36px;"></div>

    <div id="dev-grid" style="display:grid;grid-template-columns:repeat(5,1fr);gap:4px;margin-bottom:8px;"></div>

    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
      <button onclick="devDeal()" style="background:linear-gradient(135deg,#c8a84b,#9a7828);color:#1a1208;border:none;border-radius:6px;padding:6px 14px;font-size:.82rem;font-weight:600;cursor:pointer;font-family:inherit;">▶ Deal these cards</button>
      <button onclick="devClearAll()" style="background:none;border:1px solid rgba(255,255,255,0.14);color:#7a9a7a;font-size:.7rem;padding:5px 10px;border-radius:6px;cursor:pointer;font-family:inherit;">Clear all</button>
      <span id="dev-status" style="font-size:.68rem;color:#7a9a7a;"></span>
    </div>

    <div style="margin-top:5px;font-size:.6rem;color:#3a6a3a;line-height:1.7;">
      Select a target → click cards to assign. Unassigned slots = random. Scores kept, round resets.
    </div>
  </div>`;

  _devNames = { p0: myName, p1: oppName, trump: 'Trump' };
  _buildTargetBtns();
  _buildGrid();
  _refreshAll();
};

let _devNames = {};

function _buildTargetBtns() {
  const wrap = document.getElementById('dev-targets');
  if (!wrap) return;
  ['p0', 'p1', 'trump'].forEach(t => {
    const btn = document.createElement('button');
    btn.id = `dev-tb-${t}`;
    btn.textContent = _devNames[t] || t;
    btn.onclick = () => devSetTarget(t);
    Object.assign(btn.style, {
      border: '1.5px solid', borderRadius: '6px',
      padding: '3px 10px', fontSize: '.7rem',
      cursor: 'pointer', fontFamily: 'inherit', transition: 'all .12s',
    });
    wrap.appendChild(btn);
  });
}

function _buildGrid() {
  const grid = document.getElementById('dev-grid');
  if (!grid) return;
  DEV_SUITS.forEach(suit => {
    DEV_RANKS.forEach(rank => {
      const id  = rank + suit.id;
      const btn = document.createElement('button');
      btn.id      = `dev-c-${id}`;
      btn.onclick = () => devClickCard(id);
      btn.innerHTML = `${rank}<br><span style="font-size:1em">${suit.sym}</span>`;
      btn.dataset.color = suit.color;
      Object.assign(btn.style, {
        background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.12)',
        borderRadius: '5px', padding: '4px 2px', cursor: 'pointer',
        color: suit.color, fontSize: '.72rem', fontFamily: 'monospace',
        lineHeight: '1.3', transition: 'all .1s',
      });
      grid.appendChild(btn);
    });
  });
}

function _refreshAll() {
  _refreshTargetBtns();
  _refreshGrid();
  _refreshPreview();
}

function _refreshTargetBtns() {
  ['p0', 'p1', 'trump'].forEach(t => {
    const btn = document.getElementById(`dev-tb-${t}`);
    if (!btn) return;
    const c = DEV_TARGET_COLORS[t];
    const active = _devTarget === t;
    btn.style.background   = active ? c + '33' : 'none';
    btn.style.borderColor  = active ? c : 'rgba(255,255,255,0.14)';
    btn.style.color        = active ? c : '#7a9a7a';
    btn.style.fontWeight   = active ? '600' : '400';
  });
}

function _refreshGrid() {
  DEV_SUITS.forEach(suit => {
    DEV_RANKS.forEach(rank => {
      const id  = rank + suit.id;
      const btn = document.getElementById(`dev-c-${id}`);
      if (!btn) return;
      const who = _devAssign[id];
      if (who) {
        const c = DEV_TARGET_COLORS[who];
        btn.style.background  = c + '44';
        btn.style.borderColor = c;
        btn.style.boxShadow   = `0 0 0 1px ${c}66`;
      } else {
        btn.style.background  = 'rgba(255,255,255,0.05)';
        btn.style.borderColor = 'rgba(255,255,255,0.12)';
        btn.style.boxShadow   = 'none';
      }
    });
  });
}

function _refreshPreview() {
  const wrap = document.getElementById('dev-preview');
  if (!wrap) return;
  wrap.innerHTML = '';
  ['p0', 'p1', 'trump'].forEach(t => {
    const c       = DEV_TARGET_COLORS[t];
    const max     = DEV_MAX[t];
    const cards   = _getAssigned(t);
    const div     = document.createElement('div');
    div.style.cssText = 'display:flex;flex-direction:column;gap:3px;';
    const label   = document.createElement('span');
    label.textContent = (_devNames[t] || t).toUpperCase();
    label.style.cssText = `font-size:.58rem;color:${c};letter-spacing:.08em;`;
    div.appendChild(label);
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;gap:3px;flex-wrap:wrap;';
    for (let i = 0; i < max; i++) {
      const pip = document.createElement('span');
      if (cards[i]) {
        const suit = DEV_SUITS.find(s => cards[i].endsWith(s.id));
        const rank = cards[i].slice(0, -1);
        pip.textContent = `${rank}${suit?.sym}`;
        pip.title       = 'Click to unassign';
        pip.onclick     = () => devUnassign(cards[i]);
        pip.style.cssText = `background:rgba(255,255,255,0.06);border:1px solid ${c}88;border-radius:4px;padding:1px 6px;color:${suit?.color || '#e8e0d0'};font-size:.78rem;font-family:monospace;cursor:pointer;`;
      } else {
        pip.textContent = '—';
        pip.style.cssText = `background:rgba(255,255,255,0.02);border:1px dashed rgba(255,255,255,0.1);border-radius:4px;padding:1px 6px;color:#3a5a3a;font-size:.78rem;`;
      }
      row.appendChild(pip);
    }
    div.appendChild(row);
    wrap.appendChild(div);
  });
}

function _getAssigned(target) {
  return Object.entries(_devAssign)
    .filter(([, v]) => v === target).map(([k]) => k);
}


// ─── Public actions ───────────────────────────────────────────────────────────

window.devSetTarget = function(t) {
  _devTarget = t;
  _refreshTargetBtns();
};

window.devClickCard = function(id) {
  if (_devAssign[id]) {
    // Already assigned — unassign
    delete _devAssign[id];
  } else {
    // Assign to current target; evict oldest if slot full
    const assigned = _getAssigned(_devTarget);
    if (assigned.length >= DEV_MAX[_devTarget]) {
      delete _devAssign[assigned[0]];
    }
    _devAssign[id] = _devTarget;
  }
  _refreshGrid();
  _refreshPreview();
};

window.devUnassign = function(id) {
  delete _devAssign[id];
  _refreshGrid();
  _refreshPreview();
};

window.devClearAll = function() {
  _devAssign = {};
  _refreshGrid();
  _refreshPreview();
};

window.devDeal = async function() {
  const statusEl = document.getElementById('dev-status');
  if (statusEl) statusEl.textContent = 'Dealing…';

  const p0    = _getAssigned('p0');
  const p1    = _getAssigned('p1');
  const trump = _getAssigned('trump')[0] || null;

  const data = await api(ROOM('dev/deal_specific'), {
    player_idx: State.myPlayerIdx,
    p0, p1, trump,
  });

  if (!data?.state) {
    if (statusEl) statusEl.textContent = '✗ Failed';
    return;
  }

  _devAssign = {};
  if (statusEl) statusEl.textContent = '✓ Dealt!';
  setTimeout(() => { if (statusEl) statusEl.textContent = ''; }, 2000);

  State.prevHandKeys  = ['', ''];
  State.prevPlayedKey = '';
  State.gameState     = data.state;
  render();
};