"""O6: Real-time Interactive Orbit Demo.

Serves a self-contained HTML page with Canvas-based orbit visualization.
- Adjust r_p, swingby parameters via sliders
- Real-time computation and rendering in browser
- No external dependencies beyond a web browser

Usage:
    python o6_interactive.py
    Then open http://localhost:7861 in a browser.
"""

import http.server
import json
import os

HTML_CONTENT = r'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Orbit Simulator — Solar Return Interactive Demo</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0a0a1a; color: #e0e0e0; display: flex; height: 100vh; }
  #controls { width: 320px; padding: 20px; background: #12122a; overflow-y: auto; }
  #controls h2 { font-size: 16px; margin: 16px 0 8px; color: #88ccff; }
  #controls h2:first-child { margin-top: 0; }
  label { display: block; margin: 12px 0 4px; font-size: 13px; color: #aaa; }
  input[type=range] { width: 100%; }
  .val { float: right; color: #ffcc66; font-weight: bold; }
  #main { flex: 1; display: flex; flex-direction: column; }
  canvas { flex: 1; display: block; }
  #info { padding: 10px 20px; background: #12122a; font-size: 13px; }
  #info span { margin-right: 24px; }
  .pass { color: #66ff88; }
  .fail { color: #ff6666; }
  button { margin-top: 16px; padding: 8px 20px; cursor: pointer; width: 100%;
           background: #3344aa; color: white; border: none; border-radius: 4px; font-size: 14px; }
  button:hover { background: #4455cc; }
</style>
</head>
<body>
<div id="controls">
  <h2>Orbit Parameters</h2>
  <label>Perihelion r<sub>p</sub> (AU) <span class="val" id="rp_val">0.25</span></label>
  <input type="range" id="rp" min="0.10" max="0.50" step="0.01" value="0.25">

  <label>Launch Date <span class="val" id="date_val">2026-07-03</span></label>
  <input type="range" id="day" min="0" max="364" step="1" value="183">

  <h2>Lunar Swingby</h2>
  <label><input type="checkbox" id="lunar" checked> Enable</label>
  <label>Pericynthion r<sub>m</sub> (km) <span class="val" id="rm_val">5000</span></label>
  <input type="range" id="rm" min="2000" max="50000" step="100" value="5000">
  <label>Swingby Side</label>
  <select id="side"><option value="trailing" selected>Trailing (accel)</option>
    <option value="leading">Leading (decel)</option></select>

  <button onclick="update()">Update Orbit</button>
  <button onclick="animateOrbit()" style="background:#226633;">Animate</button>

  <h2>Info</h2>
  <p style="font-size:12px;color:#888;">Scroll to zoom. Drag to pan.<br>
  Yellow = Sun, Blue = Earth orbit, Red = transfer ellipse.<br>
  Click "Animate" to see the rocket fly.</p>
</div>
<div id="main">
  <canvas id="canvas"></canvas>
  <div id="info">
    <span>Δv<sub>total</sub>: <b id="dv">—</b> km/s</span>
    <span>Δv<sub>launch</sub>: <b id="dv_launch">—</b></span>
    <span>Δv<sub>reentry</sub>: <b id="dv_re">—</b></span>
    <span>Flight: <b id="T">—</b> yr</span>
    <span>e: <b id="ecc">—</b></span>
    <span>Constraints: <b id="cons">—</b></span>
  </div>
</div>

<script>
const AU = 1.495978707e8;
const R_SUN = 6.96e5;
const MU_SUN = 1.32712440018e11;
const MU_EARTH = 3.986004418e5;
const MU_MOON = 4.9048695e3;
const R_EARTH = 6378.137;
const R_MOON = 1737.4;

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let zoom = 0.95;
let offsetX = 0, offsetY = 0;
let animId = null;

function resize() {
  canvas.width = canvas.parentElement.clientWidth;
  canvas.height = canvas.parentElement.clientHeight;
  update();
}
window.addEventListener('resize', resize);

canvas.addEventListener('wheel', e => {
  e.preventDefault();
  zoom *= e.deltaY > 0 ? 0.9 : 1.1;
  zoom = Math.min(Math.max(zoom, 0.2), 3.0);
  update();
});
let dragging = false, dragStart = {};
canvas.addEventListener('mousedown', e => { dragging = true; dragStart = {x: e.offsetX, y: e.offsetY, ox: offsetX, oy: offsetY}; });
canvas.addEventListener('mousemove', e => { if (dragging) { offsetX = dragStart.ox + (e.offsetX - dragStart.x) / canvas.width * 2 * zoom; offsetY = dragStart.oy - (e.offsetY - dragStart.y) / canvas.height * 2 * zoom; update(); }});
canvas.addEventListener('mouseup', () => dragging = false);

// Slider live updates
['rp', 'day', 'rm'].forEach(id => {
  document.getElementById(id).addEventListener('input', update);
});
document.getElementById('lunar').addEventListener('change', update);
document.getElementById('side').addEventListener('change', update);
document.getElementById('rp').addEventListener('input', () => {
  document.getElementById('rp_val').textContent = document.getElementById('rp').value;
});
document.getElementById('day').addEventListener('input', () => {
  const d = parseInt(document.getElementById('day').value);
  const date = new Date(2026, 0, 1 + d);
  document.getElementById('date_val').textContent = date.toISOString().split('T')[0];
});
document.getElementById('rm').addEventListener('input', () => {
  document.getElementById('rm_val').textContent = document.getElementById('rm').value;
});

function worldToScreen(x_au, y_au) {
  const cx = canvas.width / 2, cy = canvas.height / 2;
  const s = Math.min(canvas.width, canvas.height) / (2.5 * zoom);
  return [cx + (x_au + offsetX) * s, cy - (y_au + offsetY) * s];
}

function update() {
  const rp_au = parseFloat(document.getElementById('rp').value);
  const useLunar = document.getElementById('lunar').checked;
  const r_m = parseFloat(document.getElementById('rm').value);
  const side = document.getElementById('side').value;

  document.getElementById('rp_val').textContent = rp_au.toFixed(2);

  const rp = rp_au * AU;
  const r1 = AU;
  const a = (rp + r1) / 2;
  const e = (r1 - rp) / (r1 + rp);

  // Heliocentric Δv
  const v_earth = Math.sqrt(MU_SUN / r1);
  const v_a = Math.sqrt(MU_SUN * (2 / r1 - 1 / a));
  const dv_dep_helio = Math.abs(v_a - v_earth);

  // Launch and reentry Δv from LEO
  const r_leo = R_EARTH + 200;
  const v_esc = Math.sqrt(2 * MU_EARTH / r_leo);
  const v_circ = Math.sqrt(MU_EARTH / r_leo);
  const dv_launch = Math.sqrt(dv_dep_helio * dv_dep_helio + v_esc * v_esc) - v_circ;
  const dv_reentry = dv_dep_helio <= 15 ? dv_launch : Infinity;
  const dv_total = dv_launch + dv_reentry;

  const T_years = 2 * Math.PI * Math.sqrt(a * a * a / MU_SUN) / (365.25 * 86400);
  const ok = rp > R_SUN && T_years <= 2 && dv_dep_helio <= 15;

  document.getElementById('dv').textContent = dv_total.toFixed(2);
  document.getElementById('dv_launch').textContent = dv_launch.toFixed(2);
  document.getElementById('dv_re').textContent = dv_reentry.toFixed(2);
  document.getElementById('T').textContent = T_years.toFixed(2);
  document.getElementById('ecc').textContent = e.toFixed(4);
  document.getElementById('cons').textContent = ok ? 'ALL PASS' : 'FAIL';
  document.getElementById('cons').className = ok ? 'pass' : 'fail';

  draw(rp_au, e, a / AU);
}

function draw(rp_au, e, a_au) {
  const w = canvas.width, h = canvas.height;
  ctx.fillStyle = '#0a0a1a';
  ctx.fillRect(0, 0, w, h);

  // Grid
  ctx.strokeStyle = '#1a1a3a';
  ctx.lineWidth = 0.5;
  for (let i = -2; i <= 2; i += 0.5) {
    const [x1, y1] = worldToScreen(i, -2);
    const [x2, y2] = worldToScreen(i, 2);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    const [x3, y3] = worldToScreen(-2, i);
    const [x4, y4] = worldToScreen(2, i);
    ctx.beginPath(); ctx.moveTo(x3, y3); ctx.lineTo(x4, y4); ctx.stroke();
  }

  // Transfer ellipse
  ctx.strokeStyle = '#ff4444';
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let th = 0; th <= 2 * Math.PI; th += 0.02) {
    const r = a_au * (1 - e * e) / (1 + e * Math.cos(th));
    const [x, y] = worldToScreen(r * Math.cos(th), r * Math.sin(th));
    th === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Earth orbit
  ctx.strokeStyle = '#3366cc';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 6]);
  ctx.beginPath();
  for (let th = 0; th <= 2 * Math.PI; th += 0.02) {
    const [x, y] = worldToScreen(Math.cos(th), Math.sin(th));
    th === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.setLineDash([]);

  // Sun
  const [sx, sy] = worldToScreen(0, 0);
  ctx.fillStyle = '#ffcc00';
  ctx.beginPath(); ctx.arc(sx, sy, 10, 0, 2 * Math.PI); ctx.fill();
  ctx.fillStyle = '#ff8800';
  ctx.beginPath(); ctx.arc(sx, sy, 6, 0, 2 * Math.PI); ctx.fill();

  // Earth at departure
  const [ex, ey] = worldToScreen(1, 0);
  ctx.fillStyle = '#4488ff';
  ctx.beginPath(); ctx.arc(ex, ey, 5, 0, 2 * Math.PI); ctx.fill();

  // Perihelion
  const peri_x = a_au * (1 - e);
  const [px, py] = worldToScreen(peri_x, 0);
  ctx.fillStyle = '#ff44ff';
  ctx.beginPath(); ctx.arc(px, py, 4, 0, 2 * Math.PI); ctx.fill();

  // Labels
  ctx.fillStyle = '#ffcc00'; ctx.font = '12px sans-serif';
  ctx.fillText('Sun', sx + 14, sy + 4);
  ctx.fillStyle = '#4488ff';
  ctx.fillText('Earth (departure)', ex + 8, ey - 8);
  ctx.fillStyle = '#ff44ff';
  ctx.fillText(`rp=${rp_au.toFixed(2)} AU`, px + 8, py - 8);
  ctx.fillStyle = '#ff4444';
  ctx.fillText('Transfer Orbit', 20, 30);
  ctx.fillStyle = '#3366cc';
  ctx.fillText('Earth Orbit (1 AU)', 20, 50);
}

function animateOrbit() {
  if (animId) { cancelAnimationFrame(animId); animId = null; return; }
  const rp_au = parseFloat(document.getElementById('rp').value);
  const e = (1 - rp_au) / (1 + rp_au);
  const a_au = (rp_au + 1) / 2;
  const T = 2 * Math.PI * Math.sqrt(a_au * a_au * a_au);

  const N = 500;
  let frame = 0;

  function step() {
    const th = (frame / N) * 2 * Math.PI - Math.PI;
    const r = a_au * (1 - e * e) / (1 + e * Math.cos(th));
    const rx = r * Math.cos(th), ry = r * Math.sin(th);

    draw(rp_au, e, a_au);
    const [sx, sy] = worldToScreen(rx, ry);
    ctx.fillStyle = '#ff6600';
    ctx.beginPath(); ctx.arc(sx, sy, 4, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = '#ff8800';
    ctx.fillText('Rocket', sx + 8, sy - 8);

    frame = (frame + 2) % N;
    animId = requestAnimationFrame(step);
  }
  step();
}

resize();
</script>
</body>
</html>'''


def main():
    import webbrowser

    port = 7861
    html_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'orbit_demo.html')

    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    with open(html_path, 'w') as f:
        f.write(HTML_CONTENT)

    print(f'Interactive demo saved to: {html_path}')
    print(f'Open in browser: file://{html_path}')
    print(f'Or start a server: python3 -m http.server {port} -d {os.path.dirname(html_path)}')

    try:
        webbrowser.open(f'file://{html_path}')
    except Exception:
        pass

    # Also start a simple HTTP server
    import socketserver
    os.chdir(os.path.dirname(html_path))
    handler = http.server.SimpleHTTPRequestHandler

    with socketserver.TCPServer(('', port), handler) as httpd:
        print(f'Serving at http://localhost:{port}/orbit_demo.html')
        print('Press Ctrl+C to stop.')
        httpd.serve_forever()


if __name__ == '__main__':
    main()
