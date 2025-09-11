// Folding Brownian Graph — edges connect/disconnect by distance (with hysteresis).
// Controls: number of points, movement amplitude, movement frequency,
//           connect distance (min), break distance (max).

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import GUI from 'https://cdn.jsdelivr.net/npm/lil-gui@0.20/+esm';

// ---------- scene ----------
const container = document.getElementById('scene-container');

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
renderer.outputColorSpace = THREE.SRGBColorSpace;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

const camera = new THREE.PerspectiveCamera(38, container.clientWidth / container.clientHeight, 0.01, 100);
camera.position.set(3.2, 2.2, 5.4);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.minDistance = 2.0;
controls.maxDistance = 12.0;

// ---------- helpers ----------
const tmpV1 = new THREE.Vector3();
const tmpV2 = new THREE.Vector3();
const tmpV3 = new THREE.Vector3();

function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}
function randn3(out) {
  out.set(randn(), randn(), randn());
  return out;
}
function randInSphere(radius = 1) {
  let v = new THREE.Vector3();
  do {
    v.set(Math.random()*2-1, Math.random()*2-1, Math.random()*2-1);
  } while (v.lengthSq() > 1);
  return v.multiplyScalar(radius);
}

// ---------- Brownian graph ----------
class BrownianGraph {
  constructor(opts) {
    this.params = { ...opts };
    this.group = new THREE.Group();
    scene.add(this.group);

    this.N = 0;
    this.pos = null;      // Float32Array (N*3)
    this.vel = null;      // Float32Array (N*3)
    this.anc = null;      // Float32Array (N*3)
    this.state = null;    // Uint8Array (N*N), only j>i used

    this.maxEdges = 0;
    this.pointsMesh = null;
    this.linesMesh = null;
    this.linesGeom = null;

    this.setCount(this.params.count);
  }

  setCount(n) {
    n = Math.max(2, Math.floor(n));

    if (this.pointsMesh) {
      this.group.remove(this.pointsMesh);
      this.pointsMesh.geometry.dispose();
      this.pointsMesh.material.dispose();
      this.pointsMesh = null;
    }
    if (this.linesMesh) {
      this.group.remove(this.linesMesh);
      this.linesGeom.dispose();
      this.linesMesh.material.dispose();
      this.linesMesh = null;
      this.linesGeom = null;
    }

    this.N = n;
    this.pos = new Float32Array(n * 3);
    this.vel = new Float32Array(n * 3);
    this.anc = new Float32Array(n * 3);
    this.state = new Uint8Array(n * n); // 0/1 connection state

    // anchors & initial positions
    const R = 1.6;
    for (let i = 0; i < n; i++) {
      const a = randInSphere(R);
      const p = a.clone().add(randInSphere(0.1));
      this.anc[i*3+0] = a.x; this.anc[i*3+1] = a.y; this.anc[i*3+2] = a.z;
      this.pos[i*3+0] = p.x; this.pos[i*3+1] = p.y; this.pos[i*3+2] = p.z;
      this.vel[i*3+0] = 0;   this.vel[i*3+1] = 0;   this.vel[i*3+2] = 0;
    }

    // points cloud
    const pg = new THREE.BufferGeometry();
    pg.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    const pm = new THREE.PointsMaterial({
      color: 0xffffff,
      size: 0.035,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.95
    });
    this.pointsMesh = new THREE.Points(pg, pm);
    this.group.add(this.pointsMesh);

    // lines capacity
    this.maxEdges = n * (n - 1) / 2;
    const linePos = new Float32Array(this.maxEdges * 2 * 3);
    const lineCol = new Float32Array(this.maxEdges * 2 * 3);

    this.linesGeom = new THREE.BufferGeometry();
    this.linesGeom.setAttribute('position', new THREE.BufferAttribute(linePos, 3));
    this.linesGeom.setAttribute('color',    new THREE.BufferAttribute(lineCol, 3));
    this.linesGeom.setDrawRange(0, 0);

    const lm = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.95 });
    this.linesMesh = new THREE.LineSegments(this.linesGeom, lm);
    this.group.add(this.linesMesh);
  }

  // i<j mapping
  _idx(i, j) { return i * this.N + j; }

  update(dt) {
    const N = this.N;
    const { amp, freq, connectDist, breakDist } = this.params;

    // OU-like motion around anchor -> "Brownian" vibe, bounded.
    const beta = THREE.MathUtils.clamp(1.2 * freq, 0.0, 10.0);  // velocity damping
    const k    = THREE.MathUtils.clamp(1.5 * freq, 0.0, 12.0);  // spring to anchor
    const sigma = amp * (0.8 + 0.4 * Math.random());            // noise intensity (jitter)
    const sqrtDt = Math.sqrt(Math.max(1e-6, dt));
    const n3 = new THREE.Vector3();

    for (let i = 0; i < N; i++) {
      const ix = i*3;
      // read
      const px = this.pos[ix+0], py = this.pos[ix+1], pz = this.pos[ix+2];
      let vx = this.vel[ix+0], vy = this.vel[ix+1], vz = this.vel[ix+2];
      const ax = this.anc[ix+0], ay = this.anc[ix+1], az = this.anc[ix+2];

      // forces
      const toAnchorX = ax - px, toAnchorY = ay - py, toAnchorZ = az - pz;

      randn3(n3).multiplyScalar(sigma * sqrtDt); // stochastic kick
      vx += (-beta * vx + k * toAnchorX) * dt + n3.x;
      vy += (-beta * vy + k * toAnchorY) * dt + n3.y;
      vz += (-beta * vz + k * toAnchorZ) * dt + n3.z;

      // integrate & light boundary push (keep things visible)
      let nx = px + vx * dt, ny = py + vy * dt, nz = pz + vz * dt;
      const r = Math.sqrt(nx*nx + ny*ny + nz*nz);
      const limit = 2.2;
      if (r > limit) {
        const s = (limit / r);
        nx *= s; ny *= s; nz *= s;
        vx *= 0.6; vy *= 0.6; vz *= 0.6;
      }

      // write
      this.pos[ix+0] = nx; this.pos[ix+1] = ny; this.pos[ix+2] = nz;
      this.vel[ix+0] = vx; this.vel[ix+1] = vy; this.vel[ix+2] = vz;
    }

    // connections (hysteresis)
    const onR  = connectDist;
    const offR = Math.max(breakDist, onR + 1e-6);

    let eCount = 0;
    const P = this.pos;
    const linePos = this.linesGeom.getAttribute('position').array;
    const lineCol = this.linesGeom.getAttribute('color').array;

    for (let i = 0; i < N; i++) {
      const ix = i*3;
      const pix = P[ix+0], piy = P[ix+1], piz = P[ix+2];
      for (let j = i+1; j < N; j++) {
        const jx = j*3;
        const pjx = P[jx+0], pjy = P[jx+1], pjz = P[jx+2];

        const dx = pjx - pix, dy = pjy - piy, dz = pjz - piz;
        const d2 = dx*dx + dy*dy + dz*dz;
        const d  = Math.sqrt(d2);

        const k = this._idx(i, j);
        const isOn = this.state[k] === 1;
        if (!isOn && d <= onR) this.state[k] = 1;
        else if (isOn && d >= offR) this.state[k] = 0;

        if (this.state[k] === 1) {
          // pack into line buffer
          const base = eCount * 2 * 3;

          linePos[base+0] = pix; linePos[base+1] = piy; linePos[base+2] = piz;
          linePos[base+3] = pjx; linePos[base+4] = pjy; linePos[base+5] = pjz;

          // color by tension (near-on distance is bright)
          const t = THREE.MathUtils.clamp(1.0 - (d - onR) / (offR - onR), 0.0, 1.0);
          const c0 = 0.35 + 0.65 * t;
          const r = 0.85 * t + 0.15, g = 0.95 * c0, b = 1.0; // soft iridescent bias

          lineCol[base+0] = r; lineCol[base+1] = g; lineCol[base+2] = b;
          lineCol[base+3] = r; lineCol[base+4] = g; lineCol[base+5] = b;

          eCount++;
        }
      }
    }

    this.linesGeom.setDrawRange(0, eCount * 2);
    this.linesGeom.getAttribute('position').needsUpdate = true;
    this.linesGeom.getAttribute('color').needsUpdate = true;

    // points update
    this.pointsMesh.geometry.attributes.position.needsUpdate = true;
  }
}

// ---------- gui ----------
const uiHost = document.getElementById('ui');
const gui = new GUI({ title: 'Controls', width: 320 });
uiHost.appendChild(gui.domElement);

const params = {
  count:       28,     // number of points
  amp:         0.75,   // movement amplitude (noise strength)
  freq:        0.8,    // movement frequency (jitter bandwidth)
  connectDist: 0.70,   // connect when distance <= this
  breakDist:   0.95,   // break when distance >= this
  play:        true,
  reseed: () => {
    graph.setCount(graph.N); // re-anchor & reset states
  },
  resetCamera: () => {
    camera.position.set(3.2, 2.2, 5.4);
    controls.target.set(0,0,0);
    controls.update();
  }
};

const fSystem = gui.addFolder('System');
fSystem.add(params, 'count', 4, 160, 1).name('Points').onFinishChange(v => { graph.setCount(v); });
fSystem.add(params, 'amp', 0.0, 2.0, 0.001).name('Amplitude').onChange(v => { graph.params.amp = v; });
fSystem.add(params, 'freq', 0.0, 3.0, 0.001).name('Frequency').onChange(v => { graph.params.freq = v; });
fSystem.add(params, 'connectDist', 0.05, 2.0, 0.001).name('Connect (≤)').onChange(v => {
  graph.params.connectDist = v;
  if (params.breakDist < v) { params.breakDist = v; graph.params.breakDist = v; gui.updateDisplay(); }
});
fSystem.add(params, 'breakDist', 0.05, 2.0, 0.001).name('Break (≥)').onChange(v => {
  graph.params.breakDist = Math.max(v, params.connectDist);
  params.breakDist = graph.params.breakDist;
  gui.updateDisplay();
});
fSystem.add(params, 'reseed').name('Reseed');

const fView = gui.addFolder('View');
fView.add(params, 'play').name('Play / Pause');
fView.add({ exposure: renderer.toneMappingExposure }, 'exposure', 0.6, 1.8, 0.01).name('Exposure')
     .onChange(v => renderer.toneMappingExposure = v);
fView.add(params, 'resetCamera').name('Reset Camera');

// ---------- graph instance ----------
const graph = new BrownianGraph({
  count: params.count,
  amp: params.amp,
  freq: params.freq,
  connectDist: params.connectDist,
  breakDist: params.breakDist
});

// ---------- loop ----------
let last = performance.now();
function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  let dt = (now - last) / 1000;
  last = now;

  // clamp dt for stability
  dt = Math.min(dt, 1/30);

  if (params.play) {
    graph.params.amp = params.amp;
    graph.params.freq = params.freq;
    graph.params.connectDist = params.connectDist;
    graph.params.breakDist = params.breakDist;
    graph.update(dt);
  }

  controls.update();
  renderer.render(scene, camera);
}
animate();

// ---------- resize ----------
function onResize() {
  const w = container.clientWidth;
  const h = container.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}
window.addEventListener('resize', onResize);
new ResizeObserver(onResize).observe(container);
