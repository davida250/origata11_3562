// Origata — Brownian polygon surface + expanded Trails/RGB controls
// ---------------------------------------------------------------------------------------
// We keep your original renderer/env/material/post stack intact and only change (a) how
// geometry is generated (Brownian triangles from proximity) and (b) the trails/RGB UX.
// Baseline structure, material overlay, and post chain were derived from your original
// main.js and are preserved.  :contentReference[oaicite:1]{index=1}

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { AfterimagePass } from 'three/addons/postprocessing/AfterimagePass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { RGBShiftShader } from 'three/addons/shaders/RGBShiftShader.js';

import GUI from 'https://cdn.jsdelivr.net/npm/lil-gui@0.20/+esm';

// ---------- scene bootstrap (unchanged) ----------
const container = document.getElementById('scene-container');

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
renderer.outputColorSpace = THREE.SRGBColorSpace;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

const camera = new THREE.PerspectiveCamera(36, container.clientWidth / container.clientHeight, 0.01, 100);
camera.position.set(3.2, 1.8, 4.8);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.minDistance = 2.0;
controls.maxDistance = 9.0;

// Environment (PMREM + RoomEnvironment)
const pmrem = new THREE.PMREMGenerator(renderer);
const envRT = pmrem.fromScene(new RoomEnvironment(), 0.04);
scene.environment = envRT.texture;

// ---------- shared overlay (unchanged) ----------
function injectOverlay(material) {
  const uniforms = {
    uTime:         { value: 0 },
    uBandAngle:    { value: THREE.MathUtils.degToRad(28.0) },
    uBandSpeed:    { value: 0.25 },
    uBandFreq1:    { value: 6.0 },
    uBandFreq2:    { value: 9.5 },
    uBandAngle2:   { value: THREE.MathUtils.degToRad(82.0) },
    uBandStrength: { value: 0.52 },
    uTriScale:     { value: 1.15 },
    uWarp:         { value: 0.55 },
    uCellAmp:      { value: 0.55 },
    uCellFreq:     { value: 2.75 }
  };

  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `
        #include <common>
        varying vec3 vWorldPos;
        varying vec3 vWorldNormal;
      `)
      .replace('#include <project_vertex>', `
        #include <project_vertex>
        vWorldPos = (modelMatrix * vec4(transformed, 1.0)).xyz;
        vWorldNormal = normalize(mat3(modelMatrix) * normal);
      `);

    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `
        #include <common>
        varying vec3 vWorldPos;
        varying vec3 vWorldNormal;
        uniform float uTime;
        uniform float uBandAngle, uBandSpeed, uBandFreq1, uBandFreq2, uBandAngle2, uBandStrength;
        uniform float uTriScale, uWarp, uCellAmp, uCellFreq;

        float hash12(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453123); }
        vec2  hash22(vec2 p){
          p = vec2(dot(p,vec2(127.1,311.7)), dot(p,vec2(269.5,183.3)));
          return fract(sin(p) * 43758.5453);
        }
        float noise(vec2 p){
          vec2 i = floor(p), f = fract(p);
          vec2 u = f*f*(3.0-2.0*f);
          float a = hash12(i + vec2(0,0));
          float b = hash12(i + vec2(1,0));
          float c = hash12(i + vec2(0,1));
          float d = hash12(i + vec2(1,1));
          return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
        }
        float fbm(vec2 p){
          float s = 0.0, a = 0.5;
          for(int i=0;i<5;i++){
            s += a * noise(p);
            p = mat2(1.6,1.2,-1.2,1.6) * p;
            a *= 0.5;
          }
          return s;
        }
        float cellular(vec2 p){
          p *= uCellFreq;
          vec2 i = floor(p), f = fract(p);
          float md = 1.0;
          for(int y=-1;y<=1;y++){
            for(int x=-1;x<=1;x++){
              vec2 g = vec2(float(x), float(y));
              vec2 o = hash22(i + g) - 0.5;
              vec2 r = g + o + (f - 0.5);
              md = min(md, dot(r,r));
            }
          }
          return sqrt(md);
        }
        mat2 rot(float a){ float c = cos(a), s = sin(a); return mat2(c,-s,s,c); }
        vec3 rainbow(float t){
          const float TAU = 6.28318530718;
          vec3 phase = vec3(0.0, 0.33, 0.67) * TAU;
          return 0.5 + 0.5 * cos(TAU * t + phase);
        }
        vec3 triWeights(vec3 n){
          vec3 an = abs(normalize(n));
          an = pow(an, vec3(4.0));
          return an / (an.x + an.y + an.z + 1e-5);
        }
        vec3 stripeField(vec2 uv, float baseAngle){
          float t = uTime;
          float theta = baseAngle + t * uBandSpeed;
          mat2 R = rot(theta);

          vec2 w = uv * uTriScale;
          float w1 = fbm(w * 1.2);
          w += uWarp * vec2(w1, fbm(w + 17.1));

          float s1 = 0.5 + 0.5 * sin(dot(R * w, vec2(uBandFreq1, 0.0)));
          float s2 = 0.5 + 0.5 * sin(dot(rot(uBandAngle2) * w, vec2(uBandFreq2, 0.0)));
          float mixS = max(s1, s2 * 0.85);
          mixS = mixS * (0.72 + 0.28 * fbm(w * 0.9));

          float cells = 1.0 - smoothstep(0.0, 0.75, cellular(uv));
          float m = max(mixS, cells * uCellAmp);

          float hueShift = 0.05 * fbm(uv*2.3 + 3.1);
          vec3 col = rainbow(fract(0.6*m + 0.15*hueShift + 0.03*t));
          return col * m;
        }
      `)
      .replace('#include <emissivemap_fragment>', `
        #include <emissivemap_fragment>
        {
          vec3 wN = normalize(vWorldNormal);
          vec3 w = triWeights(wN);
          vec3 p = vWorldPos;

          vec3 colXY = stripeField(p.xy, uBandAngle);
          vec3 colXZ = stripeField(p.xz, uBandAngle);
          vec3 colYZ = stripeField(p.zy, uBandAngle);

          vec3 c = w.x * colYZ + w.y * colXZ + w.z * colXY;

          totalEmissiveRadiance += c * uBandStrength;
        }
      `);

    material.userData._overlayUniforms = uniforms;
  };
  material.needsUpdate = true;
}

// ---------------- Brownian polygon surface (faces only; no lines/points) ----------------
class BrownianSurface {
  constructor(opts) {
    this.params = { ...opts };

    this.N = 0;
    this.pos = null;   // Float32Array (N*3)
    this.vel = null;   // Float32Array (N*3)
    this.anc = null;   // Float32Array (N*3)
    this.state = null; // Uint8Array (N*N) symmetric

    this.tris = [];
    this._geomMap = null;

    const mat = new THREE.MeshPhysicalMaterial({
      color: 0x151515,
      roughness: 0.26,
      metalness: 0.0,
      envMapIntensity: 1.2,
      iridescence: 1.0,
      iridescenceIOR: 1.3,
      iridescenceThicknessRange: [120, 620],
      flatShading: true,
      side: THREE.DoubleSide
    });
    injectOverlay(mat);

    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
    this.geometry.setAttribute('normal',   new THREE.BufferAttribute(new Float32Array(0), 3));

    this.mesh = new THREE.Mesh(this.geometry, mat);
    this.group = new THREE.Group();
    this.group.add(this.mesh);
    scene.add(this.group);

    this.setCount(this.params.count);
  }

  setCount(n) {
    n = Math.max(3, Math.floor(n));
    this.N = n;
    this.pos = new Float32Array(n * 3);
    this.vel = new Float32Array(n * 3);
    this.anc = new Float32Array(n * 3);
    this.state = new Uint8Array(n * n);

    const R = 1.6;
    for (let i = 0; i < n; i++) {
      const a = randInSphere(R);
      const p = a.clone().add(randInSphere(0.12));
      const o = i * 3;
      this.anc[o+0] = a.x; this.anc[o+1] = a.y; this.anc[o+2] = a.z;
      this.pos[o+0] = p.x; this.pos[o+1] = p.y; this.pos[o+2] = p.z;
      this.vel[o+0] = 0;   this.vel[o+1] = 0;   this.vel[o+2] = 0;
    }
    this._rebuildFromTriangles([]);
  }

  _computeTriangles() {
    const N = this.N, S = this.state;
    const tris = [];
    const neigh = Array.from({ length: N }, () => []);
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        if (S[i * N + j] === 1) { neigh[i].push(j); neigh[j].push(i); }
      }
    }
    for (let i = 0; i < N; i++) {
      const Ni = neigh[i];
      for (let a = 0; a < Ni.length; a++) {
        const j = Ni[a];
        if (j <= i) continue;
        for (let b = a + 1; b < Ni.length; b++) {
          const k = Ni[b];
          if (k <= j) continue;
          if (S[j * N + k] === 1) tris.push([i, j, k]);
        }
      }
    }
    return tris;
  }

  _rebuildFromTriangles(tris) {
    this.tris = tris;
    const triCount = tris.length;

    const positions = new Float32Array(triCount * 3 * 3);
    const normals   = new Float32Array(triCount * 3 * 3);
    this._geomMap   = new Int32Array(triCount * 3);

    for (let t = 0; t < triCount; t++) {
      const base = t * 3;
      const [i, j, k] = tris[t];
      this._geomMap[base + 0] = i;
      this._geomMap[base + 1] = j;
      this._geomMap[base + 2] = k;
    }

    this.geometry.dispose();
    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    this.geometry.setAttribute('normal',   new THREE.BufferAttribute(normals,   3));
    this.geometry.setIndex([...Array(triCount * 3).keys()]);
    this.geometry.computeBoundingSphere();

    this.mesh.geometry = this.geometry;
  }

  _updatePositionsAndNormals() {
    const posAttr = this.geometry.getAttribute('position');
    const nrmAttr = this.geometry.getAttribute('normal');
    if (!posAttr || !nrmAttr) return;

    const P = this.pos;
    const arr = posAttr.array;
    const nArr = nrmAttr.array;

    for (let v = 0; v < this._geomMap.length; v++) {
      const pi = this._geomMap[v] * 3;
      const w  = v * 3;
      arr[w+0] = P[pi+0];
      arr[w+1] = P[pi+1];
      arr[w+2] = P[pi+2];
    }
    posAttr.needsUpdate = true;

    const triCount = this._geomMap.length / 3;
    const v0 = new THREE.Vector3(), v1 = new THREE.Vector3(), v2 = new THREE.Vector3();
    const e1 = new THREE.Vector3(), e2 = new THREE.Vector3(), n  = new THREE.Vector3();
    for (let t = 0; t < triCount; t++) {
      const i0 = (t * 3 + 0) * 3;
      const i1 = (t * 3 + 1) * 3;
      const i2 = (t * 3 + 2) * 3;
      v0.set(arr[i0], arr[i0+1], arr[i0+2]);
      v1.set(arr[i1], arr[i1+1], arr[i1+2]);
      v2.set(arr[i2], arr[i2+1], arr[i2+2]);
      e1.subVectors(v1, v0);
      e2.subVectors(v2, v0);
      n.copy(e1).cross(e2).normalize();
      nArr[i0] = nArr[i1] = nArr[i2] = n.x;
      nArr[i0+1] = nArr[i1+1] = nArr[i2+1] = n.y;
      nArr[i0+2] = nArr[i1+2] = nArr[i2+2] = n.z;
    }
    nrmAttr.needsUpdate = true;
  }

  update(dt, now) {
    // Motion (OU-flavored Brownian around anchors)
    const N = this.N;
    const { amp, freq } = this.params;
    const beta = THREE.MathUtils.clamp(1.2 * freq + 0.2, 0, 12);
    const k    = THREE.MathUtils.clamp(1.6 * freq + 0.3, 0, 14);
    const sigma  = amp;
    const sqrtDt = Math.sqrt(Math.max(1e-6, dt));

    for (let i = 0; i < N; i++) {
      const o = i * 3;
      const nx = randn() * sigma * sqrtDt;
      const ny = randn() * sigma * sqrtDt;
      const nz = randn() * sigma * sqrtDt;

      const px = this.pos[o+0], py = this.pos[o+1], pz = this.pos[o+2];
      let vx = this.vel[o+0], vy = this.vel[o+1], vz = this.vel[o+2];
      const ax = this.anc[o+0], ay = this.anc[o+1], az = this.anc[o+2];

      vx += (-beta * vx + k * (ax - px)) * dt + nx;
      vy += (-beta * vy + k * (ay - py)) * dt + ny;
      vz += (-beta * vz + k * (az - pz)) * dt + nz;

      let nxp = px + vx * dt, nyp = py + vy * dt, nzp = pz + vz * dt;
      const r = Math.hypot(nxp, nyp, nzp), limit = 2.25;
      if (r > limit) {
        const s = limit / r;
        nxp *= s; nyp *= s; nzp *= s;
        vx *= 0.65; vy *= 0.65; vz *= 0.65;
      }

      this.vel[o+0] = vx; this.vel[o+1] = vy; this.vel[o+2] = vz;
      this.pos[o+0] = nxp; this.pos[o+1] = nyp; this.pos[o+2] = nzp;
    }

    // Adjacency with hysteresis
    const onR  = this.params.connectDist;
    const offR = Math.max(this.params.breakDist, onR + 1e-4);
    const S = this.state;

    for (let i = 0; i < N; i++) {
      const ix = i * 3;
      const pix = this.pos[ix], piy = this.pos[ix+1], piz = this.pos[ix+2];
      for (let j = i + 1; j < N; j++) {
        const jx = j * 3;
        const pjx = this.pos[jx], pjy = this.pos[jx+1], pjz = this.pos[jx+2];
        const dx = pjx - pix, dy = pjy - piy, dz = pjz - piz;
        const d = Math.sqrt(dx*dx + dy*dy + dz*dz);
        const idx = i * N + j;
        const on = S[idx] === 1;
        if (!on && d <= onR) { S[i*N+j] = S[j*N+i] = 1; }
        else if (on && d >= offR) { S[i*N+j] = S[j*N+i] = 0; }
      }
    }

    // Triangles = 3‑cliques of S
    const tris = this._computeTriangles();

    // Capacity rebuild if tri count changed
    if (tris.length * 3 !== (this._geomMap ? this._geomMap.length : 0)) {
      this._rebuildFromTriangles(tris);
    } else {
      this.tris = tris;
      for (let t = 0; t < tris.length; t++) {
        const [i, j, k] = tris[t];
        const base = t * 3;
        this._geomMap[base+0] = i;
        this._geomMap[base+1] = j;
        this._geomMap[base+2] = k;
      }
    }

    if (this._geomMap.length > 0) this._updatePositionsAndNormals();

    // Drive shader time + gentle presentation rotation
    const uniforms = this.mesh.material.userData._overlayUniforms;
    if (uniforms) uniforms.uTime.value = now;
    const yaw   = 0.22 * Math.sin(2 * Math.PI * 0.03 * now);
    const pitch = 0.17 * Math.sin(2 * Math.PI * 0.021 * now + 1.2);
    this.group.rotation.set(pitch, yaw, 0, 'YXZ');
  }
}

// ---------- utils ----------
function randInSphere(r=1) {
  const v = new THREE.Vector3();
  do { v.set(Math.random()*2-1, Math.random()*2-1, Math.random()*2-1); }
  while (v.lengthSq() > 1);
  return v.multiplyScalar(r);
}
function randn() { // Gaussian
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ---------- post pipeline (original + added "Tone" pass for trail boosting) ----------
const composer = new EffectComposer(renderer);
const renderPass = new RenderPass(scene, camera);
composer.addPass(renderPass);

// Afterimage (ghosting)
let afterimagePass = new AfterimagePass();
afterimagePass.enabled = false;
afterimagePass.uniforms['damp'].value = 0.96; // long trails = closer to 1.0
composer.addPass(afterimagePass);

// Simple tone map/gain for emphasizing trails (no-op at defaults)
const ToneShader = {
  uniforms: {
    tDiffuse:   { value: null },
    uGain:      { value: 1.0 },
    uGamma:     { value: 1.0 },
    uSaturation:{ value: 1.0 }
  },
  vertexShader: `
    varying vec2 vUv;
    void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }
  `,
  fragmentShader: `
    uniform sampler2D tDiffuse;
    uniform float uGain, uGamma, uSaturation;
    varying vec2 vUv;
    void main() {
      vec4 col = texture2D(tDiffuse, vUv);
      float l = dot(col.rgb, vec3(0.2126, 0.7152, 0.0722));
      col.rgb = mix(vec3(l), col.rgb, uSaturation);
      col.rgb = pow(max(col.rgb, 0.0), vec3(1.0 / max(0.0001, uGamma)));
      col.rgb *= uGain;
      gl_FragColor = col;
    }
  `
};
const tonePass = new ShaderPass(ToneShader);
tonePass.enabled = false;
composer.addPass(tonePass);

// Bloom (unchanged)
const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(container.clientWidth, container.clientHeight),
  0.0,
  0.2,
  0.9
);
bloomPass.enabled = false;
composer.addPass(bloomPass);

// RGB shift (unchanged shader, expanded UI)
const rgbShiftPass = new ShaderPass(RGBShiftShader);
rgbShiftPass.enabled = false;
rgbShiftPass.uniforms['amount'].value = 0.0;
rgbShiftPass.uniforms['angle'].value  = 0.0;
composer.addPass(rgbShiftPass);

// ---------- GUI ----------
const uiHost = document.getElementById('ui');
const gui = new GUI({ title: 'Controls', width: 320 });
uiHost.appendChild(gui.domElement);

const params = {
  // Brownian surface
  count:        36,
  amp:          0.85,
  freq:         0.9,
  connectDist:  0.78,
  breakDist:    0.98,

  // Animation
  play:       true,
  timeScale:  1.0,

  // Bloom (unchanged)
  bloomStrength: 0.0,
  bloomThreshold:0.9,
  bloomRadius:   0.2,

  // Trails (expanded)
  trailEnabled:    false,
  trailPersistence:96.0,  // % of previous frame kept each step (0..99.9). 96% ≈ long trails
  trailHalfLife:   0.0,   // seconds; if >0, overrides persistence each frame via exp(-ln2*dt/halfLife)
  trailBoost:      1.0,   // post gain (emphasize trails)
  trailGamma:      1.0,   // gamma (1=no change)
  trailSaturation: 1.0,   // saturation (1=neutral)
  trailClear:      () => { _clearTrailsNext = true; },

  // RGB offset (expanded)
  rgbAmount:     0.0,     // 0..0.1+
  rgbAngle:      0.0,     // degrees
  rgbAnimate:    false,
  rgbSpinHz:     0.0,     // cycles per second for angle
  rgbPulseAmp:   0.0,     // add on top of base amount
  rgbPulseHz:    0.5,     // pulse frequency

  // View
  exposure: renderer.toneMappingExposure,
  resetCamera: () => {
    camera.position.set(3.2, 1.8, 4.8);
    controls.target.set(0,0,0);
    controls.update();
  }
};

const fSys = gui.addFolder('Brownian Surface');
fSys.add(params, 'count', 4, 200, 1).name('Vertices').onFinishChange(v => surface.setCount(v));
fSys.add(params, 'amp', 0.0, 2.0, 0.001).name('Amplitude').onChange(v => surface.params.amp = v);
fSys.add(params, 'freq', 0.0, 3.0, 0.001).name('Frequency').onChange(v => surface.params.freq = v);
fSys.add(params, 'connectDist', 0.05, 2.0, 0.001).name('Connect (≤)').onChange(v => {
  surface.params.connectDist = v;
  if (params.breakDist < v) { params.breakDist = v; surface.params.breakDist = v; gui.updateDisplay(); }
});
fSys.add(params, 'breakDist', 0.05, 2.0, 0.001).name('Break (≥)').onChange(v => {
  surface.params.breakDist = Math.max(v, params.connectDist);
  params.breakDist = surface.params.breakDist;
  gui.updateDisplay();
});

const fAnim = gui.addFolder('Animation');
fAnim.add(params, 'play').name('Play / Pause');
fAnim.add(params, 'timeScale', 0.0, 3.0, 0.01).name('Time Scale');

const fGlow = gui.addFolder('Glow (Bloom)');
fGlow.add(params, 'bloomStrength', 0.0, 2.5, 0.01).name('Strength').onChange(v => {
  bloomPass.strength = v;
  bloomPass.enabled = v > 0.0;
});
fGlow.add(params, 'bloomThreshold', 0.0, 1.0, 0.001).name('Threshold').onChange(v => bloomPass.threshold = v);
fGlow.add(params, 'bloomRadius', 0.0, 1.0, 0.001).name('Radius').onChange(v => bloomPass.radius = v);

// --- Trails / Ghosting (expanded) ---
const fTrail = gui.addFolder('Trails (Ghosting)');
fTrail.add(params, 'trailEnabled').name('Enable').onChange(v => afterimagePass.enabled = v);
fTrail.add(params, 'trailPersistence', 0.0, 99.9, 0.1).name('Persistence (%)');
fTrail.add(params, 'trailHalfLife', 0.0, 6.0, 0.01).name('Half-life (s)');
fTrail.add(params, 'trailBoost', 0.1, 4.0, 0.01).name('Boost').onChange(updateTonePass);
fTrail.add(params, 'trailGamma', 0.5, 2.5, 0.01).name('Gamma').onChange(updateTonePass);
fTrail.add(params, 'trailSaturation', 0.0, 2.0, 0.01).name('Saturation').onChange(updateTonePass);
fTrail.add(params, 'trailClear').name('Clear Trails');

// --- RGB offset (expanded) ---
const fRGB = gui.addFolder('RGB Offset');
fRGB.add(params, 'rgbAmount', 0.0, 0.10, 0.0001).name('Amount').onChange(v => {
  rgbShiftPass.uniforms['amount'].value = v;
  rgbShiftPass.enabled = (v > 0) || params.rgbAnimate || params.rgbPulseAmp > 0;
});
fRGB.add(params, 'rgbAngle', 0.0, 360.0, 0.1).name('Angle (°)').onChange(v => {
  rgbShiftPass.uniforms['angle'].value = THREE.MathUtils.degToRad(v);
});
fRGB.add(params, 'rgbAnimate').name('Spin Angle');
fRGB.add(params, 'rgbSpinHz', 0.0, 3.0, 0.001).name('Spin (Hz)');
fRGB.add(params, 'rgbPulseAmp', 0.0, 0.10, 0.0001).name('Pulse Amp').onChange(v => {
  rgbShiftPass.enabled = (params.rgbAmount + v) > 0 || params.rgbAnimate;
});
fRGB.add(params, 'rgbPulseHz', 0.0, 5.0, 0.001).name('Pulse (Hz)');

const fView = gui.addFolder('View');
fView.add(params, 'exposure', 0.6, 1.8, 0.01).name('Exposure').onChange(v => renderer.toneMappingExposure = v);
fView.add(params, 'resetCamera').name('Reset Camera');

// Advanced surface overlay (kept)
const surfaceAdv = { angle: 28, angle2: 82, speed: 0.25, freq1: 6.0, freq2: 9.5, strength: 0.52, triScale: 1.15, warp: 0.55, cellAmp: 0.55, cellFreq: 2.75 };
const U = () => surface?.mesh.material.userData._overlayUniforms;
const fSurface = gui.addFolder('Surface (Advanced)');
fSurface.add(surfaceAdv, 'angle', 0, 180, 0.1).name('Stripe Angle (°)').onChange(v => { const u = U(); if (u) u.uBandAngle.value  = THREE.MathUtils.degToRad(v); });
fSurface.add(surfaceAdv, 'angle2', 0, 180, 0.1).name('Stripe2 Angle (°)').onChange(v => { const u = U(); if (u) u.uBandAngle2.value = THREE.MathUtils.degToRad(v); });
fSurface.add(surfaceAdv, 'speed', 0, 2, 0.001).name('Stripe Rot Speed').onChange(v => { const u = U(); if (u) u.uBandSpeed.value = v; });
fSurface.add(surfaceAdv, 'freq1', 1, 20, 0.1).name('Stripe Freq 1').onChange(v => { const u = U(); if (u) u.uBandFreq1.value = v; });
fSurface.add(surfaceAdv, 'freq2', 1, 20, 0.1).name('Stripe Freq 2').onChange(v => { const u = U(); if (u) u.uBandFreq2.value = v; });
fSurface.add(surfaceAdv, 'strength', 0, 1, 0.01).name('Emissive Strength').onChange(v => { const u = U(); if (u) u.uBandStrength.value = v; });
fSurface.add(surfaceAdv, 'triScale', 0.2, 4, 0.01).name('Tri-Planar Scale').onChange(v => { const u = U(); if (u) u.uTriScale.value = v; });
fSurface.add(surfaceAdv, 'warp', 0, 1.5, 0.01).name('Domain Warp').onChange(v => { const u = U(); if (u) u.uWarp.value = v; });
fSurface.add(surfaceAdv, 'cellAmp', 0, 1, 0.01).name('Cellular Mix').onChange(v => { const u = U(); if (u) u.uCellAmp.value = v; });
fSurface.add(surfaceAdv, 'cellFreq', 0.5, 8, 0.01).name('Cellular Freq').onChange(v => { const u = U(); if (u) u.uCellFreq.value = v; });
fSurface.close();

// keep tone pass identity at defaults
function updateTonePass() {
  tonePass.uniforms.uGain.value       = params.trailBoost;
  tonePass.uniforms.uGamma.value      = params.trailGamma;
  tonePass.uniforms.uSaturation.value = params.trailSaturation;
  tonePass.enabled = params.trailEnabled && (
    Math.abs(params.trailBoost - 1.0) > 1e-4 ||
    Math.abs(params.trailGamma - 1.0) > 1e-4 ||
    Math.abs(params.trailSaturation - 1.0) > 1e-4
  );
}
updateTonePass();

// ---------- create surface ----------
const surface = new BrownianSurface({
  count: params.count,
  amp: params.amp,
  freq: params.freq,
  connectDist: params.connectDist,
  breakDist: params.breakDist
});

// ---------- loop ----------
let tPrev = performance.now();
let _clearTrailsNext = false;

function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  let dt = (now - tPrev) / 1000;
  tPrev = now;

  dt = Math.min(dt, 1/30) * params.timeScale;

  if (params.play) {
    surface.params.amp         = params.amp;
    surface.params.freq        = params.freq;
    surface.params.connectDist = params.connectDist;
    surface.params.breakDist   = params.breakDist;
    surface.update(dt, now / 1000);
  }

  // Trails: correct interpretation — higher damp => longer trails.
  // Two ways to set damp:
  //  • trailHalfLife > 0  → damp = exp(-ln2 * dt / halfLife)
  //  • otherwise          → damp = trailPersistence / 100
  if (params.trailEnabled) {
    let dampTarget = (params.trailHalfLife > 0)
      ? Math.exp(-Math.LN2 * Math.max(1e-6, dt) / params.trailHalfLife)
      : THREE.MathUtils.clamp(params.trailPersistence / 100.0, 0.0, 0.9999);

    if (_clearTrailsNext) { dampTarget = 0.0; _clearTrailsNext = false; }
    // clamp just below 1.0 to allow new pixels to seep in
    afterimagePass.uniforms['damp'].value = Math.min(dampTarget, 0.9999);
  }

  // RGB offset extras
  let rgbAngle = THREE.MathUtils.degToRad(params.rgbAngle);
  if (params.rgbAnimate && params.rgbSpinHz > 0) {
    rgbAngle += 2.0 * Math.PI * params.rgbSpinHz * (now / 1000);
  }
  const pulse = (params.rgbPulseAmp > 0)
    ? params.rgbPulseAmp * (0.5 + 0.5 * Math.sin(2.0 * Math.PI * params.rgbPulseHz * (now / 1000)))
    : 0.0;
  const rgbAmt = params.rgbAmount + pulse;
  rgbShiftPass.uniforms['angle'].value  = rgbAngle;
  rgbShiftPass.uniforms['amount'].value = rgbAmt;
  rgbShiftPass.enabled = (rgbAmt > 0) || params.rgbAnimate;

  controls.update();
  composer.render();
}
animate();

// ---------- resize ----------
function onResize() {
  const w = container.clientWidth;
  const h = container.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  composer.setSize(w, h);
  bloomPass.setSize(w, h);
}
window.addEventListener('resize', onResize);
new ResizeObserver(onResize).observe(container);
