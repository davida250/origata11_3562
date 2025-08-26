/**
 * Origami — Rigid‑Hinge Folding with Crisp Creases
 * - Global progress ∈ [0..1] maps sequentially across all creases.
 * - Speed slider is a multiplier for *all* auto motion (Play + Look/Auto):
 *     left=×0.2 (5× slower), mid=×1, right=×5 (5× faster).
 * - Presets: Book Fold, Gate Fold, Crane (Demo). No FOLD imports.
 * - More paper‑realistic: geometry is cut along every crease line so triangles
 *   never straddle a crease → crisp hinges, piecewise‑planar panels.
 *
 * Notes:
 * • MIT’s simulator folds *all* creases simultaneously via a GPU solver on a triangulated mesh; we keep a sequential driver but align with the same rigid, piecewise‑planar view (creases as edges). :contentReference[oaicite:3]{index=3}
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { FXAAShader } from 'three/addons/shaders/FXAAShader.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import GUI from 'lil-gui';

// ---------- Renderer / Scene ----------
const app = document.getElementById('app');
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;
app.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x050509, 6, 36);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 200);
camera.position.set(0, 1.8, 5.2);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// ---------- Post ----------
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.35, 0.6, 0.2);
composer.addPass(bloom);
const fxaa = new ShaderPass(FXAAShader);
fxaa.material.uniforms.resolution.value.set(1 / window.innerWidth, 1 / window.innerHeight);
composer.addPass(fxaa);
composer.addPass(new OutputPass());

// ---------- Parameters ----------
const SIZE = 3.0;            // square sheet edge length
const BASE_SEG = 48;         // base grid before cutting along creases

// ---------- Math helpers ----------
const tmp = { v: new THREE.Vector3(), u: new THREE.Vector3() };
function signedDistance2(p /*Vec2|Vec3*/, a /*Vec3*/, d /*unit Vec3*/) {
  const px = p.x - a.x, py = p.y - a.y;
  return d.x * py - d.y * px; // z-component of 2D cross
}
function rotatePointAroundLine(p, a, axisUnit, ang) {
  tmp.v.copy(p).sub(a).applyAxisAngle(axisUnit, ang).add(a);
  p.copy(tmp.v);
}
function rotateVectorAxis(v, axisUnit, ang) { v.applyAxisAngle(axisUnit, ang); }
function clamp(x, lo, hi){ return Math.max(lo, Math.min(hi, x)); }
function clamp01(x){ return clamp(x, 0, 1); }
const Easings = {
  linear: t => t,
  smoothstep: t => t*t*(3-2*t),
  easeInOutCubic: t => (t<0.5? 4*t*t*t : 1 - Math.pow(-2*t+2,3)/2)
};

// ---------- Creases + MASKS ----------
const MAX_CREASES = 24;
const MAX_MASKS_PER = 4;
const VALLEY = +1, MOUNTAIN = -1;

const base = {
  count: 0,
  A: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3()),
  D: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3(1,0,0)),
  amp:  new Array(MAX_CREASES).fill(0),      // |angle| in radians
  sign: new Array(MAX_CREASES).fill(1),      // +1=valley, -1=mountain
  mCount: new Array(MAX_CREASES).fill(0),
  mA:  Array.from({ length: MAX_CREASES }, () => Array.from({ length: MAX_MASKS_PER }, () => new THREE.Vector3())),
  mD:  Array.from({ length: MAX_CREASES }, () => Array.from({ length: MAX_MASKS_PER }, () => new THREE.Vector3(1,0,0))),
};
function resetBase(){
  base.count = 0;
  for (let i=0;i<MAX_CREASES;i++){
    base.A[i].set(0,0,0); base.D[i].set(1,0,0);
    base.amp[i]=0; base.sign[i]=1; base.mCount[i]=0;
    for (let m=0;m<MAX_MASKS_PER;m++){ base.mA[i][m].set(0,0,0); base.mD[i][m].set(1,0,0); }
  }
}
function addCrease({ Ax=0, Ay=0, Dx=1, Dy=0, deg=180, sign=+1, masks=[] }){
  if (base.count >= MAX_CREASES) return;
  const i = base.count++;
  const d = new THREE.Vector2(Dx, Dy).normalize();
  base.A[i].set(Ax, Ay, 0);
  base.D[i].set(d.x, d.y, 0);
  base.amp[i]  = THREE.MathUtils.degToRad(clamp(Math.abs(deg), 0, 180));
  base.sign[i] = sign >= 0 ? +1 : -1;
  base.mCount[i] = Math.min(MAX_MASKS_PER, masks.length);
  for (let m=0;m<base.mCount[i];m++){
    const mk = masks[m];
    const dd = new THREE.Vector2(mk.Dx, mk.Dy).normalize();
    base.mA[i][m].set(mk.Ax, mk.Ay, 0);
    base.mD[i][m].set(dd.x, dd.y, 0);
  }
}

// ---------- Effective axes + masks (sequential propagation) ----------
const eff = {
  A: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3()),
  D: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3(1,0,0)),
  ang: new Float32Array(MAX_CREASES),
  mCount: new Int32Array(MAX_CREASES),
  mA:  Array.from({ length: MAX_CREASES }, () => Array.from({ length: MAX_MASKS_PER }, () => new THREE.Vector3())),
  mD:  Array.from({ length: MAX_CREASES }, () => Array.from({ length: MAX_MASKS_PER }, () => new THREE.Vector3(1,0,0))),
};

// ---------- Drive (global) ----------
const drive = {
  animate:false,       // ping-pong 0↔1
  baseSpeed:0.25,      // progress units per second *before* multiplier
  progress:0.0,        // global progress across all folds
  easing:'smoothstep',
  dir:+1,
  stepCount:1
};
function speedMultiplierFromSlider(x01){
  // map [0..1] → [1/5 .. 5], mid=1 (exponential for symmetric perception)
  return Math.pow(5, (x01 - 0.5) * 2.0);
}

// Map global progress → per-crease angles, *sequentially*.
// If N creases, local progress for crease i is clamp(N*p - i, 0..1).
function computeAngles(){
  const E = Easings[drive.easing] || Easings.linear;
  const N = Math.max(1, base.count);
  const p = clamp01(drive.progress);
  for (let i=0;i<base.count;i++){
    const segT = clamp01(N*p - i); // 0..1 window per crease
    const localT = E(segT);
    eff.ang[i]    = base.sign[i] * base.amp[i] * localT;
    eff.mCount[i] = base.mCount[i];
  }
}

// propagate axes and mask lines by earlier folds (crisp hinge: only + side moves)
function computeEffectiveFrames(){
  // base copy
  for (let i=0;i<base.count;i++){
    eff.A[i].copy(base.A[i]); eff.D[i].copy(base.D[i]).normalize();
    for (let m=0;m<MAX_MASKS_PER;m++){
      eff.mA[i][m].copy(base.mA[i][m]);
      eff.mD[i][m].copy(base.mD[i][m]).normalize();
    }
  }
  // sequentially rotate future creases & their masks
  for (let j=0;j<base.count;j++){
    const Aj = eff.A[j]; const Dj = eff.D[j].clone().normalize();
    const ang = eff.ang[j]; if (Math.abs(ang) < 1e-7) continue;
    for (let k=j+1;k<base.count;k++){
      const sd = signedDistance2(eff.A[k], Aj, Dj);
      if (sd > 0.0){
        rotatePointAroundLine(eff.A[k], Aj, Dj, ang);
        rotateVectorAxis(eff.D[k], Dj, ang); eff.D[k].normalize();
        for (let m=0;m<MAX_MASKS_PER;m++){
          const sdM = signedDistance2(eff.mA[k][m], Aj, Dj);
          if (sdM > 0.0){
            rotatePointAroundLine(eff.mA[k][m], Aj, Dj, ang);
            rotateVectorAxis(eff.mD[k][m], Dj, ang); eff.mD[k][m].normalize();
          }
        }
      }
    }
  }
}

// ---------- Shaders ----------
const vs = /* glsl */`
  #define MAX_CREASES ${MAX_CREASES}
  #define MAX_MASKS_PER ${MAX_MASKS_PER}
  precision highp float;

  uniform int   uCreaseCount;
  uniform vec3  uAeff[MAX_CREASES];
  uniform vec3  uDeff[MAX_CREASES];
  uniform float uAng[MAX_CREASES];

  uniform vec3  uMaskA[MAX_CREASES*MAX_MASKS_PER];
  uniform vec3  uMaskD[MAX_CREASES*MAX_MASKS_PER];
  uniform float uMaskOn[MAX_CREASES*MAX_MASKS_PER];

  varying vec3 vPos; varying vec3 vN; varying vec3 vLocal; varying vec2 vUv;

  vec3 rotateAroundLine(vec3 p, vec3 a, vec3 u, float ang){
    vec3 v = p - a; float c = cos(ang), s = sin(ang);
    return a + v*c + cross(u, v)*s + u*dot(u, v)*(1.0 - c);
  }
  vec3 rotateVector(vec3 v, vec3 u, float ang){
    float c = cos(ang), s = sin(ang);
    return v*c + cross(u, v)*s + u*dot(u, v)*(1.0 - c);
  }
  float signedDistanceToLine(vec2 p, vec2 a, vec2 d){
    return d.x*(p.y - a.y) - d.y*(p.x - a.x);
  }
  bool inMask(int i, vec2 p){
    for (int m=0; m<MAX_MASKS_PER; m++){
      int idx = i*MAX_MASKS_PER + m;
      if (uMaskOn[idx] > 0.5) {
        vec2 a = uMaskA[idx].xy;
        vec2 d = normalize(uMaskD[idx].xy);
        float sd = d.x*(p.y - a.y) - d.y*(p.x - a.x);
        if (sd <= 0.0) return false;
      }
    }
    return true;
  }

  void main(){
    vUv = uv;
    vec3 p = position;
    vec3 n = normalize(normal);

    for (int i=0; i<MAX_CREASES; i++){
      if (i >= uCreaseCount) break;

      vec3 a = uAeff[i];
      vec3 d = normalize(uDeff[i]);

      float sd = signedDistanceToLine(p.xy, a.xy, d.xy);
      if (sd > 0.0 && inMask(i, p.xy)){
        p = rotateAroundLine(p, a, d, uAng[i]);
        n = normalize(rotateVector(n, d, uAng[i]));
      }
    }

    vLocal = p;
    vec4 world = modelMatrix * vec4(p, 1.0);
    vPos = world.xyz;
    vN   = normalize(mat3(modelMatrix) * n);
    gl_Position = projectionMatrix * viewMatrix * world;
  }
`;

const fs = /* glsl */`
  #define MAX_CREASES ${MAX_CREASES}
  precision highp float;
  uniform float uTime;
  uniform float uSectors, uHueShift;
  uniform float uIridescence, uFilmIOR, uFilmNm, uFiber, uEdgeGlow;
  uniform int   uCreaseCount;
  uniform vec3  uAeff[MAX_CREASES];
  uniform vec3  uDeff[MAX_CREASES];
  varying vec3 vPos; varying vec3 vN; varying vec3 vLocal; varying vec2 vUv;

  #define PI 3.14159265359

  float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453123); }
  float noise(vec2 p){
    vec2 i=floor(p), f=fract(p);
    float a=hash(i), b=hash(i+vec2(1,0)), c=hash(i+vec2(0,1)), d=hash(i+vec2(1,1));
    vec2 u=f*f*(3.0-2.0*f);
    return mix(a,b,u.x)+ (c-a)*u.y*(1.0-u.x) + (d-b)*u.x*u.y;
  }
  float fbm(vec2 p){
    float v=0.0, a=0.5;
    for(int i=0;i<5;i++){ v+=a*noise(p); p*=2.0; a*=0.5; }
    return v;
  }
  vec3 hsv2rgb(vec3 c){
    vec3 rgb = clamp(abs(mod(c.x*6.0+vec3(0.,4.,2.), 6.)-3.)-1., 0., 1.);
    return c.z * mix(vec3(1.0), rgb, c.y);
  }
  vec3 thinFilm(float cosTheta, float ior, float nm){
    vec3 lambda = vec3(680.0, 550.0, 440.0);
    vec3 phase  = 4.0 * PI * ior * nm * cosTheta / lambda;
    return 0.5 + 0.5*cos(phase);
  }
  float signedDistanceToLine(vec2 p, vec2 a, vec2 d){
    return d.x*(p.y - a.y) - d.y*(p.x - a.x);
  }

  void main(){
    float theta = atan(vPos.z, vPos.x);
    float r = length(vPos.xz) * 0.55;
    float seg = 2.0*PI / max(3.0, uSectors);
    float a = mod(theta, seg); a = abs(a - 0.5*seg);
    vec2 k = vec2(cos(a), sin(a)) * r;

    vec2 q = k*2.0 + vec2(0.15*uTime, -0.1*uTime);
    q += 0.5*vec2(noise(q+13.1), noise(q+71.7));
    float n = noise(q*2.0) * 0.75 + 0.25*noise(q*5.0);
    float hue = fract(n + 0.15*sin(uTime*0.3) + uHueShift);
    vec3 baseCol = hsv2rgb(vec3(hue, 0.9, smoothstep(0.25, 1.0, n)));

    float fiberLines = 0.0;
    {
      float warp = fbm(vLocal.xy*4.0 + vec2(0.2*uTime, -0.1*uTime));
      float l = sin(vLocal.y*420.0 + warp*8.0);
      float widthAA = fwidth(l);
      fiberLines = smoothstep(0.6, 0.6 - widthAA, abs(l));
    }
    float grain = fbm(vLocal.xy*25.0);
    baseCol *= 1.0 + uFiber*(0.06*grain - 0.03) + uFiber*0.08*fiberLines;

    float minD = 1e9;
    for (int i=0; i<MAX_CREASES; i++){
      if (i >= uCreaseCount) break;
      vec2 a2 = uAeff[i].xy;
      vec2 d2 = normalize(uDeff[i].xy);
      float sd = abs(signedDistanceToLine(vLocal.xy, a2, d2));
      minD = min(minD, sd);
    }
    float aa = fwidth(minD);
    float edge = 1.0 - smoothstep(0.0025, 0.0025 + aa, minD);

    vec3 V = normalize(cameraPosition - vPos);
    vec3 N = normalize(vN);
    float cosT = clamp(dot(N, V), 0.0, 1.0);
    vec3 film = thinFilm(cosT, uFilmIOR, uFilmNm);
    float F = pow(1.0 - cosT, 5.0);
    vec3 col = mix(baseCol, mix(baseCol, film, uIridescence), F);

    col += uEdgeGlow * edge * film * 0.6;

    float vign = smoothstep(1.2, 0.2, length(vUv-0.5)*1.2);
    gl_FragColor = vec4(col*vign, 1.0);
  }
`;

const uniforms = {
  uTime:       { value: 0 },
  uSectors:    { value: 10.0 },
  uHueShift:   { value: 0.0 },
  uIridescence:{ value: 0.65 },
  uFilmIOR:    { value: 1.35 },
  uFilmNm:     { value: 360.0 },
  uFiber:      { value: 0.35 },
  uEdgeGlow:   { value: 0.8 },

  uCreaseCount: { value: 0 },
  uAeff:  { value: Array.from({length: MAX_CREASES}, () => new THREE.Vector3()) },
  uDeff:  { value: Array.from({length: MAX_CREASES}, () => new THREE.Vector3(1,0,0)) },
  uAng:   { value: new Float32Array(MAX_CREASES) },

  uMaskA: { value: Array.from({length: MAX_CREASES*MAX_MASKS_PER}, () => new THREE.Vector3()) },
  uMaskD: { value: Array.from({length: MAX_CREASES*MAX_MASKS_PER}, () => new THREE.Vector3(1,0,0)) },
  uMaskOn:{ value: new Float32Array(MAX_CREASES*MAX_MASKS_PER) }
};

function pushEffToUniforms(){
  uniforms.uCreaseCount.value = base.count;
  uniforms.uAeff.value = eff.A.map(v => v.clone());
  uniforms.uDeff.value = eff.D.map(v => v.clone());
  uniforms.uAng.value  = Float32Array.from(eff.ang);

  const flatA = []; const flatD = []; const on = [];
  for (let i=0;i<base.count;i++){
    for (let m=0;m<MAX_MASKS_PER;m++){
      flatA.push(eff.mA[i][m].clone());
      flatD.push(eff.mD[i][m].clone());
      on.push(m < eff.mCount[i] ? 1 : 0);
    }
  }
  const remain = MAX_CREASES*MAX_MASKS_PER - flatA.length;
  for (let r=0;r<remain;r++){ flatA.push(new THREE.Vector3()); flatD.push(new THREE.Vector3(1,0,0)); on.push(0); }
  uniforms.uMaskA.value = flatA;
  uniforms.uMaskD.value = flatD;
  uniforms.uMaskOn.value = Float32Array.from(on);

  mat.uniformsNeedUpdate = true;
}

// ---------- Mesh (with crease‑aligned cutting) ----------
const mat = new THREE.ShaderMaterial({
  vertexShader: vs, fragmentShader: fs, uniforms,
  side: THREE.DoubleSide, extensions: { derivatives: true }
});
let sheet = new THREE.Mesh(new THREE.BufferGeometry(), mat);
scene.add(sheet);

// Background dome
scene.add(new THREE.Mesh(
  new THREE.SphereGeometry(50, 32, 32),
  new THREE.MeshBasicMaterial({ color: 0x070711, side: THREE.BackSide })
));

// Build a plane subdivided into BASE_SEG with triangles, then cut along crease lines
function buildCutSheetGeometry(size, seg, lines /* array of {A:Vec3, D:Vec3} */){
  // Build base grid as non-indexed triangles
  const tris = [];
  for (let iy=0; iy<seg; iy++){
    const v0 = -0.5 + iy/seg, v1 = -0.5 + (iy+1)/seg;
    const y0 = v0 * size, y1 = v1 * size;
    for (let ix=0; ix<seg; ix++){
      const u0 = -0.5 + ix/seg, u1 = -0.5 + (ix+1)/seg;
      const x0 = u0 * size, x1 = u1 * size;
      // tri A: (x0,y0)-(x1,y0)-(x1,y1)
      tris.push({
        p:[ new THREE.Vector3(x0,y0,0), new THREE.Vector3(x1,y0,0), new THREE.Vector3(x1,y1,0) ],
        uv:[ new THREE.Vector2(u0+0.5, v0+0.5), new THREE.Vector2(u1+0.5, v0+0.5), new THREE.Vector2(u1+0.5, v1+0.5) ]
      });
      // tri B: (x0,y0)-(x1,y1)-(x0,y1)
      tris.push({
        p:[ new THREE.Vector3(x0,y0,0), new THREE.Vector3(x1,y1,0), new THREE.Vector3(x0,y1,0) ],
        uv:[ new THREE.Vector2(u0+0.5, v0+0.5), new THREE.Vector2(u1+0.5, v1+0.5), new THREE.Vector2(u0+0.5, v1+0.5) ]
      });
    }
  }

  const EPS = 1e-9;
  function sd(p, a, d){ return d.x*(p.y - a.y) - d.y*(p.x - a.x); }
  function lerpV2(a,b,t){ return new THREE.Vector2(a.x + (b.x-a.x)*t, a.y + (b.y-a.y)*t); }
  function lerpV3(a,b,t){ return new THREE.Vector3(a.x + (b.x-a.x)*t, a.y + (b.y-a.y)*t, a.z + (b.z-a.z)*t); }
  function intersect(a, b, sa, sb){
    const t = sa / (sa - sb);
    return { p: lerpV3(a, b, t), uv: lerpV2(a.uv, b.uv, t) };
  }
  // Split one triangle by a line (keep both sides)
  function splitTriByLine(tri, a, d){
    const P = tri.p, U = tri.uv;
    const s = [ sd(P[0],a,d), sd(P[1],a,d), sd(P[2],a,d) ];
    const pos = [], neg = [], zer = [];
    for (let i=0;i<3;i++){
      if (s[i] >  EPS) pos.push(i);
      else if (s[i] < -EPS) neg.push(i);
      else zer.push(i);
    }
    if ((pos.length===0 && neg.length===0) || pos.length===0 || neg.length===0){
      // entirely on one side or exactly on the line: no split
      return [tri];
    }
    // Helper to package a tri
    const mk = (a0,a1,a2) => ({ p:[a0.p||a0, a1.p||a1, a2.p||a2], uv:[a0.uv||U[a0], a1.uv||U[a1], a2.uv||U[a2]] });

    if (pos.length===1 && neg.length===2){
      const ip = pos[0], in1 = neg[0], in2 = neg[1];
      const A0 = {p:P[ip], uv:U[ip]}, B1 = {p:P[in1], uv:U[in1]}, B2 = {p:P[in2], uv:U[in2]};
      const I1 = intersect(A0, B1, s[ip], s[in1]);
      const I2 = intersect(A0, B2, s[ip], s[in2]);
      return [
        mk(A0, I1, I2),           // positive side (triangle)
        mk(B1, B2, I2),           // negative side (quad split into two tris)
        mk(B1, I2, I1)
      ];
    }
    if (pos.length===2 && neg.length===1){
      const ip = neg[0], in1 = pos[0], in2 = pos[1];
      const A0 = {p:P[ip], uv:U[ip]}, B1 = {p:P[in1], uv:U[in1]}, B2 = {p:P[in2], uv:U[in2]};
      const I1 = intersect(B1, A0, s[in1], s[ip]);
      const I2 = intersect(B2, A0, s[in2], s[ip]);
      return [
        mk(B1, B2, I2),           // positive side (quad → two tris)
        mk(B1, I2, I1),
        mk(A0, I1, I2)            // negative side (triangle)
      ];
    }
    // degenerate (collinear with vertex): don't split
    return [tri];
  }

  // Cut along each line
  let cur = tris;
  for (const L of lines){
    const a = L.A, d = L.D.clone().normalize();
    const next = [];
    for (const tri of cur){
      const parts = splitTriByLine(tri, a, d);
      for (const t of parts) next.push(t);
    }
    cur = next;
  }

  // Bake to BufferGeometry
  const pos = [], uv = [];
  for (const t of cur){
    for (let k=0;k<3;k++){
      const v = t.p[k], q = t.uv[k];
      pos.push(v.x, v.y, v.z);
      uv.push(q.x, q.y);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('uv',       new THREE.Float32BufferAttribute(uv,   2));
  g.computeVertexNormals();
  return g;
}

function rebuildSheetGeometry(){
  const lines = [];
  for (let i=0;i<base.count;i++){
    lines.push({ A: base.A[i].clone(), D: base.D[i].clone().normalize() });
  }
  const g = buildCutSheetGeometry(SIZE, BASE_SEG, lines);
  if (sheet.geometry) sheet.geometry.dispose();
  sheet.geometry = g;
}

// ---------- GUI: Look controls + Auto oscillators ----------
const gui = new GUI();
const looks  = gui.addFolder('Look');
const autos  = []; // { get,set,min,max,rate,dir,on,ctrl,label,integer }
function registerAuto(ctrl, label, get, set, min, max, { integer=false, rate=null }={}){
  const range = max - min;
  const entry = { get, set, min, max, integer, dir:+1, rate: (rate ?? range/6), ctrl, on:false, label };
  const autoState = { Auto:false };
  const autoCtrl  = looks.add(autoState, 'Auto').name(label + ' Auto');
  autoCtrl.onChange(v => entry.on = !!v);
  autos.push(entry);
  return entry;
}
function updateLookAutos(dt, speedMul){
  const dtEff = dt * speedMul;
  for (const a of autos){
    if (!a.on) continue;
    let v = a.get() + a.dir * a.rate * dtEff;
    if (v >= a.max){ v = a.max; a.dir = -1; }
    if (v <= a.min){ v = a.min; a.dir = +1; }
    if (a.integer) v = Math.round(v);
    a.set(v);
    a.ctrl.updateDisplay(); // live UI movement
  }
}
// Sliders + Auto toggles (every Look slider has an Auto)
const cSectors = looks.add(uniforms.uSectors, 'value', 3, 24, 1).name('kaleidoSectors');
registerAuto(cSectors, 'kaleidoSectors', () => uniforms.uSectors.value, v => uniforms.uSectors.value = v, 3, 24, { integer:true });
const cHue     = looks.add(uniforms.uHueShift, 'value', 0, 1, 0.001).name('hueShift');
registerAuto(cHue, 'hueShift', () => uniforms.uHueShift.value, v => uniforms.uHueShift.value = v, 0, 1, {});
const cIri     = looks.add(uniforms.uIridescence, 'value', 0, 1, 0.001).name('iridescence');
registerAuto(cIri, 'iridescence', () => uniforms.uIridescence.value, v => uniforms.uIridescence.value = v, 0, 1, {});
const cIOR     = looks.add(uniforms.uFilmIOR, 'value', 1.0, 2.333, 0.001).name('filmIOR');
registerAuto(cIOR, 'filmIOR', () => uniforms.uFilmIOR.value, v => uniforms.uFilmIOR.value = v, 1.0, 2.333, {});
const cNm      = looks.add(uniforms.uFilmNm, 'value', 100, 800, 1).name('filmThickness(nm)');
registerAuto(cNm, 'filmThickness(nm)', () => uniforms.uFilmNm.value, v => uniforms.uFilmNm.value = v, 100, 800, {});
const cFiber   = looks.add(uniforms.uFiber, 'value', 0, 1, 0.001).name('paperFiber');
registerAuto(cFiber, 'paperFiber', () => uniforms.uFiber.value, v => uniforms.uFiber.value = v, 0, 1, {});
const cEdge    = looks.add(uniforms.uEdgeGlow, 'value', 0.0, 2.0, 0.01).name('edgeGlow');
registerAuto(cEdge, 'edgeGlow', () => uniforms.uEdgeGlow.value, v => uniforms.uEdgeGlow.value = v, 0.0, 2.0, {});
const cBloomS  = looks.add(bloom, 'strength', 0.0, 2.5, 0.01).name('bloomStrength');
registerAuto(cBloomS, 'bloomStrength', () => bloom.strength, v => (bloom.strength = v), 0.0, 2.5, {});
const cBloomR  = looks.add(bloom, 'radius', 0.0, 1.5, 0.01).name('bloomRadius');
registerAuto(cBloomR, 'bloomRadius', () => bloom.radius, v => (bloom.radius = v), 0.0, 1.5, {});
looks.open();

// ---------- Presets (3 total) ----------
function preset_half_vertical_valley(){
  resetBase();
  addCrease({ Ax:0, Ay:0, Dx:0, Dy:1, deg:180, sign:VALLEY });
  drive.stepCount = base.count;
  rebuildSheetGeometry();
}
function preset_gate_valley(){
  resetBase();
  const x = SIZE*0.25;
  addCrease({ Ax:+x, Ay:0, Dx:0, Dy:1, deg:180, sign:VALLEY });
  addCrease({ Ax:-x, Ay:0, Dx:0, Dy:1, deg:180, sign:VALLEY });
  drive.stepCount = base.count;
  rebuildSheetGeometry();
}
// Crane (Demo) — sequential masked folds (kinematic approximation)
function preset_crane_demo(){
  resetBase();
  const s = SIZE/2;

  // Step 0: diagonal valley (pre-crease / collapse bias)
  addCrease({ Ax:0, Ay:0, Dx:1, Dy:1, deg:180, sign:VALLEY });

  // Step 1: opposite diagonal valley
  addCrease({ Ax:0, Ay:0, Dx:1, Dy:-1, deg:180, sign:VALLEY });

  // Step 2: central vertical valley — masked top half (start neck/tail split)
  addCrease({
    Ax:0, Ay:0, Dx:0, Dy:1, deg:150, sign:VALLEY,
    masks:[
      { Ax:0, Ay:0.00, Dx:0, Dy:1 },
      { Ax:-0.0001, Ay:-0.0001, Dx: 1, Dy: 1 },
      { Ax: 0.0001, Ay:-0.0001, Dx:-1, Dy: 1 },
    ]
  });

  // Step 3: central vertical mountain — masked bottom half
  addCrease({
    Ax:0, Ay:0, Dx:0, Dy:1, deg:150, sign:MOUNTAIN,
    masks:[
      { Ax:0, Ay:0.00, Dx:0, Dy:-1 },
      { Ax:-0.0001, Ay: 0.0001, Dx: 1, Dy:-1 },
      { Ax: 0.0001, Ay: 0.0001, Dx:-1, Dy:-1 },
    ]
  });

  // Step 4: wing right — diagonal valley masked right triangle
  addCrease({
    Ax:0, Ay:0, Dx:1, Dy:-1, deg:120, sign:VALLEY,
    masks:[ { Ax:0, Ay:0, Dx:0, Dy:1 }, { Ax:0, Ay:0, Dx:1, Dy:0 } ]
  });

  // Step 5: wing left — diagonal valley masked left triangle
  addCrease({
    Ax:0, Ay:0, Dx:1, Dy:1, deg:120, sign:VALLEY,
    masks:[ { Ax:0, Ay:0, Dx:0, Dy:1 }, { Ax:0, Ay:0, Dx:-1, Dy:0 } ]
  });

  // Step 6: tail inside-reverse (approx)
  addCrease({
    Ax:0.0, Ay:-0.3*s, Dx:1, Dy:-0.15, deg:140, sign:VALLEY,
    masks:[ { Ax:0, Ay:-0.1, Dx:0, Dy:-1 }, { Ax: 0.0, Ay:0.0, Dx:1, Dy:0 } ]
  });

  // Step 7: neck inside-reverse (approx)
  addCrease({
    Ax:0.0, Ay:-0.3*s, Dx:-1, Dy:-0.15, deg:140, sign:VALLEY,
    masks:[ { Ax:0, Ay:-0.1, Dx:0, Dy:-1 }, { Ax: 0.0, Ay:0.0, Dx:-1, Dy:0 } ]
  });

  // Step 8: head — small mountain
  addCrease({
    Ax:-0.45*s, Ay:-0.65*s, Dx:1, Dy:-0.2, deg:90, sign:MOUNTAIN,
    masks:[ { Ax:-0.1, Ay:-0.2, Dx:-1, Dy:-1 }, { Ax:-0.2, Ay:-0.2, Dx:-1, Dy: 0 } ]
  });

  drive.stepCount = base.count;
  rebuildSheetGeometry();
}

// ---------- DOM controls ----------
const presetSel = document.getElementById('preset');
const btnApply  = document.getElementById('btnApply');
const btnAnim   = document.getElementById('btnAnim');
const btnReset  = document.getElementById('btnReset');
const btnPrev   = document.getElementById('btnStepPrev');
const btnNext   = document.getElementById('btnStepNext');
const btnSnap   = document.getElementById('btnSnap');
const progress  = document.getElementById('progress');
const speed     = document.getElementById('speed');
const easingSel = document.getElementById('easing');
const stepInfo  = document.getElementById('stepInfo');

function setProgress(p){ drive.progress = clamp01(p); progress.value = String(drive.progress.toFixed(3)); }
function stepFromProgress(){ const N = Math.max(1, drive.stepCount); return Math.min(N, Math.floor(N*drive.progress) + 1); }
function updateStepInfo(){ stepInfo.textContent = `Step ${stepFromProgress()}/${drive.stepCount}`; }

btnApply.onclick = () => {
  const v = presetSel.value;

  // presets (no FOLD imports)
  if (v === 'half-vertical-valley') preset_half_vertical_valley();
  else if (v === 'gate-valley')     preset_gate_valley();
  else if (v === 'crane-demo')      preset_crane_demo();

  setProgress(0.0); updateStepInfo();
  camera.position.x += (Math.random()-0.5) * 0.03;
  camera.position.y += (Math.random()-0.5) * 0.03;
};
presetSel.addEventListener('change', () => btnApply.click());

btnAnim.onclick = () => {
  drive.animate = !drive.animate;
  btnAnim.textContent = drive.animate ? 'Pause' : 'Play';
};
btnReset.onclick = () => {
  drive.animate = false; btnAnim.textContent = 'Play';
  setProgress(0.0); updateStepInfo();
};
btnPrev.onclick = () => {
  const N = Math.max(1, drive.stepCount);
  const k = Math.floor(N*drive.progress);
  setProgress((Math.max(0, k-1))/N);
  updateStepInfo();
};
btnNext.onclick = () => {
  const N = Math.max(1, drive.stepCount);
  const k = Math.floor(N*drive.progress);
  setProgress((Math.min(N, k+1))/N);
  updateStepInfo();
};
progress.addEventListener('input', () => { setProgress(parseFloat(progress.value)); updateStepInfo(); });
speed.addEventListener('input',   () => {/* multiplier is read each frame */});
easingSel.addEventListener('change', () => { drive.easing = easingSel.value; });

btnSnap.onclick = () => {
  renderer.domElement.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'origami.png'; a.click();
    URL.revokeObjectURL(url);
  }, 'image/png', 1.0);
};

// ---------- Start ----------
presetSel.value = 'crane-demo';
btnApply.click();
progress.value = String(drive.progress);

// ---------- Per-frame update ----------
function updateFolding(){
  computeAngles();
  computeEffectiveFrames();
  // push to uniforms
  uniforms.uCreaseCount.value = base.count;
  uniforms.uAeff.value = eff.A.map(v => v.clone());
  uniforms.uDeff.value = eff.D.map(v => v.clone());
  uniforms.uAng.value  = Float32Array.from(eff.ang);

  const flatA = []; const flatD = []; const on = [];
  for (let i=0;i<base.count;i++){
    for (let m=0;m<MAX_MASKS_PER;m++){
      flatA.push(eff.mA[i][m].clone());
      flatD.push(eff.mD[i][m].clone());
      on.push(m < eff.mCount[i] ? 1 : 0);
    }
  }
  const remain = MAX_CREASES*MAX_MASKS_PER - flatA.length;
  for (let r=0;r<remain;r++){ flatA.push(new THREE.Vector3()); flatD.push(new THREE.Vector3(1,0,0)); on.push(0); }
  uniforms.uMaskA.value = flatA;
  uniforms.uMaskD.value = flatD;
  uniforms.uMaskOn.value = Float32Array.from(on);

  mat.uniformsNeedUpdate = true;
}
function updateProgressAuto(dt, speedMul){
  if (!drive.animate) return;
  let p = drive.progress + drive.dir * drive.baseSpeed * speedMul * dt;
  if (p >= 1){ p = 1; drive.dir = -1; }
  if (p <= 0){ p = 0; drive.dir = +1; }
  setProgress(p);
}
function tick(t){
  const tSec = t * 0.001;
  uniforms.uTime.value = tSec;

  if (!tick._prev) tick._prev = tSec;
  const dt = clamp(tSec - tick._prev, 0, 0.1); tick._prev = tSec;

  // speed multiplier affects *all* autos
  const speedMul = speedMultiplierFromSlider(parseFloat(speed.value || '0.5'));

  updateProgressAuto(dt, speedMul);
  updateLookAutos(dt, speedMul);

  updateFolding();
  controls.update();
  composer.render();
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

// ---------- Resize ----------
window.addEventListener('resize', () => {
  const w = window.innerWidth, h = window.innerHeight;
  camera.aspect = w / h; camera.updateProjectionMatrix();
  renderer.setSize(w, h); composer.setSize(w, h);
  fxaa.material.uniforms.resolution.value.set(1 / w, 1 / h);
});

// ---------- Conventions ----------
// valley = +°, mountain = −°. (Rigid origami angles are dihedral signs between adjacent panels.) :contentReference[oaicite:4]{index=4}
