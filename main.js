/**
 * Psychedelic Origami — Sequential Folding + Presets + FOLD
 * Look stack unchanged: ACES filmic + Bloom + FXAA + psychedelic iridescent paper shader.
 * BIG CHANGE: folds are now SEQUENTIAL. On the CPU we propagate earlier folds into later
 * crease axes (O(N^2) per frame). The vertex shader then rotates vertices around those
 * already-transformed axes in order. This better matches step-by-step folding.
 *
 * FOLD import: honors edges_assignment (M/V) and edges_foldAngle (degrees, V positive, M negative)
 * per Origami Simulator's Design Tips. See origamisimulator.org and the FOLD spec for details.
 * Sources: Origami Simulator site (simultaneous GPU solver & FOLD semantics), FOLD format repo. 
 * (MIT / permissive licenses; we are not embedding their solver.) 
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
renderer.toneMappingExposure = 1.15;
app.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x050509, 5, 30);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 200);
camera.position.set(0, 1.8, 5.2);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// ---------- Post ----------
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 1.0, 0.6, 0.15);
composer.addPass(bloom);
const fxaa = new ShaderPass(FXAAShader);
fxaa.material.uniforms['resolution'].value.set(1 / window.innerWidth, 1 / window.height || 1 / window.innerHeight);
composer.addPass(fxaa);
composer.addPass(new OutputPass());

// ---------- Paper geometry ----------
const WIDTH = 4.0, HEIGHT = 2.4;
const SEG_X = 180, SEG_Y = 140;      // dense grid for sharper creases
const geo = new THREE.PlaneGeometry(WIDTH, HEIGHT, SEG_X, SEG_Y);
geo.rotateX(-0.25);

// ---------- Utilities ----------
const tmp = {
  v1: new THREE.Vector3(),
  v2: new THREE.Vector3(),
  v3: new THREE.Vector3(),
  q: new THREE.Quaternion()
};
function signedDistance2(p /*Vec3*/, a /*Vec3*/, d /*unit Vec3*/){
  // Use only XY (paper plane) for side test (consistent with base definitions)
  const px = p.x - a.x, py = p.y - a.y;
  // 2D cross z-component: d.x*(p.y-a.y) - d.y*(p.x-a.x)
  return d.x * py - d.y * px;
}
function smooth01(edge0, edge1, x){
  const t = THREE.MathUtils.clamp((x - edge0) / Math.max(1e-9, edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}
function rotatePointAroundLine(p, a, axisUnit, ang){
  // (p - a) rotated by axis, then +a
  tmp.v1.copy(p).sub(a).applyAxisAngle(axisUnit, ang).add(a);
  p.copy(tmp.v1);
}
function rotateVectorAxis(v, axisUnit, ang){
  v.applyAxisAngle(axisUnit, ang);
}

// ---------- Crease store (base definitions) ----------
const MAX_CREASES = 64;
const base = {
  count: 0,
  A: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3()),
  D: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3(1,0,0)), // direction (not necessarily unit)
  amp: new Array(MAX_CREASES).fill(0),   // target amplitude (radians)
  phase: new Array(MAX_CREASES).fill(0), // animation phase
  band: new Array(MAX_CREASES).fill(0),  // smoothing band in world units
  mv:   new Array(MAX_CREASES).fill(1)   // +1 mountain, -1 valley (sign convention for angles)
};
function resetBase(){
  base.count = 0;
  for(let i=0;i<MAX_CREASES;i++){
    base.A[i].set(0,0,0);
    base.D[i].set(1,0,0);
    base.amp[i] = 0;
    base.phase[i] = 0;
    base.band[i] = 0;
    base.mv[i] = 1;
  }
}
function addCrease(Ax, Ay, Dx, Dy, deg=90, mv=1, bandFrac=0.006, phaseRand=true){
  if (base.count >= MAX_CREASES) return;
  const i = base.count++;
  const d = new THREE.Vector2(Dx, Dy).normalize();
  base.A[i].set(Ax, Ay, 0);
  base.D[i].set(d.x, d.y, 0);
  base.amp[i] = THREE.MathUtils.degToRad(deg);
  base.mv[i] = mv >= 0 ? 1 : -1;
  base.band[i] = bandFrac * WIDTH;
  base.phase[i] = phaseRand ? Math.random() * Math.PI * 2 : 0;
}

// ---------- Uniforms ----------
const uniforms = {
  // time/drive
  uTime:       { value: 0 },
  uAnimOn:     { value: 0 },    // 0/1
  uSpeed:      { value: 0.9 },
  uProgress:   { value: 0.7 },  // manual fold progress (0..1)

  // look controls (shader unchanged)
  uSectors:    { value: 10.0 },
  uHueShift:   { value: 0.0 },
  uIridescence:{ value: 0.65 },
  uFilmIOR:    { value: 1.35 },
  uFilmNm:     { value: 360.0 },
  uFiber:      { value: 0.35 },
  uEdgeGlow:   { value: 0.9 },

  // sequential fold data (effective axes + angles)
  uCreaseCount: { value: 0 },
  uAeff:  { value: Array.from({length: MAX_CREASES}, () => new THREE.Vector3()) },
  uDeff:  { value: Array.from({length: MAX_CREASES}, () => new THREE.Vector3(1,0,0)) },
  uAng:   { value: new Float32Array(MAX_CREASES) }, // signed angle per crease
  uBand:  { value: new Float32Array(MAX_CREASES) }  // band half-width per crease
};

// ---------- Shaders ----------
// Vertex: fold sequentially using pre-transformed axes + angles.
// (Fragment shader = same look as before.)
const vs = /* glsl */`
  #define MAX_CREASES ${MAX_CREASES}
  precision highp float;

  uniform int   uCreaseCount;
  uniform vec3  uAeff[MAX_CREASES];
  uniform vec3  uDeff[MAX_CREASES];
  uniform float uAng[MAX_CREASES];
  uniform float uBand[MAX_CREASES];

  varying vec3 vPos;
  varying vec3 vN;
  varying vec3 vLocal;
  varying vec2 vUv;

  vec3 rotateAroundLine(vec3 p, vec3 a, vec3 u, float ang){
    vec3 v = p - a;
    float c = cos(ang), s = sin(ang);
    return a + v*c + cross(u, v)*s + u*dot(u, v)*(1.0 - c);
  }
  vec3 rotateVector(vec3 v, vec3 u, float ang){
    float c = cos(ang), s = sin(ang);
    return v*c + cross(u, v)*s + u*dot(u, v)*(1.0 - c);
  }
  float signedDistanceToLine(vec2 p, vec2 a, vec2 d){
    return d.x*(p.y - a.y) - d.y*(p.x - a.x);
  }
  float sstep(float e0, float e1, float x){
    float t = clamp((x - e0) / max(1e-6, e1 - e0), 0.0, 1.0);
    return t*t*(3.0 - 2.0*t);
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
      float m = sstep(0.0, uBand[i], sd);   // rotate one side with a soft hinge band
      float ang = uAng[i] * m;

      p = rotateAroundLine(p, a, d, ang);
      n = normalize(rotateVector(n, d, ang));
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

  varying vec3 vPos;
  varying vec3 vN;
  varying vec3 vLocal;
  varying vec2 vUv;

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
    // Kaleidoscopic mapping in XZ (unchanged look)
    float theta = atan(vPos.z, vPos.x);
    float r = length(vPos.xz) * 0.55;
    float seg = 2.0*PI / max(3.0, uSectors);
    float a = mod(theta, seg);
    a = abs(a - 0.5*seg);
    vec2 k = vec2(cos(a), sin(a)) * r;

    vec2 q = k*2.0 + vec2(0.15*uTime, -0.1*uTime);
    q += 0.5*vec2(noise(q+13.1), noise(q+71.7));
    float n = noise(q*2.0) * 0.75 + 0.25*noise(q*5.0);
    float hue = fract(n + 0.15*sin(uTime*0.3) + uHueShift);
    vec3 baseCol = hsv2rgb(vec3(hue, 0.9, smoothstep(0.25, 1.0, n)));

    // Paper fibers + grain (unchanged)
    float fiberLines = 0.0;
    {
      float warp = fbm(vLocal.xy*4.0 + vec2(0.2*uTime, -0.1*uTime));
      float l = sin(vLocal.y*420.0 + warp*8.0);
      float widthAA = fwidth(l);
      fiberLines = smoothstep(0.6, 0.6 - widthAA, abs(l));
    }
    float grain = fbm(vLocal.xy*25.0);
    baseCol *= 1.0 + uFiber*(0.06*grain - 0.03) + uFiber*0.08*fiberLines;

    // Crease glow based on distance to nearest effective crease
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

    // Iridescence (unchanged)
    vec3 V = normalize(cameraPosition - vPos);
    vec3 N = normalize(vN);
    float cosT = clamp(dot(N, V), 0.0, 1.0);
    vec3 film = thinFilm(cosT, uFilmIOR, uFilmNm);
    float F = pow(1.0 - cosT, 5.0);
    vec3 col = mix(baseCol, mix(baseCol, film, uIridescence), F);

    col += uEdgeGlow * edge * film * 0.7;

    float vign = smoothstep(1.2, 0.2, length(vUv-0.5)*1.2);
    gl_FragColor = vec4(col*vign, 1.0);
  }
`;

const mat = new THREE.ShaderMaterial({
  vertexShader: vs,
  fragmentShader: fs,
  uniforms,
  side: THREE.DoubleSide,
  extensions: { derivatives: true }
});
const sheet = new THREE.Mesh(geo, mat);
scene.add(sheet);

// background dome
scene.add(new THREE.Mesh(
  new THREE.SphereGeometry(50, 32, 32),
  new THREE.MeshBasicMaterial({ color: 0x070711, side: THREE.BackSide })
));

// ---------- GUI ----------
const gui = new GUI();
const folds = gui.addFolder('Folding');
folds.add(uniforms.uProgress, 'value', 0, 1, 0.001).name('progress');
folds.add(uniforms.uAnimOn, 'value', 0, 1, 1).name('animate 0/1');
folds.add(uniforms.uSpeed, 'value', 0.1, 2.5, 0.01).name('speed');
folds.open();

const looks = gui.addFolder('Look');
looks.add(uniforms.uSectors, 'value', 3, 24, 1).name('kaleidoSectors');
looks.add(uniforms.uHueShift, 'value', 0, 1, 0.001).name('hueShift');
looks.add(bloom, 'strength', 0.0, 2.5, 0.01).name('bloomStrength');
looks.add(bloom, 'radius', 0.0, 1.5, 0.01).name('bloomRadius');
looks.add(uniforms.uIridescence, 'value', 0, 1, 0.001).name('iridescence');
looks.add(uniforms.uFilmIOR, 'value', 1.0, 2.333, 0.001).name('filmIOR');
looks.add(uniforms.uFilmNm, 'value', 100, 800, 1).name('filmThickness(nm)');
looks.add(uniforms.uFiber, 'value', 0, 1, 0.001).name('paperFiber');
looks.add(uniforms.uEdgeGlow, 'value', 0.0, 2.0, 0.01).name('edgeGlow');
looks.open();

// ---------- Sequential effective axes (CPU O(N^2) each frame) ----------
const eff = {
  A: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3()),
  D: Array.from({ length: MAX_CREASES }, () => new THREE.Vector3(1,0,0)),
  ang: new Float32Array(MAX_CREASES)
};

function computeAngles(time){
  // drive_i = progress OR animated sine
  for (let i=0;i<base.count;i++){
    const drive = (uniforms.uAnimOn.value > 0.5)
      ? 0.5 + 0.5 * Math.sin(time * uniforms.uSpeed.value + base.phase[i])
      : uniforms.uProgress.value;
    eff.ang[i] = base.mv[i] * base.amp[i] * drive;
  }
  for (let i=base.count;i<MAX_CREASES;i++) eff.ang[i] = 0;
}

function computeEffectiveAxes(){
  // start from base; copy into eff
  for (let i=0;i<base.count;i++){
    eff.A[i].copy(base.A[i]);
    eff.D[i].copy(base.D[i]).normalize();
  }
  // propagate: for j from 0..count-1, rotate all later axes k>j around j by j's angle,
  // using side test at A_k and a soft band = base.band[j]
  for (let j=0;j<base.count;j++){
    const Aj = eff.A[j];
    const Dj = eff.D[j].clone().normalize(); // axis unit vector
    const angleJ = eff.ang[j];
    if (Math.abs(angleJ) < 1e-7) continue;

    for (let k=j+1; k<base.count; k++){
      const sd = signedDistance2(eff.A[k], Aj, Dj);
      const m = smooth01(0, base.band[j], sd);
      const ang = angleJ * m;
      if (Math.abs(ang) < 1e-7) continue;

      // rotate anchor point and direction of later crease k
      rotatePointAroundLine(eff.A[k], Aj, Dj, ang);
      rotateVectorAxis(eff.D[k], Dj, ang);
      eff.D[k].normalize();
    }
  }
}

function pushEffToUniforms(){
  uniforms.uCreaseCount.value = base.count;
  // replace uniform array OBJECTS so WebGL re-uploads reliably
  uniforms.uAeff.value = eff.A.slice(0, MAX_CREASES).map(v => v.clone());
  uniforms.uDeff.value = eff.D.slice(0, MAX_CREASES).map(v => v.clone());
  uniforms.uAng.value  = Float32Array.from(eff.ang);
  uniforms.uBand.value = Float32Array.from(base.band);
  mat.uniformsNeedUpdate = true;
}

// ---------- Presets ----------
const DEGREES_DEFAULT = 95;
const BAND_DEFAULT = 0.006;
const M = +1, V = -1;

function lineThroughPoints(x1,y1,x2,y2,deg,mv){
  const dx = x2-x1, dy = y2-y1;
  addCrease(x1,y1,dx,dy,deg,mv,BAND_DEFAULT,true);
}

function presetRandom(){
  resetBase();
  const count = 12 + Math.floor(Math.random()*12);
  for (let i=0;i<count;i++){
    const x1 = THREE.MathUtils.lerp(-WIDTH/2, WIDTH/2, Math.random());
    const y1 = THREE.MathUtils.lerp(-HEIGHT/2, HEIGHT/2, Math.random());
    const ang = Math.random()*Math.PI;
    const dx = Math.cos(ang), dy = Math.sin(ang);
    const deg = THREE.MathUtils.lerp(40, 110, Math.random());
    const mv  = Math.random()<0.5 ? M : V;
    addCrease(x1,y1,dx,dy,deg,mv,BAND_DEFAULT,true);
  }
}

function presetPleats(n=18){
  resetBase();
  const s = WIDTH/(n+1);
  for (let i=1;i<=n;i++){
    const x = -WIDTH/2 + i*s;
    const mv = (i%2===0)? M : V;
    addCrease(x, 0, 0, 1, DEGREES_DEFAULT, mv, BAND_DEFAULT, false);
  }
}

function presetWaterbomb(n=12){
  resetBase();
  const s = Math.min(WIDTH, HEIGHT)/(n);
  for (let i=-n; i<=n; i++){
    const c = i*s;
    const x0 = 0, y0 = x0 + c;
    const mv = (i%2===0)? M : V;
    addCrease(x0, y0, 1, 1, DEGREES_DEFAULT, mv, BAND_DEFAULT, false);
  }
  for (let i=-n; i<=n; i++){
    const c = i*s;
    const x0 = 0, y0 = -x0 + c;
    const mv = (i%2===0)? V : M;
    addCrease(x0, y0, 1, -1, DEGREES_DEFAULT, mv, BAND_DEFAULT, false);
  }
}

function presetMiura(cols=10, rows=6, alphaDeg=62){
  resetBase();
  const alpha = THREE.MathUtils.degToRad(alphaDeg);
  const sx = WIDTH/cols, sy = HEIGHT/rows;
  for (let r=0; r<=rows; r++){
    const y = -HEIGHT/2 + r*sy;
    for (let c=0; c<=cols; c++){
      const x = -WIDTH/2 + c*sx;
      { const mv = ((r+c)%2===0)? M : V;
        addCrease(x, y, Math.cos(alpha),  Math.sin(alpha), DEGREES_DEFAULT, mv, BAND_DEFAULT, false); }
      { const mv = ((r+c)%2===0)? V : M;
        addCrease(x, y, Math.cos(alpha), -Math.sin(alpha), DEGREES_DEFAULT, mv, BAND_DEFAULT, false); }
    }
  }
  for (let r=1; r<rows; r++){
    const y = -HEIGHT/2 + r*sy;
    const mv = (r%2===0)? M : V;
    addCrease(0, y, 1, 0, 40, mv, BAND_DEFAULT*0.7, false);
  }
}

function presetBirdBase(){
  resetBase();
  addCrease(0, 0,  1,  1, 110, V, BAND_DEFAULT, false);
  addCrease(0, 0,  1, -1, 110, V, BAND_DEFAULT, false);
  addCrease(0, 0,  1,  0, 110, M, BAND_DEFAULT, false);
  addCrease(0, 0,  0,  1, 110, M, BAND_DEFAULT, false);
  const k = 0.35*WIDTH;
  lineThroughPoints(0, -HEIGHT/2,  k, 0, 90, V);
  lineThroughPoints(0, -HEIGHT/2, -k, 0, 90, V);
  lineThroughPoints(0,  HEIGHT/2,  k, 0, 90, V);
  lineThroughPoints(0,  HEIGHT/2, -k, 0, 90, V);
}

function presetCraneApprox(){
  presetBirdBase();
  const off = 0.18*WIDTH;
  addCrease( off, 0, 1, 0, 70, M, BAND_DEFAULT*0.8, false);
  addCrease(-off, 0, 1, 0, 70, M, BAND_DEFAULT*0.8, false);
  addCrease(0, 0.05*HEIGHT, 1, 0.35, 60, V, BAND_DEFAULT, false);
  addCrease(0,-0.05*HEIGHT, 1,-0.35, 60, V, BAND_DEFAULT, false);
}

function presetRadial(n=18){
  resetBase();
  for (let i=0;i<n;i++){
    const ang = (i/n)*Math.PI;
    const mv = (i%2===0)? M : V;
    addCrease(0,0, Math.cos(ang), Math.sin(ang), DEGREES_DEFAULT, mv, BAND_DEFAULT, false);
  }
}
function presetCross(){
  resetBase();
  addCrease(0,0, 1, 0, 100, M, BAND_DEFAULT, false);
  addCrease(0,0, 0, 1, 100, V, BAND_DEFAULT, false);
}
function presetSingle(){
  resetBase();
  addCrease(0,0, 1, 0.2, 100, M, BAND_DEFAULT, false);
}

// ---------- FOLD import (sequential base) ----------
// Uses: vertices_coords, edges_vertices, edges_assignment, optional edges_foldAngle (degrees).
// Positive foldAngle = valley; negative = mountain (per Origami Simulator Design Tips).
function buildFromFOLD(foldObj){
  try{
    const Vc = foldObj.vertices_coords;
    const Ev = foldObj.edges_vertices;
    const Ea = foldObj.edges_assignment;
    const Ef = foldObj.edges_foldAngle;

    if (!Vc || !Ev || !Ea) throw new Error('Missing required FOLD fields');

    // bbox → scale to our paper
    let minX=+Infinity, minY=+Infinity, maxX=-Infinity, maxY=-Infinity;
    for (const v of Vc){ const x=v[0], y=v[1]; if(x<minX)minX=x; if(x>maxX)maxX=x; if(y<minY)minY=y; if(y>maxY)maxY=y; }
    const w = maxX-minX || 1, h = maxY-minY || 1;
    const s = Math.min(WIDTH / w, HEIGHT / h);
    const cx = (minX+maxX)/2, cy = (minY+maxY)/2;

    resetBase();
    const N = Math.min(Ev.length, MAX_CREASES);
    for (let i=0;i<N;i++){
      const assign = (Ea[i]||'').toUpperCase();
      if (assign !== 'M' && assign !== 'V') continue;
      const [ia, ib] = Ev[i];
      const vA = Vc[ia], vB = Vc[ib];
      if (!vA || !vB) continue;

      const Ax = (vA[0]-cx)*s;
      const Ay = (vA[1]-cy)*s;
      const Dx = (vB[0]-vA[0]);
      const Dy = (vB[1]-vA[1]);

      let mv = assign === 'M' ? M : V;
      let deg = DEGREES_DEFAULT;
      if (Ef && typeof Ef[i] === 'number'){
        deg = Math.min(180, Math.abs(Ef[i]));
        mv = (Ef[i] >= 0) ? V : M; // valley positive, mountain negative
      }
      addCrease(Ax, Ay, Dx, Dy, deg, mv, BAND_DEFAULT, true);
    }
  } catch(err){
    console.error('FOLD parse error:', err);
    alert('Could not parse FOLD file. Need vertices_coords, edges_vertices, edges_assignment (optional edges_foldAngle).');
  }
}

// ---------- Frame update ----------
function updateSequentialData(tSec){
  computeAngles(tSec);
  computeEffectiveAxes();
  pushEffToUniforms();
}

// ---------- UI wiring ----------
const btnSnap = document.getElementById('btnSnap');
const btnAnim = document.getElementById('btnAnim');
const presetSel = document.getElementById('preset');
const btnApply = document.getElementById('btnApply');
const foldFile = document.getElementById('foldFile');

btnSnap.onclick = () => {
  renderer.domElement.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'origami.png';
    a.click();
    URL.revokeObjectURL(url);
  }, 'image/png', 1.0);
};
btnAnim.onclick = () => {
  uniforms.uAnimOn.value = uniforms.uAnimOn.value > 0.5 ? 0.0 : 1.0;
  btnAnim.textContent = 'Animate: ' + (uniforms.uAnimOn.value > 0.5 ? 'On' : 'Off');
};

function applySelectedPreset(){
  const v = presetSel.value;
  if (v === 'random') presetRandom();
  else if (v === 'pleats') presetPleats();
  else if (v === 'waterbomb') presetWaterbomb();
  else if (v === 'miura') presetMiura();
  else if (v === 'bird') presetBirdBase();
  else if (v === 'crane') presetCraneApprox();
  else if (v === 'radial') presetRadial();
  else if (v === 'cross') presetCross();
  else if (v === 'single') presetSingle();

  // tiny camera twitch for feedback
  camera.position.x += (Math.random()-0.5)*0.03;
  camera.position.y += (Math.random()-0.5)*0.03;
}
btnApply.onclick = applySelectedPreset;
presetSel.addEventListener('change', applySelectedPreset);

// load FOLD
foldFile.addEventListener('change', (e) => {
  const file = e.target.files && e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const text = reader.result.toString();
      const obj = JSON.parse(text);
      buildFromFOLD(obj);
    } catch (err){
      console.error(err);
      alert('Invalid JSON in FOLD file.');
    }
  };
  reader.readAsText(file);
});

// ---------- Start with Miura (nice baseline) ----------
presetMiura();

// ---------- Render loop ----------
function tick(t){
  const tSec = t * 0.001;
  uniforms.uTime.value = tSec;
  // recompute sequential axes/angles each frame (needed for animation & interactive progress)
  updateSequentialData(tSec);
  controls.update();
  composer.render();
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

// ---------- Resize ----------
window.addEventListener('resize', () => {
  const w = window.innerWidth, h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  composer.setSize(w, h);
  fxaa.material.uniforms['resolution'].value.set(1 / w, 1 / h);
});
