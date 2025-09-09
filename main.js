// Iridescent Folding Crystal — single evolving shape
// No bundler required. Uses CDNs for three.js, examples, and lil-gui.

import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';
import { OrbitControls } from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';
import { ConvexGeometry } from 'https://unpkg.com/three@0.160.0/examples/jsm/geometries/ConvexGeometry.js';
import { EffectComposer } from 'https://unpkg.com/three@0.160.0/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'https://unpkg.com/three@0.160.0/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'https://unpkg.com/three@0.160.0/examples/jsm/postprocessing/UnrealBloomPass.js';
import GUI from 'https://cdn.jsdelivr.net/npm/lil-gui@0.19/+esm';

// ---------------------------------------------------------
// Minimal, focused controls to steer look & motion
// ---------------------------------------------------------
const params = {
  speed: 1.0,            // animation rate
  fold: 0.85,            // fold intensity
  noiseAmp: 0.38,        // vertex displacement amplitude
  noiseScale: 1.8,       // spatial frequency of noise
  stripeFreq: 8.0,       // diagonal stripe frequency
  iridescence: 0.85,     // thin-film strength
  hueShift: 0.0,         // global hue offset (-1..+1)
  pastel: 0.65,          // push toward pastel (0..1)
  rotate: true,          // auto-rotate the shard
  reseed: () => reseed(true), // rebuild base shape
};

// ---------------------------------------------------------
// Renderer / Scene / Camera
// ---------------------------------------------------------
const canvas = /** @type {HTMLCanvasElement} */ (document.getElementById('scene'));
const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  alpha: false,
  powerPreference: 'high-performance',
});
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
renderer.setClearColor(0x000000, 1);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
camera.position.set(0, 0, 4.8);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.rotateSpeed = 0.6;

// ---------------------------------------------------------
// Postprocessing (subtle glow to echo the photos)
// ---------------------------------------------------------
let composer, bloom;
function setupComposer() {
  const size = new THREE.Vector2();
  renderer.getSize(size);
  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  bloom = new UnrealBloomPass(size, 0.32, 0.7, 0.0); // mild
  composer.addPass(bloom);
}
setupComposer();

// ---------------------------------------------------------
// Utilities
// ---------------------------------------------------------
function mulberry32(a) {
  return function () {
    let t = (a += 0x6D2B79F5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------
// Shaders (folding + thin-film interference-like palette)
// ---------------------------------------------------------
const vertexShader = /* glsl */ `
  varying vec3 vWorldPos;
  varying vec3 vNormalW;
  varying vec3 vObj;

  uniform float uTime;
  uniform float uFold;
  uniform float uNoiseAmp;
  uniform float uNoiseScale;
  uniform vec3  uFoldN1;
  uniform vec3  uFoldN2;
  uniform float uSeed;

  // 3D simplex noise (iq)
  vec3 mod289(vec3 x){ return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 mod289(vec4 x){ return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 permute(vec4 x){ return mod289(((x*34.0)+1.0)*x); }
  vec4 taylorInvSqrt(vec4 r){ return 1.79284291400159 - 0.85373472095314 * r; }
  float snoise(vec3 v){
    const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
    const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);
    vec3 i  = floor(v + dot(v, C.yyy) );
    vec3 x0 = v - i + dot(i, C.xxx) ;
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min( g.xyz, l.zxy );
    vec3 i2 = max( g.xyz, l.zxy );
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;
    i = mod289(i);
    vec4 p = permute( permute( permute(
              i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
            + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
            + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));
    float n_ = 0.142857142857; // 1.0/7.0
    vec3  ns = n_ * D.wyz - D.xzx;
    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );
    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);
    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;
    vec3 p0 = vec3(a0.xy,h.x);
    vec3 p1 = vec3(a0.zw,h.y);
    vec3 p2 = vec3(a1.xy,h.z);
    vec3 p3 = vec3(a1.zw,h.w);
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1),
                            dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1),
                                  dot(p2,x2), dot(p3,x3) ) );
  }

  // Triangular-wave folding along plane normal `n`
  vec3 foldSpace(vec3 p, vec3 n, float period, float intensity) {
    float d = dot(p, n); // signed distance along n
    float tri = abs(mod(d + period, 2.0*period) - period) - 0.5*period;
    float delta = (tri - d) * intensity;
    return p + n * delta;
  }

  void main() {
    vec3 p = position;
    vObj = p;

    // Animate fold plane directions softly
    float t = uTime * 0.5 + uSeed * 17.0;
    vec3 n1 = normalize(mix(uFoldN1, vec3(sin(t*0.7), cos(t*0.9), sin(t*0.5)), 0.35));
    vec3 n2 = normalize(mix(uFoldN2, vec3(cos(t*0.6), sin(t*0.8), cos(t*0.4)), 0.35));

    // Time-eased fold strength (breathing)
    float foldI = uFold * (0.7 + 0.3 * sin(uTime*0.6 + uSeed));

    // Apply two folding passes with different periods
    p = foldSpace(p, n1, 1.35, foldI);
    p = foldSpace(p, n2, 1.05, foldI * 0.66);

    // Animated simplex noise displacement along geometric normal
    float ns = snoise(p * uNoiseScale + vec3(0.0, t*0.6, t*0.3) + uSeed);
    p += normalize(normal) * (uNoiseAmp * ns);

    vec4 world = modelMatrix * vec4(p, 1.0);
    vWorldPos  = world.xyz;
    vNormalW   = normalize(mat3(modelMatrix) * normal);
    gl_Position = projectionMatrix * viewMatrix * world;
  }
`;

const fragmentShader = /* glsl */ `
  precision highp float;
  varying vec3 vWorldPos;
  varying vec3 vNormalW;
  varying vec3 vObj;

  uniform float uTime;
  uniform float uStripeFreq;
  uniform float uIri;
  uniform float uHueShift;
  uniform float uPastel;
  uniform float uSeed;

  // hsv -> rgb
  vec3 hsv2rgb(vec3 c){
    vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0/3.0, 1.0/3.0)) * 6.0 - 3.0);
    vec3 rgb = clamp(p - 1.0, 0.0, 1.0);
    return c.z * mix(vec3(1.0), rgb, c.y);
  }

  // Approximate thin-film color driven by view angle + phase
  vec3 filmColor(float ndotv, float phase, float pastelBias) {
    float hue = fract(phase);
    float sat = mix(0.2, 0.65, pow(1.0 - ndotv, 0.5));
    float val = mix(0.75, 1.0, 0.6 + 0.4 * (1.0 - ndotv));
    vec3 c = hsv2rgb(vec3(hue, sat, val));
    c = mix(c, vec3(1.0), clamp(pastelBias, 0.0, 1.0)); // pastel lift
    return c;
  }

  // Diagonal interference-like stripes with slight rotation/warp
  float stripes(vec3 p, float freq, float t, float seed) {
    float a = seed * 6.2831853 + t * 0.05;
    mat2 R = mat2(cos(a), -sin(a), sin(a), cos(a));
    vec2 q = R * p.xy + vec2(0.35 * sin(t*0.3), 0.22 * cos(t*0.27));
    float s = sin(q.x * freq + q.y * 0.6 + t * 0.7) * 0.5 + 0.5;
    return smoothstep(0.25, 0.95, s);
  }

  void main() {
    vec3 V = normalize(cameraPosition - vWorldPos);
    vec3 N = normalize(vNormalW);
    float ndv = clamp(dot(N, V), 0.0, 1.0);

    // Phase rolls slowly with time and position to create chromatic sweep
    float phase = uHueShift + uTime * 0.03 + (1.0 - ndv) * 1.8 + dot(vWorldPos, vec3(0.08, 0.02, 0.04));
    vec3 iri = filmColor(ndv, phase, uPastel);
    vec3 base = hsv2rgb(vec3(0.78 + uHueShift * 0.15, 0.25, 0.9)); // lavender/mint base

    float st = stripes(vWorldPos * 0.7 + N * 0.25, uStripeFreq, uTime, uSeed);
    vec3 color = mix(base, iri, 0.65 + 0.35 * uIri);
    color = mix(color * 0.85, color * 1.15, st); // modulate by stripes

    float rim = pow(1.0 - ndv, 3.0);
    color += rim * (0.25 + 0.25 * uIri); // soft rim glow

    color = pow(color, vec3(1.0/2.2));   // gamma-ish
    gl_FragColor = vec4(color, 1.0);
  }
`;

// ---------------------------------------------------------
// Material & Geometry
// ---------------------------------------------------------
function makeMaterial(seed) {
  const uniforms = {
    uTime:        { value: 0 },
    uFold:        { value: params.fold },
    uNoiseAmp:    { value: params.noiseAmp },
    uNoiseScale:  { value: params.noiseScale },
    uStripeFreq:  { value: params.stripeFreq },
    uIri:         { value: params.iridescence },
    uHueShift:    { value: params.hueShift },
    uPastel:      { value: params.pastel },
    uFoldN1:      { value: new THREE.Vector3(1, 0, 0).normalize() },
    uFoldN2:      { value: new THREE.Vector3(0, 1, 0).normalize() },
    uSeed:        { value: seed },
  };
  const mat = new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    uniforms,
    side: THREE.DoubleSide,
    depthTest: true,
    depthWrite: true,
    transparent: false,
  });
  return mat;
}

function makeConvexShard(seed, scale = 1.0) {
  const rng = mulberry32(Math.floor(seed * 1e6));
  const pts = [];
  const NUM = 12 + Math.floor(rng() * 8); // 12..19 points
  for (let i = 0; i < NUM; i++) {
    const r = 0.8 + rng() * 0.65;
    const p = new THREE.Vector3(
      (rng() * 2 - 1) * r,
      (rng() * 2 - 1) * r,
      (rng() * 2 - 1) * r
    );
    // Clamp some axes to create planar facets reminiscent of the reference
    if (rng() < 0.30) p.x = Math.round(p.x * 2.0) * 0.40;
    if (rng() < 0.30) p.y = Math.round(p.y * 2.0) * 0.40;
    if (rng() < 0.30) p.z = Math.round(p.z * 2.0) * 0.40;
    pts.push(p);
  }
  let geom = new ConvexGeometry(pts);
  // Flat shading: drop indexing so each face has its own vertices/normals
  geom = geom.toNonIndexed();
  geom.computeVertexNormals();
  geom.center();
  geom.scale(scale, scale, scale);
  geom.computeBoundingSphere();
  return geom;
}

// ---------------------------------------------------------
// Create the shard (one evolving object)
// ---------------------------------------------------------
let seed = 0.1375;
const material = makeMaterial(seed);
const shard = new THREE.Mesh(makeConvexShard(seed, 1.0), material);
shard.rotation.set(0.32, -0.18, 0.12);
scene.add(shard);

// ---------------------------------------------------------
// GUI (minimal footprint, right side)
// ---------------------------------------------------------
const gui = new GUI({ title: 'Controls', width: 300 });
gui.add(params, 'speed', 0.0, 3.0, 0.01).name('Speed');
gui.add(params, 'fold', 0.0, 1.5, 0.01).name('Fold Intensity').onChange(syncUniforms);
gui.add(params, 'noiseAmp', 0.0, 1.2, 0.01).name('Noise Amplitude').onChange(syncUniforms);
gui.add(params, 'noiseScale', 0.2, 4.0, 0.01).name('Noise Scale').onChange(syncUniforms);
gui.add(params, 'stripeFreq', 1.0, 20.0, 0.1).name('Stripe Frequency').onChange(syncUniforms);
gui.add(params, 'iridescence', 0.0, 1.5, 0.01).name('Iridescence').onChange(syncUniforms);
gui.add(params, 'hueShift', -1.0, 1.0, 0.001).name('Hue Shift').onChange(syncUniforms);
gui.add(params, 'pastel', 0.0, 1.0, 0.001).name('Pastel Bias').onChange(syncUniforms);
gui.add(params, 'rotate').name('Auto Rotate');
gui.add(params, 'reseed').name('Reseed Shape');

// reflect GUI changes into shader uniforms
function syncUniforms() {
  material.uniforms.uFold.value       = params.fold;
  material.uniforms.uNoiseAmp.value   = params.noiseAmp;
  material.uniforms.uNoiseScale.value = params.noiseScale;
  material.uniforms.uStripeFreq.value = params.stripeFreq;
  material.uniforms.uIri.value        = params.iridescence;
  material.uniforms.uHueShift.value   = params.hueShift;
  material.uniforms.uPastel.value     = params.pastel;
}
syncUniforms();

// Rebuild base geometry + randomize fold axes
function reseed(rebuildGeometry = false) {
  seed = Math.random() * 10.0 + 0.01;
  material.uniforms.uSeed.value = seed;
  material.uniforms.uFoldN1.value.set(Math.random()*2-1, Math.random()*2-1, Math.random()*2-1).normalize();
  material.uniforms.uFoldN2.value.set(Math.random()*2-1, Math.random()*2-1, Math.random()*2-1).normalize();
  if (rebuildGeometry) {
    const g = makeConvexShard(seed, 1.0);
    shard.geometry.dispose();
    shard.geometry = g;
  }
}

// ---------------------------------------------------------
// Resize handling (keep room for the control panel)
// ---------------------------------------------------------
function getPanelWidth() {
  const el = document.querySelector('.lil-gui.root');
  return el ? Math.ceil(el.getBoundingClientRect().width) : 300;
}
function resizeRenderer() {
  const panelW = getPanelWidth();
  const width = Math.max(1, window.innerWidth - panelW);
  const height = Math.max(1, window.innerHeight);
  renderer.setSize(width, height, false);
  composer.setSize(width, height);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resizeRenderer);
resizeRenderer();

// ---------------------------------------------------------
// Animation loop — single shape continuously evolving
// ---------------------------------------------------------
const clock = new THREE.Clock();
let time = 0;

function animate() {
  const dt = clock.getDelta();
  time += dt * (0.6 + params.speed * 1.4);

  // update uniforms
  material.uniforms.uTime.value = time;

  // gentle autorotation (you can also orbit with the mouse)
  if (params.rotate) {
    shard.rotation.y += dt * 0.25;
    shard.rotation.x += dt * 0.07;
  }

  controls.update();
  composer.render();
  requestAnimationFrame(animate);
}

requestAnimationFrame(animate);
