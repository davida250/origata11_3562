// Iridescent Folding Crystal — single evolving shape with thin-film interference.
// Copy into the same folder as index.html and open index.html.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ConvexGeometry } from 'three/addons/geometries/ConvexGeometry.js';
import GUI from 'https://cdn.jsdelivr.net/npm/lil-gui@0.19/+esm';

// ---------------- Parameters (compact, right-side sliders) ----------------
const params = {
  // Motion / shape
  speed: 1.0,          // animation rate
  autoRotate: true,    // auto-rotate the shard
  fold: 0.95,          // fold intensity
  noiseAmp: 0.35,      // vertex displacement amplitude
  noiseScale: 1.8,     // spatial frequency of displacement

  // Texture (thin-film)
  texFreq: 10.0,       // stripe/field frequency
  texWarp: 0.45,       // warping of the texture field
  baseNm: 380.0,       // base film thickness in nm (colors come from 250..700 nm range)
  ampNm: 260.0,        // variation amplitude of thickness (nm)
  intensity: 1.0,      // interference contrast
  pastel: 0.28,        // push toward pastel (0..1)

  reseed: () => reseed(true), // rebuild the base shape
};

// ---------------- Renderer / Scene / Camera ----------------
const canvas = /** @type {HTMLCanvasElement} */ (document.getElementById('scene'));
const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  alpha: false,
  powerPreference: 'high-performance',
});
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
renderer.setClearColor(0x000000, 1);
// Keep color handling simple and predictable; let renderer do sRGB conversion.
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.NoToneMapping;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
camera.position.set(0, 0, 4.8);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.rotateSpeed = 0.6;

// ---------------- Utilities ----------------
function mulberry32(a) {
  return function () {
    let t = (a += 0x6D2B79F5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------- Shaders ----------------
const vertexShader = /* glsl */`
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
    const vec2  C = vec2(1.0/6.0, 1.0/3.0);
    const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);
    vec3 i  = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;
    i = mod289(i);
    vec4 p = permute( permute( permute(
              i.z + vec4(0.0, i1.z, i2.z, 1.0))
            + i.y + vec4(0.0, i1.y, i2.y, 1.0))
            + i.x + vec4(0.0, i1.x, i2.x, 1.0));
    float n_ = 0.142857142857;
    vec3  ns = n_ * D.wyz - D.xzx;
    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_);
    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);
    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
    vec3 p0 = vec3(a0.xy,h.x);
    vec3 p1 = vec3(a0.zw,h.y);
    vec3 p2 = vec3(a1.xy,h.z);
    vec3 p3 = vec3(a1.zw,h.w);
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
    p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
  }

  // Triangular-wave folding along plane normal n
  vec3 foldSpace(vec3 p, vec3 n, float period, float intensity) {
    float d = dot(p, n);
    float tri = abs(mod(d + period, 2.0*period) - period) - 0.5*period;
    float delta = (tri - d) * intensity;
    return p + n * delta;
  }

  void main() {
    vec3 p = position;
    vObj = p;

    float t = uTime * 0.5 + uSeed * 17.0;
    vec3 n1 = normalize(mix(uFoldN1, vec3(sin(t*0.7), cos(t*0.9), sin(t*0.5)), 0.35));
    vec3 n2 = normalize(mix(uFoldN2, vec3(cos(t*0.6), sin(t*0.8), cos(t*0.4)), 0.35));

    float foldI = uFold * (0.7 + 0.3 * sin(uTime*0.6 + uSeed));
    p = foldSpace(p, n1, 1.35, foldI);
    p = foldSpace(p, n2, 1.05, foldI * 0.66);

    float ns = snoise(p * uNoiseScale + vec3(0.0, t*0.6, t*0.3) + uSeed);
    p += normalize(normal) * (uNoiseAmp * ns);

    vec4 world = modelMatrix * vec4(p, 1.0);
    vWorldPos  = world.xyz;
    vNormalW   = normalize(mat3(modelMatrix) * normal);
    gl_Position = projectionMatrix * viewMatrix * world;
  }
`;

const fragmentShader = /* glsl */`
  precision highp float;
  varying vec3 vWorldPos;
  varying vec3 vNormalW;
  varying vec3 vObj;

  uniform float uTime;
  uniform float uSeed;

  // Thin-film controls
  uniform float uTexFreq;
  uniform float uTexWarp;
  uniform float uBaseNm;
  uniform float uAmpNm;
  uniform float uIntensity;
  uniform float uPastel;

  // Rotate a 2D vector
  vec2 rot2(vec2 p, float a){
    float c = cos(a), s = sin(a);
    return mat2(c,-s,s,c)*p;
  }

  // Compute simple domain-warped stripe field in object/world space
  float stripeField(vec3 p, float freq, float warp, float t, float seed) {
    float a = seed * 6.2831853 + t * 0.15;
    vec2 q = rot2(p.xy + vec2(0.35*sin(t*0.33), 0.22*cos(t*0.27)), a);
    float s = sin(q.x * freq + 0.65*q.y + 0.7*sin(q.y*0.5 + t*0.5));
    float w = sin((q.x+q.y) * freq * 0.35 + 1.7*sin(q.x*0.8 + t*0.2));
    return 0.5 + 0.5 * mix(s, w, clamp(warp, 0.0, 1.0));
  }

  // Thin-film reflectance approximation (unpolarized, 3 wavelengths)
  // n1: air (1.0), n2: film (~1.35), n3: substrate (~1.5)
  vec3 thinFilm(float ndv, float d_nm, float n2) {
    float n1 = 1.0;
    float n3 = 1.50;

    // Snell: cos(theta2) inside the film
    float sin2 = max(0.0, 1.0 - ndv*ndv);
    float cos2 = sqrt(max(0.0, 1.0 - sin2/(n2*n2)));

    // Fresnel reflectances at interfaces (amplitude squared, unpolarized approx)
    float R12 = pow((n1 - n2)/(n1 + n2), 2.0);
    float R23 = pow((n2 - n3)/(n2 + n3), 2.0);
    float A = 2.0 * sqrt(R12 * R23);

    // Wavelengths in nm
    float lR = 650.0;
    float lG = 510.0;
    float lB = 440.0;

    float phiR = 4.0*3.14159265*n2*d_nm*cos2 / lR;
    float phiG = 4.0*3.14159265*n2*d_nm*cos2 / lG;
    float phiB = 4.0*3.14159265*n2*d_nm*cos2 / lB;

    float rR = clamp(R12 + R23 + A * cos(phiR), 0.0, 1.0);
    float rG = clamp(R12 + R23 + A * cos(phiG), 0.0, 1.0);
    float rB = clamp(R12 + R23 + A * cos(phiB), 0.0, 1.0);

    return vec3(rR, rG, rB);
  }

  void main() {
    vec3 V = normalize(cameraPosition - vWorldPos);
    vec3 N = normalize(vNormalW);
    float ndv = clamp(dot(N, V), 0.0, 1.0);

    // Domain-warped stripe mask that drives thickness (d in nm)
    float f = stripeField(vWorldPos * 0.7 + N * 0.25, uTexFreq, uTexWarp, uTime, uSeed);
    // Thickness sweeps across stripes; base around 300–450nm gives rich pastels
    float d_nm = uBaseNm + uAmpNm * (f - 0.5) * 2.0;

    // Film index slightly above 1 (e.g., soap-like / plastic)
    vec3 filmRGB = thinFilm(ndv, d_nm, 1.35);

    // Contrast / intensity and pastel bias towards white
    vec3 color = mix(vec3(0.5), filmRGB, clamp(uIntensity, 0.0, 2.0));
    color = mix(color, vec3(1.0), clamp(uPastel, 0.0, 1.0));

    // Rim brightening for crystalline edges (soft, not bloom)
    float rim = pow(1.0 - ndv, 3.5);
    color += rim * 0.25 * uIntensity;

    // Output linear; renderer will convert to sRGB
    gl_FragColor = vec4(color, 1.0);
  }
`;

// ---------------- Material & Geometry ----------------
function makeMaterial(seed) {
  const uniforms = {
    uTime:       { value: 0 },
    uSeed:       { value: seed },

    // folding / displacement
    uFold:       { value: params.fold },
    uNoiseAmp:   { value: params.noiseAmp },
    uNoiseScale: { value: params.noiseScale },
    uFoldN1:     { value: new THREE.Vector3(1, 0, 0).normalize() },
    uFoldN2:     { value: new THREE.Vector3(0, 1, 0).normalize() },

    // thin-film texture
    uTexFreq:    { value: params.texFreq },
    uTexWarp:    { value: params.texWarp },
    uBaseNm:     { value: params.baseNm },
    uAmpNm:      { value: params.ampNm },
    uIntensity:  { value: params.intensity },
    uPastel:     { value: params.pastel },
  };

  return new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    uniforms,
    side: THREE.DoubleSide,
    depthTest: true,
    depthWrite: true,
    transparent: false
  });
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
  if (geom.index) geom = geom.toNonIndexed(); // flat-shaded facets
  geom.computeVertexNormals();
  geom.center();
  geom.scale(scale, scale, scale);
  geom.computeBoundingSphere();
  return geom;
}

// ---------------- Create the shard (one evolving object) ----------------
let seed = 0.1375;
const material = makeMaterial(seed);
const shard = new THREE.Mesh(makeConvexShard(seed, 1.0), material);
shard.rotation.set(0.32, -0.18, 0.12);
scene.add(shard);

// ---------------- GUI (organized & minimal) ----------------
const gui = new GUI({ title: 'Controls', width: 300 });
const fTex = gui.addFolder('Texture');
fTex.add(params, 'texFreq', 3.0, 24.0, 0.1).name('Frequency').onChange(()=>syncUniforms());
fTex.add(params, 'texWarp', 0.0, 1.0, 0.01).name('Warp').onChange(()=>syncUniforms());
fTex.add(params, 'baseNm', 250.0, 550.0, 1.0).name('Base (nm)').onChange(()=>syncUniforms());
fTex.add(params, 'ampNm', 0.0, 350.0, 1.0).name('Variation (nm)').onChange(()=>syncUniforms());
fTex.add(params, 'intensity', 0.0, 2.0, 0.01).name('Intensity').onChange(()=>syncUniforms());
fTex.add(params, 'pastel', 0.0, 1.0, 0.01).name('Pastel').onChange(()=>syncUniforms());

const fShape = gui.addFolder('Shape');
fShape.add(params, 'fold', 0.0, 1.6, 0.01).name('Fold').onChange(()=>syncUniforms());
fShape.add(params, 'noiseAmp', 0.0, 1.2, 0.01).name('Noise Amp').onChange(()=>syncUniforms());
fShape.add(params, 'noiseScale', 0.2, 4.0, 0.01).name('Noise Scale').onChange(()=>syncUniforms());

const fMotion = gui.addFolder('Motion');
fMotion.add(params, 'speed', 0.0, 3.0, 0.01).name('Speed');
fMotion.add(params, 'autoRotate').name('Auto Rotate');

gui.add(params, 'reseed').name('Reseed Shape');

function syncUniforms() {
  material.uniforms.uFold.value       = params.fold;
  material.uniforms.uNoiseAmp.value   = params.noiseAmp;
  material.uniforms.uNoiseScale.value = params.noiseScale;

  material.uniforms.uTexFreq.value    = params.texFreq;
  material.uniforms.uTexWarp.value    = params.texWarp;
  material.uniforms.uBaseNm.value     = params.baseNm;
  material.uniforms.uAmpNm.value      = params.ampNm;
  material.uniforms.uIntensity.value  = params.intensity;
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

// ---------------- Resize handling (respect right-side panel) ----------------
function getPanelWidth() {
  const el = document.querySelector('.lil-gui.root');
  return el ? Math.ceil(el.getBoundingClientRect().width) : 300;
}
function resizeRenderer() {
  const panelW = getPanelWidth();
  const width = Math.max(1, window.innerWidth - panelW);
  const height = Math.max(1, window.innerHeight);
  renderer.setSize(width, height, false);
  renderer.domElement.style.width = width + 'px';
  renderer.domElement.style.height = height + 'px';
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resizeRenderer);
resizeRenderer();

// ---------------- Animation loop — single shape, evolving texture ----------------
const clock = new THREE.Clock();
let time = 0;
function animate() {
  const dt = clock.getDelta();
  time += dt * (0.6 + params.speed * 1.4);
  material.uniforms.uTime.value = time;

  if (params.autoRotate) {
    shard.rotation.y += dt * 0.25;
    shard.rotation.x += dt * 0.07;
  }

  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);
