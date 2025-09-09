import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { ConvexGeometry } from "three/addons/geometries/ConvexGeometry.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { AfterimagePass } from "three/addons/postprocessing/AfterimagePass.js";
import GUI from "lil-gui";

/* -------------------------------------------------------------
   DOM READY (prevents 'appendChild of null')
------------------------------------------------------------- */
await new Promise((resolve) => {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", resolve, { once: true });
  } else {
    resolve();
  }
});

/* -------------------------------------------------------------
   HOST CONTAINER (fallback if missing)
------------------------------------------------------------- */
const host = (() => {
  const el = document.getElementById("sceneHost");
  if (el) return el;
  const div = document.createElement("div");
  div.id = "sceneHost";
  div.style.position = "fixed";
  div.style.left = "0"; div.style.top = "0"; div.style.bottom = "0"; div.style.right = "300px";
  document.body.appendChild(div);
  return div;
})();

/* -------------------------------------------------------------
   RENDERER / SCENE / CAMERA / CONTROLS
------------------------------------------------------------- */
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(host.clientWidth, host.clientHeight);
renderer.setClearColor(0x000000, 1);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;
renderer.outputColorSpace = THREE.SRGBColorSpace;
host.appendChild(renderer.domElement);

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(45, host.clientWidth / host.clientHeight, 0.1, 100);
camera.position.set(2.7, 1.8, 3.6);
camera.lookAt(0, 0, 0);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.minDistance = 1.2;
controls.maxDistance = 8;

/* -------------------------------------------------------------
   LIGHTS
------------------------------------------------------------- */
const hemi = new THREE.HemisphereLight(0xffffff, 0x223344, 0.4);
scene.add(hemi);
const dir = new THREE.DirectionalLight(0xffffff, 1.0);
dir.position.set(3.5, 5.0, 2.0);
scene.add(dir);

/* -------------------------------------------------------------
   POST-PRO (Afterimage time-delay trails)
   (EffectComposer / RenderPass / AfterimagePass) — docs/examples
------------------------------------------------------------- */
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const afterPass = new AfterimagePass(0.80); // 'damp' uniform; lower = longer trails
composer.addPass(afterPass);

/* -------------------------------------------------------------
   PARAMETERS + GUI
------------------------------------------------------------- */
const params = {
  seed: 7,
  faces: 26,

  foldAmplitudeDeg: 38, // how far each facet hinges (degrees)
  speed: 0.55,          // fold tempo (shader speed)
  phaseSpread: 1.25,    // how de-synchronized faces are
  inflate: 0.02,        // push faces out along normals to avoid z-fight

  stripeDensity: 3.0,   // bands per world unit
  stripeWarp: 1.35,     // domain warp strength
  stripeSharp: 1.6,     // band sharpness

  iridescence: 0.95,    // fresnel-based hue shift strength
  filmNmMin: 180.0,     // thin-film thickness range (nm)
  filmNmMax: 420.0,
  rimPower: 2.0,        // chromatic rim power

  trails: 0.82,         // AfterimagePass 'damp' (0 off .. ~0.98)
  rotateY: 0.12,        // idle spin

  regenerate: () => rebuildCrystal(),
};

const gui = new GUI({ container: document.getElementById("gui") });
gui.add(params, "seed", 1, 9999, 1).name("Random Seed").onChange(() => params.regenerate());
gui.add(params, "faces", 8, 64, 1).name("Face Count").onChange(() => params.regenerate());
gui.add(params, "foldAmplitudeDeg", 0, 80, 1).name("Amplitude (°)");
gui.add(params, "speed", 0.05, 2.0, 0.01).name("Speed");
gui.add(params, "phaseSpread", 0.0, 3.0, 0.01).name("Phase Spread");
gui.add(params, "inflate", 0.0, 0.08, 0.001).name("Face Offset");
gui.add(params, "trails", 0.0, 0.98, 0.01).name("Trails (damp)");
gui.add(params, "stripeDensity", 0.5, 7.0, 0.05).name("Stripe Density");
gui.add(params, "stripeWarp", 0.0, 3.0, 0.01).name("Stripe Warp");
gui.add(params, "stripeSharp", 0.8, 3.0, 0.01).name("Stripe Sharpness");
gui.add(params, "iridescence", 0.0, 1.5, 0.01).name("Iridescence");
gui.add(params, "filmNmMin", 100.0, 350.0, 1).name("Film nm Min");
gui.add(params, "filmNmMax", 250.0, 700.0, 1).name("Film nm Max");
gui.add(params, "rimPower", 0.5, 5.0, 0.05).name("Rim Power");
gui.add(params, "rotateY", 0.0, 0.8, 0.01).name("Idle Spin");
gui.add(params, "regenerate").name("Regenerate");

/* -------------------------------------------------------------
   UTILS
------------------------------------------------------------- */
function mulberry32(seed) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function pickInUnitSphere(rand) {
  // Marsaglia method
  let x, y, z, s;
  do {
    x = rand() * 2 - 1;
    y = rand() * 2 - 1;
    s = x * x + y * y;
  } while (s >= 1 || s === 0);
  const factor = Math.sqrt(1 - s);
  z = 2 * s - 1;
  return new THREE.Vector3(2 * x * factor, 2 * y * factor, z).normalize();
}

/* -------------------------------------------------------------
   SHADERS
   - Vertex: per-face hinge rotation around a pivot + inflate
   - Fragment: tri-planar stripes + thin-film (Zucconi spectral)
------------------------------------------------------------- */
const crystalVertex = /* glsl */ `
precision highp float;

uniform float uTime;
uniform float uSpeed;
uniform float uAmp;        // radians
uniform float uInflate;    // along normal
uniform float uPhaseSpread;

attribute vec3 aCenter;    // per-face pivot
attribute vec3 aAxis;      // per-face hinge axis
attribute float aPhase;    // per-face phase seed
attribute float aRand;     // per-face random [0,1]

varying vec3 vWorldPos;
varying vec3 vWorldNormal;
varying vec3 vObjectPos;
varying float vFaceRnd;

// Rodrigues rotation around normalized axis
vec3 rotateAroundAxis(vec3 p, vec3 axis, float ang){
  float s = sin(ang), c = cos(ang);
  return p * c + cross(axis, p) * s + axis * dot(axis, p) * (1.0 - c);
}

void main(){
  vec3 axis = normalize(aAxis);
  float t = uTime * uSpeed + aPhase * uPhaseSpread;
  // pause near extremes, snap through the middle (origami-ish)
  float w = sin(t);
  float snap = smoothstep(0.0, 1.0, abs(w));
  float theta = uAmp * w * (0.55 + 0.45 * snap);

  // rotate vertex around its face centroid
  vec3 local = position - aCenter;
  local = rotateAroundAxis(local, axis, theta);

  // inflate along rotated normal (approximate by rotating the attribute normal)
  vec3 n = normal;
  n = rotateAroundAxis(n, axis, theta);
  local += n * uInflate * (0.5 + 0.8 * (aRand - 0.5));

  vec3 newPos = aCenter + local;

  vObjectPos = newPos;
  vFaceRnd = aRand;

  vec3 worldPos = (modelMatrix * vec4(newPos, 1.0)).xyz;
  vWorldPos = worldPos;
  vec3 worldNormal = normalize(mat3(modelMatrix) * n);
  vWorldNormal = worldNormal;

  gl_Position = projectionMatrix * viewMatrix * vec4(worldPos, 1.0);
}
`;

const crystalFragment = /* glsl */ `
precision highp float;

uniform float uTime;
uniform float uStripeDensity;
uniform float uStripeWarp;
uniform float uStripeSharp;
uniform float uIridescence;
uniform float uFilmNmMin;
uniform float uFilmNmMax;
uniform float uRimPower;
uniform vec3  uLightDir;
uniform vec3  uLightColor;
uniform vec3  uAmbient;

varying vec3 vWorldPos;
varying vec3 vWorldNormal;
varying vec3 vObjectPos;
varying float vFaceRnd;

// --- Tri-planar helpers ---
vec3 triplanarWeights(vec3 n) {
  n = abs(n);
  n = max(n, 1e-5);
  return n / (n.x + n.y + n.z);
}

// Stripe field with domain warp; returns [0,1]
float stripes2D(vec2 uv, float density, float sharp, float time, float rot, float warp) {
  // rotate uv
  float s = sin(rot), c = cos(rot);
  uv = mat2(c,-s,s,c) * uv;

  // domain warp
  if (warp > 0.0) {
    float ww = sin(uv.x * 1.2 + uv.y * 1.7 + time*0.15);
    uv += vec2(ww, sin(uv.x*0.7 + time*0.11)) * (warp * 0.6);
  }

  float v = sin(uv.x * density + time*0.4);
  v = 0.5 + 0.5 * v;
  v = pow(v, sharp); // sharpen to get crisp bands
  return v;
}

// spectral_zucconi6: wavelength (nm) -> RGB
// Based on Alan Zucconi's "Improving the Rainbow"
vec3 bump3y(vec3 x, vec3 y) {
  vec3 y2 = 1.0 - x * x;
  y2 = max(y2 - y, 0.0);
  return y2;
}
vec3 spectral_zucconi6(float w) {
  float x = clamp((w - 400.0)/300.0, 0.0, 1.0);
  vec3 c1 = vec3(3.54585104, 2.93225262, 2.41593945);
  vec3 x1 = vec3(0.69549072, 0.49228336, 0.27699880);
  vec3 y1 = vec3(0.02312639, 0.15225084, 0.52607955);
  vec3 c2 = vec3(3.90307140, 3.21182957, 3.96587128);
  vec3 x2 = vec3(0.97901834, 0.50353000, 0.06436224);
  vec3 y2 = vec3(0.04555408, 0.07691347, 0.12848300);
  return clamp(bump3y(c1 * (x - x1), y1) + bump3y(c2 * (x - x2), y2), 0.0, 1.0);
}

void main() {
  vec3 N = normalize(vWorldNormal);
  vec3 V = normalize(cameraPosition - vWorldPos);
  vec3 L = normalize(uLightDir);

  // Tri-planar procedural stripes in world space
  vec3 w = triplanarWeights(N);
  float density = uStripeDensity;
  float sharp   = uStripeSharp;
  float time    = uTime;
  float warp    = uStripeWarp;

  vec3 wp = vWorldPos;

  float sx = stripes2D(wp.yz, density, sharp, time, 0.35, warp);
  float sy = stripes2D(wp.xz, density, sharp, time, 1.55, warp);
  float sz = stripes2D(wp.xy, density, sharp, time, -0.65, warp);
  float sBlend = (sx*w.x + sy*w.y + sz*w.z) / (w.x + w.y + w.z);

  // Fresnel for angle-dependent effects
  float ndv = clamp(dot(N, V), 0.0, 1.0);
  float fres = pow(1.0 - ndv, 2.0);

  // Thin-film thickness (nm), modulated by stripes and view angle
  float filmNm = mix(uFilmNmMin, uFilmNmMax, clamp(sBlend, 0.0, 1.0));
  filmNm += uIridescence * (100.0 * fres); // angle dependence

  vec3 filmColor = spectral_zucconi6(filmNm);

  // Lighting
  float NdotL = clamp(dot(N, L), 0.0, 1.0);
  vec3 diffuse = filmColor * (0.35 + 0.65 * NdotL);

  vec3 H = normalize(L + V);
  float spec = pow(max(dot(N, H), 0.0), 80.0) * (0.25 + 0.75 * fres);
  vec3 specCol = mix(vec3(1.0), filmColor, 0.6) * spec;

  float rim = pow(1.0 - ndv, uRimPower);
  vec3 rimCol = vec3(0.9, 0.4, 1.2) * rim * 0.25;

  vec3 color = uAmbient * filmColor + uLightColor * diffuse + specCol + rimCol;
  color = pow(color, vec3(0.9)); // mild artistic gamma tweak

  gl_FragColor = vec4(color, 1.0);
}
`;

/* -------------------------------------------------------------
   CRYSTAL BUILD
   - Random convex poly via ConvexGeometry
   - Non-indexed geometry to allow *per-face* attributes
------------------------------------------------------------- */
let crystal = null;
let crystalMat = null;

function buildCrystalGeometry(seed, targetFaceCount = 24) {
  const rand = mulberry32(seed);

  // Random points on/near a sphere, jitter radius for irregular facets
  const pts = [];
  const OUTER = 1.0;
  for (let i = 0; i < targetFaceCount; i++) {
    const dir = pickInUnitSphere(rand);
    const r = OUTER * (0.6 + 0.5 * rand()); // 0.6..1.1
    pts.push(new THREE.Vector3().copy(dir).multiplyScalar(r));
  }

  const base = new ConvexGeometry(pts); // convex hull (docs) 
  const geom = base.toNonIndexed();     // tri-soup for per-face attributes
  geom.computeVertexNormals();

  const pos = geom.attributes.position.array;
  const vcount = pos.length / 3;
  const faceCount = vcount / 3;

  const centers = new Float32Array(vcount * 3);
  const axes    = new Float32Array(vcount * 3);
  const phases  = new Float32Array(vcount);
  const rands   = new Float32Array(vcount);

  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  const e0 = new THREE.Vector3(), e1 = new THREE.Vector3(), norm = new THREE.Vector3();
  const centroid = new THREE.Vector3();

  for (let f = 0; f < faceCount; f++) {
    const i0 = f * 9, i1 = i0 + 3, i2 = i0 + 6;

    a.set(pos[i0], pos[i0+1], pos[i0+2]);
    b.set(pos[i1], pos[i1+1], pos[i1+2]);
    c.set(pos[i2], pos[i2+1], pos[i2+2]);

    centroid.copy(a).add(b).add(c).multiplyScalar(1/3);
    e0.copy(b).sub(a);
    e1.copy(c).sub(a);
    norm.copy(e0).cross(e1).normalize();

    // Hinge axis (choose an in-plane direction, lightly randomized)
    const pick = rand();
    const axis = new THREE.Vector3();
    if (pick < 0.34) axis.copy(e0).normalize();
    else if (pick < 0.67) axis.copy(e1).normalize();
    else axis.copy(e0).add(e1).normalize();

    const phase = rand() * Math.PI * 2.0; // independent phase
    const rnd   = rand();

    for (let k = 0; k < 3; k++) {
      const vIndex = f * 3 + k;

      centers[vIndex*3+0] = centroid.x;
      centers[vIndex*3+1] = centroid.y;
      centers[vIndex*3+2] = centroid.z;

      axes[vIndex*3+0] = axis.x;
      axes[vIndex*3+1] = axis.y;
      axes[vIndex*3+2] = axis.z;

      phases[vIndex] = phase;
      rands[vIndex]  = rnd;
    }
  }

  geom.setAttribute("aCenter", new THREE.BufferAttribute(centers, 3));
  geom.setAttribute("aAxis",   new THREE.BufferAttribute(axes, 3));
  geom.setAttribute("aPhase",  new THREE.BufferAttribute(phases, 1));
  geom.setAttribute("aRand",   new THREE.BufferAttribute(rands, 1));

  // Normalize scale
  geom.computeBoundingSphere();
  const s = 1.2 / (geom.boundingSphere?.radius || 1.0);
  geom.scale(s, s, s);

  return geom;
}

function buildCrystalMaterial() {
  const uniforms = {
    uTime:          { value: 0 },
    uSpeed:         { value: params.speed },
    uAmp:           { value: THREE.MathUtils.degToRad(params.foldAmplitudeDeg) },
    uInflate:       { value: params.inflate },
    uPhaseSpread:   { value: params.phaseSpread },

    uStripeDensity: { value: params.stripeDensity },
    uStripeWarp:    { value: params.stripeWarp },
    uStripeSharp:   { value: params.stripeSharp },

    uIridescence:   { value: params.iridescence },
    uFilmNmMin:     { value: params.filmNmMin },
    uFilmNmMax:     { value: params.filmNmMax },
    uRimPower:      { value: params.rimPower },

    uLightDir:      { value: new THREE.Vector3(0.6, 1.0, 0.25).normalize() },
    uLightColor:    { value: new THREE.Color(1.0, 0.95, 1.0) },
    uAmbient:       { value: new THREE.Color(0.04, 0.06, 0.08) },
  };

  const mat = new THREE.ShaderMaterial({
    vertexShader: crystalVertex,
    fragmentShader: crystalFragment,
    uniforms,
    transparent: false,
    side: THREE.DoubleSide, // show backfaces when folding
    dithering: true
  });

  return mat;
}

function rebuildCrystal() {
  if (crystal) {
    scene.remove(crystal);
    crystal.geometry.dispose();
    crystal.material.dispose();
  }
  const geom = buildCrystalGeometry(params.seed, params.faces);
  crystalMat = buildCrystalMaterial();
  crystal = new THREE.Mesh(geom, crystalMat);
  scene.add(crystal);
}
rebuildCrystal();

/* -------------------------------------------------------------
   RESIZE
------------------------------------------------------------- */
function resize() {
  const w = host.clientWidth;
  const h = host.clientHeight;
  renderer.setSize(w, h);
  composer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
const ro = new ResizeObserver(resize);
ro.observe(host);

/* -------------------------------------------------------------
   ANIMATION LOOP
------------------------------------------------------------- */
const clock = new THREE.Clock();
function animate() {
  const dt = clock.getDelta();
  const t = clock.elapsedTime;

  if (crystalMat) {
    const u = crystalMat.uniforms;
    u.uTime.value = t;
    u.uSpeed.value = params.speed;
    u.uAmp.value = THREE.MathUtils.degToRad(params.foldAmplitudeDeg);
    u.uInflate.value = params.inflate;
    u.uPhaseSpread.value = params.phaseSpread;

    u.uStripeDensity.value = params.stripeDensity;
    u.uStripeWarp.value = params.stripeWarp;
    u.uStripeSharp.value = params.stripeSharp;

    u.uIridescence.value = params.iridescence;
    u.uFilmNmMin.value = params.filmNmMin;
    u.uFilmNmMax.value = params.filmNmMax;
    u.uRimPower.value = params.rimPower;
  }

  afterPass.uniforms["damp"].value = params.trails; // AfterimagePass control (examples)

  if (crystal) crystal.rotation.y += params.rotateY * dt;

  controls.update();
  composer.render();
  requestAnimationFrame(animate);
}
animate();
