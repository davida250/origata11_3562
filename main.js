// Iridescent folding crystal — continuous folds (no separation),
// vibrant thin‑film stripes + real reflections/refraction (MeshPhysicalMaterial).

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ConvexGeometry } from 'three/addons/geometries/ConvexGeometry.js';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js'; // <-- correct import
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import GUI from 'https://cdn.jsdelivr.net/npm/lil-gui@0.19/+esm';

/* ===== Controls (minimal, right-side) ===== */
const P = {
  // Motion / shape
  speed: 1.0,
  fold: 1.10,
  noiseAmp: 0.26,
  noiseScale: 1.7,
  autoRotate: true,

  // Thin‑film overlay (vibrant texture)
  texFreq: 14.0,
  texWarp: 0.55,
  baseNm: 430.0,   // base thickness (nm)
  ampNm: 380.0,    // variation (nm)
  vibrance: 1.6,   // emissive boost
  pastel: 0.22,    // lift toward white (0..1)

  // Physical material (reflections/refraction/iridescence)
  envIntensity: 1.6,
  transmission: 0.82,
  thickness: 0.8,
  ior: 1.36,
  roughness: 0.06,
  clearcoat: 1.0,
  clearcoatRoughness: 0.12,
  iridescence: 1.0,
  iriMinNm: 250.0,
  iriMaxNm: 900.0,

  reseed: () => reseed(true),
};

/* ===== Renderer / Scene ===== */
const canvas = /** @type {HTMLCanvasElement} */ (document.getElementById('scene'));
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
renderer.setClearColor(0x000000, 1);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;
renderer.physicallyCorrectLights = true;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

// Clean, neutral environment for reflections/refraction (no external HDR)
const pmrem = new THREE.PMREMGenerator(renderer);
const envRT = pmrem.fromScene(new RoomEnvironment(renderer), 0.04);
scene.environment = envRT.texture;

/* ===== Camera / Controls ===== */
const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
camera.position.set(0, 0, 4.8);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.08; controls.rotateSpeed = 0.6;

/* ===== Utilities ===== */
function mulberry32(a) {
  return function () {
    let t = (a += 0x6D2B79F5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ===== Geometry: convex "crystal" with shared vertices =====
   - ConvexGeometry starts non-indexed; we merge vertices to index it.
   - We bake a continuous smooth normal attribute `aSmooth` for displacement. */
function makeConvexShard(seed, scale = 1.0) {
  const rng = mulberry32(Math.floor(seed * 1e6));
  const pts = [];
  const NUM = 16 + Math.floor(rng() * 8); // 16..23 pts
  for (let i = 0; i < NUM; i++) {
    const r = 0.8 + rng() * 0.65;
    const p = new THREE.Vector3((rng()*2-1)*r, (rng()*2-1)*r, (rng()*2-1)*r);
    // Snap some axes to suggest planar facets
    if (rng() < 0.32) p.x = Math.round(p.x * 2.0) * 0.42;
    if (rng() < 0.32) p.y = Math.round(p.y * 2.0) * 0.42;
    if (rng() < 0.32) p.z = Math.round(p.z * 2.0) * 0.42;
    pts.push(p);
  }

  // Start as non-indexed convex, then merge to create shared vertices.
  let geom = new ConvexGeometry(pts);
  geom = mergeVertices(geom, 1e-4); // <-- continuous hull (no separation)
  geom.computeVertexNormals();       // smooth normals for continuous displacement
  geom.center();
  geom.scale(scale, scale, scale);
  geom.computeBoundingSphere();

  // Bake a copy of smooth normals for displacement, even if we shade flat.
  const smooth = geom.getAttribute('normal').array.slice();
  geom.setAttribute('aSmooth', new THREE.BufferAttribute(new Float32Array(smooth), 3));
  return geom;
}

/* ===== Material: MeshPhysical + shader patch (safe) =====
   - Displacement happens along `aSmooth` (continuous).
   - Emissive adds thin-film stripes (view dependent).
   - We keep Three's PBR stack for reflections/refraction/iridescence. */
let seed = 0.1375;

const material = new THREE.MeshPhysicalMaterial({
  color: 0xffffff,
  metalness: 0.0,
  roughness: P.roughness,
  clearcoat: P.clearcoat,
  clearcoatRoughness: P.clearcoatRoughness,
  transmission: P.transmission,
  thickness: P.thickness,
  ior: P.ior,
  iridescence: P.iridescence,
  iridescenceIOR: 1.30,
  iridescenceThicknessRange: [P.iriMinNm, P.iriMaxNm],
  envMapIntensity: P.envIntensity,
  side: THREE.DoubleSide,
  flatShading: true // faceted look while geometry remains continuous
});

// Inject folding + thin‑film in a robust way
material.onBeforeCompile = (shader) => {
  // GPU uniforms
  Object.assign(shader.uniforms, {
    uTime:       { value: 0.0 },
    uSeed:       { value: seed },
    uFold:       { value: P.fold },
    uNoiseAmp:   { value: P.noiseAmp },
    uNoiseScale: { value: P.noiseScale },
    uFoldN1:     { value: new THREE.Vector3(1,0,0) },
    uFoldN2:     { value: new THREE.Vector3(0,1,0) },
    uTexFreq:    { value: P.texFreq },
    uTexWarp:    { value: P.texWarp },
    uBaseNm:     { value: P.baseNm },
    uAmpNm:      { value: P.ampNm },
    uVibrance:   { value: P.vibrance },
    uPastel:     { value: P.pastel },
  });

  // Varyings/attributes/uniforms in vertex
  shader.vertexShader =
  `
    attribute vec3 aSmooth;
    varying vec3 vWorldPos;
    varying vec3 vNormalW;

    uniform float uTime;
    uniform float uSeed;
    uniform float uFold;
    uniform float uNoiseAmp;
    uniform float uNoiseScale;
    uniform vec3  uFoldN1;
    uniform vec3  uFoldN2;
  ` + shader.vertexShader;

  // Helpers + main() hook
  shader.vertexShader = shader.vertexShader.replace(
    'void main() {',
    `
    // Simplex noise (iq)
    vec3 mod289(vec3 x){ return x - floor(x * (1.0/289.0)) * 289.0; }
    vec4 mod289(vec4 x){ return x - floor(x * (1.0/289.0)) * 289.0; }
    vec4 permute(vec4 x){ return mod289(((x*34.0)+1.0)*x); }
    vec4 taylorInvSqrt(vec4 r){ return 1.79284291400159 - 0.85373472095314 * r; }
    float snoise(vec3 v){
      const vec2 C = vec2(1.0/6.0, 1.0/3.0);
      const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
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

    // Triangular-wave folding mapping along plane normal n
    vec3 foldSpace(vec3 p, vec3 n, float period, float intensity) {
      float d = dot(p, n);
      float tri = abs(mod(d + period, 2.0*period) - period) - 0.5*period;
      float delta = (tri - d) * intensity;
      return p + n * delta;
    }

    void main() {
    `
  );

  // Apply folding & continuous displacement; write varyings
  shader.vertexShader = shader.vertexShader.replace(
    '#include <begin_vertex>',
    `
      #include <begin_vertex>
      {
        vec3 p = transformed;
        float t = uTime * 0.5 + uSeed * 17.0;

        // Animated fold axes
        vec3 dyn1 = normalize(vec3(sin(t*0.7), cos(t*0.9), sin(t*0.5)));
        vec3 dyn2 = normalize(vec3(cos(t*0.6), sin(t*0.8), cos(t*0.4)));
        vec3 n1 = normalize(uFoldN1 * 0.65 + dyn1 * 0.35);
        vec3 n2 = normalize(uFoldN2 * 0.65 + dyn2 * 0.35);

        float foldI = uFold * (0.7 + 0.3 * sin(uTime*0.6 + uSeed));
        p = foldSpace(p, n1, 1.35, foldI);
        p = foldSpace(p, n2, 1.05, foldI * 0.66);

        // Continuous displacement along baked smooth normal (no cracks)
        vec3 smoothN = normalize(aSmooth);
        float ns = snoise(p * uNoiseScale + vec3(0.0, t*0.6, t*0.3) + uSeed);
        p += smoothN * (uNoiseAmp * ns);

        transformed = p;

        // World-space varyings
        vWorldPos = (modelMatrix * vec4(transformed, 1.0)).xyz;
        vNormalW  = normalize(mat3(modelMatrix) * smoothN);
      }
    `
  );

  // Fragment prelude: uniforms + varyings + helpers (no macro defines)
  shader.fragmentShader =
  `
    uniform float uTime, uSeed;
    uniform float uTexFreq, uTexWarp, uBaseNm, uAmpNm, uVibrance, uPastel;
    varying vec3 vWorldPos;
    varying vec3 vNormalW;

    vec2 rot2(vec2 p, float a){ float c=cos(a), s=sin(a); return mat2(c,-s,s,c)*p; }

    float stripeField(vec3 p, float freq, float warp, float t, float seed) {
      float a = seed * 6.2831853 + t * 0.15;
      vec2 q = rot2(p.xy + vec2(0.35*sin(t*0.33), 0.22*cos(t*0.27)), a);
      float s = sin(q.x * freq + 0.65*q.y + 0.7*sin(q.y*0.5 + t*0.5));
      float w = sin((q.x+q.y) * freq * 0.35 + 1.7*sin(q.x*0.8 + t*0.2));
      return 0.5 + 0.5 * mix(s, w, clamp(warp, 0.0, 1.0));
    }

    vec3 thinFilm(float ndv, float d_nm, float n2) {
      float n1 = 1.0, n3 = 1.50;
      float sin2 = max(0.0, 1.0 - ndv*ndv);
      float cos2 = sqrt(max(0.0, 1.0 - sin2/(n2*n2)));
      float R12 = pow((n1 - n2)/(n1 + n2), 2.0);
      float R23 = pow((n2 - n3)/(n2 + n3), 2.0);
      float A = 2.0 * sqrt(R12 * R23);
      float PI4 = 12.56637061435917295385;
      float lR = 650.0, lG = 510.0, lB = 440.0;
      float phiR = PI4*n2*d_nm*cos2 / lR;
      float phiG = PI4*n2*d_nm*cos2 / lG;
      float phiB = PI4*n2*d_nm*cos2 / lB;
      float rR = clamp(R12 + R23 + A * cos(phiR), 0.0, 1.0);
      float rG = clamp(R12 + R23 + A * cos(phiG), 0.0, 1.0);
      float rB = clamp(R12 + R23 + A * cos(phiB), 0.0, 1.0);
      return vec3(rR, rG, rB);
    }
  ` + shader.fragmentShader;

  // Add emissive contribution (vibrant stripes)
  shader.fragmentShader = shader.fragmentShader.replace(
    '#include <emissivemap_fragment>',
    `
      #include <emissivemap_fragment>
      {
        vec3 N = normalize(vNormalW);
        vec3 V = normalize(cameraPosition - vWorldPos);
        float ndv = clamp(dot(N, V), 0.0, 1.0);

        float f = stripeField(vWorldPos * 0.7 + N * 0.25, uTexFreq, uTexWarp, uTime, uSeed);
        float d_nm = uBaseNm + uAmpNm * (f - 0.5) * 2.0;

        vec3 filmRGB = thinFilm(ndv, d_nm, 1.35);
        filmRGB = mix(filmRGB, vec3(1.0), clamp(uPastel, 0.0, 1.0));

        totalEmissiveRadiance += filmRGB * uVibrance;
      }
    `
  );

  material.userData.shader = shader;
};
material.needsUpdate = true;

/* ===== Mesh ===== */
const shard = new THREE.Mesh(makeConvexShard(seed, 1.0), material);
shard.rotation.set(0.32, -0.18, 0.12);
scene.add(shard);

/* ===== GUI ===== */
const gui = new GUI({ title: 'Controls', width: 300 });
const fTex = gui.addFolder('Texture');
fTex.add(P, 'texFreq', 3.0, 28.0, 0.1).name('Frequency').onChange(syncUniforms);
fTex.add(P, 'texWarp', 0.0, 1.0, 0.01).name('Warp').onChange(syncUniforms);
fTex.add(P, 'baseNm', 250.0, 600.0, 1.0).name('Base (nm)').onChange(syncUniforms);
fTex.add(P, 'ampNm', 0.0, 500.0, 1.0).name('Variation (nm)').onChange(syncUniforms);
fTex.add(P, 'vibrance', 0.0, 3.0, 0.01).name('Vibrance').onChange(syncUniforms);
fTex.add(P, 'pastel', 0.0, 1.0, 0.01).name('Pastel').onChange(syncUniforms);

const fShape = gui.addFolder('Shape');
fShape.add(P, 'fold', 0.0, 1.8, 0.01).name('Fold').onChange(syncUniforms);
fShape.add(P, 'noiseAmp', 0.0, 1.2, 0.01).name('Noise Amp').onChange(syncUniforms);
fShape.add(P, 'noiseScale', 0.2, 4.0, 0.01).name('Noise Scale').onChange(syncUniforms);

const fMat = gui.addFolder('Material');
fMat.add(P, 'envIntensity', 0.0, 3.0, 0.01).name('Env Intensity').onChange(()=> material.envMapIntensity = P.envIntensity);
fMat.add(P, 'transmission', 0.0, 1.0, 0.01).name('Transmission').onChange(()=> material.transmission = P.transmission);
fMat.add(P, 'thickness', 0.0, 2.0, 0.01).name('Thickness').onChange(()=> material.thickness = P.thickness);
fMat.add(P, 'ior', 1.0, 2.0, 0.001).name('IOR').onChange(()=> material.ior = P.ior);
fMat.add(P, 'roughness', 0.0, 1.0, 0.001).name('Roughness').onChange(()=> material.roughness = P.roughness);
fMat.add(P, 'clearcoat', 0.0, 1.0, 0.001).name('Clearcoat').onChange(()=> material.clearcoat = P.clearcoat);
fMat.add(P, 'clearcoatRoughness', 0.0, 1.0, 0.001).name('Clearcoat Rough.').onChange(()=> material.clearcoatRoughness = P.clearcoatRoughness);
fMat.add(P, 'iridescence', 0.0, 1.0, 0.001).name('PBR Iridescence').onChange(()=> material.iridescence = P.iridescence);
fMat.add(P, 'iriMinNm', 0.0, 1000.0, 1.0).name('Iri Min (nm)').onChange(()=> material.iridescenceThicknessRange = [P.iriMinNm, P.iriMaxNm]);
fMat.add(P, 'iriMaxNm', 0.0, 2000.0, 1.0).name('Iri Max (nm)').onChange(()=> material.iridescenceThicknessRange = [P.iriMinNm, P.iriMaxNm]);

const fMotion = gui.addFolder('Motion');
fMotion.add(P, 'speed', 0.0, 3.0, 0.01).name('Speed');
fMotion.add(P, 'autoRotate').name('Auto Rotate');

gui.add(P, 'reseed').name('Reseed Shape');

function syncUniforms() {
  const sh = material.userData.shader; if (!sh) return;
  sh.uniforms.uFold.value       = P.fold;
  sh.uniforms.uNoiseAmp.value   = P.noiseAmp;
  sh.uniforms.uNoiseScale.value = P.noiseScale;

  sh.uniforms.uTexFreq.value    = P.texFreq;
  sh.uniforms.uTexWarp.value    = P.texWarp;
  sh.uniforms.uBaseNm.value     = P.baseNm;
  sh.uniforms.uAmpNm.value      = P.ampNm;
  sh.uniforms.uVibrance.value   = P.vibrance;
  sh.uniforms.uPastel.value     = P.pastel;
}

function reseed(rebuildGeometry = false) {
  seed = Math.random() * 10 + 0.01;
  const sh = material.userData.shader;
  if (sh) {
    sh.uniforms.uSeed.value = seed;
    sh.uniforms.uFoldN1.value.set(Math.random()*2-1, Math.random()*2-1, Math.random()*2-1).normalize();
    sh.uniforms.uFoldN2.value.set(Math.random()*2-1, Math.random()*2-1, Math.random()*2-1).normalize();
  }
  if (rebuildGeometry) {
    const g = makeConvexShard(seed, 1.0);
    shard.geometry.dispose();
    shard.geometry = g;
  }
}

/* ===== Resize ===== */
function panelWidth() {
  const el = document.querySelector('.lil-gui.root');
  return el ? Math.ceil(el.getBoundingClientRect().width) : 300;
}
function onResize() {
  const w = Math.max(1, window.innerWidth - panelWidth());
  const h = Math.max(1, window.innerHeight);
  renderer.setSize(w, h, false);
  renderer.domElement.style.width = w + 'px';
  renderer.domElement.style.height = h + 'px';
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', onResize);
onResize();

/* ===== Animate ===== */
const clock = new THREE.Clock();
let time = 0;
function animate() {
  const dt = clock.getDelta();
  time += dt * (0.6 + P.speed * 1.4);

  const sh = material.userData.shader;
  if (sh) sh.uniforms.uTime.value = time;

  if (P.autoRotate) {
    shard.rotation.y += dt * 0.25;
    shard.rotation.x += dt * 0.07;
  }

  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);
