// main.js
// A 4x4 contact sheet of one animated folding shape with time offsets.
// Clean, robust ES module with minimal external deps (CDNs).

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/controls/OrbitControls.js';
import { RoomEnvironment } from 'https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/environments/RoomEnvironment.js';
import GUI from 'https://cdn.jsdelivr.net/npm/lil-gui@0.20/+esm';

// ---------- constants ----------
const COLS = 4;
const ROWS = 4;
const PANELS = COLS * ROWS;

// Phase offsets spanning one "cycle" of the fold animation
const phaseOffsets = new Array(PANELS).fill(0).map((_, i) => i / PANELS);

// ---------- renderer / scene / camera ----------
const container = document.getElementById('app');
const renderer = new THREE.WebGLRenderer({
  antialias: true,
  alpha: false,
  powerPreference: 'high-performance',
});
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
// Environment for realistic reflections/refractions (PMREM)
const pmrem = new THREE.PMREMGenerator(renderer);
const env = new RoomEnvironment(renderer);
scene.environment = pmrem.fromScene(env).texture;
scene.background = new THREE.Color(0x000000);

const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
camera.position.set(2.8, 1.9, 3.4);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 1.6;
controls.maxDistance = 7.0;

// ---------- geometry & material ----------
const geometry = new THREE.IcosahedronGeometry(1.0, 5); // dense enough to displace smoothly

// Physical surface w/ thin-film iridescence & transmission
const material = new THREE.MeshPhysicalMaterial({
  color: 0xffffff,
  metalness: 0.0,
  roughness: 0.12,
  transmission: 0.92,          // glassy/transmissive
  thickness: 0.6,               // needed for transmission attenuation
  attenuationColor: new THREE.Color(0xa9ffc9),
  attenuationDistance: 1.25,
  clearcoat: 1.0,
  clearcoatRoughness: 0.1,
  ior: 1.45,
  iridescence: 1.0,             // thin-film color shift
  iridescenceIOR: 1.3,
  iridescenceThicknessRange: [180, 520],
  envMapIntensity: 1.3,
});

// Inject a **folding** displacement into the vertex stage while keeping PBR shading.
// Uses triangle waves along several oblique planes + power shaping for sharp creases.
// References: triangle-wave shaping & space folding ideas.
let shaderRef = null;
material.onBeforeCompile = (shader) => {
  shader.uniforms.uTime = { value: 0.0 };
  shader.uniforms.uAmp = { value: 0.45 };        // displacement amplitude
  shader.uniforms.uFreq = { value: 2.3 };        // waves per unit along planes
  shader.uniforms.uSharp = { value: 2.0 };       // crease sharpness (power)
  shader.uniforms.uRot = { value: 0.35 };        // slow internal rotation in shader
  shader.uniforms.uPlanes = { value: [
    new THREE.Vector3( 1.0,  0.32,  0.05),
    new THREE.Vector3(-0.25, 1.00,  0.48),
    new THREE.Vector3( 0.07, 0.26,  1.00),
  ]};
  shader.uniforms.uMix = { value: 0.66 };        // how much we combine plane contributions

  // Add helpers + uniforms to the shader
  shader.vertexShader = shader.vertexShader
    .replace(
      '#include <common>',
      `
      #include <common>
      uniform float uTime, uAmp, uFreq, uSharp, uRot, uMix;
      uniform vec3 uPlanes[3];

      // Triangle wave: 0..1..0..1 (period 1). Great for repeating creases.
      // (See triangle-wave & shaping function references)
      float tri(float x) {
        return abs(fract(x) - 0.5) * 2.0;
      }

      // Rotate a vector around an axis by angle (Rodrigues' rotation)
      vec3 rotateAroundAxis(vec3 p, vec3 axis, float angle) {
        axis = normalize(axis);
        float s = sin(angle), c = cos(angle);
        return p * c + cross(axis, p) * s + axis * dot(axis, p) * (1.0 - c);
      }
      `
    )
    .replace(
      '#include <begin_vertex>',
      `
      #include <begin_vertex>

      // Work in object space.
      vec3 p = transformed;
      vec3 n = objectNormal;

      // Gently rotate object-space positions inside the shader for richer folding.
      p = rotateAroundAxis(p, vec3(0.0,1.0,0.0), uRot * uTime);
      p = rotateAroundAxis(p, vec3(1.0,0.0,0.0), 0.37 * uRot * uTime);

      // Project onto three oblique fold planes and combine triangle waves.
      float d0 = dot(p, normalize(uPlanes[0]));
      float d1 = dot(p, normalize(uPlanes[1]));
      float d2 = dot(p, normalize(uPlanes[2]));

      // Phase offsets ensure the folds don't align perfectly (more "organic").
      float w0 = tri(d0 * uFreq + uTime * 0.90);
      float w1 = tri(d1 * (uFreq * 1.13) - uTime * 0.70);
      float w2 = tri(d2 * (uFreq * 0.73) + uTime * 0.55);

      // Sharpen into creases and normalize contribution.
      float crease = pow((w0 + w1 + w2) / 3.0, max(uSharp, 0.0001));

      // Displace along the surface normal -> preserves planar facets aesthetic.
      float disp = uAmp * (crease - 0.5) * 2.0 * uMix;

      transformed += n * disp;
      `
    );

  shaderRef = shader;
};

const mesh = new THREE.Mesh(geometry, material);
scene.add(mesh);

// Add a faint directional light to emphasize edges (env handles most lighting)
const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
dirLight.position.set(5, 8, 4);
scene.add(dirLight);

// ---------- GUI ----------
const params = {
  // timing
  speed: 0.6,
  phaseSpread: 1.0,    // how much the 16 panels span in "time units"
  rotSpeed: 0.28,

  // fold shaping
  amplitude: 0.45,
  frequency: 2.3,
  sharpness: 2.0,
  mix: 0.66,

  // material look
  transmission: 0.92,
  thickness: 0.6,
  roughness: 0.12,
  envMapIntensity: 1.3,
  iridescence: 1.0,
  iridescenceIOR: 1.3,
  iridescenceMin: 180,
  iridescenceMax: 520,

  randomizePlanes: () => {
    // Re-orient the three fold planes for new looks
    if (!shaderRef) return;
    for (let i = 0; i < 3; i++) {
      const v = new THREE.Vector3(
        (Math.random() * 2 - 1),
        (Math.random() * 2 - 1),
        (Math.random() * 2 - 1)
      ).normalize();
      shaderRef.uniforms.uPlanes.value[i].copy(v);
    }
  },
};

const gui = new GUI({ title: 'Controls' });
const fTiming = gui.addFolder('Timing');
fTiming.add(params, 'speed', 0.0, 3.0, 0.01).name('Speed');
fTiming.add(params, 'phaseSpread', 0.0, 4.0, 0.01).name('Spread (phases)');
fTiming.add(params, 'rotSpeed', 0.0, 1.0, 0.01).name('Rotation');

const fFold = gui.addFolder('Folding');
fFold.add(params, 'amplitude', 0.0, 1.2, 0.01).name('Amplitude');
fFold.add(params, 'frequency', 0.1, 8.0, 0.01).name('Frequency');
fFold.add(params, 'sharpness', 0.1, 6.0, 0.01).name('Sharpness');
fFold.add(params, 'mix', 0.0, 1.0, 0.01).name('Mix');
fFold.add(params, 'randomizePlanes').name('Randomize planes');

const fMat = gui.addFolder('Material');
fMat.add(material, 'transmission', 0.0, 1.0, 0.01).name('Transmission');
fMat.add(material, 'thickness', 0.0, 2.0, 0.01).name('Thickness');
fMat.add(material, 'roughness', 0.0, 1.0, 0.01).name('Roughness');
fMat.add(material, 'envMapIntensity', 0.0, 3.0, 0.01).name('Env Intensity');
fMat.add(material, 'iridescence', 0.0, 1.0, 0.01).name('Iridescence');
fMat.add(material, 'iridescenceIOR', 1.0, 2.0, 0.01).name('Iridescence IOR');
fMat.add(params, 'iridescenceMin', 0, 1200, 1).name('Iridescence Min')
  .onChange(() => { material.iridescenceThicknessRange = [params.iridescenceMin, params.iridescenceMax]; });
fMat.add(params, 'iridescenceMax', 0, 1200, 1).name('Iridescence Max')
  .onChange(() => { material.iridescenceThicknessRange = [params.iridescenceMin, params.iridescenceMax]; });

// ---------- resize ----------
function resizeRendererToDisplaySize() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  renderer.setSize(w, h, false);
}
window.addEventListener('resize', resizeRendererToDisplaySize);
resizeRendererToDisplaySize();

// ---------- render loop ----------
const clock = new THREE.Clock();

function render() {
  const baseTime = clock.getElapsedTime();
  controls.update();

  // We’ll render the scene 16 times into 16 scissored viewports.
  renderer.setScissorTest(true);

  const canvas = renderer.domElement;
  const fullW = canvas.width;
  const fullH = canvas.height;

  const cellW = Math.floor(fullW / COLS);
  const cellH = Math.floor(fullH / ROWS);

  for (let r = 0, idx = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++, idx++) {

      const x = c * cellW;
      const y = r * cellH;
      // Y-flip viewport/scissor (origin bottom-left)
      const vx = x;
      const vy = fullH - (y + cellH);

      renderer.setViewport(vx, vy, cellW, cellH);
      renderer.setScissor(vx, vy, cellW, cellH);

      // Set camera aspect per cell
      camera.aspect = cellW / cellH;
      camera.updateProjectionMatrix();

      // Time offset per panel
      const t = baseTime * params.speed + phaseOffsets[idx] * params.phaseSpread;

      // Update the injected uniforms
      if (shaderRef) {
        shaderRef.uniforms.uTime.value = t;
        shaderRef.uniforms.uAmp.value = params.amplitude;
        shaderRef.uniforms.uFreq.value = params.frequency;
        shaderRef.uniforms.uSharp.value = params.sharpness;
        shaderRef.uniforms.uRot.value = params.rotSpeed;
        shaderRef.uniforms.uMix.value = params.mix;
      }

      // Also rotate the mesh at the scene level (global motion matched to time)
      const rot = t * params.rotSpeed;
      mesh.rotation.set(rot * 0.7, rot * 1.1, rot * 0.2);

      renderer.render(scene, camera);
    }
  }

  renderer.setScissorTest(false);
  requestAnimationFrame(render);
}
requestAnimationFrame(render);

// ---------- cleanup on hot-reload / page hide ----------
window.addEventListener('beforeunload', () => {
  pmrem.dispose();
  env.dispose?.();
  geometry.dispose();
  material.dispose();
  renderer.dispose();
});
