// Twisting cube with feedback + blur + RGBA delay (TouchDesigner-style).
// - Single continuous mesh (no separating polygons).
// - Feedback accumulation creates the vibrant “folding” texture.
// - RGBA delay picks different history frames per channel (a la rgbaDelay).
// Open index.html in a modern browser.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import GUI from 'https://cdn.jsdelivr.net/npm/lil-gui@0.19/+esm';

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

const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
camera.position.set(0, 0, 4.6);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.08; controls.rotateSpeed = 0.6;

// Neutral, clean environment (for reflections/refraction)
const pmrem = new THREE.PMREMGenerator(renderer);
const envRT = pmrem.fromScene(new RoomEnvironment(renderer), 0.04);
scene.environment = envRT.texture;

/* =======================
   Parameters / GUI
   ======================= */
const P = {
  // Cube deformation & motion
  twist: 3.0,       // radians per unit height
  spin: 0.35,       // auto spin
  autoRotate: true,

  // Feedback pipeline
  feedbackDecay: 0.92,  // < 1.0 keeps trails
  feedbackGain: 0.95,   // strength of previous frame
  currentGain: 1.0,     // strength of current frame
  warpRotate: 0.01,     // rotation of feedback each frame
  warpZoom: 0.002,      // zoom of feedback each frame
  warpJitter: 0.0025,   // procedural jitter on feedback UVs
  blur: 1.6,            // blur radius (separable)
  dispersion: 0.0018,   // chromatic UV shift on final composite

  // RGBA delay (frames)
  delayR: 0,
  delayG: 2,
  delayB: 4,

  // Material (reflections / refraction)
  roughness: 0.06,
  transmission: 0.74,
  thickness: 0.75,
  ior: 1.36,
  clearcoat: 1.0,
  clearcoatRoughness: 0.12,
  iridescence: 1.0,
  envIntensity: 1.5,

  speed: 1.0,       // global speed scalar
  reseed: () => (seed = Math.random() * 1000.0)
};

const gui = new GUI({ title: 'Controls', width: 300 });
const fShape = gui.addFolder('Shape');
fShape.add(P, 'twist', 0.0, 8.0, 0.01).name('Twist');
fShape.add(P, 'spin', 0.0, 2.0, 0.01).name('Spin');
fShape.add(P, 'autoRotate').name('Auto Rotate');

const fFb = gui.addFolder('Feedback');
fFb.add(P, 'feedbackDecay', 0.80, 0.995, 0.001).name('Decay');
fFb.add(P, 'feedbackGain', 0.0, 1.5, 0.01).name('Feedback Gain');
fFb.add(P, 'currentGain', 0.0, 2.0, 0.01).name('Current Gain');
fFb.add(P, 'warpRotate', -0.05, 0.05, 0.0005).name('Warp Rotate');
fFb.add(P, 'warpZoom', -0.02, 0.02, 0.0001).name('Warp Zoom');
fFb.add(P, 'warpJitter', 0.0, 0.01, 0.0001).name('Warp Jitter');
fFb.add(P, 'blur', 0.0, 4.0, 0.01).name('Blur');

const fDelay = gui.addFolder('RGBA Delay');
fDelay.add(P, 'delayR', 0, 7, 1).name('Delay R');
fDelay.add(P, 'delayG', 0, 7, 1).name('Delay G');
fDelay.add(P, 'delayB', 0, 7, 1).name('Delay B');
fDelay.add(P, 'dispersion', 0.0, 0.01, 0.0001).name('Dispersion');

const fMat = gui.addFolder('Material');
fMat.add(P, 'roughness', 0.0, 1.0, 0.001).name('Roughness').onChange(()=> material.roughness = P.roughness);
fMat.add(P, 'transmission', 0.0, 1.0, 0.001).name('Transmission').onChange(()=> material.transmission = P.transmission);
fMat.add(P, 'thickness', 0.0, 2.0, 0.01).name('Thickness').onChange(()=> material.thickness = P.thickness);
fMat.add(P, 'ior', 1.0, 2.0, 0.001).name('IOR').onChange(()=> material.ior = P.ior);
fMat.add(P, 'clearcoat', 0.0, 1.0, 0.001).name('Clearcoat').onChange(()=> material.clearcoat = P.clearcoat);
fMat.add(P, 'clearcoatRoughness', 0.0, 1.0, 0.001).name('Clearcoat Rough.').onChange(()=> material.clearcoatRoughness = P.clearcoatRoughness);
fMat.add(P, 'iridescence', 0.0, 1.0, 0.001).name('PBR Iridescence').onChange(()=> material.iridescence = P.iridescence);
fMat.add(P, 'envIntensity', 0.0, 3.0, 0.01).name('Env Intensity').onChange(()=> material.envMapIntensity = P.envIntensity);

gui.add(P, 'speed', 0.0, 3.0, 0.01).name('Speed');
gui.add(P, 'reseed').name('Reseed');

let seed = 123.456;

/* =======================
   Twisting cube (continuous)
   ======================= */
const geo = new THREE.BoxGeometry(2.2, 2.2, 2.2, 80, 120, 80); // many segments for smooth twist
const material = new THREE.MeshPhysicalMaterial({
  color: 0xffffff,
  metalness: 0.0,
  roughness: P.roughness,
  transmission: P.transmission,
  thickness: P.thickness,
  ior: P.ior,
  clearcoat: P.clearcoat,
  clearcoatRoughness: P.clearcoatRoughness,
  iridescence: P.iridescence,
  envMapIntensity: P.envIntensity,
  side: THREE.DoubleSide,
  flatShading: true // faceted shading while positions remain continuous
});
const cube = new THREE.Mesh(geo, material);
scene.add(cube);

// Inject a simple, continuous TWIST deformation in the vertex stage (no cracks)
material.onBeforeCompile = (sh) => {
  sh.uniforms.uTwist = { value: P.twist };
  sh.uniforms.uTime  = { value: 0.0 };

  sh.vertexShader = `
    uniform float uTwist;
    uniform float uTime;
  ` + sh.vertexShader;

  sh.vertexShader = sh.vertexShader.replace(
    '#include <begin_vertex>',
    `
      #include <begin_vertex>
      {
        // continuous twist about Y; angle scales with height
        float angle = uTwist * transformed.y;
        float s = sin(angle), c = cos(angle);
        vec3 p = transformed;
        p.x = c * transformed.x - s * transformed.z;
        p.z = s * transformed.x + c * transformed.z;
        transformed = p;
      }
    `
  );

  material.userData.shader = sh;
};
material.needsUpdate = true;

/* =======================
   Post pipeline (feedback)
   ======================= */
const postScene = new THREE.Scene();
const postCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ color: 0xffffff }));
postScene.add(quad);

const RTScale = 1.0;
let W = 1, H = 1;

function makeRT() {
  return new THREE.WebGLRenderTarget(
    Math.max(2, Math.floor(W * RTScale)),
    Math.max(2, Math.floor(H * RTScale)),
    {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer: false,
      stencilBuffer: false
    }
  );
}

let rtScene = makeRT();
let rtAccumA = makeRT();
let rtAccumB = makeRT();
let rtTemp = makeRT();

const HISTORY = 8; // matches GUI max delays
const history = new Array(HISTORY).fill(0).map(() => makeRT());
let histIndex = 0;

// Common fullscreen VS
const fsVS = /* glsl */`
  varying vec2 vUv;
  void main(){
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

// Simple 2D noise (cheap) for warp
const noiseGLSL = /* glsl */`
  float hash(vec2 p){ p = fract(p*vec2(123.34, 345.45)); p += dot(p, p+34.345); return fract(p.x*p.y); }
  float noise(vec2 p){
    vec2 i = floor(p), f = fract(p);
    float a = hash(i), b = hash(i + vec2(1.0,0.0));
    float c = hash(i + vec2(0.0,1.0)), d = hash(i + vec2(1.0,1.0));
    vec2 u = f*f*(3.0-2.0*f);
    return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
  }
`;

// Warp the previous accumulation slightly (rotate/zoom/jitter)
const warpMat = new THREE.ShaderMaterial({
  vertexShader: fsVS,
  fragmentShader: /* glsl */`
    varying vec2 vUv;
    uniform sampler2D tPrev;
    uniform float uRot, uZoom, uJitter, uTime;
    ${noiseGLSL}
    void main(){
      vec2 uv = vUv - 0.5;
      float c = cos(uRot), s = sin(uRot);
      uv = mat2(c, -s, s, c) * uv;
      uv *= (1.0 + uZoom);
      uv += 0.5;

      // Tiny domain warp for organic motion
      float n = noise(vUv * 4.0 + uTime * 0.15);
      vec2 jitter = (vec2(noise(vUv*8.0 + 13.1 + uTime*0.2), noise(vUv*8.0 + 91.7 - uTime*0.2)) - 0.5) * uJitter * (0.5 + n);
      vec4 col = texture2D(tPrev, uv + jitter);
      gl_FragColor = col;
    }
  `,
  uniforms: {
    tPrev: { value: null },
    uRot: { value: 0 },
    uZoom: { value: 0 },
    uJitter: { value: 0 },
    uTime: { value: 0 }
  }
});

// Composite current scene with warped feedback (with decay)
const compositeMat = new THREE.ShaderMaterial({
  vertexShader: fsVS,
  fragmentShader: /* glsl */`
    varying vec2 vUv;
    uniform sampler2D tWarped;
    uniform sampler2D tCurr;
    uniform float uDecay;
    uniform float uFbGain;
    uniform float uCurrGain;
    void main(){
      vec4 fb = texture2D(tWarped, vUv) * uDecay * uFbGain;
      vec4 cur = texture2D(tCurr, vUv) * uCurrGain;
      // additive-ish blend (looks closer to TD feedback aesthetics)
      vec4 col = clamp(fb + cur, 0.0, 1.0);
      gl_FragColor = col;
    }
  `,
  uniforms: {
    tWarped: { value: null },
    tCurr: { value: null },
    uDecay: { value: 0.95 },
    uFbGain: { value: 1.0 },
    uCurrGain: { value: 1.0 }
  }
});

// Separable blur (one pass does H or V)
const blurMat = new THREE.ShaderMaterial({
  vertexShader: fsVS,
  fragmentShader: /* glsl */`
    varying vec2 vUv;
    uniform sampler2D tInput;
    uniform vec2 uDir;    // (1,0) or (0,1)
    uniform float uRadius;
    void main(){
      vec2 texel = uDir / vec2(textureSize(tInput, 0));
      float r = uRadius;
      vec4 acc = vec4(0.0);
      float wsum = 0.0;
      // 9-tap Gaussian-ish
      for(int i=-4;i<=4;i++){
        float x = float(i);
        float w = exp(-0.5 * (x*x) / (r*r + 1e-5));
        vec2 off = x * texel;
        acc += texture2D(tInput, vUv + off) * w;
        wsum += w;
      }
      gl_FragColor = acc / max(wsum, 1e-5);
    }
  `,
  uniforms: {
    tInput: { value: null },
    uDir: { value: new THREE.Vector2(1,0) },
    uRadius: { value: 1.0 }
  }
});

// Copy (for pushing into history slots)
const copyMat = new THREE.ShaderMaterial({
  vertexShader: fsVS,
  fragmentShader: /* glsl */`
    varying vec2 vUv;
    uniform sampler2D tInput;
    void main(){ gl_FragColor = texture2D(tInput, vUv); }
  `,
  uniforms: { tInput: { value: null } }
});

// Final RGBA delay + slight dispersion
const finalMat = new THREE.ShaderMaterial({
  vertexShader: fsVS,
  fragmentShader: /* glsl */`
    varying vec2 vUv;
    uniform sampler2D t0; uniform sampler2D t1; uniform sampler2D t2; uniform sampler2D t3;
    uniform sampler2D t4; uniform sampler2D t5; uniform sampler2D t6; uniform sampler2D t7;
    uniform float wR[8]; uniform float wG[8]; uniform float wB[8];
    uniform float uDispersion;

    vec4 sampleIndex(int k, vec2 uv){
      if(k==0) return texture2D(t0, uv);
      if(k==1) return texture2D(t1, uv);
      if(k==2) return texture2D(t2, uv);
      if(k==3) return texture2D(t3, uv);
      if(k==4) return texture2D(t4, uv);
      if(k==5) return texture2D(t5, uv);
      if(k==6) return texture2D(t6, uv);
      return texture2D(t7, uv);
    }

    void main(){
      // Read all history frames once
      vec4 c0 = texture2D(t0, vUv);
      vec4 c1 = texture2D(t1, vUv);
      vec4 c2 = texture2D(t2, vUv);
      vec4 c3 = texture2D(t3, vUv);
      vec4 c4 = texture2D(t4, vUv);
      vec4 c5 = texture2D(t5, vUv);
      vec4 c6 = texture2D(t6, vUv);
      vec4 c7 = texture2D(t7, vUv);

      // Weighted pick per channel (selects one history slot by setting weight=1)
      float r =
          c0.r*wR[0] + c1.r*wR[1] + c2.r*wR[2] + c3.r*wR[3] +
          c4.r*wR[4] + c5.r*wR[5] + c6.r*wR[6] + c7.r*wR[7];

      // Slight dispersion: shift UVs a little per channel for iridescent edges
      vec2 off = vec2(uDispersion, -uDispersion);
      float g =
          texture2D(t0, vUv+off).g*wG[0] + texture2D(t1, vUv+off).g*wG[1] +
          texture2D(t2, vUv+off).g*wG[2] + texture2D(t3, vUv+off).g*wG[3] +
          texture2D(t4, vUv+off).g*wG[4] + texture2D(t5, vUv+off).g*wG[5] +
          texture2D(t6, vUv+off).g*wG[6] + texture2D(t7, vUv+off).g*wG[7];

      vec2 off2 = vec2(-uDispersion, uDispersion);
      float b =
          texture2D(t0, vUv+off2).b*wB[0] + texture2D(t1, vUv+off2).b*wB[1] +
          texture2D(t2, vUv+off2).b*wB[2] + texture2D(t3, vUv+off2).b*wB[3] +
          texture2D(t4, vUv+off2).b*wB[4] + texture2D(t5, vUv+off2).b*wB[5] +
          texture2D(t6, vUv+off2).b*wB[6] + texture2D(t7, vUv+off2).b*wB[7];

      gl_FragColor = vec4(r, g, b, 1.0);
    }
  `,
  uniforms: {
    t0: { value: null }, t1: { value: null }, t2: { value: null }, t3: { value: null },
    t4: { value: null }, t5: { value: null }, t6: { value: null }, t7: { value: null },
    wR: { value: new Array(8).fill(0) },
    wG: { value: new Array(8).fill(0) },
    wB: { value: new Array(8).fill(0) },
    uDispersion: { value: P.dispersion }
  }
});

/* =======================
   Render helpers
   ======================= */
function renderTo(target, mat) {
  quad.material = mat;
  renderer.setRenderTarget(target);
  renderer.render(postScene, postCam);
  renderer.setRenderTarget(null);
}

function resize() {
  const panel = document.querySelector('.lil-gui.root');
  const pad = panel ? Math.ceil(panel.getBoundingClientRect().width) : 300;
  W = Math.max(1, window.innerWidth - pad);
  H = Math.max(1, window.innerHeight);

  renderer.setSize(W, H, false);
  renderer.domElement.style.width = `${W}px`;
  renderer.domElement.style.height = `${H}px`;
  camera.aspect = W / H;
  camera.updateProjectionMatrix();

  rtScene.setSize(W, H);
  rtAccumA.setSize(W, H);
  rtAccumB.setSize(W, H);
  rtTemp.setSize(W, H);
  for (const rt of history) rt.setSize(W, H);
}
window.addEventListener('resize', resize);
resize();

/* =======================
   Animation loop
   ======================= */
const clock = new THREE.Clock();
let tAccum = rtAccumA;

function setDelayWeights() {
  // zero all weights
  finalMat.uniforms.wR.value = finalMat.uniforms.wR.value.map(() => 0);
  finalMat.uniforms.wG.value = finalMat.uniforms.wG.value.map(() => 0);
  finalMat.uniforms.wB.value = finalMat.uniforms.wB.value.map(() => 0);
  finalMat.uniforms.wR.value[P.delayR] = 1;
  finalMat.uniforms.wG.value[P.delayG] = 1;
  finalMat.uniforms.wB.value[P.delayB] = 1;
  finalMat.uniforms.uDispersion.value = P.dispersion;
}

function animate() {
  const dt = clock.getDelta();
  const t = clock.elapsedTime * (0.6 + 1.4 * P.speed);

  // Update twist + spin on the cube (continuous deformation — no cracks)
  cube.rotation.y += (P.autoRotate ? P.spin * dt : 0);
  const sh = material.userData.shader;
  if (sh) {
    sh.uniforms.uTwist.value = P.twist;
    sh.uniforms.uTime.value = t;
  }

  // Render current 3D scene to texture
  renderer.setRenderTarget(rtScene);
  renderer.render(scene, camera);
  renderer.setRenderTarget(null);

  // WARP previous accumulation
  warpMat.uniforms.tPrev.value = tAccum.texture;
  warpMat.uniforms.uRot.value = P.warpRotate;
  warpMat.uniforms.uZoom.value = P.warpZoom;
  warpMat.uniforms.uJitter.value = P.warpJitter;
  warpMat.uniforms.uTime.value = t + seed;
  renderTo(rtTemp, warpMat);

  // COMPOSITE with current frame
  compositeMat.uniforms.tWarped.value = rtTemp.texture;
  compositeMat.uniforms.tCurr.value = rtScene.texture;
  compositeMat.uniforms.uDecay.value = P.feedbackDecay;
  compositeMat.uniforms.uFbGain.value = P.feedbackGain;
  compositeMat.uniforms.uCurrGain.value = P.currentGain;
  renderTo(rtAccumB, compositeMat);

  // BLUR (H then V)
  if (P.blur > 0.001) {
    blurMat.uniforms.tInput.value = rtAccumB.texture;
    blurMat.uniforms.uDir.value.set(1, 0);
    blurMat.uniforms.uRadius.value = P.blur;
    renderTo(rtTemp, blurMat);

    blurMat.uniforms.tInput.value = rtTemp.texture;
    blurMat.uniforms.uDir.value.set(0, 1);
    renderTo(rtAccumB, blurMat);
  }

  // Push into history ring buffer (copy once into the current slot)
  copyMat.uniforms.tInput.value = rtAccumB.texture;
  renderTo(history[histIndex], copyMat);
  histIndex = (histIndex + 1) % HISTORY;

  // Prepare final RGBA delay weights & bindings
  setDelayWeights();
  // Re-bind t0..t7 so that index 0 = newest, 7 = oldest
  for (let i = 0; i < HISTORY; i++) {
    // newest at index 0
    const slot = (histIndex - 1 - i + HISTORY) % HISTORY;
    finalMat.uniforms['t'+i].value = history[slot].texture;
  }

  // Draw final to screen
  quad.material = finalMat;
  renderer.setRenderTarget(null);
  renderer.render(postScene, postCam);

  // Ping-pong accumulators
  const tmp = tAccum; tAccum = rtAccumB; rtAccumB = tmp;

  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);
