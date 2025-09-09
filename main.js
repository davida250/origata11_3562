// Deforming iridescent cube that folds into itself (continuous twist),
// with facet-aligned thin-film stripes + feedback + blur + RGBA delay.
// Open index.html in a modern browser.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import GUI from 'https://cdn.jsdelivr.net/npm/lil-gui@0.19/+esm';

/* =======================
   Renderer / Scene / Camera
   ======================= */
const canvas = /** @type {HTMLCanvasElement} */ (document.getElementById('scene'));
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
renderer.setClearColor(0x000000, 1);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
renderer.physicallyCorrectLights = true;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
camera.position.set(0, 0, 4.6);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.08; controls.rotateSpeed = 0.6;

// Neutral PMREM environment for clean reflections/refraction
const pmrem = new THREE.PMREMGenerator(renderer);
const envRT = pmrem.fromScene(new RoomEnvironment(renderer), 0.04);
scene.environment = envRT.texture;

/* =======================
   Parameters / GUI
   ======================= */
const P = {
  // Twisting deformation (continuous)
  twist: 3.25,       // radians per unit height — tuned towards your sheet
  spin: 0.35,
  autoRotate: true,

  // Thin-film stripes (facet-aligned)
  filmBaseNm: 430.0, // nm
  filmAmpNm: 280.0,  // nm
  stripeFreq: 12.8,  // tuned for diagonal stripes like the sheet
  stripeWarp: 0.52,
  filmVibrance: 1.55,
  filmPastel: 0.22,

  // Feedback pipeline (TouchDesigner-style)
  feedbackDecay: 0.91,
  feedbackGain: 1.00,
  currentGain: 1.00,
  warpRotate: 0.015,
  warpZoom:   0.003,
  warpJitter: 0.0018,
  blur: 1.8,                    // separable blur radius
  maskHardness: 1.05,           // 1=hard mask, <1 feathered
  dispersion: 0.0020,           // per-channel UV shift in final

  // RGBA delay in frames (0..7)
  delayR: 1,
  delayG: 3,
  delayB: 5,

  // PBR material
  roughness: 0.06,
  transmission: 0.78,
  thickness: 0.80,
  ior: 1.36,
  clearcoat: 1.0,
  clearcoatRoughness: 0.12,
  iridescence: 1.0,
  envIntensity: 1.6,

  // Global
  speed: 1.0
};

const gui = new GUI({ title: 'Controls', width: 300 });
const fShape = gui.addFolder('Shape');
fShape.add(P, 'twist', 0.0, 8.0, 0.01).name('Twist');
fShape.add(P, 'spin', 0.0, 2.0, 0.01).name('Spin');
fShape.add(P, 'autoRotate').name('Auto Rotate');

const fFilm = gui.addFolder('Facet Texture');
fFilm.add(P, 'stripeFreq', 3.0, 24.0, 0.1).name('Stripe Freq');
fFilm.add(P, 'stripeWarp', 0.0, 1.0, 0.01).name('Stripe Warp');
fFilm.add(P, 'filmBaseNm', 250.0, 600.0, 1.0).name('Film Base (nm)');
fFilm.add(P, 'filmAmpNm', 0.0, 400.0, 1.0).name('Film Amp (nm)');
fFilm.add(P, 'filmVibrance', 0.0, 3.0, 0.01).name('Vibrance');
fFilm.add(P, 'filmPastel', 0.0, 1.0, 0.01).name('Pastel');

const fFb = gui.addFolder('Feedback');
fFb.add(P, 'feedbackDecay', 0.80, 0.995, 0.001).name('Decay');
fFb.add(P, 'feedbackGain', 0.0, 1.5, 0.01).name('Feedback Gain');
fFb.add(P, 'currentGain', 0.0, 2.0, 0.01).name('Current Gain');
fFb.add(P, 'warpRotate', -0.05, 0.05, 0.0005).name('Warp Rotate');
fFb.add(P, 'warpZoom', -0.02, 0.02, 0.0001).name('Warp Zoom');
fFb.add(P, 'warpJitter', 0.0, 0.01, 0.0001).name('Warp Jitter');
fFb.add(P, 'blur', 0.0, 4.0, 0.01).name('Blur');
fFb.add(P, 'maskHardness', 0.5, 1.5, 0.01).name('Mask Hardness');

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

/* =======================
   Twisting cube (continuous) + facet iridescence
   ======================= */
const geo = new THREE.BoxGeometry(2.2, 2.2, 2.2, 96, 144, 96); // many segments → smooth twist
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
  flatShading: true // crisp facets while geometry stays continuous
});

const cube = new THREE.Mesh(geo, material);
scene.add(cube);

// Mask mesh (same deformation) to confine feedback inside the silhouette
const maskMat = new THREE.ShaderMaterial({
  vertexShader: /* glsl */`
    uniform float uTwist;
    void main(){
      vec3 p = position;
      float ang = uTwist * p.y;
      float s = sin(ang), c = cos(ang);
      vec3 q = vec3(c*p.x - s*p.z, p.y, s*p.x + c*p.z);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(q, 1.0);
    }
  `,
  fragmentShader: /* glsl */` void main(){ gl_FragColor = vec4(1.0); } `,
  uniforms: { uTwist: { value: P.twist } },
  depthTest: true, depthWrite: true
});
const cubeMask = new THREE.Mesh(geo, maskMat);
cubeMask.matrixAutoUpdate = true;

const maskScene = new THREE.Scene();
maskScene.add(cubeMask);

// Inject twist + facet-aligned thin-film into the physical material
material.onBeforeCompile = (sh) => {
  // uniforms
  sh.uniforms.uTwist = { value: P.twist };
  sh.uniforms.uStripeFreq = { value: P.stripeFreq };
  sh.uniforms.uStripeWarp = { value: P.stripeWarp };
  sh.uniforms.uFilmBase = { value: P.filmBaseNm };
  sh.uniforms.uFilmAmp  = { value: P.filmAmpNm };
  sh.uniforms.uVibrance = { value: P.filmVibrance };
  sh.uniforms.uPastel   = { value: P.filmPastel };

  // vertex twist (continuous)
  sh.vertexShader = `
    uniform float uTwist;
  ` + sh.vertexShader;

  sh.vertexShader = sh.vertexShader.replace(
    '#include <begin_vertex>',
    `
      #include <begin_vertex>
      {
        float a = uTwist * transformed.y;
        float s = sin(a), c = cos(a);
        vec3 p = transformed;
        p.x = c * transformed.x - s * transformed.z;
        p.z = s * transformed.x + c * transformed.z;
        transformed = p;
      }
    `
  );

  // varyings for world pos/normal
  sh.vertexShader = sh.vertexShader.replace(
    'void main() {',
    `
    varying vec3 vWorldPos;
    varying vec3 vNormalW;
    void main() {
    `
  );
  sh.vertexShader = sh.vertexShader.replace(
    '#include <project_vertex>',
    `
      #include <project_vertex>
      vWorldPos = (modelMatrix * vec4(transformed,1.0)).xyz;
      vNormalW = normalize(mat3(modelMatrix) * normal);
    `
  );

  // thin-film helpers + emissive add
  sh.fragmentShader = `
    uniform float uStripeFreq, uStripeWarp, uFilmBase, uFilmAmp, uVibrance, uPastel;
    varying vec3 vWorldPos; varying vec3 vNormalW;

    vec2 rot2(vec2 p, float a){ float c=cos(a), s=sin(a); return mat2(c,-s,s,c)*p; }

    float stripeField(vec3 p, float freq, float warp) {
      // world/object anchored stripes with a little domain warp
      vec2 q = p.xy + vec2(0.6*p.z, 0.35*p.x);
      q = rot2(q, 0.785398); // 45°
      float s1 = sin(q.x*freq + 0.7*q.y);
      float s2 = sin((q.x+q.y)*freq*0.35 + 1.7*sin(q.x*0.6));
      return 0.5 + 0.5 * mix(s1, s2, clamp(warp,0.0,1.0));
    }

    vec3 thinFilm(float ndv, float d_nm, float n2) {
      float n1 = 1.0, n3 = 1.50;
      float sin2 = max(0.0, 1.0 - ndv*ndv);
      float cos2 = sqrt(max(0.0, 1.0 - sin2/(n2*n2)));
      float R12 = pow((n1 - n2)/(n1 + n2), 2.0);
      float R23 = pow((n2 - n3)/(n2 + n3), 2.0);
      float A = 2.0 * sqrt(R12 * R23);
      float lR = 650.0, lG = 510.0, lB = 440.0;
      float phiR = 12.5663706*n2*d_nm*cos2 / lR;
      float phiG = 12.5663706*n2*d_nm*cos2 / lG;
      float phiB = 12.5663706*n2*d_nm*cos2 / lB;
      vec3 r = vec3(
        clamp(R12 + R23 + A*cos(phiR), 0.0, 1.0),
        clamp(R12 + R23 + A*cos(phiG), 0.0, 1.0),
        clamp(R12 + R23 + A*cos(phiB), 0.0, 1.0)
      );
      return r;
    }
  ` + sh.fragmentShader;

  sh.fragmentShader = sh.fragmentShader.replace(
    '#include <emissivemap_fragment>',
    `
      #include <emissivemap_fragment>
      {
        vec3 N = normalize(vNormalW);
        vec3 V = normalize(cameraPosition - vWorldPos);
        float ndv = clamp(dot(N, V), 0.0, 1.0);
        float f = stripeField(vWorldPos*0.7 + N*0.25, uStripeFreq, uStripeWarp);
        float d_nm = uFilmBase + uFilmAmp * (f - 0.5) * 2.0;
        vec3 filmRGB = thinFilm(ndv, d_nm, 1.35);
        filmRGB = mix(filmRGB, vec3(1.0), clamp(uPastel, 0.0, 1.0));
        totalEmissiveRadiance += filmRGB * uVibrance;
      }
    `
  );

  material.userData.shader = sh;
};
material.needsUpdate = true;

/* =======================
   Post pipeline (feedback + blur + RGBA delay)
   ======================= */
const postScene = new THREE.Scene();
const postCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial({ color: 0xffffff }));
postScene.add(quad);

let W = 2, H = 2;
function makeRT() {
  return new THREE.WebGLRenderTarget(W, H, {
    minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter,
    depthBuffer: false, stencilBuffer: false
  });
}
let rtScene = makeRT();     // current scene color
let rtMask  = makeRT();     // shape mask (confines feedback to the cube)
let rtA = makeRT(), rtB = makeRT(), rtTemp = makeRT();

const HISTORY = 8;
const history = new Array(HISTORY).fill(0).map(()=> makeRT());
let histIndex = 0;

// Fullscreen VS
const fsVS = /* glsl */`
  varying vec2 vUv;
  void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

// Cheap noise (warp jitter)
const noiseGLSL = /* glsl */`
  float hash(vec2 p){ p = fract(p*vec2(123.34,345.45)); p += dot(p,p+34.345); return fract(p.x*p.y); }
  float noise(vec2 p){
    vec2 i=floor(p), f=fract(p);
    float a=hash(i), b=hash(i+vec2(1,0)), c=hash(i+vec2(0,1)), d=hash(i+vec2(1,1));
    vec2 u=f*f*(3.0-2.0*f);
    return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
  }
`;

// Warp previous frame a little (rotate/zoom/jitter)
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
      uv = mat2(c,-s,s,c)*uv;
      uv *= (1.0 + uZoom);
      uv += 0.5;
      vec2 jitter = (vec2(noise(vUv*8.0 + uTime*0.20), noise(vUv*8.0 - uTime*0.22)) - 0.5) * uJitter;
      gl_FragColor = texture2D(tPrev, uv + jitter);
    }
  `,
  uniforms: { tPrev: { value: null }, uRot:{ value:0 }, uZoom:{value:0}, uJitter:{value:0}, uTime:{value:0} }
});

// Composite warped feedback with current frame, masked to the cube
const compositeMat = new THREE.ShaderMaterial({
  vertexShader: fsVS,
  fragmentShader: /* glsl */`
    varying vec2 vUv;
    uniform sampler2D tWarped, tCurr, tMask;
    uniform float uDecay, uFbGain, uCurrGain, uHard;
    void main(){
      vec4 fb  = texture2D(tWarped, vUv) * uDecay * uFbGain;
      float m  = pow(texture2D(tMask, vUv).r, uHard); // confine feedback
      vec4 cur = texture2D(tCurr, vUv) * uCurrGain;
      gl_FragColor = clamp(fb*m + cur, 0.0, 1.0);
    }
  `,
  uniforms: {
    tWarped: { value: null }, tCurr: { value: null }, tMask: { value: null },
    uDecay: { value: P.feedbackDecay }, uFbGain: { value: P.feedbackGain }, uCurrGain: { value: P.currentGain },
    uHard: { value: P.maskHardness }
  }
});

// Separable blur (WebGL1-safe: provide texel size)
const blurMat = new THREE.ShaderMaterial({
  vertexShader: fsVS,
  fragmentShader: /* glsl */`
    varying vec2 vUv;
    uniform sampler2D tInput;
    uniform vec2 uTexel;   // 1/size
    uniform vec2 uDir;     // (1,0) or (0,1)
    uniform float uRadius;
    void main(){
      vec2 stepv = uDir * uTexel;
      float r = uRadius;
      vec4 acc = vec4(0.0); float wsum = 0.0;
      for(int i=-4;i<=4;i++){
        float x = float(i);
        float w = exp(-0.5 * (x*x) / (r*r + 1e-5));
        acc += texture2D(tInput, vUv + x*stepv) * w;
        wsum += w;
      }
      gl_FragColor = acc / max(wsum, 1e-5);
    }
  `,
  uniforms: {
    tInput: { value: null },
    uTexel: { value: new THREE.Vector2(1/Math.max(2,W), 1/Math.max(2,H)) },
    uDir:   { value: new THREE.Vector2(1,0) },
    uRadius:{ value: P.blur }
  }
});

// Copy material (push frames into history)
const copyMat = new THREE.ShaderMaterial({
  vertexShader: fsVS,
  fragmentShader: /* glsl */`
    varying vec2 vUv; uniform sampler2D tInput;
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
    uniform vec2  uTexel;       // for dispersion
    uniform float uDispersion;

    void main(){
      // Slight per-channel UV shift (dispersion) for iridescent fringes
      vec2 ur = vUv + vec2( uDispersion, -uDispersion) * uTexel;
      vec2 ug = vUv;
      vec2 ub = vUv + vec2(-uDispersion,  uDispersion) * uTexel;

      // Weighted pick per channel (TouchDesigner-like discrete delays)
      float r =
          texture2D(t0, ur).r*wR[0]+texture2D(t1, ur).r*wR[1]+texture2D(t2, ur).r*wR[2]+texture2D(t3, ur).r*wR[3]+
          texture2D(t4, ur).r*wR[4]+texture2D(t5, ur).r*wR[5]+texture2D(t6, ur).r*wR[6]+texture2D(t7, ur).r*wR[7];

      float g =
          texture2D(t0, ug).g*wG[0]+texture2D(t1, ug).g*wG[1]+texture2D(t2, ug).g*wG[2]+texture2D(t3, ug).g*wG[3]+
          texture2D(t4, ug).g*wG[4]+texture2D(t5, ug).g*wG[5]+texture2D(t6, ug).g*wG[6]+texture2D(t7, ug).g*wG[7];

      float b =
          texture2D(t0, ub).b*wB[0]+texture2D(t1, ub).b*wB[1]+texture2D(t2, ub).b*wB[2]+texture2D(t3, ub).b*wB[3]+
          texture2D(t4, ub).b*wB[4]+texture2D(t5, ub).b*wB[5]+texture2D(t6, ub).b*wB[6]+texture2D(t7, ub).b*wB[7];

      gl_FragColor = vec4(r, g, b, 1.0);
    }
  `,
  uniforms: {
    t0:{value:null}, t1:{value:null}, t2:{value:null}, t3:{value:null},
    t4:{value:null}, t5:{value:null}, t6:{value:null}, t7:{value:null},
    wR:{value:new Array(8).fill(0)}, wG:{value:new Array(8).fill(0)}, wB:{value:new Array(8).fill(0)},
    uTexel:{value:new THREE.Vector2(1/Math.max(2,W), 1/Math.max(2,H))},
    uDispersion:{value:P.dispersion}
  }
});

/* =======================
   Resize & utilities
   ======================= */
function panelWidth(){
  const el = document.querySelector('.lil-gui.root');
  return el ? Math.ceil(el.getBoundingClientRect().width) : 300;
}
function resize(){
  const w = Math.max(1, window.innerWidth - panelWidth());
  const h = Math.max(1, window.innerHeight);
  renderer.setSize(w, h, false);
  renderer.domElement.style.width = `${w}px`;
  renderer.domElement.style.height = `${h}px`;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();

  W = w; H = h;
  const setSize = (rt)=> rt.setSize(W, H);
  setSize(rtScene); setSize(rtMask); setSize(rtA); setSize(rtB); setSize(rtTemp);
  history.forEach(setSize);

  blurMat.uniforms.uTexel.value.set(1/W, 1/H);
  finalMat.uniforms.uTexel.value.set(1/W, 1/H);
}
window.addEventListener('resize', resize);
resize();

function setDelayWeights(){
  const zr = new Array(8).fill(0), zg = new Array(8).fill(0), zb = new Array(8).fill(0);
  zr[P.delayR] = 1; zg[P.delayG] = 1; zb[P.delayB] = 1;
  finalMat.uniforms.wR.value = zr;
  finalMat.uniforms.wG.value = zg;
  finalMat.uniforms.wB.value = zb;
  finalMat.uniforms.uDispersion.value = P.dispersion;
}
function renderTo(target, mat){
  quad.material = mat;
  renderer.setRenderTarget(target);
  renderer.render(postScene, postCam);
  renderer.setRenderTarget(null);
}

/* =======================
   Animation / feedback
   ======================= */
const clock = new THREE.Clock();
function animate(){
  const dt = clock.getDelta();
  const t  = clock.elapsedTime * (0.6 + 1.4 * P.speed);

  // Update cube transform + uniforms
  if (P.autoRotate) cube.rotation.y += P.spin * dt;
  const sh = material.userData.shader;
  if (sh) {
    sh.uniforms.uTwist.value      = P.twist;
    sh.uniforms.uStripeFreq.value = P.stripeFreq;
    sh.uniforms.uStripeWarp.value = P.stripeWarp;
    sh.uniforms.uFilmBase.value   = P.filmBaseNm;
    sh.uniforms.uFilmAmp.value    = P.filmAmpNm;
    sh.uniforms.uVibrance.value   = P.filmVibrance;
    sh.uniforms.uPastel.value     = P.filmPastel;
  }
  cubeMask.rotation.copy(cube.rotation);
  maskMat.uniforms.uTwist.value = P.twist;

  // 1) Render scene color
  renderer.setRenderTarget(rtScene);
  renderer.render(scene, camera);
  renderer.setRenderTarget(null);

  // 2) Render mask (white shape on black)
  renderer.setRenderTarget(rtMask);
  renderer.clear();
  renderer.render(maskScene, camera);
  renderer.setRenderTarget(null);

  // 3) Warp previous accumulation
  warpMat.uniforms.tPrev.value = rtA.texture;
  warpMat.uniforms.uRot.value = P.warpRotate;
  warpMat.uniforms.uZoom.value = P.warpZoom;
  warpMat.uniforms.uJitter.value = P.warpJitter;
  warpMat.uniforms.uTime.value = t;
  renderTo(rtTemp, warpMat);

  // 4) Composite with current, masked into the cube
  compositeMat.uniforms.tWarped.value = rtTemp.texture;
  compositeMat.uniforms.tCurr.value   = rtScene.texture;
  compositeMat.uniforms.tMask.value   = rtMask.texture;
  compositeMat.uniforms.uDecay.value  = P.feedbackDecay;
  compositeMat.uniforms.uFbGain.value = P.feedbackGain;
  compositeMat.uniforms.uCurrGain.value = P.currentGain;
  compositeMat.uniforms.uHard.value   = P.maskHardness;
  renderTo(rtB, compositeMat);

  // 5) Blur (H then V)
  if (P.blur > 0.001) {
    blurMat.uniforms.uRadius.value = P.blur;
    blurMat.uniforms.tInput.value = rtB.texture;
    blurMat.uniforms.uDir.value.set(1,0);
    renderTo(rtTemp, blurMat);

    blurMat.uniforms.tInput.value = rtTemp.texture;
    blurMat.uniforms.uDir.value.set(0,1);
    renderTo(rtB, blurMat);
  }

  // 6) Push into history ring
  copyMat.uniforms.tInput.value = rtB.texture;
  renderTo(history[histIndex], copyMat);
  histIndex = (histIndex + 1) % HISTORY;

  // 7) Prepare RGBA delay bindings (0=newest)
  setDelayWeights();
  for (let i=0;i<HISTORY;i++){
    const slot = (histIndex - 1 - i + HISTORY) % HISTORY;
    finalMat.uniforms['t'+i].value = history[slot].texture;
  }

  // 8) Draw final to screen
  quad.material = finalMat;
  renderer.setRenderTarget(null);
  renderer.render(postScene, postCam);

  // 9) Ping-pong accumulation
  const tmp = rtA; rtA = rtB; rtB = tmp;

  controls.update();
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);

/* =======================
   Optional: 4x4 contact-sheet capture
   Press 'C' to capture 16 frames and download a PNG.
   ======================= */
window.addEventListener('keydown', async (e)=>{
  if (e.key.toLowerCase() !== 'c') return;
  const rows=4, cols=4, pad=8, cw=256, ch=256;
  const W2 = cols*cw + (cols+1)*pad, H2 = rows*ch + (rows+1)*pad;
  const off = document.createElement('canvas'); off.width=W2; off.height=H2;
  const ctx = off.getContext('2d'); ctx.fillStyle='#000'; ctx.fillRect(0,0,W2,H2);

  // Sample 16 frames at small time offsets (like your sheet)
  const samples = [];
  for (let i=0;i<rows*cols;i++){
    const bitmap = await createImageBitmap(renderer.domElement);
    samples.push(bitmap);
    await new Promise(r => setTimeout(r, 50));
  }
  let idx=0;
  for (let r=0;r<rows;r++){
    for (let c=0;c<cols;c++){
      const x = pad + c*(cw+pad);
      const y = pad + r*(ch+pad);
      ctx.drawImage(samples[idx++], 0,0,renderer.domElement.width,renderer.domElement.height, x,y,cw,ch);
    }
  }
  const link = document.createElement('a');
  link.download = 'contact_sheet.png';
  link.href = off.toDataURL('image/png');
  link.click();
});
