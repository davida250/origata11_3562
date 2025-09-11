// Folding Swarm — points with smooth Brownian motion; proximity-based edges
// Full file (drop-in). Keeps the iridescent interference look on camera-facing "paper" ribbons.
// Controls: number of points, amplitude, frequency, connect (on) distance, break (off) distance.

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

// ---------- scene bootstrap ----------
const container = document.getElementById('scene-container');

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.18;
renderer.outputColorSpace = THREE.SRGBColorSpace;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

const camera = new THREE.PerspectiveCamera(36, container.clientWidth / container.clientHeight, 0.01, 100);
camera.position.set(3.0, 1.9, 5.0);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.minDistance = 2.0;
controls.maxDistance = 9.0;

// soft environment
const pmrem = new THREE.PMREMGenerator(renderer);
const envRT = pmrem.fromScene(new RoomEnvironment(), 0.04);
scene.environment = envRT.texture;

// ---------- utilities ----------
const tmpV1 = new THREE.Vector3();
const tmpV2 = new THREE.Vector3();
const tmpV3 = new THREE.Vector3();
const tmpM  = new THREE.Matrix4();
const tmpS  = new THREE.Vector3();
const randInSphere = (r=1) => {
  let v = new THREE.Vector3();
  do {
    v.set(Math.random()*2-1, Math.random()*2-1, Math.random()*2-1);
  } while (v.lengthSq() > 1);
  return v.multiplyScalar(r);
};

// ---------- Iridescent + interference overlay (for meshes) ----------
function injectSurfaceOverlay(material) {
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

// ---------- Folding Swarm ----------
class FoldingSwarm {
  constructor(opts) {
    this.params = opts;
    this.group = new THREE.Group();
    scene.add(this.group);

    this.N = 0;
    this.points = [];
    this.conn = null; // adjacency (upper triangle indices)
    this.maxEdges = 0;

    // geometries
    this.pointsGeom = null;
    this.pointsMesh = null;
    this.ribbonMesh = null;

    this.reseed();
    this.setCount(opts.count);
  }

  reseed() {
    this.seed = Math.random() * 1e9;
    // nothing else needed (using Math.random) — this knob lets user "reseed" positions/goals
  }

  setCount(n) {
    n = Math.max(2, Math.floor(n));
    // dispose old meshes
    if (this.pointsMesh) { this.group.remove(this.pointsMesh); this.pointsMesh.geometry.dispose(); this.pointsMesh.material.dispose(); this.pointsMesh = null; }
    if (this.ribbonMesh) { this.group.remove(this.ribbonMesh); this.ribbonMesh.geometry.dispose(); this.ribbonMesh.material.dispose(); this.ribbonMesh = null; }

    this.N = n;
    this.points = [];
    const radius = 1.4;
    for (let i = 0; i < n; i++) {
      const anchor = randInSphere(radius);
      const pos = anchor.clone();
      const vel = new THREE.Vector3();
      const goal = randInSphere(1.0);
      const nextT = 0; // force immediate goal pick
      this.points.push({ anchor, pos, vel, goal, nextT });
    }

    // adjacency + capacity
    this.conn = new Uint8Array(n * n); // we will read only j>i
    this.maxEdges = n * (n - 1) / 2;

    // points cloud
    this.pointsGeom = new THREE.BufferGeometry();
    const pos = new Float32Array(n * 3);
    this.pointsGeom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const pmat = new THREE.PointsMaterial({
      color: 0xffffff,
      size: 0.03,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.95
    });
    this.pointsMesh = new THREE.Points(this.pointsGeom, pmat);
    this.group.add(this.pointsMesh);

    // ribbons: camera-facing quads (instanced)
    const plane = new THREE.PlaneGeometry(1, 1, 1, 1);
    const rmat = new THREE.MeshPhysicalMaterial({
      color: 0x141414,
      roughness: 0.25,
      metalness: 0.0,
      envMapIntensity: 1.2,
      side: THREE.DoubleSide,
      flatShading: true
    });
    injectSurfaceOverlay(rmat);

    this.ribbonMesh = new THREE.InstancedMesh(plane, rmat, this.maxEdges);
    this.ribbonMesh.count = 0;
    this.group.add(this.ribbonMesh);
  }

  // upper-triangular index helper
  _I(i, j) {
    return i * this.N + j; // we always store j>i
  }

  _updateGoals(now) {
    const f = this.params.freq; // Hz
    const amp = this.params.amp; // world units
    // choose a fresh random target (goal) every ~1/f seconds per point
    const meanInterval = (f > 0) ? (1.0 / f) : Infinity;

    for (let i = 0; i < this.N; i++) {
      const p = this.points[i];
      if (now >= p.nextT) {
        p.goal.copy(randInSphere(1.0)).multiplyScalar(amp);
        // next goal time with slight jitter (so not all switch at once)
        const jitter = THREE.MathUtils.lerp(0.6, 1.4, Math.random());
        p.nextT = now + (meanInterval === Infinity ? Infinity : meanInterval * jitter);
      }
    }
  }

  update(dt, now) {
    // 1) motion (smooth random — spring-damper to a wandering goal around the anchor)
    this._updateGoals(now);
    const spring = 6.0 * Math.max(0.02, this.params.freq);  // pull toward goal scales with freq (snappier for higher)
    const damp   = 0.9;                                      // velocity damping per second
    const visc   = Math.pow(1.0 - (1.0 - damp), dt);         // exp-like decay

    for (let i = 0; i < this.N; i++) {
      const p = this.points[i];
      const target = tmpV1.copy(p.anchor).add(p.goal); // anchor + goal offset
      const toT = tmpV2.copy(target).sub(p.pos);
      // spring acceleration
      p.vel.addScaledVector(toT, spring * dt);
      // tiny random jiggle (keeps it lively)
      p.vel.addScaledVector(randInSphere(0.02), Math.min(1, this.params.freq * 0.2) * dt);
      // damping
      p.vel.multiplyScalar(visc);
      // integrate
      p.pos.addScaledVector(p.vel, dt);
    }

    // 2) connections (with hysteresis)
    const connectAt = this.params.connectDist; // on ≤
    const breakAt   = Math.max(this.params.breakDist, connectAt + 1e-4); // off ≥
    const width     = this.params.ribbonWidth;

    let eCount = 0;
    const view = camera.getWorldDirection(tmpV3).negate(); // approx. eye->scene
    const minLen = 1e-6;

    for (let i = 0; i < this.N; i++) {
      const pi = this.points[i].pos;
      for (let j = i + 1; j < this.N; j++) {
        const pj = this.points[j].pos;
        const d = pi.distanceTo(pj);
        const k = this._I(i, j);
        const isOn = this.conn[k] === 1;

        if (!isOn && d <= connectAt) this.conn[k] = 1;
        else if (isOn && d >= breakAt) this.conn[k] = 0;

        if (this.conn[k] === 1) {
          // instance transform for ribbon i-j
          tmpV1.subVectors(pj, pi);               // dir * length
          const L = tmpV1.length();
          if (L > minLen) {
            const dir = tmpV1.multiplyScalar(1 / L);        // normalize
            // camera-facing "billboard" around the edge: Y is perpendicular to view and dir
            const y = tmpV2.crossVectors(view, dir);
            if (y.lengthSq() < 1e-8) { // handle near-parallel cases
              y.set(0, 1, 0).cross(dir).normalize();
            } else {
              y.normalize();
            }
            const z = tmpV3.crossVectors(dir, y).normalize();

            tmpM.makeBasis(dir, y, z);                       // X=dir, Y=billboard up, Z=normal
            tmpS.set(L, width, 1.0);
            tmpM.scale(tmpS);
            tmpM.setPosition(tmpV1.copy(pi).add(pj).multiplyScalar(0.5));

            this.ribbonMesh.setMatrixAt(eCount++, tmpM);
          }
        }
      }
    }

    this.ribbonMesh.count = eCount;
    this.ribbonMesh.instanceMatrix.needsUpdate = true;

    // 3) write points positions
    const posAttr = this.pointsGeom.getAttribute('position');
    for (let i = 0; i < this.N; i++) {
      const p = this.points[i].pos;
      posAttr.setXYZ(i, p.x, p.y, p.z);
    }
    posAttr.needsUpdate = true;

    // 4) drive overlay time (for ribbons)
    const u = this.ribbonMesh.material.userData._overlayUniforms;
    if (u) u.uTime.value = now;
  }
}

// ---------- post processing ----------
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));

const afterimagePass = new AfterimagePass();
afterimagePass.enabled = false;
afterimagePass.uniforms['damp'].value = 1.0;
composer.addPass(afterimagePass);

const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(container.clientWidth, container.clientHeight),
  0.0,
  0.2,
  0.9
);
bloomPass.enabled = false;
composer.addPass(bloomPass);

const rgbShiftPass = new ShaderPass(RGBShiftShader);
rgbShiftPass.enabled = false;
rgbShiftPass.uniforms['amount'].value = 0.0;
rgbShiftPass.uniforms['angle'].value  = 0.0;
composer.addPass(rgbShiftPass);

// ---------- gui ----------
const uiHost = document.getElementById('ui');
const gui = new GUI({ title: 'Controls', width: 320 });
uiHost.appendChild(gui.domElement);

const params = {
  // movement + topology
  count:        24,      // number of points
  amp:          0.85,    // movement amplitude (radius around anchor)
  freq:         0.6,     // Hz — how often a new goal is chosen
  connectDist:  0.75,    // connect when distance <= this
  breakDist:    0.95,    // disconnect when distance >= this
  ribbonWidth:  0.045,   // visual width of edge ribbons

  // renderer / fx
  play:         true,
  foldSpeed:    1.0,     // global time scale (affects motion speed)
  bloomStrength:0.0,
  bloomThreshold:0.9,
  bloomRadius:  0.2,
  trailAmount:  0.0,
  rgbAmount:    0.0,
  rgbAngle:     0.0,
  exposure:     renderer.toneMappingExposure,

  // actions
  reseed: () => { swarm.reseed(); },
  resetCamera: () => {
    camera.position.set(3.0, 1.9, 5.0);
    controls.target.set(0,0,0);
    controls.update();
  }
};

const fSystem = gui.addFolder('Swarm');
fSystem.add(params, 'count', 4, 128, 1).name('Points').onFinishChange(v => { swarm.setCount(v); });
fSystem.add(params, 'amp', 0.0, 2.0, 0.001).name('Amplitude').onChange(v => { swarm.params.amp = v; });
fSystem.add(params, 'freq', 0.0, 3.0, 0.001).name('Frequency (Hz)').onChange(v => { swarm.params.freq = v; });
fSystem.add(params, 'connectDist', 0.05, 2.0, 0.001).name('Connect at ≤').onChange(v => {
  swarm.params.connectDist = v;
  if (params.breakDist < v) { params.breakDist = v; swarm.params.breakDist = v; gui.updateDisplay(); }
});
fSystem.add(params, 'breakDist', 0.05, 2.0, 0.001).name('Break at ≥').onChange(v => {
  swarm.params.breakDist = Math.max(v, params.connectDist);
  params.breakDist = swarm.params.breakDist;
  gui.updateDisplay();
});
fSystem.add(params, 'ribbonWidth', 0.005, 0.12, 0.001).name('Ribbon Width').onChange(v => { swarm.params.ribbonWidth = v; });
fSystem.add(params, 'reseed').name('Reseed');

const fAnim = gui.addFolder('Animation');
fAnim.add(params, 'play').name('Play / Pause');
fAnim.add(params, 'foldSpeed', 0.0, 3.0, 0.01).name('Time Scale');

const fGlow = gui.addFolder('Glow (Bloom)');
fGlow.add(params, 'bloomStrength', 0.0, 2.5, 0.01).name('Strength').onChange(v => {
  bloomPass.strength = v;
  bloomPass.enabled = v > 0.0;
});
fGlow.add(params, 'bloomThreshold', 0.0, 1.0, 0.001).name('Threshold').onChange(v => bloomPass.threshold = v);
fGlow.add(params, 'bloomRadius', 0.0, 1.0, 0.001).name('Radius').onChange(v => bloomPass.radius = v);

const fTrail = gui.addFolder('Trails');
fTrail.add(params, 'trailAmount', 0.0, 1.0, 0.001).name('Amount').onChange(v => {
  const damp = 1.0 - v * 0.98;
  afterimagePass.uniforms['damp'].value = damp;
  afterimagePass.enabled = v > 0.0;
});

const fRGB = gui.addFolder('RGB Offset');
fRGB.add(params, 'rgbAmount', 0.0, 0.02, 0.0001).name('Amount').onChange(v => {
  rgbShiftPass.uniforms['amount'].value = v;
  rgbShiftPass.enabled = v > 0.0;
});
fRGB.add(params, 'rgbAngle', 0.0, 180.0, 0.1).name('Angle (°)')
    .onChange(v => rgbShiftPass.uniforms['angle'].value = THREE.MathUtils.degToRad(v));

const fView = gui.addFolder('View');
fView.add(params, 'exposure', 0.6, 1.8, 0.01).name('Exposure').onChange(v => renderer.toneMappingExposure = v);
fView.add(params, 'resetCamera').name('Reset Camera');

// ---------- create swarm ----------
const swarm = new FoldingSwarm({
  count: params.count,
  amp: params.amp,
  freq: params.freq,
  connectDist: params.connectDist,
  breakDist: params.breakDist,
  ribbonWidth: params.ribbonWidth
});

// ---------- loop ----------
let t0 = performance.now();
function animate() {
  requestAnimationFrame(animate);
  const nowMs = performance.now();
  const t = (nowMs - t0) / 1000;

  if (params.play) {
    const dt = Math.min(1/30, (1/60) * params.foldSpeed); // stable delta (tied to time scale)
    swarm.params.amp        = params.amp;
    swarm.params.freq       = params.freq;
    swarm.params.connectDist= params.connectDist;
    swarm.params.breakDist  = params.breakDist;
    swarm.params.ribbonWidth= params.ribbonWidth;
    swarm.update(dt, t);
  }

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
