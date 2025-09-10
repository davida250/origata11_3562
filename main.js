// Endless Folding Decahedron (Pentagonal Dipyramid)
// - Self-contained 10-face polyhedron (triangles) with hinge-only folding.
// - Hinges are pinned: shared edges are welded (no gaps ever).
// - Iridescent base (MeshPhysicalMaterial) + richer emissive overlay (bands + cellular + fBm).
// - Right-side controls: Fold Speed, Glow (bloom, default 0), Trails (ghosting), RGB Offset.

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

// ---------- Scene bootstrap ----------
const container = document.getElementById('scene-container');

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
renderer.outputColorSpace = THREE.SRGBColorSpace;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

// Camera + controls
const camera = new THREE.PerspectiveCamera(36, container.clientWidth / container.clientHeight, 0.01, 100);
camera.position.set(3.2, 1.8, 4.8);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.minDistance = 2.0;
controls.maxDistance = 9.0;

// Environment (PMREM + RoomEnvironment)
const pmrem = new THREE.PMREMGenerator(renderer);
const envRT = pmrem.fromScene(new RoomEnvironment(), 0.04);
scene.environment = envRT.texture;

// ---------- Pentagonal Dipyramid with welded hinges ----------
class PentagonalDipyramid {
  constructor(opts = {}) {
    this.group = new THREE.Group();

    // Regular-ish pentagonal ring + top/bottom apices
    const R = 1.05;         // equator radius
    const H = 1.10;         // apex height
    const jitter = 0.03;    // slight irregularity for a more "natural" gem look

    const vTop = new THREE.Vector3(0,  H, 0);
    const vBot = new THREE.Vector3(0, -H, 0);
    const ring = [];
    for (let i = 0; i < 5; i++) {
      const a = (i / 5) * Math.PI * 2;
      const rj = R * (1.0 + jitter * (i % 2 ? -1 : 1));
      ring.push(new THREE.Vector3(Math.cos(a) * rj, 0, Math.sin(a) * rj));
    }

    // Faces: top 5 triangles + bottom 5 triangles (counter-winding for consistent normals)
    const facesIdx = [];
    for (let i = 0; i < 5; i++) {
      const i0 = i;
      const i1 = (i + 1) % 5;
      facesIdx.push([ 'T', vTop.clone(), ring[i0].clone(), ring[i1].clone() ]);
    }
    for (let i = 0; i < 5; i++) {
      const i0 = i;
      const i1 = (i + 1) % 5;
      facesIdx.push([ 'B', vBot.clone(), ring[i1].clone(), ring[i0].clone() ]);
    }

    // Build face records
    this.faces = facesIdx.map((entry, faceId) => {
      const rest = [entry[1], entry[2], entry[3]];
      return {
        id: faceId,
        tag: entry[0],      // 'T' or 'B'
        rest,
        world: [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()],
        parent: null,
        hinge: null         // { parent, parentEdge:[ia,ib], childEdge:[ia,ib], amp, freq, phase, bind: Matrix4 }
      };
    });

    // Hinge helper
    const thirdIndex = (i, j) => (0 + 1 + 2) - i - j; // 0+1+2=3; get the remaining index

    const bindHinge = (child, parent, parentEdge, childEdge, ampDeg, freq, phase) => {
      const hinge = { parent, parentEdge, childEdge, amp: THREE.MathUtils.degToRad(ampDeg), freq, phase, bind: new THREE.Matrix4() };

      // Compute bind so edges coincide & faces coplanar at angle = 0
      const pFace = this.faces[parent];
      const cFace = this.faces[child];

      const pA = pFace.rest[parentEdge[0]].clone();
      const pB = pFace.rest[parentEdge[1]].clone();
      const pT = pFace.rest[thirdIndex(parentEdge[0], parentEdge[1])].clone();

      const cA = cFace.rest[childEdge[0]].clone();
      const cB = cFace.rest[childEdge[1]].clone();
      const cT = cFace.rest[thirdIndex(childEdge[0], childEdge[1])].clone();

      // Edge unit vectors
      const up = new THREE.Vector3().subVectors(pB, pA).normalize();
      const uc = new THREE.Vector3().subVectors(cB, cA).normalize();

      // Rotate child edge onto parent edge (q1)
      const q1 = new THREE.Quaternion().setFromUnitVectors(uc, up);

      // Normals
      const nP = new THREE.Vector3().subVectors(pB, pA).cross(new THREE.Vector3().subVectors(pT, pA)).normalize();
      const nC1 = new THREE.Vector3().subVectors(cB, cA).cross(new THREE.Vector3().subVectors(cT, cA)).applyQuaternion(q1).normalize();

      // Rotate around hinge axis so normals align (q2)
      const proj = (n, axis) => n.clone().sub(axis.clone().multiplyScalar(n.dot(axis)));
      const nPp = proj(nP, up).normalize();
      const nCp = proj(nC1, up).normalize();

      const cross = new THREE.Vector3().crossVectors(nCp, nPp);
      const phi = Math.atan2(up.dot(cross), nCp.dot(nPp));
      const q2 = new THREE.Quaternion().setFromAxisAngle(up, phi);

      const q = q2.multiply(q1);
      const pos = pA.clone().sub(cA.clone().applyQuaternion(q));
      hinge.bind.compose(pos, q, new THREE.Vector3(1,1,1));

      // Attach
      this.faces[child].parent = parent;
      this.faces[child].hinge = hinge;
    };

    // Hinge tree (no loops): Top ring chain, then bottom ring chain
    // Root = top face 0
    this.faces[0].hinge = null;

    // Top chain: each shares edge (top, ring[i+1]) with previous
    bindHinge(1, 0, [0,2], [0,1], 38, 0.10, 0.00); // (top,v1)
    bindHinge(2, 1, [0,2], [0,1], 36, 0.13, 0.90);
    bindHinge(3, 2, [0,2], [0,1], 34, 0.11, 1.80);
    bindHinge(4, 3, [0,2], [0,1], 36, 0.12, 2.70);

    // Bottom chain: start from top face 0 across equator edge (v0,v1), then chain around
    bindHinge(5, 0, [1,2], [1,2], 44, 0.16, 0.50); // B0 about (v0,v1)
    bindHinge(6, 5, [1,2], [1,2], 42, 0.09, 1.35);
    bindHinge(7, 6, [1,2], [1,2], 43, 0.14, 2.10);
    bindHinge(8, 7, [1,2], [1,2], 41, 0.12, 2.85);
    bindHinge(9, 8, [1,2], [1,2], 42, 0.10, 3.60);

    this.faceOrder = [...Array(this.faces.length).keys()]; // 0..9

    // --- Geometry with duplicated vertices per face (keeps faces rigid/flat) ---
    const positions = new Float32Array(this.faces.length * 3 * 3);
    const normals   = new Float32Array(this.faces.length * 3 * 3);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('normal',   new THREE.BufferAttribute(normals,   3));
    geometry.setIndex([...Array(this.faces.length * 3).keys()]);
    geometry.computeBoundingSphere();

    // Physical material with iridescence (thin-film)
    const material = new THREE.MeshPhysicalMaterial({
      color: 0x151515,
      roughness: 0.26,
      metalness: 0.0,
      envMapIntensity: 1.2,
      iridescence: 1.0,
      iridescenceIOR: 1.3,
      iridescenceThicknessRange: [120, 620],
      flatShading: true
    });

    // Add tri-planar, domain-warped stripes + cellular overlay (emissive)
    this._injectOverlay(material);

    this.mesh = new THREE.Mesh(geometry, material);
    this.mesh.castShadow = false;
    this.mesh.receiveShadow = false;
    this.group.add(this.mesh);

    // Transforms per face
    this._M = this.faces.map(() => new THREE.Matrix4());
    this._mTA = new THREE.Matrix4();
    this._mTNegA = new THREE.Matrix4();
    this._mRot = new THREE.Matrix4();

    this.geometry = geometry;

    this.update(0, 1.0);
  }

  dispose() {
    this.geometry.dispose();
    if (this.mesh.material) this.mesh.material.dispose();
  }

  _injectOverlay(material) {
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
          uniform float uBandAngle;
          uniform float uBandSpeed;
          uniform float uBandFreq1;
          uniform float uBandFreq2;
          uniform float uBandAngle2;
          uniform float uBandStrength;
          uniform float uTriScale;
          uniform float uWarp;
          uniform float uCellAmp;
          uniform float uCellFreq;

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

  // Compose child so shared edge is pinned for any angle:
  //   Mchild = Mparent * T(Aw) * R(axis, angle) * T(-Aw) * bind
  _composePinned(Mparent, parentFace, hinge, angle, outM) {
    const Arest = parentFace.rest[hinge.parentEdge[0]].clone();
    const Brest = parentFace.rest[hinge.parentEdge[1]].clone();

    const Aw = Arest.clone().applyMatrix4(Mparent);
    const Bw = Brest.clone().applyMatrix4(Mparent);

    const axis = new THREE.Vector3().subVectors(Bw, Aw).normalize();

    this._mTA.makeTranslation(Aw.x, Aw.y, Aw.z);
    this._mTNegA.makeTranslation(-Aw.x, -Aw.y, -Aw.z);
    this._mRot.makeRotationAxis(axis, angle);

    outM.copy(Mparent)
        .multiply(this._mTA)
        .multiply(this._mRot)
        .multiply(this._mTNegA)
        .multiply(hinge.bind);
  }

  // t (seconds), speed (fold speed scalar)
  update(t, speed = 1.0) {
    const q = (f, p) => Math.sin(2 * Math.PI * (f * speed) * t + p);

    // Root transform = identity
    this._M[0].identity();

    // Child transforms (parent-first order)
    for (let i = 1; i < this.faceOrder.length; i++) {
      const id = this.faceOrder[i];
      const f  = this.faces[id];
      const h  = f.hinge;
      const Mparent = this._M[h.parent];

      const angle = h.amp * q(h.freq, h.phase)
                  + 0.10 * Math.sin(2 * Math.PI * (0.05 * speed) * t + id);
      this._composePinned(Mparent, this.faces[h.parent], h, angle, this._M[id]);
    }

    // Global gentle display yaw/pitch (not scaled by speed to keep scene calm)
    const yaw   = 0.22 * Math.sin(2 * Math.PI * 0.03 * t);
    const pitch = 0.17 * Math.sin(2 * Math.PI * 0.021 * t + 1.2);
    const Mglobal = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(pitch, yaw, 0, 'YXZ'));

    const posAttr = this.geometry.getAttribute('position');
    const nrmAttr = this.geometry.getAttribute('normal');
    const posArr  = posAttr.array;
    const nrmArr  = nrmAttr.array;

    // Pass 1: write world positions for each face's vertices
    for (let i = 0; i < this.faces.length; i++) {
      const M = new THREE.Matrix4().multiplyMatrices(Mglobal, this._M[i]);
      const fr = this.faces[i].rest;

      const w0 = this.faces[i].world[0].copy(fr[0]).applyMatrix4(M);
      const w1 = this.faces[i].world[1].copy(fr[1]).applyMatrix4(M);
      const w2 = this.faces[i].world[2].copy(fr[2]).applyMatrix4(M);

      const base = i * 3;
      posAttr.setXYZ(base + 0, w0.x, w0.y, w0.z);
      posAttr.setXYZ(base + 1, w1.x, w1.y, w1.z);
      posAttr.setXYZ(base + 2, w2.x, w2.y, w2.z);
    }

    // Pass 2 (weld): copy parent hinge vertices into child hinge vertices → zero gaps
    for (let i = 0; i < this.faces.length; i++) {
      const f = this.faces[i];
      if (!f.hinge) continue;
      const h = f.hinge;
      const parent = this.faces[h.parent];

      const pA = parent.world[h.parentEdge[0]];
      const pB = parent.world[h.parentEdge[1]];

      const base = i * 3;
      const ci0 = h.childEdge[0];
      const ci1 = h.childEdge[1];

      f.world[ci0].copy(pA);
      f.world[ci1].copy(pB);

      posAttr.setXYZ(base + ci0, pA.x, pA.y, pA.z);
      posAttr.setXYZ(base + ci1, pB.x, pB.y, pB.z);
    }

    // Pass 3: recompute flat normals from final welded positions
    const v0 = new THREE.Vector3(), v1 = new THREE.Vector3(), v2 = new THREE.Vector3();
    const e1 = new THREE.Vector3(), e2 = new THREE.Vector3(), n  = new THREE.Vector3();
    for (let i = 0; i < this.faces.length; i++) {
      const base = i * 3;
      const i0 = (base + 0) * 3;
      const i1 = (base + 1) * 3;
      const i2 = (base + 2) * 3;

      v0.set(posArr[i0], posArr[i0+1], posArr[i0+2]);
      v1.set(posArr[i1], posArr[i1+1], posArr[i1+2]);
      v2.set(posArr[i2], posArr[i2+1], posArr[i2+2]);

      e1.subVectors(v1, v0);
      e2.subVectors(v2, v0);
      n.copy(e1).cross(e2).normalize();

      nrmAttr.setXYZ(base + 0, n.x, n.y, n.z);
      nrmAttr.setXYZ(base + 1, n.x, n.y, n.z);
      nrmAttr.setXYZ(base + 2, n.x, n.y, n.z);
    }

    posAttr.needsUpdate = true;
    nrmAttr.needsUpdate = true;

    // Drive overlay time
    const uniforms = this.mesh.material.userData._overlayUniforms;
    if (uniforms) uniforms.uTime.value = t;
  }
}

// ---------- Create the object ----------
const deca = new PentagonalDipyramid();
scene.add(deca.group);

// ---------- Post-processing pipeline ----------
// Composer
const composer = new EffectComposer(renderer);
const renderPass = new RenderPass(scene, camera);
composer.addPass(renderPass);

// Trails / Afterimage (ghosting)
const afterimagePass = new AfterimagePass();
afterimagePass.enabled = false;                       // default off
afterimagePass.uniforms['damp'].value = 1.0;          // 1.0 ~ no trail
composer.addPass(afterimagePass);

// Glow / Bloom (UnrealBloomPass)
const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(container.clientWidth, container.clientHeight),
  0.0,   // strength (default 0)
  0.2,   // radius
  0.9    // threshold
);
bloomPass.enabled = false; // disabled when strength is 0
composer.addPass(bloomPass);

// RGB offset (chromatic aberration)
const rgbShiftPass = new ShaderPass(RGBShiftShader);
rgbShiftPass.enabled = false; // default off (amount 0)
rgbShiftPass.uniforms['amount'].value = 0.0; // typical useful range: 0.0005..0.01
rgbShiftPass.uniforms['angle'].value  = 0.0;
composer.addPass(rgbShiftPass);

// ---------- GUI (minimal, right panel) ----------
const uiHost = document.getElementById('ui');
const gui = new GUI({ title: 'Controls', width: 300 });
uiHost.appendChild(gui.domElement);

const params = {
  play: true,
  foldSpeed: 1.0,     // scales hinge oscillators (0..3)
  // Glow (bloom)
  bloomStrength: 0.0, // default 0 (off)
  bloomThreshold: 0.9,
  bloomRadius: 0.2,
  // Trails / Ghosting
  trailAmount: 0.0,   // 0..1; 0 off; higher = longer trails
  // RGB offset
  rgbAmount: 0.0,     // 0..0.01 typical
  rgbAngle: 0.0,      // degrees
  // Utilities
  exposure: renderer.toneMappingExposure,
  resetCamera: () => {
    camera.position.set(3.2, 1.8, 4.8);
    controls.target.set(0,0,0);
    controls.update();
  }
};

const fAnim = gui.addFolder('Animation');
fAnim.add(params, 'play').name('Play / Pause');
fAnim.add(params, 'foldSpeed', 0.0, 3.0, 0.01).name('Fold Speed');

const fGlow = gui.addFolder('Glow (Bloom)');
fGlow.add(params, 'bloomStrength', 0.0, 2.5, 0.01).name('Strength').onChange(v => {
  bloomPass.strength = v;
  bloomPass.enabled = v > 0.0;
});
fGlow.add(params, 'bloomThreshold', 0.0, 1.0, 0.001).name('Threshold').onChange(v => bloomPass.threshold = v);
fGlow.add(params, 'bloomRadius', 0.0, 1.0, 0.001).name('Radius').onChange(v => bloomPass.radius = v);

const fTrail = gui.addFolder('Trails (Ghosting)');
fTrail.add(params, 'trailAmount', 0.0, 1.0, 0.001).name('Amount').onChange(v => {
  // AfterimagePass uses 'damp' (0..1): lower damp → more trail.
  const damp = 1.0 - v * 0.98;
  afterimagePass.uniforms['damp'].value = damp;
  afterimagePass.enabled = v > 0.0;
});

const fRGB = gui.addFolder('RGB Offset');
fRGB.add(params, 'rgbAmount', 0.0, 0.02, 0.0001).name('Amount').onChange(v => {
  rgbShiftPass.uniforms['amount'].value = v;
  rgbShiftPass.enabled = v > 0.0;
});
fRGB.add(params, 'rgbAngle', 0.0, 180.0, 0.1).name('Angle (°)').onChange(v => {
  rgbShiftPass.uniforms['angle'].value = THREE.MathUtils.degToRad(v);
});

const fView = gui.addFolder('View');
fView.add(params, 'exposure', 0.6, 1.8, 0.01).name('Exposure').onChange(v => renderer.toneMappingExposure = v);
fView.add(params, 'resetCamera').name('Reset Camera');

// ---------- Render loop ----------
let t0 = performance.now();
function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  const t = (now - t0) / 1000;

  if (params.play) deca.update(t, params.foldSpeed);

  controls.update();
  composer.render();
}
animate();

// ---------- Resize handling ----------
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
