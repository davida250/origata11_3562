// Self‑Folding Decahedron — Sequential (closed start, one face moves at a time)
// Shapes: Pentagonal Dipyramid (10 triangles) / Pentagonal Trapezohedron (10 kites).
// Edges are pinned + welded; faces remain rigid and flat-shaded.
// Post: Bloom (default 0), Trails (Afterimage), RGB offset. Import-mapped modules; no bundler.

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

// ---------- Scene ----------
const container = document.getElementById('scene-container');

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.12;
renderer.outputColorSpace = THREE.SRGBColorSpace;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

// Camera / Controls
const camera = new THREE.PerspectiveCamera(36, container.clientWidth / container.clientHeight, 0.01, 100);
camera.position.set(3.0, 1.7, 4.4);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.minDistance = 2.0;
controls.maxDistance = 9.0;

// PMREM env for PBR / iridescence
const pmrem = new THREE.PMREMGenerator(renderer);
const envRT = pmrem.fromScene(new RoomEnvironment(), 0.04);
scene.environment = envRT.texture;

// ---------- Rigid hinge mesh ----------
class RigidHingeMesh {
  /**
   * @param {Object} spec
   *   faces:  Array<{ rest: THREE.Vector3[] }>
   *   hinges: Array<{ child, parent, parentEdge:[i,j], childEdge:[i,j], group:'top'|'bottom' }>
   */
  constructor(spec) {
    this.group = new THREE.Group();

    // Copy faces
    this.faces = spec.faces.map((f, id) => ({
      id,
      rest: f.rest.map(v => v.clone()),
      world: f.rest.map(v => v.clone()),
      parent: null,
      hinge: null
    }));

    // Triangulate faces (fan) -> geometry mapping
    this._geomMap = []; // geometry vertex -> { face, local }
    const triangles = [];
    for (let fi = 0; fi < this.faces.length; fi++) {
      const m = this.faces[fi].rest.length;
      for (let k = 1; k < m - 1; k++) {
        triangles.push([[fi,0],[fi,k],[fi,k+1]]);
        this._geomMap.push({ face: fi, local: 0 });
        this._geomMap.push({ face: fi, local: k });
        this._geomMap.push({ face: fi, local: k+1 });
      }
    }
    const triCount = triangles.length;

    // Geometry
    const positions = new Float32Array(triCount * 3 * 3);
    const normals   = new Float32Array(triCount * 3 * 3);
    const geometry  = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('normal',   new THREE.BufferAttribute(normals,   3));
    geometry.setIndex([...Array(triCount * 3).keys()]);
    geometry.computeBoundingSphere();

    // Base material (PBR + iridescence)
    const material = new THREE.MeshPhysicalMaterial({
      color: 0x151515,
      roughness: 0.26,
      metalness: 0.0,
      envMapIntensity: 1.15,
      iridescence: 1.0,
      iridescenceIOR: 1.3,
      iridescenceThicknessRange: [120, 620],
      flatShading: true
    });
    this._injectOverlay(material); // tri-planar domain-warped interference bands + cellular

    this.mesh = new THREE.Mesh(geometry, material);
    this.mesh.castShadow = this.mesh.receiveShadow = false;
    this.group.add(this.mesh);

    this.geometry = geometry;
    this.triCount = triCount;

    // Per-face transforms + helpers
    this._M = this.faces.map(() => new THREE.Matrix4());
    this._mTA = new THREE.Matrix4();
    this._mTNegA = new THREE.Matrix4();
    this._mRot = new THREE.Matrix4();

    // Bind hinges (ordered list kept for sequential driver)
    this.hinges = [];
    for (const h of spec.hinges) this._bindHinge(h);
    this.faceOrder = this._topoOrder();
  }

  dispose() { this.geometry.dispose(); if (this.mesh.material) this.mesh.material.dispose(); }

  // ---- Shader overlay (tri-planar interference bands + cellular, domain-warped) ----
  _injectOverlay(material) {
    const uniforms = {
      uTime:         { value: 0 },
      uBandAngle:    { value: THREE.MathUtils.degToRad(24.0) },
      uBandSpeed:    { value: 0.22 },
      uBandFreq1:    { value: 6.0 },
      uBandFreq2:    { value: 9.0 },
      uBandAngle2:   { value: THREE.MathUtils.degToRad(82.0) },
      uBandStrength: { value: 0.52 },
      uTriScale:     { value: 1.1 },
      uWarp:         { value: 0.55 },
      uCellAmp:      { value: 0.55 },
      uCellFreq:     { value: 2.6 }
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

  _pickThirdIndex(face, i, j) {
    const m = face.rest.length;
    for (let k = 0; k < m; k++) if (k !== i && k !== j) return k;
    return 0;
  }

  _bindHinge(h) {
    const child  = this.faces[h.child];
    const parent = this.faces[h.parent];

    const pA = parent.rest[h.parentEdge[0]].clone();
    const pB = parent.rest[h.parentEdge[1]].clone();
    const pT = parent.rest[this._pickThirdIndex(parent, h.parentEdge[0], h.parentEdge[1])].clone();

    const cA = child.rest[h.childEdge[0]].clone();
    const cB = child.rest[h.childEdge[1]].clone();
    const cT = child.rest[this._pickThirdIndex(child, h.childEdge[0], h.childEdge[1])].clone();

    const up = new THREE.Vector3().subVectors(pB, pA).normalize();
    const uc = new THREE.Vector3().subVectors(cB, cA).normalize();

    const q1 = new THREE.Quaternion().setFromUnitVectors(uc, up);

    const nP = new THREE.Vector3().subVectors(pB, pA).cross(new THREE.Vector3().subVectors(pT, pA)).normalize();
    const nC1 = new THREE.Vector3().subVectors(cB, cA).cross(new THREE.Vector3().subVectors(cT, cA)).applyQuaternion(q1).normalize();

    const proj = (n, axis) => n.clone().sub(axis.clone().multiplyScalar(n.dot(axis)));
    const nPp = proj(nP, up).normalize();
    const nCp = proj(nC1, up).normalize();

    const cross = new THREE.Vector3().crossVectors(nCp, nPp);
    const phi = Math.atan2(up.dot(cross), nCp.dot(nPp));
    const q2 = new THREE.Quaternion().setFromAxisAngle(up, phi);
    const q  = q2.multiply(q1);

    const pos = pA.clone().sub(cA.clone().applyQuaternion(q));
    const bind = new THREE.Matrix4().compose(pos, q, new THREE.Vector3(1,1,1));

    child.parent = h.parent;
    child.hinge  = { parent: h.parent, parentEdge: h.parentEdge.slice(0), childEdge: h.childEdge.slice(0), group: h.group, bind };
    this.hinges.push({ face: child.id, group: h.group }); // store order for sequential animation
  }

  _topoOrder() {
    const order = [];
    const visited = new Array(this.faces.length).fill(false);
    const dfs = (i) => {
      visited[i] = true;
      for (let j = 0; j < this.faces.length; j++) if (!visited[j] && this.faces[j].parent === i) dfs(j);
      order.push(i);
    };
    for (let i = 0; i < this.faces.length; i++) if (this.faces[i].parent === null) dfs(i);
    return order.reverse(); // parents before children
  }

  _composePinned(Mparent, parentFace, hinge, angle, outM) {
    // Mchild = Mparent * T(Aw) * R(axis,angle) * T(-Aw) * bind
    const Arest = parentFace.rest[hinge.parentEdge[0]].clone();
    const Brest = parentFace.rest[hinge.parentEdge[1]].clone();

    const Aw = Arest.clone().applyMatrix4(Mparent);
    const Bw = Brest.clone().applyMatrix4(Mparent);
    const axis = new THREE.Vector3().subVectors(Bw, Aw).normalize();

    this._mTA.makeTranslation(Aw.x, Aw.y, Aw.z);
    this._mTNegA.makeTranslation(-Aw.x, -Aw.y, -Aw.z);
    this._mRot.makeRotationAxis(axis, angle);

    outM.copy(Mparent).multiply(this._mTA).multiply(this._mRot).multiply(this._mTNegA).multiply(hinge.bind);
  }

  /**
   * Sequential driver: exactly one hinge opens->closes at a time.
   * @param {number} timeSec
   * @param {object} cfg { speed, foldMaxRad, stepSec, spinAmp }
   */
  update(timeSec, cfg) {
    const speed = cfg?.speed ?? 1;
    const foldMaxRad = cfg?.foldMaxRad ?? THREE.MathUtils.degToRad(20);
    const stepSec = Math.max(0.05, cfg?.stepSec ?? 1.2);    // per-face time window
    const spinAmp = cfg?.spinAmp ?? 0.0;                    // 0 = no global spin (keeps it self‑contained)

    // Root transforms
    for (let i = 0; i < this._M.length; i++) this._M[i].identity();

    // Active hinge + angle
    const smooth = (x) => x * x * (3 - 2 * x);             // smoothstep
    const t = timeSec * speed;
    const idx = Math.floor(t / stepSec) % this.hinges.length;
    const phase = (t % stepSec) / stepSec;                 // 0..1 within step
    const tri = 1 - Math.abs(2 * phase - 1);               // 0 -> 1 -> 0
    const a = foldMaxRad * smooth(tri);                    // 0..A..0

    // Angle resolver: only the active hinge moves; others are closed
    const active = this.hinges[idx];
    const angleForFace = (fid, grp) => (fid === active.face ? (grp === 'bottom' ? -a : +a) : 0);

    // Compose transforms (parents first)
    for (let i = 0; i < this.faceOrder.length; i++) {
      const id = this.faceOrder[i];
      const f  = this.faces[id];
      if (!f.hinge) continue;
      const h = f.hinge;
      const Mparent = this._M[h.parent];
      const angle = angleForFace(id, h.group || 'top');
      this._composePinned(Mparent, this.faces[h.parent], h, angle, this._M[id]);
    }

    // Global *very* small presentation spin (default 0)
    const yaw   = spinAmp * Math.sin(2 * Math.PI * 0.04 * timeSec);
    const pitch = spinAmp * Math.sin(2 * Math.PI * 0.033 * timeSec + 0.8);
    const Mglobal = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(pitch, yaw, 0, 'YXZ'));

    // Pass 1: world positions per face
    for (let i = 0; i < this.faces.length; i++) {
      const M = new THREE.Matrix4().multiplyMatrices(Mglobal, this._M[i]);
      const fr = this.faces[i].rest;
      for (let k = 0; k < fr.length; k++) this.faces[i].world[k].copy(fr[k]).applyMatrix4(M);
    }

    // Pass 2: weld shared edges (eliminate numeric drift)
    for (let i = 0; i < this.faces.length; i++) {
      const f = this.faces[i];
      if (!f.hinge) continue;
      const h = f.hinge;
      const parent = this.faces[h.parent];
      f.world[h.childEdge[0]].copy(parent.world[h.parentEdge[0]]);
      f.world[h.childEdge[1]].copy(parent.world[h.parentEdge[1]]);
    }

    // Pass 3: write to geometry
    const pos = this.geometry.getAttribute('position');
    for (let vi = 0; vi < this._geomMap.length; vi++) {
      const m = this._geomMap[vi];
      const w = this.faces[m.face].world[m.local];
      pos.setXYZ(vi, w.x, w.y, w.z);
    }
    pos.needsUpdate = true;

    // Pass 4: recompute flat normals
    const nrm = this.geometry.getAttribute('normal');
    const arr = pos.array, nArr = nrm.array;
    const v0 = new THREE.Vector3(), v1 = new THREE.Vector3(), v2 = new THREE.Vector3();
    const e1 = new THREE.Vector3(), e2 = new THREE.Vector3(), n  = new THREE.Vector3();
    for (let tIdx = 0; tIdx < this.triCount; tIdx++) {
      const i0 = (tIdx * 3 + 0) * 3, i1 = (tIdx * 3 + 1) * 3, i2 = (tIdx * 3 + 2) * 3;
      v0.set(arr[i0], arr[i0+1], arr[i0+2]);
      v1.set(arr[i1], arr[i1+1], arr[i1+2]);
      v2.set(arr[i2], arr[i2+1], arr[i2+2]);
      e1.subVectors(v1, v0); e2.subVectors(v2, v0); n.copy(e1).cross(e2).normalize();
      nArr[i0] = nArr[i1] = nArr[i2] = n.x;
      nArr[i0+1] = nArr[i1+1] = nArr[i2+1] = n.y;
      nArr[i0+2] = nArr[i1+2] = nArr[i2+2] = n.z;
    }
    nrm.needsUpdate = true;

    // Drive overlay time
    const uniforms = this.mesh.material.userData._overlayUniforms;
    if (uniforms) uniforms.uTime.value = timeSec;
  }
}

// ---------- Shape builders (compact decahedra) ----------
function buildPentagonalDipyramidSpec() {
  const R = 1.00, H = 1.08, jitter = 0.03;
  const top = new THREE.Vector3(0,  H, 0);
  const bot = new THREE.Vector3(0, -H, 0);
  const ring = [];
  for (let i = 0; i < 5; i++) {
    const a = (i / 5) * Math.PI * 2;
    const r = R * (1.0 + jitter * (i % 2 ? -1 : 1));
    ring.push(new THREE.Vector3(Math.cos(a) * r, 0, Math.sin(a) * r));
  }

  const faces = [];
  for (let i = 0; i < 5; i++) faces.push({ rest: [top.clone(), ring[i].clone(), ring[(i+1)%5].clone()] });
  for (let i = 0; i < 5; i++) faces.push({ rest: [bot.clone(), ring[(i+1)%5].clone(), ring[i].clone()] });

  const hinges = [];
  for (let i = 1; i < 5; i++) hinges.push({ child: i, parent: i-1, parentEdge: [0,2], childEdge: [0,1], group: 'top' });
  hinges.push({ child: 5, parent: 0, parentEdge: [1,2], childEdge: [1,2], group: 'bottom' });
  for (let k = 6; k < 10; k++) hinges.push({ child: k, parent: k-1, parentEdge: [1,2], childEdge: [1,2], group: 'bottom' });

  return { faces, hinges };
}

function buildPentagonalTrapezohedronSpec() {
  const H = 1.02, h = 0.42, R1 = 0.96, R2 = 1.04, off = Math.PI / 5;
  const N = new THREE.Vector3(0,  H, 0), S = new THREE.Vector3(0, -H, 0);
  const topR = [], botR = [];
  for (let i = 0; i < 5; i++) {
    const aTop = (i / 5) * Math.PI * 2;
    const aBot = aTop + off;
    topR.push(new THREE.Vector3(Math.cos(aTop)*R1,  h, Math.sin(aTop)*R1));
    botR.push(new THREE.Vector3(Math.cos(aBot)*R2, -h, Math.sin(aBot)*R2));
  }

  const faces = [];
  for (let i = 0; i < 5; i++) faces.push({ rest: [N.clone(), topR[i].clone(), botR[i].clone(), topR[(i+1)%5].clone()] });
  for (let i = 0; i < 5; i++) faces.push({ rest: [S.clone(), botR[i].clone(), topR[i].clone(), botR[(i+1)%5].clone()] });

  const hinges = [];
  for (let i = 1; i < 5; i++) hinges.push({ child: i, parent: i-1, parentEdge: [0,3], childEdge: [0,1], group: 'top' });
  hinges.push({ child: 5, parent: 0, parentEdge: [1,2], childEdge: [1,2], group: 'bottom' });
  for (let k = 6; k < 10; k++) hinges.push({ child: k, parent: k-1, parentEdge: [0,3], childEdge: [0,1], group: 'bottom' });

  return { faces, hinges };
}

// ---------- Shape instance management ----------
let shape = null;
function makeShape(kind) {
  return new RigidHingeMesh(kind === 'trapezohedron' ? buildPentagonalTrapezohedronSpec()
                                                     : buildPentagonalDipyramidSpec());
}
function switchShape(kind) {
  if (shape) { scene.remove(shape.group); shape.dispose(); shape = null; }
  shape = makeShape(kind);
  scene.add(shape.group);

  // Force an initial closed frame (t=0) so the scene starts closed.
  shape.update(0, {
    speed: 0,
    foldMaxRad: THREE.MathUtils.degToRad(params.foldMaxDeg),
    stepSec: params.stepTime,
    spinAmp: params.spinAmplitude
  });
}

// ---------- Post‑processing ----------
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));

const afterimagePass = new AfterimagePass();
afterimagePass.enabled = false;              // default off (trails)
afterimagePass.uniforms['damp'].value = 1.0; // 1.0 ~ no trails
composer.addPass(afterimagePass);

const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(container.clientWidth, container.clientHeight),
  0.0,  // strength (0 => disabled)
  0.2,  // radius
  0.9   // threshold
);
bloomPass.enabled = false;
composer.addPass(bloomPass);

const rgbShiftPass = new ShaderPass(RGBShiftShader);
rgbShiftPass.enabled = false;
rgbShiftPass.uniforms['amount'].value = 0.0;
rgbShiftPass.uniforms['angle'].value  = 0.0;
composer.addPass(rgbShiftPass);

// ---------- GUI ----------
const uiHost = document.getElementById('ui');
const gui = new GUI({ title: 'Controls', width: 330 });
uiHost.appendChild(gui.domElement);

const params = {
  objectType: 'dipyramid',              // 'dipyramid' | 'trapezohedron'
  play: true,

  // Sequential folding
  foldSpeed: 1.0,                       // global time scale (up to 20×)
  foldMaxDeg: 22,                       // peak angle for the active hinge (°)
  stepTime: 1.2,                        // seconds per face (open->close)
  spinAmplitude: 0.0,                   // small presentation spin (0 = off, default)

  // Glow (Bloom)
  bloomStrength: 0.0,                   // default 0
  bloomThreshold: 0.9,
  bloomRadius: 0.2,

  // Trails
  trailAmount: 0.0,                     // 0..1 (0 = off)

  // RGB offset
  rgbAmount: 0.0,                       // 0..0.02 typical
  rgbAngle: 0.0,                        // degrees

  // View
  exposure: renderer.toneMappingExposure,
  resetCamera: () => { camera.position.set(3.0, 1.7, 4.4); controls.target.set(0,0,0); controls.update(); }
};

const fShape = gui.addFolder('Shape');
fShape.add(params, 'objectType', {
  'Pentagonal Dipyramid (10 triangles)': 'dipyramid',
  'Pentagonal Trapezohedron (10 kites)': 'trapezohedron'
}).name('Object Type').onChange(v => switchShape(v));

const fFold = gui.addFolder('Folding (Sequential)');
fFold.add(params, 'foldSpeed', 0.0, 20.0, 0.01).name('Speed (×)');
fFold.add(params, 'foldMaxDeg', 0.0, 60.0, 0.1).name('Peak Angle (°)');
fFold.add(params, 'stepTime', 0.05, 6.0, 0.01).name('Step Time (s)');
fFold.add(params, 'spinAmplitude', 0.0, 0.3, 0.001).name('Spin Amplitude').listen();

const fGlow = gui.addFolder('Glow (Bloom)');
fGlow.add(params, 'bloomStrength', 0.0, 2.5, 0.01).name('Strength').onChange(v => { bloomPass.strength = v; bloomPass.enabled = v > 0.0; });
fGlow.add(params, 'bloomThreshold', 0.0, 1.0, 0.001).name('Threshold').onChange(v => bloomPass.threshold = v);
fGlow.add(params, 'bloomRadius', 0.0, 1.0, 0.001).name('Radius').onChange(v => bloomPass.radius = v);

const fTrail = gui.addFolder('Trails (Ghosting)');
fTrail.add(params, 'trailAmount', 0.0, 1.0, 0.001).name('Amount').onChange(v => {
  const damp = 1.0 - v * 0.98;
  afterimagePass.uniforms['damp'].value = damp;
  afterimagePass.enabled = v > 0.0;
});

const fRGB = gui.addFolder('RGB Offset');
fRGB.add(params, 'rgbAmount', 0.0, 0.02, 0.0001).name('Amount').onChange(v => { rgbShiftPass.uniforms['amount'].value = v; rgbShiftPass.enabled = v > 0.0; });
fRGB.add(params, 'rgbAngle', 0.0, 180.0, 0.1).name('Angle (°)').onChange(v => { rgbShiftPass.uniforms['angle'].value = THREE.MathUtils.degToRad(v); });

const fView = gui.addFolder('View');
fView.add(params, 'exposure', 0.6, 1.8, 0.01).name('Exposure').onChange(v => renderer.toneMappingExposure = v);
fView.add(params, 'resetCamera').name('Reset Camera');

// ---------- Boot + loop ----------
switchShape(params.objectType); // creates the mesh and commits a closed first frame

let t0 = performance.now();
function animate() {
  requestAnimationFrame(animate);
  const t = (performance.now() - t0) / 1000;

  if (params.play && shape) {
    shape.update(t, {
      speed: params.foldSpeed,
      foldMaxRad: THREE.MathUtils.degToRad(params.foldMaxDeg),
      stepSec: params.stepTime,
      spinAmp: params.spinAmplitude
    });
  }

  controls.update();
  composer.render();
}
animate();

// Resize
function onResize() {
  const w = container.clientWidth, h = container.clientHeight;
  camera.aspect = w / h; camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  composer.setSize(w, h);
  bloomPass.setSize(w, h);
}
window.addEventListener('resize', onResize);
new ResizeObserver(onResize).observe(container);
