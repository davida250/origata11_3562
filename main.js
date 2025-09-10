// Self‑Folding Decahedron (compact, no flailing)
// - Shapes: Pentagonal Dipyramid (10 triangles), Pentagonal Trapezohedron (10 kites)
// - Faces are rigid; hinges are pinned and edges are welded => no gaps.
// - One‑parameter "self‑fold" driver: all top hinges use +alpha(t), bottoms use −alpha(t).
// - Right‑panel controls: Shape, Fold Speed, Fold Range, Glow (Bloom, default 0), Trails (Afterimage), RGB Offset.
// - Material: MeshPhysicalMaterial with iridescence + tri‑planar domain‑warped interference overlay.

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

// Neutral environment (PMREM) for PBR / iridescence
const pmrem = new THREE.PMREMGenerator(renderer);
const envRT = pmrem.fromScene(new RoomEnvironment(), 0.04);
scene.environment = envRT.texture;

// ---------- Rigid hinge mesh (generic; supports triangles or quads (kites)) ----------
class RigidHingeMesh {
  /**
   * @param {Object} spec
   *   faces: Array<{ rest: THREE.Vector3[] }>
   *   hinges: Array<{ child, parent, parentEdge:[i,j], childEdge:[i,j], group:'top'|'bottom' }>
   */
  constructor(spec) {
    this.group = new THREE.Group();
    // Make a deep copy of faces
    this.faces = spec.faces.map((f, id) => ({
      id,
      rest: f.rest.map(v => v.clone()),
      world: f.rest.map(v => v.clone()),
      parent: null,
      hinge: null
    }));

    // Triangulate each face (fan) so we can render rigid quads as one planar face
    this._geomMap = []; // each entry maps geometry vertex -> { face, local }
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

    // Geometry buffers
    const triCount = triangles.length;
    const positions = new Float32Array(triCount * 3 * 3);
    const normals   = new Float32Array(triCount * 3 * 3);
    const geometry  = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('normal',   new THREE.BufferAttribute(normals,   3));
    geometry.setIndex([...Array(triCount * 3).keys()]);
    geometry.computeBoundingSphere();

    // Physical material with iridescence (thin‑film) — see three.js docs
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
    this._injectOverlay(material); // adds tri‑planar interference emissive overlay

    this.mesh = new THREE.Mesh(geometry, material);
    this.mesh.castShadow = this.mesh.receiveShadow = false;
    this.group.add(this.mesh);

    this.geometry = geometry;
    this.triCount = triCount;

    // Build hinges with bind transforms so edges are pinned at angle=0
    this._M = this.faces.map(() => new THREE.Matrix4());
    this._mTA = new THREE.Matrix4();
    this._mTNegA = new THREE.Matrix4();
    this._mRot = new THREE.Matrix4();

    for (const h of spec.hinges) this._bindHinge(h);
    this.faceOrder = this._topoOrder();

    this.update(0, 1, {min: 8*Math.PI/180, max: 22*Math.PI/180}); // initial fold
  }

  dispose() { this.geometry.dispose(); if (this.mesh.material) this.mesh.material.dispose(); }

  // ----- Material overlay (iridescent‑friendly bands + cellular, tri‑planar, domain‑warped) -----
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
  }

  _topoOrder() {
    const order = [];
    const visited = new Array(this.faces.length).fill(false);
    const dfs = (i) => {
      visited[i] = true;
      for (let j = 0; j < this.faces.length; j++) {
        if (!visited[j] && this.faces[j].parent === i) dfs(j);
      }
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

  // Single‑parameter fold: angles for 'top' group = +alpha(t); for 'bottom' group = −alpha(t).
  update(t, speed = 1.0, foldRange = {min: 8*Math.PI/180, max: 22*Math.PI/180}) {
    const smoothstep = (x) => x*x*(3 - 2*x);
    const q01 = 0.5 + 0.5 * Math.sin(2 * Math.PI * (0.10 * speed) * t); // 0..1
    const s = smoothstep(q01);
    const alpha = THREE.MathUtils.lerp(foldRange.min, foldRange.max, s);
    const beta  = -alpha;

    // Reset transforms (roots first)
    for (let i = 0; i < this._M.length; i++) this._M[i].identity();

    // Compose child transforms hierarchically
    for (let i = 0; i < this.faceOrder.length; i++) {
      const id = this.faceOrder[i];
      const f  = this.faces[id];
      if (!f.hinge) continue;

      const h = f.hinge;
      const Mparent = this._M[h.parent];
      const angle = (h.group === 'bottom') ? beta : alpha; // default 'top' if not specified
      this._composePinned(Mparent, this.faces[h.parent], h, angle, this._M[id]);
    }

    // Gentle presentation (very small so the object stays "self‑contained")
    const yaw   = 0.12 * Math.sin(2 * Math.PI * 0.03 * t);
    const pitch = 0.10 * Math.sin(2 * Math.PI * 0.021 * t + 0.8);
    const Mglobal = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(pitch, yaw, 0, 'YXZ'));

    // Pass 1: transform all face-local vertices to world
    for (let i = 0; i < this.faces.length; i++) {
      const M = new THREE.Matrix4().multiplyMatrices(Mglobal, this._M[i]);
      const fr = this.faces[i].rest;
      for (let k = 0; k < fr.length; k++) this.faces[i].world[k].copy(fr[k]).applyMatrix4(M);
    }

    // Pass 2: weld shared hinge vertices (kill numeric drift)
    for (let i = 0; i < this.faces.length; i++) {
      const f = this.faces[i];
      if (!f.hinge) continue;
      const h = f.hinge;
      const parent = this.faces[h.parent];
      f.world[h.childEdge[0]].copy(parent.world[h.parentEdge[0]]);
      f.world[h.childEdge[1]].copy(parent.world[h.parentEdge[1]]);
    }

    // Pass 3: write positions to geometry and recompute flat normals
    const pos = this.geometry.getAttribute('position');
    for (let vi = 0; vi < this._geomMap.length; vi++) {
      const m = this._geomMap[vi];
      const w = this.faces[m.face].world[m.local];
      pos.setXYZ(vi, w.x, w.y, w.z);
    }
    pos.needsUpdate = true;

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
      nArr[i0] = nArr[i1] = nArr[i2] = n.x; nArr[i0+1] = nArr[i1+1] = nArr[i2+1] = n.y; nArr[i0+2] = nArr[i1+2] = nArr[i2+2] = n.z;
    }
    nrm.needsUpdate = true;

    // Drive overlay animation
    const uniforms = this.mesh.material.userData._overlayUniforms;
    if (uniforms) uniforms.uTime.value = t;
  }
}

// ---------- Shape builders (compact, self‑contained) ----------
function buildPentagonalDipyramidSpec() {
  // 10 triangles (convex) => decahedron; slight irregularity keeps it “organic”
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
  for (let i = 0; i < 5; i++) faces.push({ rest: [top.clone(), ring[i].clone(), ring[(i+1)%5].clone()] }); // top 5
  for (let i = 0; i < 5; i++) faces.push({ rest: [bot.clone(), ring[(i+1)%5].clone(), ring[i].clone()] }); // bottom 5

  const hinges = [];
  // Root: face 0 (top wedge). Short chains to minimize compounding.
  // Top wedges: each hinged to the previous around the apex edge [top, ring[i]].
  for (let i = 1; i < 5; i++) {
    hinges.push({ child: i, parent: i-1, parentEdge: [0,2], childEdge: [0,1], group: 'top' });
  }
  // Bottom wedges: start from face5 hinged to face0 across equator edge [ring0, ring1] and chain.
  hinges.push({ child: 5, parent: 0, parentEdge: [1,2], childEdge: [1,2], group: 'bottom' });
  for (let k = 6; k < 10; k++) {
    hinges.push({ child: k, parent: k-1, parentEdge: [1,2], childEdge: [1,2], group: 'bottom' });
  }

  return { faces, hinges };
}

function buildPentagonalTrapezohedronSpec() {
  // 10 kites (convex) => decahedron (D10 dice shape)
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
  for (let i = 0; i < 5; i++) faces.push({ rest: [N.clone(), topR[i].clone(), botR[i].clone(), topR[(i+1)%5].clone()] }); // upper kites
  for (let i = 0; i < 5; i++) faces.push({ rest: [S.clone(), botR[i].clone(), topR[i].clone(), botR[(i+1)%5].clone()] }); // lower kites

  const hinges = [];
  // Root = face 0. Short chains around the pole for compact motion.
  for (let i = 1; i < 5; i++) hinges.push({ child: i, parent: i-1, parentEdge: [0,3], childEdge: [0,1], group: 'top' });
  hinges.push({ child: 5, parent: 0, parentEdge: [1,2], childEdge: [1,2], group: 'bottom' });
  for (let k = 6; k < 10; k++) hinges.push({ child: k, parent: k-1, parentEdge: [0,3], childEdge: [0,1], group: 'bottom' });

  return { faces, hinges };
}

// ---------- Instance management ----------
let shape = null;
function makeShape(kind) {
  return new RigidHingeMesh(kind === 'trapezohedron' ? buildPentagonalTrapezohedronSpec()
                                                     : buildPentagonalDipyramidSpec());
}
function switchShape(kind) {
  if (shape) { scene.remove(shape.group); shape.dispose(); shape = null; }
  shape = makeShape(kind);
  scene.add(shape.group);
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
  0.0,  // strength (default 0 => disabled)
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
  foldSpeed: 1.0,                       // scales the 1‑DOF driver
  foldMin: 8,                           // degrees (compactness)
  foldMax: 22,                          // degrees

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

const fAnim = gui.addFolder('Folding');
fAnim.add(params, 'play').name('Play / Pause');
fAnim.add(params, 'foldSpeed', 0.0, 3.0, 0.01).name('Speed');
fAnim.add(params, 'foldMin', 0.0, 45.0, 0.1).name('Range Min (°)');
fAnim.add(params, 'foldMax', 1.0, 60.0, 0.1).name('Range Max (°)');

const fGlow = gui.addFolder('Glow (Bloom)');
fGlow.add(params, 'bloomStrength', 0.0, 2.5, 0.01).name('Strength').onChange(v => { bloomPass.strength = v; bloomPass.enabled = v > 0.0; });
fGlow.add(params, 'bloomThreshold', 0.0, 1.0, 0.001).name('Threshold').onChange(v => bloomPass.threshold = v);
fGlow.add(params, 'bloomRadius', 0.0, 1.0, 0.001).name('Radius').onChange(v => bloomPass.radius = v);

const fTrail = gui.addFolder('Trails (Ghosting)');
fTrail.add(params, 'trailAmount', 0.0, 1.0, 0.001).name('Amount').onChange(v => {
  const damp = 1.0 - v * 0.98; // lower damp -> longer trails
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
switchShape(params.objectType);

let t0 = performance.now();
function animate() {
  requestAnimationFrame(animate);
  const t = (performance.now() - t0) / 1000;

  if (params.play && shape) {
    const foldRange = { min: THREE.MathUtils.degToRad(params.foldMin), max: THREE.MathUtils.degToRad(params.foldMax) };
    shape.update(t, params.foldSpeed, foldRange);
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
