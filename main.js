// Endless Folding Polyhedra — Dodecahedron-first
// Full replacement for prior script. Starts on a regular dodecahedron (12 pentagons),
// and keeps the origami-like, rigid-hinge animation system.
// Provenance (user uploads): HTML shell :contentReference[oaicite:2]{index=2} • Original code baseline :contentReference[oaicite:3]{index=3}
//
// Key changes:
//   • Added buildDodecahedronSpec() which reconstructs pentagonal faces + hinges from THREE.DodecahedronGeometry.
//   • Shape selector now includes "Dodecahedron (12 pentagons)" and defaults to it.
//   • Hinge tree is built via a BFS spanning tree over face adjacency, with alternating hinge groups for symmetric folding.
//
// Notes:
//   • Faces are rigid and planar; edges are welded (pinned) — evoking paper origami creases.
//   • In "Self-fold" mode, all hinges move in one DOF with mirrored top/bottom groups.
//   • In "Free" mode, each hinge has its own amplitude/frequency/phase for playful motion.

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
renderer.toneMappingExposure = 1.18;
renderer.outputColorSpace = THREE.SRGBColorSpace;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

// Camera + controls
const camera = new THREE.PerspectiveCamera(36, container.clientWidth / container.clientHeight, 0.01, 100);
camera.position.set(3.0, 1.9, 5.0);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.minDistance = 2.0;
controls.maxDistance = 9.0;

// Environment (PMREM + RoomEnvironment)
const pmrem = new THREE.PMREMGenerator(renderer);
const envRT = pmrem.fromScene(new RoomEnvironment(), 0.04);
scene.environment = envRT.texture;

// ---------- Generic Rigid Hinge Mesh (supports arbitrary polygonal faces) ----------
class RigidHingeMesh {
  /**
   * @param {Object} spec
   *   spec.faces: Array<{ rest: THREE.Vector3[], tag?: string }>
   *   spec.hinges: Array<{ child:number, parent:number, parentEdge:[number,number], childEdge:[number,number], ampDeg:number, freq:number, phase:number, group?:'top'|'bottom' }>
   *   spec.material?: THREE.Material
   */
  constructor(spec) {
    this.group = new THREE.Group();

    // Deep copy face data
    this.faces = spec.faces.map((f, id) => ({
      id,
      tag: f.tag || '',
      rest: f.rest.map(v => v.clone()),
      world: f.rest.map(v => v.clone()),
      parent: null,
      hinge: null
    }));

    // Triangulate each face (fan) and map geometry vertices -> (face, local index)
    this._geomMap = [];
    const tris = [];
    for (let fi = 0; fi < this.faces.length; fi++) {
      const m = this.faces[fi].rest.length;
      for (let k = 1; k < m - 1; k++) {
        tris.push([[fi, 0], [fi, k], [fi, k + 1]]);
        this._geomMap.push({ face: fi, local: 0 });
        this._geomMap.push({ face: fi, local: k });
        this._geomMap.push({ face: fi, local: k + 1 });
      }
    }

    // Create geometry
    const triCount = tris.length;
    const positions = new Float32Array(triCount * 3 * 3);
    const normals   = new Float32Array(triCount * 3 * 3);
    const geometry  = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('normal',   new THREE.BufferAttribute(normals,   3));
    geometry.setIndex([...Array(triCount * 3).keys()]);
    geometry.computeBoundingSphere();

    // Material (physical with iridescence); overlay added later
    const mat = (spec.material instanceof THREE.Material) ? spec.material : new THREE.MeshPhysicalMaterial({
      color: 0x151515,
      roughness: 0.26,
      metalness: 0.0,
      envMapIntensity: 1.2,
      iridescence: 1.0,
      iridescenceIOR: 1.3,
      iridescenceThicknessRange: [120, 620],
      flatShading: true
    });

    this._injectOverlay(mat);

    const mesh = new THREE.Mesh(geometry, mat);
    mesh.castShadow = mesh.receiveShadow = false;
    this.group.add(mesh);

    this.mesh = mesh;
    this.geometry = geometry;
    this.triCount = triCount;

    // Working matrices
    this._M = this.faces.map(() => new THREE.Matrix4());
    this._mTA = new THREE.Matrix4();
    this._mTNegA = new THREE.Matrix4();
    this._mRot = new THREE.Matrix4();

    // Attach hinges
    const hinges = spec.hinges || [];
    for (const h of hinges) this._bindHinge(h);

    // Order (parent before child)
    this.faceOrder = this._topoOrder();

    this.update(0, 1.0);
  }

  dispose() {
    this.geometry.dispose();
    if (this.mesh.material) this.mesh.material.dispose();
  }

  // ---------- Overlay (tri-planar, domain-warped interference + cellular) ----------
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
    child.hinge  = {
      parent: h.parent,
      parentEdge: h.parentEdge.slice(0),
      childEdge:  h.childEdge.slice(0),
      amp: THREE.MathUtils.degToRad(h.ampDeg ?? 36),
      freq: h.freq ?? 0.12,
      phase: h.phase ?? 0.0,
      group: h.group ?? 'top',
      bind
    };
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
    return order.reverse();
  }

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

  update(t, speed = 1.0) {
    const ease = (x) => x*x*(3 - 2*x);
    const q01 = 0.5 + 0.5 * Math.sin(2 * Math.PI * (0.10 * speed) * t);
    const s   = ease(q01);
    const aMin = THREE.MathUtils.degToRad(params.foldMinDeg);
    const aMax = THREE.MathUtils.degToRad(params.foldMaxDeg);
    const alpha = THREE.MathUtils.lerp(aMin, aMax, s);
    const beta  = -alpha;
    const q = (f, p) => Math.sin(2 * Math.PI * (f * speed) * t + p);

    for (let i = 0; i < this._M.length; i++) this._M[i].identity();

    for (let i = 0; i < this.faceOrder.length; i++) {
      const id = this.faceOrder[i];
      const f  = this.faces[id];
      if (!f.hinge) continue;

      const h = f.hinge;
      const Mparent = this._M[h.parent];

      let angle;
      if (params.foldMode === 'self-fold') {
        const g = h.group || 'top';
        angle = (g === 'bottom') ? beta : alpha;
      } else {
        angle = h.amp * q(h.freq, h.phase) + 0.10 * Math.sin(2 * Math.PI * (0.05 * speed) * t + id);
      }

      this._composePinned(Mparent, this.faces[h.parent], h, angle, this._M[id]);
    }

    // Gentle presentation rotation
    const yaw   = 0.22 * Math.sin(2 * Math.PI * 0.03 * t);
    const pitch = 0.17 * Math.sin(2 * Math.PI * 0.021 * t + 1.2);
    const Mglobal = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(pitch, yaw, 0, 'YXZ'));

    // Pass 1: transform face vertices
    for (let i = 0; i < this.faces.length; i++) {
      const M = new THREE.Matrix4().multiplyMatrices(Mglobal, this._M[i]);
      const fr = this.faces[i].rest;
      for (let k = 0; k < fr.length; k++) {
        this.faces[i].world[k].copy(fr[k]).applyMatrix4(M);
      }
    }

    // Pass 2: weld hinge vertices from parent to child
    for (let i = 0; i < this.faces.length; i++) {
      const f = this.faces[i];
      if (!f.hinge) continue;
      const h = f.hinge;
      const parent = this.faces[h.parent];

      const pA = parent.world[h.parentEdge[0]];
      const pB = parent.world[h.parentEdge[1]];

      f.world[h.childEdge[0]].copy(pA);
      f.world[h.childEdge[1]].copy(pB);
    }

    // Pass 3: write positions
    const posAttr = this.geometry.getAttribute('position');
    for (let gv = 0; gv < this._geomMap.length; gv++) {
      const m = this._geomMap[gv];
      const w = this.faces[m.face].world[m.local];
      posAttr.setXYZ(gv, w.x, w.y, w.z);
    }
    posAttr.needsUpdate = true;

    // Pass 4: recompute flat normals
    const nrmAttr = this.geometry.getAttribute('normal');
    const arr = posAttr.array;
    const nArr = nrmAttr.array;
    const v0 = new THREE.Vector3(), v1 = new THREE.Vector3(), v2 = new THREE.Vector3();
    const e1 = new THREE.Vector3(), e2 = new THREE.Vector3(), n  = new THREE.Vector3();
    for (let tIdx = 0; tIdx < this.triCount; tIdx++) {
      const i0 = (tIdx * 3 + 0) * 3;
      const i1 = (tIdx * 3 + 1) * 3;
      const i2 = (tIdx * 3 + 2) * 3;
      v0.set(arr[i0], arr[i0+1], arr[i0+2]);
      v1.set(arr[i1], arr[i1+1], arr[i1+2]);
      v2.set(arr[i2], arr[i2+1], arr[i2+2]);
      e1.subVectors(v1, v0);
      e2.subVectors(v2, v0);
      n.copy(e1).cross(e2).normalize();
      nArr[i0] = nArr[i1] = nArr[i2] = n.x;
      nArr[i0+1] = nArr[i1+1] = nArr[i2+1] = n.y;
      nArr[i0+2] = nArr[i1+2] = nArr[i2+2] = n.z;
    }
    nrmAttr.needsUpdate = true;

    const uniforms = this.mesh.material.userData._overlayUniforms;
    if (uniforms) uniforms.uTime.value = t;
  }
}

// ---------- Dodecahedron builder (12 pentagons) ----------
function buildDodecahedronSpec() {
  // Source geometry (detail 0). We'll reconstruct pentagon faces from its triangles.
  const g = new THREE.DodecahedronGeometry(1.0, 0);

  const posAttr = g.getAttribute('position');
  const idxAttr = g.getIndex();
  const positions = [];
  for (let i = 0; i < posAttr.count; i++) {
    positions.push(new THREE.Vector3().fromArray(posAttr.array, i * 3));
  }
  const indices = Array.from(idxAttr.array);

  const quant = (x, q = 1e4) => Math.round(x * q) / q;

  // Group triangles by (plane normal, plane constant) to get 12 planar clusters
  const clusters = new Map();
  for (let t = 0; t < indices.length; t += 3) {
    const i0 = indices[t], i1 = indices[t+1], i2 = indices[t+2];
    const p0 = positions[i0], p1 = positions[i1], p2 = positions[i2];

    const n = new THREE.Vector3().subVectors(p1, p0).cross(new THREE.Vector3().subVectors(p2, p0)).normalize();
    const d = -n.dot(p0);

    const key = `${quant(n.x)},${quant(n.y)},${quant(n.z)}|${quant(d)}`;
    if (!clusters.has(key)) clusters.set(key, { normal: n.clone(), tris: [], verts: new Set() });
    const c = clusters.get(key);
    c.tris.push([i0, i1, i2]);
    c.verts.add(i0); c.verts.add(i1); c.verts.add(i2);
  }

  // Build faces as ordered CCW pentagons
  const faces = [];
  const faceIndices = []; // local->global vertex index map per face
  const faceCenters = [];
  const scale = 1.15; // make it nicely visible

  for (const c of clusters.values()) {
    const uniq = Array.from(c.verts);
    if (uniq.length !== 5) continue; // safety; for dodecahedron all should be 5

    // Order the 5 vertices CCW in the plane
    // Basis on the plane
    const n = c.normal.clone().normalize();
    const centroid = uniq.reduce((acc, i) => acc.add(positions[i]), new THREE.Vector3()).multiplyScalar(1 / uniq.length);
    const ref = (Math.abs(n.y) < 0.99) ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
    const u = new THREE.Vector3().crossVectors(ref, n).normalize();
    const v = new THREE.Vector3().crossVectors(n, u);

    const ordered = uniq.map((idx) => {
      const p = positions[idx];
      const r = p.clone().sub(centroid);
      const ang = Math.atan2(r.dot(v), r.dot(u));
      return { idx, ang };
    }).sort((a, b) => a.ang - b.ang).map(o => o.idx);

    const faceVerts = ordered.map(i => positions[i].clone().multiplyScalar(scale));
    faces.push({ rest: faceVerts, tag: 'D' });
    faceIndices.push(ordered);
    faceCenters.push(centroid.clone().multiplyScalar(scale));
  }

  // Adjacency via shared edges (two faces per edge)
  const edgeMap = new Map(); // key: "min|max" -> [{face, local:[i,i+1], glob:[a,b]}...]
  for (let f = 0; f < faceIndices.length; f++) {
    const order = faceIndices[f];
    const m = order.length;
    for (let k = 0; k < m; k++) {
      const a = order[k], b = order[(k + 1) % m];
      const key = (a < b) ? `${a}|${b}` : `${b}|${a}`;
      if (!edgeMap.has(key)) edgeMap.set(key, []);
      edgeMap.get(key).push({ face: f, local: [k, (k + 1) % m], glob: [a, b] });
    }
  }

  // Build neighbor lists
  const neighbors = Array.from({ length: faces.length }, () => []);
  for (const [_, list] of edgeMap) {
    if (list.length === 2) {
      const A = list[0], B = list[1];
      neighbors[A.face].push({ other: B.face, parentEdge: A.local, childLocal: B.local, gParent: A.glob, gChild: B.glob });
      neighbors[B.face].push({ other: A.face, parentEdge: B.local, childLocal: A.local, gParent: B.glob, gChild: A.glob });
    }
  }

  // Choose a root: the face with the highest center.y
  let root = 0, maxY = -Infinity;
  for (let i = 0; i < faceCenters.length; i++) {
    const y = faceCenters[i].y;
    if (y > maxY) { maxY = y; root = i; }
  }

  // BFS spanning tree to create hinges (avoid cycles)
  const visited = new Array(faces.length).fill(false);
  const depth   = new Array(faces.length).fill(0);
  const hinges = [];

  const q = [root];
  visited[root] = true; depth[root] = 0;

  while (q.length) {
    const parent = q.shift();
    for (const link of neighbors[parent]) {
      const child = link.other;
      if (visited[child]) continue;

      // Make child's local edge orientation match parent's global edge order
      const [ga, gb] = link.gParent;
      const [gc, gd] = link.gChild;
      const childEdge = (gc === ga && gd === gb) ? [link.childLocal[0], link.childLocal[1]]
                                                 : [link.childLocal[1], link.childLocal[0]];

      const group = (depth[parent] % 2 === 0) ? 'top' : 'bottom';
      hinges.push({
        child,
        parent,
        parentEdge: [link.parentEdge[0], link.parentEdge[1]],
        childEdge,
        group,
        ampDeg: 40,                 // used in "Free" mode
        freq: 0.12 + 0.02 * (child % 3),
        phase: 0.5 * (1 + (child % 5))
      });

      visited[child] = true;
      depth[child] = depth[parent] + 1;
      q.push(child);
    }
  }

  return { faces, hinges };
}

// ---------- Other Shape builders (kept for comparison / selection) ----------
function buildPentagonalDipyramidSpec() {
  const R = 1.05, H = 1.10, jitter = 0.025;
  const top = new THREE.Vector3(0,  H, 0);
  const bot = new THREE.Vector3(0, -H, 0);
  const ring = [];
  for (let i = 0; i < 5; i++) {
    const a = (i / 5) * Math.PI * 2;
    const r = R * (1.0 + jitter * (i % 2 ? -1 : 1));
    ring.push(new THREE.Vector3(Math.cos(a) * r, 0, Math.sin(a) * r));
  }

  const faces = [];
  for (let i = 0; i < 5; i++) faces.push({ rest: [top.clone(), ring[i].clone(), ring[(i+1)%5].clone()], tag: 'T' });
  for (let i = 0; i < 5; i++) faces.push({ rest: [bot.clone(), ring[(i+1)%5].clone(), ring[i].clone()], tag: 'B' });

  const hinges = [];
  for (let i = 1; i < 5; i++) {
    hinges.push({
      child: i, parent: i-1,
      parentEdge: [0,2], childEdge: [0,1], group: 'top',
      ampDeg: 36 - i, freq: 0.10 + 0.02*i, phase: 0.6*i
    });
  }
  hinges.push({ child: 5, parent: 0, parentEdge: [1,2], childEdge: [1,2], group: 'bottom', ampDeg: 44, freq: 0.16, phase: 0.5 });
  for (let k = 6; k < 10; k++) {
    hinges.push({
      child: k, parent: k-1,
      parentEdge: [1,2], childEdge: [1,2], group: 'bottom',
      ampDeg: 42 - (k-6), freq: 0.09 + 0.02*(k-6), phase: 0.85 + 0.75*(k-6)
    });
  }
  return { faces, hinges };
}

function buildPentagonalTrapezohedronSpec() {
  const H = 1.05, h = 0.42, R1 = 0.95, R2 = 1.05, off = Math.PI / 5;
  const N = new THREE.Vector3(0,  H, 0);
  const S = new THREE.Vector3(0, -H, 0);
  const topR = [], botR = [];
  for (let i = 0; i < 5; i++) {
    const aTop = (i / 5) * Math.PI * 2;
    const aBot = aTop + off;
    topR.push(new THREE.Vector3(Math.cos(aTop)*R1,  h, Math.sin(aTop)*R1));
    botR.push(new THREE.Vector3(Math.cos(aBot)*R2, -h, Math.sin(aBot)*R2));
  }
  const faces = [];
  for (let i = 0; i < 5; i++) faces.push({ rest: [N.clone(), topR[i].clone(), botR[i].clone(), topR[(i+1)%5].clone()], tag: 'U' });
  for (let i = 0; i < 5; i++) faces.push({ rest: [S.clone(), botR[i].clone(), topR[i].clone(), botR[(i+1)%5].clone()], tag: 'L' });

  const hinges = [];
  for (let i = 1; i < 5; i++) {
    hinges.push({
      child: i, parent: i-1,
      parentEdge: [0,3], childEdge: [0,1], group: 'top',
      ampDeg: 34 - i, freq: 0.11 + 0.02*i, phase: 0.8*i
    });
  }
  hinges.push({ child: 5, parent: 0, parentEdge: [1,2], childEdge: [1,2], group: 'bottom', ampDeg: 40, freq: 0.15, phase: 0.5 });
  for (let k = 6; k < 10; k++) {
    hinges.push({
      child: k, parent: k-1,
      parentEdge: [0,3], childEdge: [0,1], group: 'bottom',
      ampDeg: 38 - (k-6), freq: 0.10 + 0.02*(k-6), phase: 1.1 + 0.7*(k-6)
    });
  }
  return { faces, hinges };
}

// ---------- Shape instance management ----------
let shape = null;

function makeShape(kind) {
  const spec = (kind === 'trapezohedron') ? buildPentagonalTrapezohedronSpec()
              : (kind === 'dipyramid')    ? buildPentagonalDipyramidSpec()
              :                              buildDodecahedronSpec(); // default
  const inst = new RigidHingeMesh(spec);
  return inst;
}

function switchShape(kind) {
  const prevUniforms = shape?.mesh.material.userData._overlayUniforms || null;

  if (shape) {
    scene.remove(shape.group);
    shape.dispose();
    shape = null;
  }
  shape = makeShape(kind);
  scene.add(shape.group);

  // Reapply overlay uniforms from previous instance so UI values persist
  if (prevUniforms) {
    const u = shape.mesh.material.userData._overlayUniforms;
    if (u) {
      u.uBandAngle.value    = prevUniforms.uBandAngle.value;
      u.uBandSpeed.value    = prevUniforms.uBandSpeed.value;
      u.uBandFreq1.value    = prevUniforms.uBandFreq1.value;
      u.uBandFreq2.value    = prevUniforms.uBandFreq2.value;
      u.uBandAngle2.value   = prevUniforms.uBandAngle2.value;
      u.uBandStrength.value = prevUniforms.uBandStrength.value;
      u.uTriScale.value     = prevUniforms.uTriScale.value;
      u.uWarp.value         = prevUniforms.uWarp.value;
      u.uCellAmp.value      = prevUniforms.uCellAmp.value;
      u.uCellFreq.value     = prevUniforms.uCellFreq.value;
    }
  }
}

// ---------- Post-processing pipeline ----------
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

// ---------- GUI ----------
const uiHost = document.getElementById('ui');
const gui = new GUI({ title: 'Controls', width: 320 });
uiHost.appendChild(gui.domElement);

const params = {
  objectType: 'dodecahedron', // default (NEW): 'dodecahedron' | 'dipyramid' | 'trapezohedron'
  play: true,
  foldSpeed: 1.0,
  foldMode: 'self-fold',   // 'self-fold' | 'free'
  foldMinDeg: 0.0,         // start from flat-ish
  foldMaxDeg: 55.0,        // < 63.4349° (π - dihedral) to keep paper-like motion

  // Glow (bloom)
  bloomStrength: 0.0,
  bloomThreshold: 0.9,
  bloomRadius: 0.2,

  // Trails / Ghosting
  trailAmount: 0.0,

  // RGB offset
  rgbAmount: 0.0,
  rgbAngle: 0.0,

  // View
  exposure: renderer.toneMappingExposure,
  resetCamera: () => {
    camera.position.set(3.0, 1.9, 5.0);
    controls.target.set(0,0,0);
    controls.update();
  }
};

const fShape = gui.addFolder('Shape');
fShape.add(params, 'objectType', {
  'Dodecahedron (12 pentagons)': 'dodecahedron',
  'Pentagonal Dipyramid (10 triangles)': 'dipyramid',
  'Pentagonal Trapezohedron (10 kites)': 'trapezohedron'
}).name('Object Type').onChange(v => switchShape(v));

fShape.add(params, 'foldMode', { 'Self-fold (1‑DOF)': 'self-fold', 'Free (multi‑DOF)': 'free' }).name('Fold Mode');
fShape.add(params, 'foldMinDeg', 0, 89, 0.1).name('Fold Min (°)');
fShape.add(params, 'foldMaxDeg', 1, 89, 0.1).name('Fold Max (°)');

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

// Surface (Advanced)
const surface = { angle: 28, angle2: 82, speed: 0.25, freq1: 6.0, freq2: 9.5, strength: 0.52, triScale: 1.15, warp: 0.55, cellAmp: 0.55, cellFreq: 2.75 };
const U = () => shape?.mesh.material.userData._overlayUniforms;
const fSurface = gui.addFolder('Surface (Advanced)');
fSurface.add(surface, 'angle', 0, 180, 0.1).name('Stripe Angle (°)').onChange(v => { const u = U(); if (u) u.uBandAngle.value  = THREE.MathUtils.degToRad(v); });
fSurface.add(surface, 'angle2', 0, 180, 0.1).name('Stripe2 Angle (°)').onChange(v => { const u = U(); if (u) u.uBandAngle2.value = THREE.MathUtils.degToRad(v); });
fSurface.add(surface, 'speed', 0, 2, 0.001).name('Stripe Rot Speed').onChange(v => { const u = U(); if (u) u.uBandSpeed.value = v; });
fSurface.add(surface, 'freq1', 1, 20, 0.1).name('Stripe Freq 1').onChange(v => { const u = U(); if (u) u.uBandFreq1.value = v; });
fSurface.add(surface, 'freq2', 1, 20, 0.1).name('Stripe Freq 2').onChange(v => { const u = U(); if (u) u.uBandFreq2.value = v; });
fSurface.add(surface, 'strength', 0, 1, 0.01).name('Emissive Strength').onChange(v => { const u = U(); if (u) u.uBandStrength.value = v; });
fSurface.add(surface, 'triScale', 0.2, 4, 0.01).name('Tri-Planar Scale').onChange(v => { const u = U(); if (u) u.uTriScale.value = v; });
fSurface.add(surface, 'warp', 0, 1.5, 0.01).name('Domain Warp').onChange(v => { const u = U(); if (u) u.uWarp.value = v; });
fSurface.add(surface, 'cellAmp', 0, 1, 0.01).name('Cellular Mix').onChange(v => { const u = U(); if (u) u.uCellAmp.value = v; });
fSurface.add(surface, 'cellFreq', 0.5, 8, 0.01).name('Cellular Freq').onChange(v => { const u = U(); if (u) u.uCellFreq.value = v; });
fSurface.close();

// ---------- Create initial shape (Dodecahedron) ----------
switchShape(params.objectType);

// ---------- Render loop ----------
let t0 = performance.now();
function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  const t = (now - t0) / 1000;

  if (params.play && shape) shape.update(t, params.foldSpeed);

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
