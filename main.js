// Endless Folding Decahedron — with Shape Selector
// - Shapes: Pentagonal Dipyramid (10 triangles), Pentagonal Trapezohedron (10 kites)
// - Rigid faces only; hinges are pinned -> shared edges never separate.
// - Iridescent base (MeshPhysicalMaterial) + rich tri-planar, domain-warped interference overlay.
// - Controls (right): Shape, Fold Speed, Glow (Bloom, default 0), Trails (Afterimage), RGB Offset.
//
// References used while building (docs/examples):
//   - Pentagonal dipyramid (decahedron of 10 triangles).  Wikipedia.          // https://en.wikipedia.org/wiki/Pentagonal_bipyramid
//   - Pentagonal trapezohedron (10 kites), D10 shape.   Wikipedia/MathWorld.   // https://en.wikipedia.org/wiki/Pentagonal_trapezohedron
//   - MeshPhysicalMaterial iridescence (thin-film).      three.js docs.        // https://threejs.org/docs/api/en/materials/MeshPhysicalMaterial.html
//   - EffectComposer + passes: UnrealBloom, Afterimage, RGBShift.              // https://threejs.org/docs/examples/en/postprocessing/EffectComposer.html
//   - Import maps (bare specifiers) + ES Module Shims.                         // MDN + ESMS https://developer.mozilla.org/.../script/type/importmap

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

// ---------- Generic Rigid Hinge Mesh (supports polygonal faces: triangles or kites) ----------
class RigidHingeMesh {
  /**
   * @param {Object} spec
   *   spec.faces: Array<{ rest: THREE.Vector3[], tag?: string }>
   *   spec.hinges: Array<{ child:number, parent:number, parentEdge:[number,number], childEdge:[number,number], ampDeg:number, freq:number, phase:number }>
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

    // Triangulate each face (fan): store mapping from geometry vertex -> (face, local-vertex)
    this._geomMap = []; // array of {face, local}
    const tris = [];
    for (let fi = 0; fi < this.faces.length; fi++) {
      const m = this.faces[fi].rest.length;
      for (let k = 1; k < m - 1; k++) {
        // triangle (0, k, k+1)
        tris.push([ [fi,0], [fi,k], [fi,k+1] ]);
        this._geomMap.push({ face: fi, local: 0 });
        this._geomMap.push({ face: fi, local: k });
        this._geomMap.push({ face: fi, local: k+1 });
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
      iridescence: 1.0,                // three.js iridescence (thin-film) — see docs
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

    // Build hinges (compute bind for each)
    this._M = this.faces.map(() => new THREE.Matrix4());
    this._mTA = new THREE.Matrix4();
    this._mTNegA = new THREE.Matrix4();
    this._mRot = new THREE.Matrix4();

    // Attach hinges
    const hinges = spec.hinges || [];
    for (const h of hinges) this._bindHinge(h);

    // Order (parent before child). Here we just make a simple topological order.
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
    return 0; // fallback
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
      amp: THREE.MathUtils.degToRad(h.ampDeg),
      freq: h.freq,
      phase: h.phase,
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
    // find roots (no parent)
    for (let i = 0; i < this.faces.length; i++) if (this.faces[i].parent === null) dfs(i);
    // parents before children
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
    const q = (f, p) => Math.sin(2 * Math.PI * (f * speed) * t + p);

    // Root faces (no parent)
    for (let i = 0; i < this._M.length; i++) this._M[i].identity();

    // Child transforms
    for (let i = 0; i < this.faceOrder.length; i++) {
      const id = this.faceOrder[i];
      const f  = this.faces[id];
      if (!f.hinge) continue;

      const h = f.hinge;
      const Mparent = this._M[h.parent];
      const angle = h.amp * q(h.freq, h.phase) + 0.10 * Math.sin(2 * Math.PI * (0.05 * speed) * t + id);
      this._composePinned(Mparent, this.faces[h.parent], h, angle, this._M[id]);
    }

    // Global gentle presentation rotation
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

    // Pass 2 (weld): copy parent's hinge vertices into child hinge vertices
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

    // Pass 3: write positions per geometry vertex from face/world map
    const posAttr = this.geometry.getAttribute('position');
    for (let gv = 0; gv < this._geomMap.length; gv++) {
      const m = this._geomMap[gv];
      const w = this.faces[m.face].world[m.local];
      posAttr.setXYZ(gv, w.x, w.y, w.z);
    }
    posAttr.needsUpdate = true;

    // Pass 4: recompute flat normals per triangle
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

    // Drive overlay time
    const uniforms = this.mesh.material.userData._overlayUniforms;
    if (uniforms) uniforms.uTime.value = t;
  }
}

// ---------- Shape builders ----------
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
  // Top 5 triangles
  for (let i = 0; i < 5; i++) {
    faces.push({ rest: [top.clone(), ring[i].clone(), ring[(i+1)%5].clone()], tag: 'T' });
  }
  // Bottom 5 triangles (winding to keep outward normals)
  for (let i = 0; i < 5; i++) {
    faces.push({ rest: [bot.clone(), ring[(i+1)%5].clone(), ring[i].clone()], tag: 'B' });
  }

  const hinges = [];
  // Root = face 0 (top wedge)
  // Top chain around apex: share edge [top, ring[i]]
  for (let i = 1; i < 5; i++) {
    hinges.push({
      child: i, parent: i-1,
      parentEdge: [0,2], // in parent face (top, ring[i])
      childEdge:  [0,1],
      ampDeg: 36 - i, freq: 0.10 + 0.02*i, phase: 0.6*i
    });
  }
  // Bottom chain starting at face 5 hinged to face 0 along edge [ring0, ring1] -> in top face [1,2], in bottom [1,2]
  hinges.push({ child: 5, parent: 0, parentEdge: [1,2], childEdge: [1,2], ampDeg: 44, freq: 0.16, phase: 0.5 });
  for (let k = 6; k < 10; k++) {
    hinges.push({
      child: k, parent: k-1,
      parentEdge: [1,2], childEdge: [1,2],
      ampDeg: 42 - (k-6), freq: 0.09 + 0.02*(k-6), phase: 0.85 + 0.75*(k-6)
    });
  }

  return { faces, hinges };
}

function buildPentagonalTrapezohedronSpec() {
  // Two polar vertices + two offset 5-gon rings -> 10 kite faces
  const H = 1.05;      // pole height
  const h = 0.42;      // ring heights
  const R1 = 0.95;     // top ring radius
  const R2 = 1.05;     // bottom ring radius (asymmetry -> kite)
  const off = Math.PI / 5; // 36° offset between rings

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
  // Top ring of 5 kites: [N, Top[i], Bot[i], Top[i+1]]
  for (let i = 0; i < 5; i++) {
    faces.push({ rest: [N.clone(), topR[i].clone(), botR[i].clone(), topR[(i+1)%5].clone()], tag: 'U' });
  }
  // Bottom ring of 5 kites: [S, Bot[i], Top[i], Bot[i+1]]
  for (let i = 0; i < 5; i++) {
    faces.push({ rest: [S.clone(), botR[i].clone(), topR[i].clone(), botR[(i+1)%5].clone()], tag: 'L' });
  }

  const hinges = [];
  // Root face 0 (upper kite)
  // Upper chain: share edge [N, Top[i]] : in parent (0..4), the shared edge is [0,3]; in child it's [0,1]
  for (let i = 1; i < 5; i++) {
    hinges.push({
      child: i, parent: i-1,
      parentEdge: [0,3], childEdge: [0,1],
      ampDeg: 34 - i, freq: 0.11 + 0.02*i, phase: 0.8*i
    });
  }
  // Lower chain starts at face 5 hinged to face 0 along edge [Top0, Bot0] which is [1,2] in both
  hinges.push({ child: 5, parent: 0, parentEdge: [1,2], childEdge: [1,2], ampDeg: 40, freq: 0.15, phase: 0.5 });
  for (let k = 6; k < 10; k++) {
    hinges.push({
      child: k, parent: k-1,
      parentEdge: [0,3], // share edge [S, Bot[k-5]] with previous lower face
      childEdge:  [0,1],
      ampDeg: 38 - (k-6), freq: 0.10 + 0.02*(k-6), phase: 1.1 + 0.7*(k-6)
    });
  }

  return { faces, hinges };
}

// ---------- Shape instance management ----------
let shape = null;

function makeShape(kind) {
  const spec = (kind === 'trapezohedron') ? buildPentagonalTrapezohedronSpec()
                                          : buildPentagonalDipyramidSpec();
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

// Trails (Afterimage)
const afterimagePass = new AfterimagePass();
afterimagePass.enabled = false;                        // default off
afterimagePass.uniforms['damp'].value = 1.0;           // 1.0 ~ no trail
composer.addPass(afterimagePass);

// Glow / Bloom
const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(container.clientWidth, container.clientHeight),
  0.0, // strength default 0 (off)
  0.2, // radius
  0.9  // threshold
);
bloomPass.enabled = false;
composer.addPass(bloomPass);

// RGB offset
const rgbShiftPass = new ShaderPass(RGBShiftShader);
rgbShiftPass.enabled = false;
rgbShiftPass.uniforms['amount'].value = 0.0;  // typical 0.0005..0.01
rgbShiftPass.uniforms['angle'].value  = 0.0;
composer.addPass(rgbShiftPass);

// ---------- GUI ----------
const uiHost = document.getElementById('ui');
const gui = new GUI({ title: 'Controls', width: 320 });
uiHost.appendChild(gui.domElement);

const params = {
  objectType: 'dipyramid', // 'dipyramid' | 'trapezohedron'
  play: true,
  foldSpeed: 1.0,

  // Glow (bloom)
  bloomStrength: 0.0, // default 0
  bloomThreshold: 0.9,
  bloomRadius: 0.2,

  // Trails / Ghosting
  trailAmount: 0.0,   // 0..1 (0 = off)

  // RGB offset
  rgbAmount: 0.0,     // 0..0.02 typical
  rgbAngle: 0.0,      // degrees

  // View
  exposure: renderer.toneMappingExposure,
  resetCamera: () => {
    camera.position.set(3.2, 1.8, 4.8);
    controls.target.set(0,0,0);
    controls.update();
  }
};

const fShape = gui.addFolder('Shape');
fShape.add(params, 'objectType', {
  'Pentagonal Dipyramid (10 triangles)': 'dipyramid',
  'Pentagonal Trapezohedron (10 kites)': 'trapezohedron'
}).name('Object Type').onChange(v => switchShape(v));

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
  const damp = 1.0 - v * 0.98; // lower damp -> longer trail
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

// Optional: expose the surface overlay controls (collapsed by default)
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

// ---------- Create initial shape ----------
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
