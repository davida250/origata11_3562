// Endless Folding Iridescent Polyhedron — welded hinges + rich procedural surface
// Two big upgrades vs last version:
//   1) Hinges are "pinned": shared edges are locked exactly → no polygon separation.
//   2) Surface = tri-planar mix of iridescent stripes + cellular (Worley-style) + fBm domain warp.
//
// References used: MeshPhysicalMaterial iridescence & onBeforeCompile (three.js docs),
// tri-planar mapping, Worley/cellular noise, fBm and domain warping (see notes at end).

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
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

// Camera
const camera = new THREE.PerspectiveCamera(36, container.clientWidth / container.clientHeight, 0.01, 100);
camera.position.set(3.2, 1.8, 4.8);

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.minDistance = 2.0;
controls.maxDistance = 9.0;

// Environment (PMREM + RoomEnvironment)
const pmrem = new THREE.PMREMGenerator(renderer);
const envRT = pmrem.fromScene(new RoomEnvironment(), 0.04);
scene.environment = envRT.texture;

// ---------- Rigid-Face Foldable Object (hinge-pinned) ----------
class RigidHingePoly {
  constructor(opts = {}) {
    this.group = new THREE.Group();

    // Irregular octahedron-like rest shape (6 vertices → 8 triangular faces)
    const a = 1.00;
    const V = [
      new THREE.Vector3( 0.00,  a,   0.00),   // v0 top
      new THREE.Vector3( 0.95,  0.02, 0.35),  // v1 equator
      new THREE.Vector3(-0.55,  0.01, 0.90),  // v2 equator
      new THREE.Vector3(-0.95, -0.02,-0.35),  // v3 equator
      new THREE.Vector3( 0.60,  0.00,-0.95),  // v4 equator
      new THREE.Vector3( 0.00, -a,   0.00)    // v5 bottom
    ];

    const F = [
      [0,1,2],
      [0,2,3],
      [0,3,4],
      [0,4,1],
      [5,2,1],
      [5,3,2],
      [5,4,3],
      [5,1,4]
    ];

    const faces = F.map((tri, idx) => ({
      id: idx,
      rest: tri.map(i => V[i].clone()),
      world: [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()],
      parent: null,
      hinge: null,     // { parent, parentEdge:[ia,ib], childEdge:[ia,ib], amp, freq, phase, bind }
    }));

    // Utility: find the "third" vertex of a triangle not in edge [i,j]
    const thirdIndex = (i, j) => {
      for (let k = 0; k < 3; k++) if (k !== i && k !== j) return k;
      return 2;
    };

    // Build hinge tree + compute bind transforms so edges are pinned.
    const bind = (child, parent, parentEdge, childEdge, ampDeg, freq, phase) => {
      const hinge = { parent, parentEdge, childEdge, amp: THREE.MathUtils.degToRad(ampDeg), freq, phase, bind: new THREE.Matrix4() };

      // Compute "bind" matrix that maps child's local face into parent's local space
      // so that at angle=0 they are coplanar and share the same edge line.
      const pA = faces[parent].rest[parentEdge[0]].clone();
      const pB = faces[parent].rest[parentEdge[1]].clone();
      const cA = faces[child].rest[childEdge[0]].clone();
      const cB = faces[child].rest[childEdge[1]].clone();

      const pT = faces[parent].rest[thirdIndex(parentEdge[0], parentEdge[1])].clone();
      const cT = faces[child].rest[thirdIndex(childEdge[0], childEdge[1])].clone();

      const up = new THREE.Vector3().subVectors(pB, pA).normalize();
      const uc = new THREE.Vector3().subVectors(cB, cA).normalize();

      // Rotate child's hinge edge direction onto parent's
      const q1 = new THREE.Quaternion().setFromUnitVectors(uc, up);

      // Rotate around hinge axis so face normals align (coplanar bind)
      const np = new THREE.Vector3().subVectors(pB, pA).cross(new THREE.Vector3().subVectors(pT, pA)).normalize();
      const nc1 = new THREE.Vector3().subVectors(cB, cA).cross(new THREE.Vector3().subVectors(cT, cA)).applyQuaternion(q1).normalize();

      // Project normals off the hinge axis to get an oriented angle
      const proj = (n, axis) => n.clone().sub(axis.clone().multiplyScalar(n.dot(axis)));
      const npP = proj(np, up).normalize();
      const ncP = proj(nc1, up).normalize();

      const cross = new THREE.Vector3().crossVectors(ncP, npP);
      const phi = Math.atan2(up.dot(cross), ncP.dot(npP));
      const q2 = new THREE.Quaternion().setFromAxisAngle(up, phi);

      const q = q2.multiply(q1);
      const pos = pA.clone().sub(cA.clone().applyQuaternion(q)); // make cA land on pA
      hinge.bind.compose(pos, q, new THREE.Vector3(1,1,1));

      faces[child].parent = parent;
      faces[child].hinge = hinge;
    };

    // Root = 0
    faces[0].hinge = null;
    // Top ring
    bind(1, 0, [0,2], [0,1], 65, 0.11, 0.0);
    bind(2, 1, [0,2], [0,1], 60, 0.13, 1.1);
    bind(3, 2, [0,2], [0,1], 55, 0.10, 2.3);
    // Bottom chain
    bind(4, 0, [1,2], [1,2], 70, 0.17, 0.6);
    bind(5, 4, [0,1], [0,1], 65, 0.09, 1.7);
    bind(6, 5, [0,1], [0,1], 62, 0.14, 2.6);
    bind(7, 6, [0,1], [0,1], 68, 0.12, 3.7);

    this.faceOrder = [0,1,2,3,4,5,6,7];

    // Geometry (duplicated vertices per face for flat-shaded, rigid triangles)
    const positions = new Float32Array(faces.length * 3 * 3);
    const normals   = new Float32Array(faces.length * 3 * 3);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('normal',   new THREE.BufferAttribute(normals,   3));
    geometry.setIndex([...Array(faces.length * 3).keys()]);
    geometry.computeBoundingSphere();

    // Physical material with native iridescence (thin-film)
    const mat = (opts.material instanceof THREE.Material)
      ? opts.material
      : new THREE.MeshPhysicalMaterial({
          color: 0x151515,
          roughness: 0.26,
          metalness: 0.0,
          envMapIntensity: 1.3,
          iridescence: 1.0,
          iridescenceIOR: 1.3,
          iridescenceThicknessRange: [120, 620],
          flatShading: true
        });

    // Add tri‑planar, domain‑warped stripe+cellular emissive overlay
    this._injectRichOverlay(mat);

    const mesh = new THREE.Mesh(geometry, mat);
    mesh.castShadow = false;
    mesh.receiveShadow = false;

    this.group.add(mesh);
    this.faces = faces;
    this.mesh = mesh;
    this.geometry = geometry;

    // Reusable matrices
    this._mTA = new THREE.Matrix4();
    this._mTNegA = new THREE.Matrix4();
    this._mRot = new THREE.Matrix4();

    // Per-face object-space → world transforms
    this._M = faces.map(() => new THREE.Matrix4());

    // For hierarchical order / quick access
    this.children = new Map();
    faces.forEach((f, i) => this.children.set(i, []));
    faces.forEach((f, i) => { if (f.parent !== null) this.children.get(f.parent).push(i); });

    this.update(0);
  }

  dispose() {
    this.geometry.dispose();
    if (this.mesh.material) this.mesh.material.dispose();
  }

  // Emissive overlay: tri‑planar stripes + cellular + fBm with domain warping
  _injectRichOverlay(material) {
    const uniforms = {
      uTime:         { value: 0 },
      uBandAngle:    { value: THREE.MathUtils.degToRad(32.0) },
      uBandSpeed:    { value: 0.25 },
      uBandFreq1:    { value: 6.0 },
      uBandFreq2:    { value: 9.5 },
      uBandAngle2:   { value: THREE.MathUtils.degToRad(78.0) },
      uBandStrength: { value: 0.52 },
      uTriScale:     { value: 1.2 },   // world→UV scale for tri‑planar
      uWarp:         { value: 0.55 },  // domain-warp amount
      uNoiseAmp:     { value: 0.35 },  // fBm amplitude in mix
      uCellAmp:      { value: 0.55 },  // cellular mix amount
      uCellFreq:     { value: 2.75 }   // cellular frequency
    };

    material.onBeforeCompile = (shader) => {
      Object.assign(shader.uniforms, uniforms);

      // Vertex: export world position and world normal for tri‑planar mapping
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

      // Fragment: tri‑planar mapping + fBm + Worley-style cellular + stripes
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
          uniform float uNoiseAmp;
          uniform float uCellAmp;
          uniform float uCellFreq;

          // ————— utilities —————
          float hash11(float n){ return fract(sin(n)*43758.5453123); }
          float hash12(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453123); }
          vec2  hash22(vec2 p){
            p = vec2( dot(p,vec2(127.1,311.7)), dot(p,vec2(269.5,183.3)) );
            return fract(sin(p)*43758.5453);
          }

          // value noise
          float noise(vec2 p){
            vec2 i = floor(p), f = fract(p);
            vec2 u = f*f*(3.0-2.0*f);
            float a = hash12(i + vec2(0,0));
            float b = hash12(i + vec2(1,0));
            float c = hash12(i + vec2(0,1));
            float d = hash12(i + vec2(1,1));
            return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
          }

          // fBm
          float fbm(vec2 p){
            float s = 0.0, a = 0.5;
            for(int i=0;i<5;i++){
              s += a * noise(p);
              p = mat2(1.6,1.2,-1.2,1.6) * p; // rotate/scale
              a *= 0.5;
            }
            return s;
          }

          // Worley-style cellular distance (2D, 3x3 neighborhood)
          float cellular(vec2 p){
            p *= uCellFreq;
            vec2 i = floor(p);
            vec2 f = fract(p);
            float md = 1.0;
            for(int y=-1;y<=1;y++){
              for(int x=-1;x<=1;x++){
                vec2 g = vec2(float(x), float(y));
                vec2 o = hash22(i + g) - 0.5;
                vec2 r = g + o + (f - 0.5);
                md = min(md, dot(r,r));
              }
            }
            return sqrt(md); // 0..~1
          }

          mat2 rot(float a){ float c = cos(a), s = sin(a); return mat2(c,-s,s,c); }

          // Cosmetic rainbow palette
          vec3 rainbow(float t){
            const float TAU = 6.28318530718;
            vec3 phase = vec3(0.0, 0.33, 0.67) * TAU;
            return 0.5 + 0.5 * cos(TAU * t + phase);
          }

          // Tri‑planar helper: weights from world normal
          vec3 triWeights(vec3 n){
            vec3 an = abs(normalize(n));
            an = pow(an, vec3(4.0));
            return an / (an.x + an.y + an.z + 1e-5);
          }

          // Stripe field on a 2D plane, with domain warping & second stripe set
          vec3 stripeField(vec2 uv, float baseAngle){
            float t = uTime;
            float theta = baseAngle + t * uBandSpeed;
            mat2 R = rot(theta);

            // domain warp
            vec2 w = uv * uTriScale;
            float w1 = fbm(w * 1.2);
            w += uWarp * vec2(w1, fbm(w + 17.1));

            // two stripe sets
            float s1 = 0.5 + 0.5 * sin(dot(R * w, vec2(uBandFreq1, 0.0)));
            float s2 = 0.5 + 0.5 * sin(dot(rot(uBandAngle2) * w, vec2(uBandFreq2, 0.0)));

            // enhancement & fBm modulation
            float mixS = max(s1, s2 * 0.85);
            mixS = mixS * (0.7 + 0.3 * fbm(w * 0.9));
            float cells = 1.0 - smoothstep(0.0, 0.75, cellular(uv)); // edges of cells
            float m = mix(mixS, max(mixS, cells), uCellAmp);

            // build color via "thin‑film‑ish" palette travel
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

            // three planes: XY, XZ, YZ (tri‑planar)
            vec3 colXY = stripeField(p.xy, uBandAngle);
            vec3 colXZ = stripeField(p.xz, uBandAngle);
            vec3 colYZ = stripeField(p.zy, uBandAngle);

            vec3 c = w.x * colYZ + w.y * colXZ + w.z * colXY;

            totalEmissiveRadiance += c * uBandStrength;
          }
        `);

      material.userData._richUniforms = uniforms;
    };
    material.needsUpdate = true;
  }

  // Compose child transform that keeps shared edge pinned for all times:
  //   Mchild = Mparent * T(Aw) * R(axisAw, angle) * T(-Aw) * bind
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
        .multiply(hinge.bind); // bind maps child-local to parent-local at angle=0
  }

  // Update; t in seconds
  update(t) {
    const q = (f, p) => Math.sin(2 * Math.PI * f * t + p);

    // Root face (identity in object space)
    this._M[0].identity();

    // Hierarchically build child transforms
    for (let i = 1; i < this.faceOrder.length; i++) {
      const id = this.faceOrder[i];
      const f  = this.faces[id];
      const h  = f.hinge;
      const Mparent = this._M[h.parent];

      const angle = h.amp * q(h.freq, h.phase) + 0.12 * Math.sin(2 * Math.PI * 0.05 * t + id);
      this._composePinned(Mparent, this.faces[h.parent], h, angle, this._M[id]);
    }

    // Global gentle presentation yaw/pitch
    const yaw   = 0.25 * Math.sin(2 * Math.PI * 0.03 * t);
    const pitch = 0.18 * Math.sin(2 * Math.PI * 0.021 * t + 1.2);
    const Mglobal = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(pitch, yaw, 0, 'YXZ'));

    const pos = this.geometry.getAttribute('position');
    const nrm = this.geometry.getAttribute('normal');

    let vOffset = 0, nOffset = 0;
    const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();

    // First pass: transform vertices
    for (let i = 0; i < this.faces.length; i++) {
      const M = new THREE.Matrix4().multiplyMatrices(Mglobal, this._M[i]);
      const fr = this.faces[i].rest;

      const w0 = this.faces[i].world[0].copy(fr[0]).applyMatrix4(M);
      const w1 = this.faces[i].world[1].copy(fr[1]).applyMatrix4(M);
      const w2 = this.faces[i].world[2].copy(fr[2]).applyMatrix4(M);

      pos.setXYZ(vOffset + 0, w0.x, w0.y, w0.z);
      pos.setXYZ(vOffset + 1, w1.x, w1.y, w1.z);
      pos.setXYZ(vOffset + 2, w2.x, w2.y, w2.z);

      // Flat normal
      a.subVectors(w1, w0);
      b.subVectors(w2, w0);
      c.copy(a).cross(b).normalize();
      nrm.setXYZ(nOffset + 0, c.x, c.y, c.z);
      nrm.setXYZ(nOffset + 1, c.x, c.y, c.z);
      nrm.setXYZ(nOffset + 2, c.x, c.y, c.z);

      vOffset += 3;
      nOffset += 3;
    }

    // Second pass (weld): for each child, overwrite its two hinge-vertex positions
    // with *exact* copies of the parent's corresponding vertices → zero gap.
    vOffset = 0;
    for (let i = 0; i < this.faces.length; i++) {
      const f = this.faces[i];
      if (!f.hinge) { vOffset += 3; continue; }

      const h = f.hinge;
      const parent = this.faces[h.parent];

      // Parent world positions already computed above (and multiplied by Mglobal)
      const pA = parent.world[h.parentEdge[0]];
      const pB = parent.world[h.parentEdge[1]];

      // Overwrite child's hinge vertices in both CPU buffers and cached world vectors
      const ci0 = h.childEdge[0];
      const ci1 = h.childEdge[1];

      f.world[ci0].copy(pA);
      f.world[ci1].copy(pB);

      pos.setXYZ(vOffset + ci0, pA.x, pA.y, pA.z);
      pos.setXYZ(vOffset + ci1, pB.x, pB.y, pB.z);

      vOffset += 3;
    }

    pos.needsUpdate = true;
    nrm.needsUpdate = true;

    const uniforms = this.mesh.material.userData._richUniforms;
    if (uniforms) uniforms.uTime.value = t;
  }
}

// ---------- Create material + mesh ----------
const material = new THREE.MeshPhysicalMaterial({
  color: 0x151515,
  roughness: 0.26,
  metalness: 0.0,
  envMapIntensity: 1.3,
  iridescence: 1.0,
  iridescenceIOR: 1.3,
  iridescenceThicknessRange: [120, 620],
  flatShading: true
});
const poly = new RigidHingePoly({ material });
scene.add(poly.group);

// ---------- GUI ----------
const uiHost = document.getElementById('ui');
const gui = new GUI({ title: 'Controls', width: 300 });
uiHost.appendChild(gui.domElement);

const U = () => poly.mesh.material.userData._richUniforms;

const params = {
  play: true,
  exposure: renderer.toneMappingExposure,
  envIntensity: 1.3,
  iridescence: material.iridescence,
  ior: material.iridescenceIOR,
  filmMin: material.iridescenceThicknessRange[0],
  filmMax: material.iridescenceThicknessRange[1],

  bandAngleDeg: 32,
  bandSpeed: 0.25,
  bandFreq1: 6.0,
  bandFreq2: 9.5,
  bandAngle2Deg: 78.0,
  bandStrength: 0.52,
  triScale: 1.2,
  warp: 0.55,
  noiseAmp: 0.35,
  cellAmp: 0.55,
  cellFreq: 2.75,

  resetCamera: () => {
    camera.position.set(3.2, 1.8, 4.8);
    controls.target.set(0,0,0);
    controls.update();
  }
};

const p0 = gui.addFolder('Playback & Render');
p0.add(params, 'play').name('Play / Pause');
p0.add(params, 'exposure', 0.6, 1.8, 0.01).name('Exposure')
  .onChange(v => renderer.toneMappingExposure = v);
p0.add(params, 'envIntensity', 0.0, 3.0, 0.01).name('IBL Intensity')
  .onChange(v => (poly.mesh.material.envMapIntensity = v));
p0.add(params, 'resetCamera').name('Reset Camera');

const fIri = gui.addFolder('Iridescence');
fIri.add(params, 'iridescence', 0.0, 1.0, 0.01).name('Amount')
    .onChange(v => (poly.mesh.material.iridescence = v));
fIri.add(params, 'ior', 1.0, 2.333, 0.001).name('Iridescence IOR')
    .onChange(v => (poly.mesh.material.iridescenceIOR = v));
fIri.add(params, 'filmMin', 50, 800, 1).name('Film Min (nm)')
    .onChange(v => (poly.mesh.material.iridescenceThicknessRange[0] = v));
fIri.add(params, 'filmMax', 50, 800, 1).name('Film Max (nm)')
    .onChange(v => (poly.mesh.material.iridescenceThicknessRange[1] = v));

const fBands = gui.addFolder('Pattern Mix');
fBands.add(params, 'bandAngleDeg', 0, 180, 0.1).name('Stripe Angle (°)')
  .onChange(v => { const u = U(); if (u) u.uBandAngle.value = THREE.MathUtils.degToRad(v); });
fBands.add(params, 'bandAngle2Deg', 0, 180, 0.1).name('Stripe2 Angle (°)')
  .onChange(v => { const u = U(); if (u) u.uBandAngle2.value = THREE.MathUtils.degToRad(v); });
fBands.add(params, 'bandFreq1', 1.0, 20.0, 0.1).name('Stripe Freq 1')
  .onChange(v => { const u = U(); if (u) u.uBandFreq1.value = v; });
fBands.add(params, 'bandFreq2', 1.0, 20.0, 0.1).name('Stripe Freq 2')
  .onChange(v => { const u = U(); if (u) u.uBandFreq2.value = v; });
fBands.add(params, 'bandSpeed', 0.0, 2.0, 0.001).name('Stripe Rot Speed')
  .onChange(v => { const u = U(); if (u) u.uBandSpeed.value = v; });
fBands.add(params, 'bandStrength', 0.0, 1.0, 0.01).name('Emissive Strength')
  .onChange(v => { const u = U(); if (u) u.uBandStrength.value = v; });
fBands.add(params, 'triScale', 0.2, 4.0, 0.01).name('Tri-Planar Scale')
  .onChange(v => { const u = U(); if (u) u.uTriScale.value = v; });
fBands.add(params, 'warp', 0.0, 1.5, 0.01).name('Domain Warp')
  .onChange(v => { const u = U(); if (u) u.uWarp.value = v; });
fBands.add(params, 'noiseAmp', 0.0, 1.0, 0.01).name('fBm Amount')
  .onChange(v => { const u = U(); if (u) u.uNoiseAmp.value = v; });
fBands.add(params, 'cellAmp', 0.0, 1.0, 0.01).name('Cellular Mix')
  .onChange(v => { const u = U(); if (u) u.uCellAmp.value = v; });
fBands.add(params, 'cellFreq', 0.5, 8.0, 0.01).name('Cellular Freq')
  .onChange(v => { const u = U(); if (u) u.uCellFreq.value = v; });

// ---------- Render loop ----------
let start = performance.now();
function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  const t = (now - start) / 1000;
  if (params.play) poly.update(t);
  controls.update();
  renderer.render(scene, camera);
}
animate();

// ---------- Resize handling ----------
function onResize() {
  const w = container.clientWidth;
  const h = container.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}
window.addEventListener('resize', onResize);
new ResizeObserver(onResize).observe(container);
