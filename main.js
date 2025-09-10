// Endless Folding Iridescent Polyhedron (no morphs; rigid faces rotating on hinges)
// Ready-to-run ESM script. Drop beside index.html and open the HTML file.

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
renderer.toneMappingExposure = 1.1;
renderer.outputColorSpace = THREE.SRGBColorSpace;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

// Camera
const camera = new THREE.PerspectiveCamera(36, container.clientWidth / container.clientHeight, 0.01, 100);
camera.position.set(3.0, 1.7, 4.5);

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.minDistance = 2.0;
controls.maxDistance = 9.0;

// Environment (PMREM + RoomEnvironment)
const pmrem = new THREE.PMREMGenerator(renderer);
const envRT = pmrem.fromScene(new RoomEnvironment(), 0.04); // PMREM-preprocessed env map for correct roughness response
scene.environment = envRT.texture; // PMREM enables roughness-correlated reflections. :contentReference[oaicite:1]{index=1}

// ---------- Rigid-Face Foldable Object ----------
class RigidHingePoly {
  constructor(opts = {}) {
    this.group = new THREE.Group();

    // Irregular octahedron-like rest shape (6 vertices used by 8 triangular faces)
    const a = 1.00;
    const V = [
      new THREE.Vector3( 0.00,  a,   0.00),    // v0 top
      new THREE.Vector3( 0.95,  0.02, 0.35),   // v1 equator
      new THREE.Vector3(-0.55,  0.01, 0.90),   // v2 equator
      new THREE.Vector3(-0.95, -0.02,-0.35),   // v3 equator
      new THREE.Vector3( 0.60,  0.00,-0.95),   // v4 equator
      new THREE.Vector3( 0.00, -a,   0.00)     // v5 bottom
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
      hinge: null
    }));

    // Hinge relationships (tree)
    const bind = (child, parent, parentEdge, childEdge, ampDeg, freq, phase) => {
      faces[child].parent = parent;
      faces[child].hinge = { parent, parentEdge, childEdge, amp: THREE.MathUtils.degToRad(ampDeg), freq, phase };
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

    // Geometry (duplicated vertices per face for rigid, flat-shaded triangles)
    const positions = new Float32Array(faces.length * 3 * 3);
    const normals   = new Float32Array(faces.length * 3 * 3);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('normal',   new THREE.BufferAttribute(normals,   3));
    geometry.setIndex([...Array(faces.length * 3).keys()]);

    // Physical material with native iridescence (thin-film interference) :contentReference[oaicite:2]{index=2}
    const mat = (opts.material instanceof THREE.Material)
      ? opts.material
      : new THREE.MeshPhysicalMaterial({
          color: 0x151515,
          roughness: 0.25,
          metalness: 0.0,
          envMapIntensity: 1.15,
          iridescence: 1.0,
          iridescenceIOR: 1.3,
          iridescenceThicknessRange: [120, 620],
          flatShading: true
        });

    // Add oriented interference bands via onBeforeCompile (emissive overlay)
    this._injectStripeOverlay(mat);

    const mesh = new THREE.Mesh(geometry, mat);
    mesh.castShadow = false;
    mesh.receiveShadow = false;

    this.group.add(mesh);
    this.faces = faces;
    this.mesh = mesh;
    this.geometry = geometry;

    this._mTranslateA = new THREE.Matrix4();
    this._mTranslateNegA = new THREE.Matrix4();
    this._mRotate = new THREE.Matrix4();
    this._M = faces.map(() => new THREE.Matrix4());

    this.children = new Map();
    faces.forEach((f, i) => this.children.set(i, []));
    faces.forEach((f, i) => { if (f.parent !== null) this.children.get(f.parent).push(i); });

    this.update(0);
  }

  dispose() {
    this.geometry.dispose();
    if (this.mesh.material) this.mesh.material.dispose();
  }

  _injectStripeOverlay(material) {
    const uniforms = {
      uTime:        { value: 0 },
      uBandAngle:   { value: 32.0 * Math.PI / 180.0 },
      uBandFreq:    { value: 6.0 },
      uBandStrength:{ value: 0.45 },
      uBandSpeed:   { value: 0.25 }
    };
    material.onBeforeCompile = (shader) => {
      Object.assign(shader.uniforms, uniforms);
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', `
          #include <common>
          varying vec3 vWorldPos;
        `)
        .replace('#include <project_vertex>', `
          #include <project_vertex>
          vWorldPos = (modelMatrix * vec4(transformed, 1.0)).xyz;
        `);

      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', `
          #include <common>
          varying vec3 vWorldPos;
          uniform float uTime;
          uniform float uBandAngle;
          uniform float uBandFreq;
          uniform float uBandStrength;
          uniform float uBandSpeed;

          float n13(vec3 p) {
            p = fract(p * 0.1031);
            p += dot(p, p.yzx + 33.33);
            return fract((p.x + p.y) * p.z);
          }
          vec3 rainbow(float t) {
            const float TAU = 6.28318530718;
            vec3 phase = vec3(0.0, 0.33, 0.67) * TAU;
            return 0.5 + 0.5 * cos(TAU * t + phase);
          }
        `)
        .replace('#include <emissivemap_fragment>', `
          #include <emissivemap_fragment>
          {
            float theta = uBandAngle + uTime * uBandSpeed;
            vec2 dir = vec2(cos(theta), sin(theta));
            float coord = dot(vWorldPos.xy, dir) * uBandFreq;

            float s = 0.5 + 0.5 * sin(coord + n13(vWorldPos * 3.17));
            float band = smoothstep(0.35, 0.65, s);

            vec3 stripeColor = rainbow(fract(coord * 0.05 + 0.5));
            totalEmissiveRadiance += stripeColor * (uBandStrength * band);
          }
        `);
      material.userData._stripeUniforms = uniforms;
    };
    material.needsUpdate = true;
  }

  _composeHinge(Mparent, A, B, angle, outM) {
    const axis = new THREE.Vector3().subVectors(B, A).normalize();
    this._mTranslateA.makeTranslation(A.x, A.y, A.z);
    this._mTranslateNegA.makeTranslation(-A.x, -A.y, -A.z);
    this._mRotate.makeRotationAxis(axis, angle);
    outM.copy(Mparent)
        .multiply(this._mTranslateA)
        .multiply(this._mRotate)
        .multiply(this._mTranslateNegA);
  }

  update(t) {
    const q = (f, p) => Math.sin(2 * Math.PI * f * t + p);
    this._M[0].identity();

    for (let i = 1; i < this.faceOrder.length; i++) {
      const id = this.faceOrder[i];
      const f = this.faces[id];
      const h = f.hinge;
      const Mparent = this._M[h.parent];

      const A = this.faces[h.parent].rest[h.parentEdge[0]].clone().applyMatrix4(Mparent);
      const B = this.faces[h.parent].rest[h.parentEdge[1]].clone().applyMatrix4(Mparent);

      const angle = h.amp * q(h.freq, h.phase) + 0.12 * Math.sin(2 * Math.PI * 0.05 * t + id);
      this._composeHinge(Mparent, A, B, angle, this._M[id]);
    }

    const yaw   = 0.25 * Math.sin(2 * Math.PI * 0.03 * t);
    const pitch = 0.18 * Math.sin(2 * Math.PI * 0.021 * t + 1.2);
    const Mglobal = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(pitch, yaw, 0, 'YXZ'));

    const pos = this.geometry.getAttribute('position');
    const nrm = this.geometry.getAttribute('normal');

    let vOffset = 0, nOffset = 0;
    const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();

    for (let i = 0; i < this.faces.length; i++) {
      const M = new THREE.Matrix4().multiplyMatrices(Mglobal, this._M[i]);
      const fr = this.faces[i].rest;

      const w0 = this.faces[i].world[0].copy(fr[0]).applyMatrix4(M);
      const w1 = this.faces[i].world[1].copy(fr[1]).applyMatrix4(M);
      const w2 = this.faces[i].world[2].copy(fr[2]).applyMatrix4(M);

      pos.setXYZ(vOffset + 0, w0.x, w0.y, w0.z);
      pos.setXYZ(vOffset + 1, w1.x, w1.y, w1.z);
      pos.setXYZ(vOffset + 2, w2.x, w2.y, w2.z);

      a.subVectors(w1, w0);
      b.subVectors(w2, w0);
      c.copy(a).cross(b).normalize();

      nrm.setXYZ(nOffset + 0, c.x, c.y, c.z);
      nrm.setXYZ(nOffset + 1, c.x, c.y, c.z);
      nrm.setXYZ(nOffset + 2, c.x, c.y, c.z);

      vOffset += 3;
      nOffset += 3;
    }

    pos.needsUpdate = true;
    nrm.needsUpdate = true;

    const uniforms = this.mesh.material.userData._stripeUniforms;
    if (uniforms) uniforms.uTime.value = t;
  }
}

// ---------- Create material + mesh ----------
const material = new THREE.MeshPhysicalMaterial({
  color: 0x151515,
  roughness: 0.28,
  metalness: 0.0,
  envMapIntensity: 1.15,
  iridescence: 1.0,
  iridescenceIOR: 1.3,
  iridescenceThicknessRange: [120, 620],
  flatShading: true
});
const poly = new RigidHingePoly({ material });
scene.add(poly.group);

// ---------- GUI (minimal, right panel) ----------
const uiHost = document.getElementById('ui');
const gui = new GUI({ title: 'Controls', width: 280 });
uiHost.appendChild(gui.domElement);

const params = {
  play: true,
  exposure: renderer.toneMappingExposure,
  envIntensity: 1.15,
  iridescence: material.iridescence,
  ior: material.iridescenceIOR,
  filmMin: material.iridescenceThicknessRange[0],
  filmMax: material.iridescenceThicknessRange[1],
  bandAngleDeg: 32,
  bandFreq: 6.0,
  bandStrength: 0.45,
  bandSpeed: 0.25,
  resetCamera: () => {
    camera.position.set(3.0, 1.7, 4.5);
    controls.target.set(0, 0, 0);
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

const fBands = gui.addFolder('Interference Bands');
fBands.add(params, 'bandAngleDeg', 0, 180, 0.1).name('Angle (°)')
  .onChange(v => {
    const u = poly.mesh.material.userData._stripeUniforms;
    if (u) u.uBandAngle.value = THREE.MathUtils.degToRad(v);
  });
fBands.add(params, 'bandFreq', 1.0, 20.0, 0.1).name('Frequency')
  .onChange(v => {
    const u = poly.mesh.material.userData._stripeUniforms;
    if (u) u.uBandFreq.value = v;
  });
fBands.add(params, 'bandStrength', 0.0, 1.0, 0.01).name('Strength')
  .onChange(v => {
    const u = poly.mesh.material.userData._stripeUniforms;
    if (u) u.uBandStrength.value = v;
  });
fBands.add(params, 'bandSpeed', 0.0, 2.0, 0.001).name('Angle Speed')
  .onChange(v => {
    const u = poly.mesh.material.userData._stripeUniforms;
    if (u) u.uBandSpeed.value = v;
  });

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
