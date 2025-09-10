import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import GUI from 'lil-gui';

/* ----------------------------------------------------------------------------
   Scene / Renderer
---------------------------------------------------------------------------- */
const container = document.getElementById('viewport');

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 100);
camera.position.set(3.5, 2.2, 4.5);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;

/* ----------------------------------------------------------------------------
   Lighting + Environment (PMREM + RoomEnvironment)
---------------------------------------------------------------------------- */
const pmrem = new THREE.PMREMGenerator(renderer);
const envScene = new RoomEnvironment();
const envMap = pmrem.fromScene(envScene).texture;
scene.environment = envMap;

/* ----------------------------------------------------------------------------
   Parameters (GUI)
---------------------------------------------------------------------------- */
const params = {
  // Animation / folding
  play: true,
  speed: 0.26,
  maxAngleDeg: 82.0,
  stagger: 0.55,        // time offset between folds
  autoRotate: true,

  // Material / texture feel
  iridescence: 1.0,
  iriIOR: 1.3,
  iriMinNm: 120.0,
  iriMaxNm: 650.0,
  transmission: 0.06,
  thickness: 0.55,
  roughness: 0.08,
  clearcoat: 1.0,
  clearcoatRoughness: 0.22,
  envIntensity: 1.25,

  // Diffractive stripe overlay
  stripeScale: 22.0,
  stripeStrength: 0.55,
  rainbowShift: 0.0,
  stripeMix: 0.5
};

/* ----------------------------------------------------------------------------
   Geometry: low‑poly octahedron — cohesive fold base
---------------------------------------------------------------------------- */
const baseRadius = 1.35;
const geo = new THREE.OctahedronGeometry(baseRadius, 0);
geo.computeVertexNormals();

// keep a copy for deformation baseline (so vertices never drift)
const basePositions = geo.attributes.position.array.slice();

/* ----------------------------------------------------------------------------
   Material: iridescent physical + diffractive stripe overlay (onBeforeCompile)
---------------------------------------------------------------------------- */
const material = new THREE.MeshPhysicalMaterial({
  color: 0xffffff,
  side: THREE.DoubleSide,
  flatShading: true,
  metalness: 0.1,
  roughness: params.roughness,
  clearcoat: params.clearcoat,
  clearcoatRoughness: params.clearcoatRoughness,
  envMapIntensity: params.envIntensity,
  ior: 1.5,
  transmission: params.transmission,
  thickness: params.thickness,
  attenuationColor: new THREE.Color(0xffffff),
  attenuationDistance: 1.2,
  iridescence: params.iridescence,
  iridescenceIOR: params.iriIOR,
  iridescenceThicknessRange: [params.iriMinNm, params.iriMaxNm]
});

// Inject a diffractive grating-esque overlay using onBeforeCompile.
material.onBeforeCompile = (shader) => {
  shader.uniforms.uTime = { value: 0.0 };
  shader.uniforms.uStripeScale = { value: params.stripeScale };
  shader.uniforms.uStripeStrength = { value: params.stripeStrength };
  shader.uniforms.uRainbowShift = { value: params.rainbowShift };
  shader.uniforms.uStripeMix = { value: params.stripeMix };

  shader.vertexShader = `
    varying vec3 vWorldPos;
    varying vec3 vWorldNormal;
  ` + shader.vertexShader
    .replace('#include <beginnormal_vertex>', `
      #include <beginnormal_vertex>
      vWorldNormal = normalize( mat3( modelMatrix ) * objectNormal );
    `)
    .replace('#include <begin_vertex>', `
      #include <begin_vertex>
      vWorldPos = ( modelMatrix * vec4( transformed, 1.0 ) ).xyz;
    `);

  shader.fragmentShader = `
    uniform float uTime;
    uniform float uStripeScale;
    uniform float uStripeStrength;
    uniform float uRainbowShift;
    uniform float uStripeMix;
    varying vec3 vWorldPos;
    varying vec3 vWorldNormal;

    // Cheap spectral ramp (0..1 -> RGB rainbow)
    vec3 spectral(float x){
      x = clamp(fract(x), 0.0, 1.0);
      return clamp(vec3(
        abs(x*6.0-3.0)-1.0,
        2.0-abs(x*6.0-2.0),
        2.0-abs(x*6.0-4.0)
      ), 0.0, 1.0);
    }
  ` + shader.fragmentShader
    .replace('#include <begin_fragment>', `
      #include <begin_fragment>
      vec3 N = normalize(vWorldNormal);
      vec3 helper = (abs(N.y) > 0.98) ? vec3(1.0,0.0,0.0) : vec3(0.0,1.0,0.0);
      vec3 T = normalize(cross(N, helper));
      vec3 B = normalize(cross(N, T));

      // Two stripe sets -> moiré-like diffractive feel
      float s1 = 0.5 + 0.5 * sin( dot(vWorldPos, T) * uStripeScale + uRainbowShift + uTime*0.6 );
      float s2 = 0.5 + 0.5 * sin( dot(vWorldPos, B) * (uStripeScale * 0.63) - (uRainbowShift*1.37) + uTime*0.37 );
      float bands = mix(s1, s2, uStripeMix);
      vec3 rainbow = spectral(bands);

      diffuseColor.rgb = mix(diffuseColor.rgb, diffuseColor.rgb + rainbow, uStripeStrength);
    `);

  material.userData.shader = shader;
};

const mesh = new THREE.Mesh(geo, material);
scene.add(mesh);

/* ----------------------------------------------------------------------------
   Hinged folds along octahedron edges (no polygon separation)
---------------------------------------------------------------------------- */

// Extract unique vertices from non-indexed geometry
function uniqueVertsFromGeometry(g) {
  const pos = g.attributes.position.array;
  const map = new Map();
  const verts = [];
  for (let i = 0; i < pos.length; i += 3) {
    const x = +pos[i].toFixed(5), y = +pos[i + 1].toFixed(5), z = +pos[i + 2].toFixed(5);
    const key = `${x},${y},${z}`;
    if (!map.has(key)) {
      map.set(key, verts.length);
      verts.push(new THREE.Vector3(x, y, z));
    }
  }
  return verts;
}
const uniq = uniqueVertsFromGeometry(geo);

// Identify top / bottom / equator
let top = uniq[0], bottom = uniq[0];
for (const v of uniq) {
  if (v.y > top.y) top = v;
  if (v.y < bottom.y) bottom = v;
}
const equator = uniq.filter(v => v !== top && v !== bottom)
  .sort((a, b) => Math.atan2(a.z, a.x) - Math.atan2(b.z, b.x));

// Define hinges: 4 around the top, 4 around the bottom
const hinges = [];
for (let i = 0; i < 4; i++) {
  const E = equator[i];
  const Enext = equator[(i + 1) % 4];

  // Top hinges
  const axisTop = E.clone().sub(top).normalize();
  const faceNTop = new THREE.Vector3().crossVectors(
    E.clone().sub(top),
    Enext.clone().sub(top)
  ).normalize();
  const planeNTop = new THREE.Vector3().crossVectors(axisTop, faceNTop).normalize();
  hinges.push({
    origin: top.clone(),
    axis: axisTop.clone(),
    planeN: planeNTop.clone(),
    phase: i * params.stagger,
    sign: 1.0
  });

  // Bottom hinges (interleaved)
  const axisBot = E.clone().sub(bottom).normalize();
  const faceNBot = new THREE.Vector3().crossVectors(
    Enext.clone().sub(bottom),
    E.clone().sub(bottom)
  ).normalize();
  const planeNBot = new THREE.Vector3().crossVectors(axisBot, faceNBot).normalize();
  hinges.push({
    origin: bottom.clone(),
    axis: axisBot.clone(),
    planeN: planeNBot.clone(),
    phase: (i + 0.5) * params.stagger,
    sign: -1.0
  });
}

// Rodrigues rotation around a line (axis through origin)
const _a = new THREE.Vector3(), _r = new THREE.Vector3(), _p = new THREE.Vector3();
function rotateAroundAxisLine(point, origin, axis, angle) {
  _p.copy(point).sub(origin);
  const cos = Math.cos(angle), sin = Math.sin(angle);
  _a.copy(axis);
  _r.copy(_p).multiplyScalar(cos)
    .add(_a.clone().cross(_p).multiplyScalar(sin))
    .add(_a.multiplyScalar(_a.dot(_p) * (1.0 - cos)));
  return _r.add(origin);
}

/* Apply folds every frame (from immutable baseline for cohesion) */
const posAttr = geo.attributes.position;
const pos = posAttr.array;
const basePos = basePositions;
const EPS = 1e-6;

function applyFolds(t) {
  const maxAngle = THREE.MathUtils.degToRad(params.maxAngleDeg);
  for (let i = 0; i < pos.length; i += 3) {
    let vx = basePos[i], vy = basePos[i + 1], vz = basePos[i + 2];
    _p.set(vx, vy, vz);

    // Apply each hinge
    for (let h = 0; h < hinges.length; h++) {
      const H = hinges[h];
      const tt = (t * params.speed + H.phase);
      const tri = Math.abs((tt % 2) - 1);          // ping-pong 0..1..0
      const eased = tri * tri * (3 - 2 * tri);     // smoothstep
      const angle = (eased - 0.5) * 2.0 * maxAngle * H.sign;

      const side = (_p.clone().sub(H.origin)).dot(H.planeN);
      if (side > EPS) {
        _p.copy(rotateAroundAxisLine(_p, H.origin, H.axis, angle));
      }
    }

    pos[i] = _p.x;
    pos[i + 1] = _p.y;
    pos[i + 2] = _p.z;
  }
  posAttr.needsUpdate = true;
  geo.computeVertexNormals();
}

/* ----------------------------------------------------------------------------
   GUI
---------------------------------------------------------------------------- */
const gui = new GUI({ container: document.getElementById('ui'), width: 280, title: 'Controls' });
gui.add(params, 'play').name('Play/Pause');
gui.add(params, 'autoRotate').name('Auto rotate');
gui.add(params, 'speed', 0.02, 1.2, 0.01).name('Fold speed');
gui.add(params, 'maxAngleDeg', 5, 140, 1).name('Fold amplitude (°)');
gui.add(params, 'stagger', 0.0, 1.5, 0.01).name('Time offset');

const fMat = gui.addFolder('Material');
fMat.add(params, 'iridescence', 0.0, 1.0, 0.01).name('Iridescence').onChange(v => { material.iridescence = v; });
fMat.add(params, 'iriIOR', 1.0, 2.0, 0.01).name('Iridescence IOR').onChange(v => { material.iridescenceIOR = v; });
fMat.add(params, 'iriMinNm', 50, 500, 1).name('Iri min (nm)').onChange(() => { material.iridescenceThicknessRange = [params.iriMinNm, params.iriMaxNm]; });
fMat.add(params, 'iriMaxNm', 200, 1200, 1).name('Iri max (nm)').onChange(() => { material.iridescenceThicknessRange = [params.iriMinNm, params.iriMaxNm]; });
fMat.add(params, 'transmission', 0.0, 0.5, 0.01).name('Transmission').onChange(v => { material.transmission = v; });
fMat.add(params, 'thickness', 0.05, 2.0, 0.01).name('Thickness').onChange(v => { material.thickness = v; });
fMat.add(params, 'roughness', 0.0, 0.6, 0.001).name('Roughness').onChange(v => { material.roughness = v; });
fMat.add(params, 'clearcoat', 0.0, 1.0, 0.01).name('Clearcoat').onChange(v => { material.clearcoat = v; });
fMat.add(params, 'clearcoatRoughness', 0.0, 1.0, 0.01).name('ClearcoatRough').onChange(v => { material.clearcoatRoughness = v; });
fMat.add(params, 'envIntensity', 0.1, 3.0, 0.01).name('Env Intensity').onChange(v => { material.envMapIntensity = v; });

const fTex = gui.addFolder('Diffractive stripes');
fTex.add(params, 'stripeScale', 2.0, 60.0, 0.1).name('Scale').onChange(v => { if (material.userData.shader) material.userData.shader.uniforms.uStripeScale.value = v; });
fTex.add(params, 'stripeStrength', 0.0, 1.0, 0.01).name('Strength').onChange(v => { if (material.userData.shader) material.userData.shader.uniforms.uStripeStrength.value = v; });
fTex.add(params, 'stripeMix', 0.0, 1.0, 0.01).name('Dual mix').onChange(v => { if (material.userData.shader) material.userData.shader.uniforms.uStripeMix.value = v; });
fTex.add(params, 'rainbowShift', -Math.PI, Math.PI, 0.01).name('Phase').onChange(v => { if (material.userData.shader) material.userData.shader.uniforms.uRainbowShift.value = v; });

/* ----------------------------------------------------------------------------
   Resize & Input
---------------------------------------------------------------------------- */
window.addEventListener('resize', () => {
  const w = container.clientWidth, h = container.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
});

window.addEventListener('keydown', (e) => {
  if (e.code === 'Space') params.play = !params.play;
});

/* ----------------------------------------------------------------------------
   Animate
---------------------------------------------------------------------------- */
const clock = new THREE.Clock();

function render() {
  const t = clock.getElapsedTime();
  if (params.play) applyFolds(t);

  if (material.userData.shader) {
    material.userData.shader.uniforms.uTime.value = t;
    material.userData.shader.uniforms.uStripeScale.value = params.stripeScale;
    material.userData.shader.uniforms.uStripeStrength.value = params.stripeStrength;
    material.userData.shader.uniforms.uRainbowShift.value = params.rainbowShift;
    material.userData.shader.uniforms.uStripeMix.value = params.stripeMix;
  }

  if (params.autoRotate) mesh.rotation.y += 0.0025;

  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(render);
}
render();
