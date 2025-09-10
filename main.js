import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import * as BufferGeometryUtils from 'three/addons/utils/BufferGeometryUtils.js';
import GUI from 'lil-gui';

// Boot after DOM is ready
if (document.readyState === 'loading') window.addEventListener('DOMContentLoaded', boot);
else boot();

function boot() {
  // Mounts
  const container = document.getElementById('viewport');
  const canvas = document.createElement('canvas');
  canvas.id = 'three-canvas';
  container.appendChild(canvas);

  // GL context (WebGL2 → WebGL1)
  const glAttribs = { antialias: true, alpha: false, premultipliedAlpha: false, powerPreference: 'high-performance' };
  const gl = canvas.getContext('webgl2', glAttribs) || canvas.getContext('webgl', glAttribs);
  if (!gl) { alert('WebGL not available'); return; }

  // Renderer / scene / camera
  const renderer = new THREE.WebGLRenderer({ canvas, context: gl });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000000);

  const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 100);
  camera.position.set(3.6, 2.2, 4.8);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;

  // PMREM + RoomEnvironment for PBR reflections (recommended for PhysicalMaterial). :contentReference[oaicite:2]{index=2}
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envMap = pmrem.fromScene(new RoomEnvironment()).texture;
  scene.environment = envMap;

  // =========================
  // Parameters & GUI
  // =========================
  const params = {
    // Folding
    play: true,
    speed: 0.38,
    maxAngleDeg: 88,    // large to really "fold"
    stagger: 0.55,      // phase offset between hinges
    autoRotate: false,  // keep off so folds are obvious

    // Material & texture
    iridescence: 1.0,
    iriIOR: 1.35,
    iriMinNm: 120,
    iriMaxNm: 950,
    transmission: 0.04,
    thickness: 0.65,
    roughness: 0.12,
    clearcoat: 1.0,
    clearcoatRoughness: 0.22,
    envIntensity: 1.25,

    // Diffractive band overlay
    stripeScale: 28.0,
    stripeStrength: 0.95,
    stripeSharpness: 8.0,
    rainbowShift: 0.0,
    stripeMix: 0.55,
    edgeGlow: 1.2
  };

  // =========================
  // Geometry (indexed!) + base copy
  // =========================
  // Start from an octahedron (8 faces), then MERGE vertices to make it indexed so
  // adjacent faces really share hinge edges → cohesive folding (no cracks). :contentReference[oaicite:3]{index=3}
  let geo = new THREE.OctahedronGeometry(1.35, 0);
  geo = BufferGeometryUtils.mergeVertices(geo, 1e-5);
  geo.computeVertexNormals();

  const basePositions = geo.attributes.position.array.slice(); // immutable baseline

  // =========================
  // Material: Physical + custom diffraction overlay
  // =========================
  const material = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    side: THREE.DoubleSide,
    flatShading: true,                       // crisp facets
    metalness: 0.08,
    roughness: params.roughness,
    clearcoat: params.clearcoat,
    clearcoatRoughness: params.clearcoatRoughness,
    envMapIntensity: params.envIntensity,
    ior: 1.5,
    transmission: params.transmission,
    thickness: params.thickness,
    attenuationColor: new THREE.Color(0xffffff),
    attenuationDistance: 1.5,
    iridescence: params.iridescence,         // thin‑film iridescence in MeshPhysicalMaterial (docs). :contentReference[oaicite:4]{index=4}
    iridescenceIOR: params.iriIOR,
    iridescenceThicknessRange: [params.iriMinNm, params.iriMaxNm]
  });

  // Overlay approximate diffraction grating + edge Fresnel accent; injected via onBeforeCompile (supported pattern). :contentReference[oaicite:5]{index=5}
  material.onBeforeCompile = (shader) => {
    shader.uniforms.uTime = { value: 0.0 };
    shader.uniforms.uStripeScale = { value: params.stripeScale };
    shader.uniforms.uStripeStrength = { value: params.stripeStrength };
    shader.uniforms.uStripeSharpness = { value: params.stripeSharpness };
    shader.uniforms.uRainbowShift = { value: params.rainbowShift };
    shader.uniforms.uStripeMix = { value: params.stripeMix };
    shader.uniforms.uEdgeGlow = { value: params.edgeGlow };

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
      uniform float uStripeSharpness;
      uniform float uRainbowShift;
      uniform float uStripeMix;
      uniform float uEdgeGlow;
      varying vec3 vWorldPos;
      varying vec3 vWorldNormal;

      // GPU Gems-style spectral ramp (approx). :contentReference[oaicite:6]{index=6}
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

        // Two planar "grating" band sets with different frequencies -> moiré-like rainbow.
        float a = dot(vWorldPos, T) * uStripeScale + uRainbowShift + uTime*0.45;
        float b = dot(vWorldPos, B) * (uStripeScale * 0.67) - (uRainbowShift*1.37) + uTime*0.31;

        // Sharper stripes using smoothstep around sin (controls contrast)
        float s1 = smoothstep(0.5 - 0.5/pow(uStripeSharpness,0.5), 0.5 + 0.5/pow(uStripeSharpness,0.5), 0.5 + 0.5 * sin(a));
        float s2 = smoothstep(0.5 - 0.5/pow(uStripeSharpness,0.5), 0.5 + 0.5/pow(uStripeSharpness,0.5), 0.5 + 0.5 * sin(b));
        float bands = mix(s1, s2, uStripeMix);

        vec3 rainbow = spectral(bands);

        // Slight Fresnel rim to mimic the bright edges seen in the contact sheet
        vec3 V = normalize(cameraPosition - vWorldPos);
        float fres = pow(1.0 - max(dot(normalize(N), V), 0.0), 3.0);

        diffuseColor.rgb = mix(diffuseColor.rgb, diffuseColor.rgb + rainbow, uStripeStrength);
        diffuseColor.rgb += fres * uEdgeGlow * 0.12;
      `);

    material.userData.shader = shader;
  };

  const mesh = new THREE.Mesh(geo, material);
  scene.add(mesh);

  // =========================
  // Build hinges + precompute affected vertex sets
  // =========================
  // Unique vertex positions (from indexed geometry)
  const verts = [];
  const pos = geo.attributes.position.array;
  for (let i = 0; i < pos.length; i += 3) verts.push(new THREE.Vector3(pos[i], pos[i+1], pos[i+2]));

  // Find top/bottom and sort equator by angle
  let top = verts[0], bottom = verts[0];
  for (const v of verts) { if (v.y > top.y) top = v; if (v.y < bottom.y) bottom = v; }
  const equator = verts.filter(v => v !== top && v !== bottom)
                       .sort((a,b) => Math.atan2(a.z, a.x) - Math.atan2(b.z, b.x));

  // Hinge helper
  const hinges = [];
  const centroid = new THREE.Vector3(0,0,0);
  for (const v of verts) centroid.add(v); centroid.multiplyScalar(1/verts.length);

  function addHinge(p0, p1, phase, sign) {
    const origin = p0.clone();
    const axis = p1.clone().sub(p0).normalize();         // edge direction
    // Plane containing the axis and roughly facing away from the centroid:
    // n = cross(axis, (origin - centroid))  → guarantees axis lies in plane. 
    const planeN = new THREE.Vector3().crossVectors(axis, origin.clone().sub(centroid)).normalize();

    // Precompute which vertex indices belong to the "folding" side of this hinge
    const affected = [];
    const axisPoint = origin.clone();
    const axisDir = axis.clone();
    const AXIS_EPS = 1e-5;

    // Utility: distance from point to the hinge line
    const tmp = new THREE.Vector3(), w = new THREE.Vector3();
    function distToAxis(p) {
      w.copy(p).sub(axisPoint);
      const projLen = w.dot(axisDir);
      tmp.copy(axisDir).multiplyScalar(projLen);
      return w.sub(tmp).length();
    }

    for (let i = 0; i < verts.length; i++) {
      const p = verts[i];
      // If the vertex lies on the axis, we don't assign it (it stays fixed → true hinge).
      if (distToAxis(p) < AXIS_EPS) continue;
      const side = p.clone().sub(origin).dot(planeN);
      if (side > 0) affected.push(i);
    }

    hinges.push({ origin, axis, planeN, affected, phase, sign });
  }

  // Create 8 hinges (top↔equator[i], bottom↔equator[i]), time‑staggered to create delayed folding.
  for (let i = 0; i < 4; i++) {
    const E = equator[i];
    const Enext = equator[(i + 1) % 4];

    addHinge(top, E, i * params.stagger, +1.0);
    addHinge(bottom, E, (i + 0.5) * params.stagger, -1.0);

    // Optional: add a couple of diagonals for richer silhouettes at some times
    addHinge(E, Enext, (i + 0.25) * params.stagger, (i % 2 === 0) ? 1.0 : -1.0);
  }

  // Rotate a point around an axis line (Rodrigues formula).
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

  // Folding step (from immutable baseline → cohesive, no drift)
  const posAttr = geo.attributes.position;
  const EPS = 1e-6;

  function applyFolds(t) {
    const maxAngle = THREE.MathUtils.degToRad(params.maxAngleDeg);
    const count = posAttr.count; // number of vertices
    for (let vi = 0; vi < count; vi++) {
      // start from baseline
      const i3 = vi * 3;
      _p.set(basePositions[i3], basePositions[i3+1], basePositions[i3+2]);

      for (let h = 0; h < hinges.length; h++) {
        const H = hinges[h];
        // vertex membership test is precomputed
        if (!H.affectedSet) H.affectedSet = new Set(H.affected);
        if (!H.affectedSet.has(vi)) continue;

        // Triangular ping‑pong with smoothstep ease → “delayed” folding cadence
        const tt = (t * params.speed + H.phase);
        const tri = Math.abs((tt % 2) - 1);                 // 0..1..0
        const eased = tri * tri * (3 - 2 * tri);            // smoothstep
        const angle = (eased - 0.5) * 2.0 * maxAngle * H.sign;

        _p.copy(rotateAroundAxisLine(_p, H.origin, H.axis, angle));
      }

      pos[i3] = _p.x; pos[i3+1] = _p.y; pos[i3+2] = _p.z;
    }
    posAttr.needsUpdate = true;
    geo.computeVertexNormals(); // recalc per-frame (flat shading will keep faceted look)
  }

  // =========================
  // GUI
  // =========================
  const gui = new GUI({ container: document.getElementById('ui'), width: 280, title: 'Controls' });
  gui.add(params, 'play').name('Play/Pause');
  gui.add(params, 'autoRotate').name('Auto rotate');
  gui.add(params, 'speed', 0.02, 1.5, 0.01).name('Fold speed');
  gui.add(params, 'maxAngleDeg', 5, 140, 1).name('Fold amplitude (°)');
  gui.add(params, 'stagger', 0.0, 1.5, 0.01).name('Time offset');

  const fMat = gui.addFolder('Material');
  fMat.add(params, 'iridescence', 0.0, 1.0, 0.01).name('Iridescence').onChange(v => material.iridescence = v);
  fMat.add(params, 'iriIOR', 1.0, 2.0, 0.01).name('Iri IOR').onChange(v => material.iridescenceIOR = v);
  fMat.add(params, 'iriMinNm', 50, 500, 1).name('Iri min (nm)').onChange(() => material.iridescenceThicknessRange = [params.iriMinNm, params.iriMaxNm]);
  fMat.add(params, 'iriMaxNm', 200, 1200, 1).name('Iri max (nm)').onChange(() => material.iridescenceThicknessRange = [params.iriMinNm, params.iriMaxNm]);
  fMat.add(params, 'transmission', 0.0, 0.4, 0.01).name('Transmission').onChange(v => material.transmission = v);
  fMat.add(params, 'thickness', 0.05, 2.0, 0.01).name('Thickness').onChange(v => material.thickness = v);
  fMat.add(params, 'roughness', 0.0, 0.6, 0.001).name('Roughness').onChange(v => material.roughness = v);
  fMat.add(params, 'clearcoat', 0.0, 1.0, 0.01).name('Clearcoat').onChange(v => material.clearcoat = v);
  fMat.add(params, 'clearcoatRoughness', 0.0, 1.0, 0.01).name('ClearcoatRough').onChange(v => material.clearcoatRoughness = v);
  fMat.add(params, 'envIntensity', 0.1, 3.0, 0.01).name('Env Intensity').onChange(v => material.envMapIntensity = v);

  const fTex = gui.addFolder('Diffractive stripes');
  fTex.add(params, 'stripeScale', 4.0, 60.0, 0.1).name('Scale').onChange(v => material.userData.shader && (material.userData.shader.uniforms.uStripeScale.value = v));
  fTex.add(params, 'stripeStrength', 0.0, 1.0, 0.01).name('Strength').onChange(v => material.userData.shader && (material.userData.shader.uniforms.uStripeStrength.value = v));
  fTex.add(params, 'stripeSharpness', 1.0, 20.0, 0.1).name('Sharpness').onChange(v => material.userData.shader && (material.userData.shader.uniforms.uStripeSharpness.value = v));
  fTex.add(params, 'stripeMix', 0.0, 1.0, 0.01).name('Dual mix').onChange(v => material.userData.shader && (material.userData.shader.uniforms.uStripeMix.value = v));
  fTex.add(params, 'rainbowShift', -Math.PI, Math.PI, 0.01).name('Phase').onChange(v => material.userData.shader && (material.userData.shader.uniforms.uRainbowShift.value = v));
  fTex.add(params, 'edgeGlow', 0.0, 2.5, 0.01).name('Edge glow').onChange(v => material.userData.shader && (material.userData.shader.uniforms.uEdgeGlow.value = v));

  // =========================
  // Resize & input
  // =========================
  window.addEventListener('resize', () => {
    const w = container.clientWidth, h = container.clientHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  });
  window.addEventListener('keydown', (e) => { if (e.code === 'Space') params.play = !params.play; });

  // =========================
  // Animate
  // =========================
  const clock = new THREE.Clock();

  function frame() {
    const t = clock.getElapsedTime();
    if (params.play) applyFolds(t);
    if (material.userData.shader) {
      material.userData.shader.uniforms.uTime.value = t;
      material.userData.shader.uniforms.uStripeScale.value = params.stripeScale;
      material.userData.shader.uniforms.uStripeStrength.value = params.stripeStrength;
      material.userData.shader.uniforms.uStripeSharpness.value = params.stripeSharpness;
      material.userData.shader.uniforms.uRainbowShift.value = params.rainbowShift;
      material.userData.shader.uniforms.uStripeMix.value = params.stripeMix;
      material.userData.shader.uniforms.uEdgeGlow.value = params.edgeGlow;
    }
    if (params.autoRotate) mesh.rotation.y += 0.0023;

    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }
  frame();
}
