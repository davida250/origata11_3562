import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import GUI from 'lil-gui';

/* ============================================================================
   Boot after DOM ready
============================================================================ */
if (document.readyState === 'loading') window.addEventListener('DOMContentLoaded', start);
else start();

function start() {
  /* ------------------------------------------------------------------------ */
  /* Canvas + GL                                                              */
  /* ------------------------------------------------------------------------ */
  const container = document.getElementById('viewport');
  const canvas = document.createElement('canvas');
  canvas.id = 'three-canvas';
  container.appendChild(canvas);

  const glAttribs = { antialias: true, alpha: false, premultipliedAlpha: false, powerPreference: 'high-performance' };
  const gl = canvas.getContext('webgl2', glAttribs) || canvas.getContext('webgl', glAttribs);
  if (!gl) { alert('WebGL not available'); return; }

  /* ------------------------------------------------------------------------ */
  /* Renderer / Scene / Camera / Controls                                     */
  /* ------------------------------------------------------------------------ */
  const renderer = new THREE.WebGLRenderer({ canvas, context: gl });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000000);

  const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 100);
  camera.position.set(3.8, 2.4, 4.8);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;

  // PMREM + RoomEnvironment → correct PBR lighting for PhysicalMaterial. :contentReference[oaicite:1]{index=1}
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment()).texture;

  /* ------------------------------------------------------------------------ */
  /* Parameters                                                               */
  /* ------------------------------------------------------------------------ */
  const P = {
    // Folding
    play: true,
    speed: 0.34,            // global speed
    amplitudeDeg: 95,       // how far the folds swing
    stagger: 0.55,          // time offset between hinges
    autoRotate: false,

    // Material / “contact‑sheet” look
    iridescence: 1.0,       // thin‑film layer (MeshPhysicalMaterial) :contentReference[oaicite:2]{index=2}
    iriIOR: 1.35,
    iriMinNm: 120,
    iriMaxNm: 950,
    transmission: 0.04,     // a hint of translucency makes edges glow
    thickness: 0.65,
    roughness: 0.12,
    clearcoat: 1.0,
    clearcoatRoughness: 0.22,
    envIntensity: 1.25,

    // Birefringent / diffractive stripes (shader overlay through onBeforeCompile) :contentReference[oaicite:3]{index=3}
    bandScale: 26.0,
    bandStrength: 0.95,
    bandSharpness: 9.0,
    bandMix: 0.55,
    bandPhase: 0.0,
    edgeGlow: 1.35
  };

  /* ------------------------------------------------------------------------ */
  /* Geometry — Irregular bipyramid (N-gon ring + top/bottom)                 */
  /* Why this shape? The contact sheet silhouettes look like a rigid “paper”
     core with ears/petals. A jittered bipyramid provides the same family of
     silhouettes while staying convex at rest. We’ll fold along explicit
     edges so it self-overlaps but remains cohesive (no polygon explosion).
  ------------------------------------------------------------------------ */
  const N = 6;                 // ring vertex count (5–7 look good)
  const R = 1.25;              // ring radius
  const H = 1.25;              // half height
  const jitterR = 0.12;        // small irregularity to avoid symmetry
  const jitterY = 0.07;

  const verts = [];
  // ring
  for (let i = 0; i < N; i++) {
    const t = (i / N) * Math.PI * 2;
    const jr = (Math.random() * 2 - 1) * jitterR;
    const jy = (Math.random() * 2 - 1) * jitterY;
    verts.push(new THREE.Vector3(Math.cos(t) * (R + jr), jy, Math.sin(t) * (R + jr)));
  }
  const top = new THREE.Vector3(0, H, 0);
  const bottom = new THREE.Vector3(0, -H, 0);
  const topIndex = verts.push(top) - 1;
  const bottomIndex = verts.push(bottom) - 1;

  // Triangles: top→ring and bottom→ring
  const indices = [];
  for (let i = 0; i < N; i++) {
    const a = i;
    const b = (i + 1) % N;
    indices.push(topIndex, a, b);     // top face
    indices.push(bottomIndex, b, a);  // bottom face
  }

  // Build BufferGeometry (indexed!)
  const geo = new THREE.BufferGeometry();
  const posArr = new Float32Array(verts.length * 3);
  for (let i = 0; i < verts.length; i++) {
    posArr[i*3] = verts[i].x; posArr[i*3+1] = verts[i].y; posArr[i*3+2] = verts[i].z;
  }
  geo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  const basePositions = geo.attributes.position.array.slice(); // immutable baseline for deformation

  /* ------------------------------------------------------------------------ */
  /* Material: Physical + custom birefringence/diffraction overlay            */
  /* - Thin‑film iridescence from MeshPhysicalMaterial (official).            */
  /* - Striped interference overlay injected with onBeforeCompile.            */
  /*   (Three.js supports this for extending materials.)                      */
  /*   Bands are oriented per-facet using a TBN-like basis from normal.       */
  /*   Two band sets blended -> moiré; Fresnel rim lifts edges.               */
  /* References: onBeforeCompile usage / thin‑film & birefringence background */
  /*   docs/discussions. :contentReference[oaicite:4]{index=4}           */
  const material = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    side: THREE.DoubleSide,
    flatShading: true,
    metalness: 0.08,
    roughness: P.roughness,
    clearcoat: P.clearcoat,
    clearcoatRoughness: P.clearcoatRoughness,
    envMapIntensity: P.envIntensity,
    ior: 1.5,
    transmission: P.transmission,
    thickness: P.thickness,
    attenuationColor: new THREE.Color(0xffffff),
    attenuationDistance: 1.5,
    iridescence: P.iridescence,
    iridescenceIOR: P.iriIOR,
    iridescenceThicknessRange: [P.iriMinNm, P.iriMaxNm]
  });

  material.onBeforeCompile = (shader) => {
    shader.uniforms.uTime = { value: 0.0 };
    shader.uniforms.uBandScale = { value: P.bandScale };
    shader.uniforms.uBandStrength = { value: P.bandStrength };
    shader.uniforms.uBandSharpness = { value: P.bandSharpness };
    shader.uniforms.uBandMix = { value: P.bandMix };
    shader.uniforms.uBandPhase = { value: P.bandPhase };
    shader.uniforms.uEdgeGlow = { value: P.edgeGlow };

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
      uniform float uTime, uBandScale, uBandStrength, uBandSharpness, uBandMix, uBandPhase, uEdgeGlow;
      varying vec3 vWorldPos;
      varying vec3 vWorldNormal;

      // Approximate spectral ramp (0..1 → RGB) used in GPU literature. :contentReference[oaicite:5]{index=5}
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
        vec3 T = normalize(cross(N, helper)); // per-facet tangent
        vec3 B = normalize(cross(N, T));      // per-facet bitangent

        // Two "birefringent" band sets projected along T and B.
        float a = dot(vWorldPos, T) * uBandScale + uBandPhase + uTime*0.35;
        float b = dot(vWorldPos, B) * (uBandScale*0.66) - (uBandPhase*1.37) + uTime*0.23;

        float s1 = 0.5 + 0.5 * sin(a);
        float s2 = 0.5 + 0.5 * sin(b);

        // Sharpen bands → crisp grating, then mix.
        float k = 0.5 / max(uBandSharpness, 1.0);
        s1 = smoothstep(0.5 - k, 0.5 + k, s1);
        s2 = smoothstep(0.5 - k, 0.5 + k, s2);
        float bands = mix(s1, s2, uBandMix);

        vec3 rainbow = spectral(bands);
        diffuseColor.rgb = mix(diffuseColor.rgb, diffuseColor.rgb + rainbow, uBandStrength);

        // Fresnel-ish edge lift for the bright rims visible in the contact sheet
        vec3 V = normalize(cameraPosition - vWorldPos);
        float fres = pow(1.0 - max(dot(normalize(N), V), 0.0), 3.0);
        diffuseColor.rgb += fres * uEdgeGlow * 0.13;
      `);

    material.userData.shader = shader;
  };

  const mesh = new THREE.Mesh(geo, material);
  scene.add(mesh);

  /* ------------------------------------------------------------------------ */
  /* HINGE ENGINE — explicit edge hinges with BFS partition                   */
  /* 1) Build adjacency + edge→faces map from indexed triangles.              */
  /* 2) For each hinge (edge AB) choose the “wing” side using the triangle    */
  /*    opposite vertex as the BFS seed.                                      */
  /* 3) On each frame: reset verts to baseline, then rotate affected side     */
  /*    around the AB axis by a phase-shifted ping-pong angle.                */
  /* This produces clean folds without polygon separation.                    */
  /* References motivating the approach: rotating around arbitrary axes;      */
  /* pivot/hinge patterns; material extension via onBeforeCompile.            */
  /* :contentReference[oaicite:6]{index=6}                                 */
  const tri = geo.getIndex().array;
  const vcount = geo.attributes.position.count;

  // adjacency (undirected)
  const neighbors = Array.from({ length: vcount }, () => new Set());
  const edgeKey = (a, b) => a < b ? `${a}_${b}` : `${b}_${a}`;
  const edgeFaces = new Map(); // edgeKey -> [faceIdx0, faceIdx1]
  for (let i = 0; i < tri.length; i += 3) {
    const a = tri[i], b = tri[i+1], c = tri[i+2];
    const edges = [[a,b],[b,c],[c,a]];
    edges.forEach(([u,v]) => {
      neighbors[u].add(v); neighbors[v].add(u);
      const k = edgeKey(u,v);
      if (!edgeFaces.has(k)) edgeFaces.set(k, []);
      edgeFaces.get(k).push(i); // store face start index
    });
  }

  function bfsSide(blockA, blockB, seed) {
    // BFS from seed while NOT crossing the blocked edge (blockA<->blockB)
    const q = [seed];
    const visited = new Array(vcount).fill(false);
    visited[seed] = true;
    while (q.length) {
      const v = q.shift();
      for (const n of neighbors[v]) {
        if ((v === blockA && n === blockB) || (v === blockB && n === blockA)) continue;
        if (!visited[n]) { visited[n] = true; q.push(n); }
      }
    }
    return visited;
  }

  function addHinge(a, b, seed, phase, sign) {
    const origin = new THREE.Vector3().fromArray(basePositions, a*3);
    const dir = new THREE.Vector3().fromArray(basePositions, b*3).sub(origin).normalize();
    const affectedMask = bfsSide(a, b, seed); // boolean mask of vertices on the "folding" side
    const affected = [];
    for (let i = 0; i < vcount; i++) if (affectedMask[i] && i !== a && i !== b) affected.push(i);
    hinges.push({ a, b, origin, axis: dir, affected, phase, sign });
  }

  const hinges = [];
  // Helper: for edge (u,v), pick “seed” as the vertex opposite (u,v) on one of the adjacent faces.
  const oppositeVertexOnFace = (faceStart, u, v) => {
    const fa = tri[faceStart], fb = tri[faceStart+1], fc = tri[faceStart+2];
    if (fa !== u && fa !== v) return fa;
    if (fb !== u && fb !== v) return fb;
    return fc;
  };

  // Build three classes of hinges: top↔ring, bottom↔ring, and ring↔ring (valleys)
  for (let i = 0; i < N; i++) {
    const r = i;
    const rNext = (i + 1) % N;

    // Top-ring hinge (edge topIndex—r). Take seed as the opposite vertex on the top triangle.
    const kTop = edgeKey(topIndex, r);
    const fTop = edgeFaces.get(kTop)[0]; // there is always exactly one adjacent "top" face in our mesh
    const seedTop = oppositeVertexOnFace(fTop, topIndex, r);
    addHinge(topIndex, r, seedTop, i * P.stagger, +1.0);

    // Bottom-ring hinge (edge bottomIndex—r). Seed from bottom triangle.
    const kBot = edgeKey(bottomIndex, r);
    const fBot = edgeFaces.get(kBot)[0];
    const seedBot = oppositeVertexOnFace(fBot, bottomIndex, r);
    addHinge(bottomIndex, r, seedBot, (i + 0.5) * P.stagger, -1.0);

    // Ring-ring hinge (edge r—rNext). Seed from one of its two faces (pick the top one if present).
    const kRing = edgeKey(r, rNext);
    const faces = edgeFaces.get(kRing);
    const seedRing = oppositeVertexOnFace(faces[0], r, rNext);
    addHinge(r, rNext, seedRing, (i + 0.25) * P.stagger, (i % 2 === 0) ? 1.0 : -1.0);
  }

  // Rotate a point around an axis line using Rodrigues’ formula.
  const _a = new THREE.Vector3(), _r = new THREE.Vector3(), _p = new THREE.Vector3();
  function rotateAroundAxis(point, origin, axis, angle) {
    _p.copy(point).sub(origin);
    const cos = Math.cos(angle), sin = Math.sin(angle);
    _a.copy(axis);
    _r.copy(_p).multiplyScalar(cos)
      .add(_a.clone().cross(_p).multiplyScalar(sin))
      .add(_a.multiplyScalar(_a.dot(_p) * (1.0 - cos)));
    return _r.add(origin);
  }

  // Folding step (from immutable baseline each frame → cohesive mesh)
  const posAttr = geo.attributes.position;
  const pos = posAttr.array;
  function applyFolds(t) {
    const maxA = THREE.MathUtils.degToRad(P.amplitudeDeg);

    for (let i = 0; i < vcount; i++) {
      pos[i*3]   = basePositions[i*3];
      pos[i*3+1] = basePositions[i*3+1];
      pos[i*3+2] = basePositions[i*3+2];
    }

    for (let h = 0; h < hinges.length; h++) {
      const H = hinges[h];
      // phase‑shifted ping‑pong with smoothstep ease → clear delayed folding rhythm
      const tt = (t * P.speed + H.phase);
      const tri = Math.abs((tt % 2) - 1);                  // 0..1..0
      const eased = tri * tri * (3 - 2 * tri);             // smoothstep
      const angle = (eased - 0.5) * 2.0 * maxA * H.sign;

      // Rotate the affected side; hinge edge vertices (a,b) stay on axis.
      for (let k = 0; k < H.affected.length; k++) {
        const vi = H.affected[k];
        const i3 = vi * 3;
        const p = _p.set(basePositions[i3], basePositions[i3+1], basePositions[i3+2]);
        const q = rotateAroundAxis(p, H.origin, H.axis, angle);
        pos[i3] = q.x; pos[i3+1] = q.y; pos[i3+2] = q.z;
      }
    }
    posAttr.needsUpdate = true;
    geo.computeVertexNormals(); // keep flat facets consistent
  }

  /* ------------------------------------------------------------------------ */
  /* GUI                                                                      */
  /* ------------------------------------------------------------------------ */
  const gui = new GUI({ container: document.getElementById('ui'), width: 280, title: 'Controls' });
  gui.add(P, 'play').name('Play/Pause');
  gui.add(P, 'autoRotate').name('Auto rotate');
  gui.add(P, 'speed', 0.05, 1.5, 0.01).name('Fold speed');
  gui.add(P, 'amplitudeDeg', 10, 150, 1).name('Fold amplitude (°)');
  gui.add(P, 'stagger', 0.0, 1.5, 0.01).name('Time offset');

  const fMat = gui.addFolder('Material');
  fMat.add(P, 'iridescence', 0.0, 1.0, 0.01).name('Iridescence').onChange(v => material.iridescence = v);
  fMat.add(P, 'iriIOR', 1.0, 2.0, 0.01).name('Iridescence IOR').onChange(v => material.iridescenceIOR = v);
  fMat.add(P, 'iriMinNm', 50, 500, 1).name('Iri min (nm)').onChange(() => material.iridescenceThicknessRange = [P.iriMinNm, P.iriMaxNm]);
  fMat.add(P, 'iriMaxNm', 200, 1200, 1).name('Iri max (nm)').onChange(() => material.iridescenceThicknessRange = [P.iriMinNm, P.iriMaxNm]);
  fMat.add(P, 'transmission', 0.0, 0.5, 0.01).name('Transmission').onChange(v => material.transmission = v);
  fMat.add(P, 'thickness', 0.05, 2.0, 0.01).name('Thickness').onChange(v => material.thickness = v);
  fMat.add(P, 'roughness', 0.0, 0.6, 0.001).name('Roughness').onChange(v => material.roughness = v);
  fMat.add(P, 'clearcoat', 0.0, 1.0, 0.01).name('Clearcoat').onChange(v => material.clearcoat = v);
  fMat.add(P, 'clearcoatRoughness', 0.0, 1.0, 0.01).name('ClearcoatRough').onChange(v => material.clearcoatRoughness = v);
  fMat.add(P, 'envIntensity', 0.1, 3.0, 0.01).name('Env Intensity').onChange(v => material.envMapIntensity = v);

  const fTex = gui.addFolder('Striped interference');
  fTex.add(P, 'bandScale', 6.0, 60.0, 0.1).name('Band scale').onChange(v => material.userData.shader && (material.userData.shader.uniforms.uBandScale.value = v));
  fTex.add(P, 'bandStrength', 0.0, 1.0, 0.01).name('Band strength').onChange(v => material.userData.shader && (material.userData.shader.uniforms.uBandStrength.value = v));
  fTex.add(P, 'bandSharpness', 1.0, 20.0, 0.1).name('Band sharpness').onChange(v => material.userData.shader && (material.userData.shader.uniforms.uBandSharpness.value = v));
  fTex.add(P, 'bandMix', 0.0, 1.0, 0.01).name('Dual mix').onChange(v => material.userData.shader && (material.userData.shader.uniforms.uBandMix.value = v));
  fTex.add(P, 'bandPhase', -Math.PI, Math.PI, 0.01).name('Band phase').onChange(v => material.userData.shader && (material.userData.shader.uniforms.uBandPhase.value = v));
  fTex.add(P, 'edgeGlow', 0.0, 2.5, 0.01).name('Edge glow').onChange(v => material.userData.shader && (material.userData.shader.uniforms.uEdgeGlow.value = v));

  /* ------------------------------------------------------------------------ */
  /* Resize / Input                                                           */
  /* ------------------------------------------------------------------------ */
  window.addEventListener('resize', () => {
    const w = container.clientWidth, h = container.clientHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  });
  window.addEventListener('keydown', (e) => { if (e.code === 'Space') P.play = !P.play; });

  /* ------------------------------------------------------------------------ */
  /* Animate                                                                  */
  /* ------------------------------------------------------------------------ */
  const clock = new THREE.Clock();
  function render() {
    const t = clock.getElapsedTime();
    if (P.play) applyFolds(t);
    if (material.userData.shader) {
      material.userData.shader.uniforms.uTime.value = t;
    }
    if (P.autoRotate) mesh.rotation.y += 0.0023;

    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(render);
  }
  render();
}
