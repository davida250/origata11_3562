// Origata — Brownian polygon surface + echo trails + luma‑preserving RGB
// ---------------------------------------------------------------------------------------
// Everything (renderer/env/material/overlay/bloom/RGB) adheres to your original stack.
// Trails are now composited so the current frame NEVER dims — only echoes fade.
// Points initialize near a chosen surface (Sphere/Cube/Torus) with +/- offset.
// Baseline provenance (materials/pipeline structure) from your original main.js.  :contentReference[oaicite:1]{index=1}

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { AfterimagePass } from 'three/addons/postprocessing/AfterimagePass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { SavePass } from 'three/addons/postprocessing/SavePass.js';

import GUI from 'https://cdn.jsdelivr.net/npm/lil-gui@0.20/+esm';

// ---------- scene bootstrap (same) ----------
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

const camera = new THREE.PerspectiveCamera(36, container.clientWidth / container.clientHeight, 0.01, 100);
camera.position.set(3.2, 1.8, 4.8);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.minDistance = 2.0;
controls.maxDistance = 9.0;

const pmrem = new THREE.PMREMGenerator(renderer);
const envRT = pmrem.fromScene(new RoomEnvironment(), 0.04);
scene.environment = envRT.texture;

// ---------- overlay shader (unchanged look) ----------
function injectOverlay(material) {
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

// ---------------- Brownian polygon surface (faces only; no lines/points) ----------------
class BrownianSurface {
  constructor(opts) {
    this.params = { ...opts };

    this.surfaceType = opts.surfaceType || 'sphere';
    this.surfaceJitter = opts.surfaceJitter ?? 0.12;

    this.N = 0;
    this.pos = null;   // Float32Array (N*3)
    this.vel = null;   // Float32Array (N*3)
    this.anc = null;   // Float32Array (N*3)
    this.state = null; // Uint8Array (N*N) symmetric

    this.tris = [];
    this._geomMap = null;

    const mat = new THREE.MeshPhysicalMaterial({
      color: 0x151515,
      roughness: 0.26,
      metalness: 0.0,
      envMapIntensity: 1.2,
      iridescence: 1.0,
      iridescenceIOR: 1.3,
      iridescenceThicknessRange: [120, 620],
      flatShading: true,
      side: THREE.DoubleSide
    });
    injectOverlay(mat);

    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
    this.geometry.setAttribute('normal',   new THREE.BufferAttribute(new Float32Array(0), 3));

    this.mesh = new THREE.Mesh(this.geometry, mat);
    this.group = new THREE.Group();
    this.group.add(this.mesh);
    scene.add(this.group);

    this.setCount(this.params.count);
  }

  setSurfaceType(type) {
    this.surfaceType = type;
    this.setCount(this.N);
  }

  setSurfaceJitter(j) {
    this.surfaceJitter = j;
    this.setCount(this.N);
  }

  setCount(n) {
    n = Math.max(3, Math.floor(n));
    this.N = n;
    this.pos = new Float32Array(n * 3);
    this.vel = new Float32Array(n * 3);
    this.anc = new Float32Array(n * 3);
    this.state = new Uint8Array(n * n);

    this._initAnchorsFromSurface(this.surfaceType, this.surfaceJitter);
    this._rebuildFromTriangles([]);
  }

  _initAnchorsFromSurface(type, jitter) {
    const R = 1.45;         // presentation scale
    const half = 1.0;       // cube half-size
    const Rt = 1.10, rt = 0.45; // torus radii (major/minor)

    const v = new THREE.Vector3(), n = new THREE.Vector3();
    for (let i = 0; i < this.N; i++) {
      if (type === 'cube') {
        // pick a face uniformly
        const f = Math.floor(Math.random()*6);
        const u = (Math.random()*2 - 1) * half;
        const w = (Math.random()*2 - 1) * half;
        switch (f) {
          case 0: v.set( half, u, w); n.set( 1, 0, 0); break; // +X
          case 1: v.set(-half, u, w); n.set(-1, 0, 0); break; // -X
          case 2: v.set(u,  half, w); n.set(0,  1, 0); break; // +Y
          case 3: v.set(u, -half, w); n.set(0, -1, 0); break; // -Y
          case 4: v.set(u, w,  half); n.set(0, 0,  1); break; // +Z
          default:v.set(u, w, -half); n.set(0, 0, -1); break; // -Z
        }
        v.multiplyScalar(R / (Math.sqrt(3)*half)); // normalize to similar scale as sphere
      } else if (type === 'torus') {
        const U = Math.random() * Math.PI * 2;
        const V = Math.random() * Math.PI * 2;
        const cx = Rt * Math.cos(U), cz = Rt * Math.sin(U);
        const px = (Rt + rt * Math.cos(V)) * Math.cos(U);
        const py = rt * Math.sin(V);
        const pz = (Rt + rt * Math.cos(V)) * Math.sin(U);
        v.set(px, py, pz);
        const c = new THREE.Vector3(cx, 0, cz);
        n.copy(v).sub(c).normalize();
      } else {
        // sphere
        const u = Math.random();
        const vv = Math.random();
        const theta = 2*Math.PI*u;
        const phi = Math.acos(2*vv - 1);
        n.set(
          Math.sin(phi)*Math.cos(theta),
          Math.cos(phi),
          Math.sin(phi)*Math.sin(theta)
        ).normalize();
        v.copy(n).multiplyScalar(R);
      }

      const off = (Math.random()*2 - 1) * jitter;
      const p  = v.clone().add(n.clone().multiplyScalar(off));

      const o = i * 3;
      this.anc[o+0] = v.x; this.anc[o+1] = v.y; this.anc[o+2] = v.z;
      this.pos[o+0] = p.x; this.pos[o+1] = p.y; this.pos[o+2] = p.z;
      this.vel[o+0] = 0;   this.vel[o+1] = 0;   this.vel[o+2] = 0;
    }
  }

  _computeTriangles() {
    const N = this.N, S = this.state;
    const tris = [];
    const neigh = Array.from({ length: N }, () => []);
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) if (S[i*N+j] === 1) { neigh[i].push(j); neigh[j].push(i); }
    for (let i = 0; i < N; i++) {
      const Ni = neigh[i];
      for (let a = 0; a < Ni.length; a++) {
        const j = Ni[a]; if (j <= i) continue;
        for (let b = a + 1; b < Ni.length; b++) {
          const k = Ni[b]; if (k <= j) continue;
          if (S[j*N+k] === 1) tris.push([i, j, k]);
        }
      }
    }
    return tris;
  }

  _rebuildFromTriangles(tris) {
    this.tris = tris;
    const triCount = tris.length;

    const positions = new Float32Array(triCount * 3 * 3);
    const normals   = new Float32Array(triCount * 3 * 3);
    this._geomMap   = new Int32Array(triCount * 3);

    for (let t = 0; t < triCount; t++) {
      const base = t * 3;
      const [i, j, k] = tris[t];
      this._geomMap[base + 0] = i;
      this._geomMap[base + 1] = j;
      this._geomMap[base + 2] = k;
    }

    this.geometry.dispose();
    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    this.geometry.setAttribute('normal',   new THREE.BufferAttribute(normals,   3));
    this.geometry.setIndex([...Array(triCount * 3).keys()]);
    this.geometry.computeBoundingSphere();

    this.mesh.geometry = this.geometry;
  }

  _updatePositionsAndNormals() {
    const posAttr = this.geometry.getAttribute('position');
    const nrmAttr = this.geometry.getAttribute('normal');
    if (!posAttr || !nrmAttr) return;

    const P = this.pos, arr = posAttr.array, nArr = nrmAttr.array;
    for (let v = 0; v < this._geomMap.length; v++) {
      const pi = this._geomMap[v] * 3, w = v * 3;
      arr[w+0] = P[pi+0]; arr[w+1] = P[pi+1]; arr[w+2] = P[pi+2];
    }
    posAttr.needsUpdate = true;

    const triCount = this._geomMap.length / 3;
    const v0 = new THREE.Vector3(), v1 = new THREE.Vector3(), v2 = new THREE.Vector3();
    const e1 = new THREE.Vector3(), e2 = new THREE.Vector3(), n  = new THREE.Vector3();
    for (let t = 0; t < triCount; t++) {
      const i0 = (t * 3 + 0) * 3, i1 = (t * 3 + 1) * 3, i2 = (t * 3 + 2) * 3;
      v0.set(arr[i0], arr[i0+1], arr[i0+2]);
      v1.set(arr[i1], arr[i1+1], arr[i1+2]);
      v2.set(arr[i2], arr[i2+1], arr[i2+2]);
      e1.subVectors(v1, v0); e2.subVectors(v2, v0);
      n.copy(e1).cross(e2).normalize();
      nArr[i0] = nArr[i1] = nArr[i2] = n.x;
      nArr[i0+1] = nArr[i1+1] = nArr[i2+1] = n.y;
      nArr[i0+2] = nArr[i1+2] = nArr[i2+2] = n.z;
    }
    nrmAttr.needsUpdate = true;
  }

  update(dt, now) {
    // OU-like motion around surface anchors
    const N = this.N;
    const { amp, freq } = this.params;
    const beta = THREE.MathUtils.clamp(1.2 * freq + 0.2, 0, 12);
    const k    = THREE.MathUtils.clamp(1.6 * freq + 0.3, 0, 14);
    const sigma  = amp;
    const sqrtDt = Math.sqrt(Math.max(1e-6, dt));

    for (let i = 0; i < N; i++) {
      const o = i * 3;
      const nx = randn() * sigma * sqrtDt, ny = randn() * sigma * sqrtDt, nz = randn() * sigma * sqrtDt;

      const px = this.pos[o+0], py = this.pos[o+1], pz = this.pos[o+2];
      let vx = this.vel[o+0], vy = this.vel[o+1], vz = this.vel[o+2];
      const ax = this.anc[o+0], ay = this.anc[o+1], az = this.anc[o+2];

      vx += (-beta * vx + k * (ax - px)) * dt + nx;
      vy += (-beta * vy + k * (ay - py)) * dt + ny;
      vz += (-beta * vz + k * (az - pz)) * dt + nz;

      let nxp = px + vx * dt, nyp = py + vy * dt, nzp = pz + vz * dt;
      const r = Math.hypot(nxp, nyp, nzp), limit = 2.25;
      if (r > limit) {
        const s = limit / r;
        nxp *= s; nyp *= s; nzp *= s;
        vx *= 0.65; vy *= 0.65; vz *= 0.65;
      }

      this.vel[o+0] = vx; this.vel[o+1] = vy; this.vel[o+2] = vz;
      this.pos[o+0] = nxp; this.pos[o+1] = nyp; this.pos[o+2] = nzp;
    }

    // Proximity edges with hysteresis
    const onR  = this.params.connectDist, offR = Math.max(this.params.breakDist, onR + 1e-4);
    const S = this.state;

    for (let i = 0; i < N; i++) {
      const ix = i * 3; const pix = this.pos[ix], piy = this.pos[ix+1], piz = this.pos[ix+2];
      for (let j = i + 1; j < N; j++) {
        const jx = j * 3; const pjx = this.pos[jx], pjy = this.pos[jx+1], pjz = this.pos[jx+2];
        const dx = pjx - pix, dy = pjy - piy, dz = pjz - piz; const d = Math.sqrt(dx*dx + dy*dy + dz*dz);
        const idx = i * N + j, on = S[idx] === 1;
        if (!on && d <= onR) { S[i*N+j] = S[j*N+i] = 1; }
        else if (on && d >= offR) { S[i*N+j] = S[j*N+i] = 0; }
      }
    }

    // Triangles = 3‑cliques of S
    const tris = this._computeTriangles();

    // Rebuild capacity if tri count changed
    if (tris.length * 3 !== (this._geomMap ? this._geomMap.length : 0)) this._rebuildFromTriangles(tris);
    else {
      this.tris = tris;
      for (let t = 0; t < tris.length; t++) {
        const [i, j, k] = tris[t], base = t * 3;
        this._geomMap[base+0] = i; this._geomMap[base+1] = j; this._geomMap[base+2] = k;
      }
    }

    if (this._geomMap.length > 0) this._updatePositionsAndNormals();

    // Drive overlay time + gentle presentation rotation
    const uniforms = this.mesh.material.userData._overlayUniforms;
    if (uniforms) uniforms.uTime.value = now;
    const yaw   = 0.22 * Math.sin(2 * Math.PI * 0.03 * now);
    const pitch = 0.17 * Math.sin(2 * Math.PI * 0.021 * now + 1.2);
    this.group.rotation.set(pitch, yaw, 0, 'YXZ');
  }
}

// ---------- utils ----------
function randn() { // Gaussian
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ---------- post pipeline ----------
// We'll render the scene, SAVE the current frame, build trails with Afterimage,
// then COMPOSITE: final = base + k * (historyOnly). This keeps base brightness intact.

// Base render
const composer = new EffectComposer(renderer);
const renderPass = new RenderPass(scene, camera);
composer.addPass(renderPass);

// Save the current frame (for trail composition / "base")
const saveBaseRT = new THREE.WebGLRenderTarget(container.clientWidth, container.clientHeight, { depthBuffer: false, stencilBuffer: false });
const saveBasePass = new SavePass(saveBaseRT);
composer.addPass(saveBasePass);

// Afterimage (temporal accumulator); it will hold base * (1-damp) + damp*history
const afterimagePass = new AfterimagePass(); // we will NOT render this directly to screen
afterimagePass.enabled = false;
afterimagePass.uniforms['damp'].value = 0.96;
composer.addPass(afterimagePass);

// TrailComposite shader: blur + isolate history + add on top of base
const TrailCompositeShader = {
  uniforms: {
    tDiffuse:      { value: null },             // afterimage output
    tBase:         { value: saveBaseRT.texture },
    uDamp:         { value: afterimagePass.uniforms['damp'].value },

    uGain:         { value: 1.0 },
    uGamma:        { value: 1.0 },
    uSaturation:   { value: 1.0 },
    uBlurPx:       { value: 0.75 },
    uBlurSigma:    { value: 1.0 },
    uInvResolution:{ value: new THREE.Vector2(1/container.clientWidth, 1/container.clientHeight) }
  },
  vertexShader: `
    varying vec2 vUv;
    void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }
  `,
  fragmentShader: `
    uniform sampler2D tDiffuse;   // afterimage buffer
    uniform sampler2D tBase;      // current saved base frame
    uniform float uDamp;

    uniform float uGain, uGamma, uSaturation;
    uniform float uBlurPx, uBlurSigma;
    uniform vec2  uInvResolution;

    vec3 blur9(sampler2D tex, vec2 uv){
      float sigma = max(uBlurSigma, 1e-4);
      float w0 = 1.0;
      float w1 = exp(-0.5 * pow(1.0/sigma, 2.0));
      float w2 = exp(-0.5 * pow(1.41421356237/sigma, 2.0));
      float WS = w0 + 4.0*w1 + 4.0*w2;
      vec2 px = uInvResolution * uBlurPx;

      vec3 c = texture2D(tex, uv).rgb * w0;
      c += texture2D(tex, uv + vec2( px.x, 0.0)).rgb * w1;
      c += texture2D(tex, uv + vec2(-px.x, 0.0)).rgb * w1;
      c += texture2D(tex, uv + vec2( 0.0, px.y)).rgb * w1;
      c += texture2D(tex, uv + vec2( 0.0,-px.y)).rgb * w1;

      c += texture2D(tex, uv + vec2( px.x,  px.y)).rgb * w2;
      c += texture2D(tex, uv + vec2(-px.x,  px.y)).rgb * w2;
      c += texture2D(tex, uv + vec2( px.x, -px.y)).rgb * w2;
      c += texture2D(tex, uv + vec2(-px.x, -px.y)).rgb * w2;

      return c / WS;
    }

    void main(){
      vec3 base     = blur9(tBase,    vUv);      // blur to match trail blur
      vec3 afterImg = blur9(tDiffuse, vUv);

      // Isolate history only (remove current contribution): history ≈ afterimage - (1-damp)*base
      vec3 history = max(afterImg - (1.0 - uDamp) * base, vec3(0.0));

      // Tone controls apply to HISTORY ONLY
      float l = dot(history, vec3(0.2126, 0.7152, 0.0722));
      history = mix(vec3(l), history, uSaturation);
      history = pow(max(history, 0.0), vec3(1.0 / max(0.0001, uGamma)));
      history *= uGain;

      vec3 finalCol = clamp(base + history, 0.0, 1.0);
      gl_FragColor = vec4(finalCol, 1.0);
    }
  `
};
const trailCompositePass = new ShaderPass(TrailCompositeShader);
trailCompositePass.enabled = false;
composer.addPass(trailCompositePass);

// Bloom (kept)
const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(container.clientWidth, container.clientHeight),
  0.0, 0.2, 0.9
);
bloomPass.enabled = false;
composer.addPass(bloomPass);

// Luma‑preserving RGB shift (keeps perceived brightness constant)
const RGBShiftLumaShader = {
  uniforms: {
    tDiffuse: { value: null },
    amount:   { value: 0.0 },
    angle:    { value: 0.0 }
  },
  vertexShader: `
    varying vec2 vUv;
    void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }
  `,
  fragmentShader: `
    uniform sampler2D tDiffuse;
    uniform float amount;
    uniform float angle;
    varying vec2 vUv;

    void main(){
      vec2 off = amount * vec2(cos(angle), sin(angle));

      vec4 base = texture2D(tDiffuse, vUv);
      vec4 cr   = texture2D(tDiffuse, vUv + off);
      vec4 cb   = texture2D(tDiffuse, vUv - off);

      vec3 shifted = vec3(cr.r, base.g, cb.b);

      // preserve scene luma
      const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);
      float lBase = dot(base.rgb, LUMA);
      float lSh   = max(1e-5, dot(shifted, LUMA));
      shifted *= (lBase / lSh);

      gl_FragColor = vec4(shifted, base.a);
    }
  `
};
const rgbShiftPass = new ShaderPass(RGBShiftLumaShader);
rgbShiftPass.enabled = false;
rgbShiftPass.uniforms['amount'].value = 0.0;
rgbShiftPass.uniforms['angle'].value  = 0.0;
composer.addPass(rgbShiftPass);

// ---------- GUI ----------
const uiHost = document.getElementById('ui');
const gui = new GUI({ title: 'Controls', width: 320 });
uiHost.appendChild(gui.domElement);

const params = {
  // Brownian surface
  primitive:     'sphere',  // 'sphere' | 'cube' | 'torus'
  surfaceJitter: 0.12,      // initial +/- offset along surface normal
  count:         36,
  amp:           0.85,
  freq:          0.9,
  connectDist:   0.78,
  breakDist:     0.98,

  // Animation
  play:       true,
  timeScale:  1.0,

  // Bloom (same)
  bloomStrength: 0.0,
  bloomThreshold:0.9,
  bloomRadius:   0.2,

  // Trails (echoes that DO NOT dim the base)
  trailEnabled:     false,
  trailPersistence: 96.0, // % kept each step (0..99.9). Higher => longer history
  trailHalfLife:    0.0,  // seconds; if >0, overrides persistence each frame
  trailBlurPx:      0.75, // pixels
  trailBlurSigma:   1.0,
  trailGain:        1.0,  // brightness of echoes
  trailGamma:       1.0,
  trailSaturation:  1.0,
  trailClear:       () => { _clearTrailsNext = true; },

  // RGB offset (brightness‑neutral)
  rgbAmount:     0.0,
  rgbAngle:      0.0,
  rgbAnimate:    false,
  rgbSpinHz:     0.0,
  rgbPulseAmp:   0.0,
  rgbPulseHz:    0.5,

  // View
  exposure: renderer.toneMappingExposure,
  resetCamera: () => {
    camera.position.set(3.2, 1.8, 4.8);
    controls.target.set(0,0,0);
    controls.update();
  }
};

const surface = new BrownianSurface({
  surfaceType: params.primitive,
  surfaceJitter: params.surfaceJitter,
  count: params.count,
  amp: params.amp,
  freq: params.freq,
  connectDist: params.connectDist,
  breakDist: params.breakDist
});

const fSrc = gui.addFolder('Surface Source');
fSrc.add(params, 'primitive', { Sphere: 'sphere', Cube: 'cube', Torus: 'torus' })
   .name('Primitive')
   .onChange(v => surface.setSurfaceType(v));
fSrc.add(params, 'surfaceJitter', 0.0, 0.6, 0.001).name('± Offset').onFinishChange(v => surface.setSurfaceJitter(v));

const fSys = gui.addFolder('Brownian Surface');
fSys.add(params, 'count', 4, 200, 1).name('Vertices').onFinishChange(v => surface.setCount(v));
fSys.add(params, 'amp', 0.0, 2.0, 0.001).name('Amplitude').onChange(v => surface.params.amp = v);
fSys.add(params, 'freq', 0.0, 3.0, 0.001).name('Frequency').onChange(v => surface.params.freq = v);
fSys.add(params, 'connectDist', 0.05, 2.0, 0.001).name('Connect (≤)').onChange(v => {
  surface.params.connectDist = v;
  if (params.breakDist < v) { params.breakDist = v; surface.params.breakDist = v; gui.updateDisplay(); }
});
fSys.add(params, 'breakDist', 0.05, 2.0, 0.001).name('Break (≥)').onChange(v => {
  surface.params.breakDist = Math.max(v, params.connectDist);
  params.breakDist = surface.params.breakDist;
  gui.updateDisplay();
});

const fAnim = gui.addFolder('Animation');
fAnim.add(params, 'play').name('Play / Pause');
fAnim.add(params, 'timeScale', 0.0, 3.0, 0.01).name('Time Scale');

const fGlow = gui.addFolder('Glow (Bloom)');
fGlow.add(params, 'bloomStrength', 0.0, 2.5, 0.01).name('Strength').onChange(v => {
  bloomPass.strength = v;
  bloomPass.enabled = v > 0.0;
});
fGlow.add(params, 'bloomThreshold', 0.0, 1.0, 0.001).name('Threshold').onChange(v => bloomPass.threshold = v);
fGlow.add(params, 'bloomRadius', 0.0, 1.0, 0.001).name('Radius').onChange(v => bloomPass.radius = v);

// Trails (Echoes)
const fTrail = gui.addFolder('Trails (Echoes)');
fTrail.add(params, 'trailEnabled').name('Enable').onChange(v => updateTrailEnabled());
fTrail.add(params, 'trailPersistence', 0.0, 99.9, 0.1).name('Persistence (%)');
fTrail.add(params, 'trailHalfLife', 0.0, 6.0, 0.01).name('Half-life (s)');
fTrail.add(params, 'trailBlurPx', 0.0, 3.0, 0.01).name('Blur (px)').onChange(updateTrailUniforms);
fTrail.add(params, 'trailBlurSigma', 0.5, 3.0, 0.01).name('Blur Sigma').onChange(updateTrailUniforms);
fTrail.add(params, 'trailGain', 0.1, 4.0, 0.01).name('Echo Gain').onChange(updateTrailUniforms);
fTrail.add(params, 'trailGamma', 0.5, 2.5, 0.01).name('Echo Gamma').onChange(updateTrailUniforms);
fTrail.add(params, 'trailSaturation', 0.0, 2.0, 0.01).name('Echo Saturation').onChange(updateTrailUniforms);
fTrail.add(params, 'trailClear').name('Clear Trails');

const fRGB = gui.addFolder('RGB Offset (Luma‑neutral)');
fRGB.add(params, 'rgbAmount', 0.0, 0.10, 0.0001).name('Amount').onChange(() => updateRGBEnabled());
fRGB.add(params, 'rgbAngle', 0.0, 360.0, 0.1).name('Angle (°)').onChange(v => {
  rgbShiftPass.uniforms['angle'].value = THREE.MathUtils.degToRad(v);
});
fRGB.add(params, 'rgbAnimate').name('Spin Angle').onChange(() => updateRGBEnabled());
fRGB.add(params, 'rgbSpinHz', 0.0, 3.0, 0.001).name('Spin (Hz)');
fRGB.add(params, 'rgbPulseAmp', 0.0, 0.10, 0.0001).name('Pulse Amp').onChange(() => updateRGBEnabled());
fRGB.add(params, 'rgbPulseHz', 0.0, 5.0, 0.001).name('Pulse (Hz)');

const fView = gui.addFolder('View');
fView.add(params, 'exposure', 0.6, 1.8, 0.01).name('Exposure').onChange(v => renderer.toneMappingExposure = v);
fView.add(params, 'resetCamera').name('Reset Camera');

// overlay controls (optional)
const surfaceAdv = { angle: 28, angle2: 82, speed: 0.25, freq1: 6.0, freq2: 9.5, strength: 0.52, triScale: 1.15, warp: 0.55, cellAmp: 0.55, cellFreq: 2.75 };
const U = () => surface?.mesh.material.userData._overlayUniforms;
const fSurfaceAdv = gui.addFolder('Surface (Advanced)');
fSurfaceAdv.add(surfaceAdv, 'angle', 0, 180, 0.1).name('Stripe Angle (°)').onChange(v => { const u = U(); if (u) u.uBandAngle.value  = THREE.MathUtils.degToRad(v); });
fSurfaceAdv.add(surfaceAdv, 'angle2', 0, 180, 0.1).name('Stripe2 Angle (°)').onChange(v => { const u = U(); if (u) u.uBandAngle2.value = THREE.MathUtils.degToRad(v); });
fSurfaceAdv.add(surfaceAdv, 'speed', 0, 2, 0.001).name('Stripe Rot Speed').onChange(v => { const u = U(); if (u) u.uBandSpeed.value = v; });
fSurfaceAdv.add(surfaceAdv, 'freq1', 1, 20, 0.1).name('Stripe Freq 1').onChange(v => { const u = U(); if (u) u.uBandFreq1.value = v; });
fSurfaceAdv.add(surfaceAdv, 'freq2', 1, 20, 0.1).name('Stripe Freq 2').onChange(v => { const u = U(); if (u) u.uBandFreq2.value = v; });
fSurfaceAdv.add(surfaceAdv, 'strength', 0, 1, 0.01).name('Emissive Strength').onChange(v => { const u = U(); if (u) u.uBandStrength.value = v; });
fSurfaceAdv.add(surfaceAdv, 'triScale', 0.2, 4, 0.01).name('Tri-Planar Scale').onChange(v => { const u = U(); if (u) u.uTriScale.value = v; });
fSurfaceAdv.add(surfaceAdv, 'warp', 0, 1.5, 0.01).name('Domain Warp').onChange(v => { const u = U(); if (u) u.uWarp.value = v; });
fSurfaceAdv.add(surfaceAdv, 'cellAmp', 0, 1, 0.01).name('Cellular Mix').onChange(v => { const u = U(); if (u) u.uCellAmp.value = v; });
fSurfaceAdv.add(surfaceAdv, 'cellFreq', 0.5, 8, 0.01).name('Cellular Freq').onChange(v => { const u = U(); if (u) u.uCellFreq.value = v; });
fSurfaceAdv.close();

// helpers
function updateTrailUniforms() {
  trailCompositePass.uniforms.uGain.value        = params.trailGain;
  trailCompositePass.uniforms.uGamma.value       = params.trailGamma;
  trailCompositePass.uniforms.uSaturation.value  = params.trailSaturation;
  trailCompositePass.uniforms.uBlurPx.value      = params.trailBlurPx;
  trailCompositePass.uniforms.uBlurSigma.value   = params.trailBlurSigma;
}
function updateTrailEnabled() {
  const enabled = params.trailEnabled;
  afterimagePass.enabled       = enabled;
  trailCompositePass.enabled   = enabled;
  updateTrailUniforms();
}
function updateRGBEnabled() {
  const active = (params.rgbAmount + params.rgbPulseAmp) > 0 || params.rgbAnimate;
  rgbShiftPass.enabled = active;
}
updateTrailEnabled();
updateRGBEnabled();

// ---------- loop ----------
let tPrev = performance.now();
let _clearTrailsNext = false;

function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  let dt = (now - tPrev) / 1000;
  tPrev = now;

  dt = Math.min(dt, 1/30) * params.timeScale;

  if (params.play) {
    surface.params.amp         = params.amp;
    surface.params.freq        = params.freq;
    surface.params.connectDist = params.connectDist;
    surface.params.breakDist   = params.breakDist;
    surface.update(dt, now / 1000);
  }

  // Trails: compute damp (per-frame), push into both passes and isolate history on composite.
  if (params.trailEnabled) {
    let damp = (params.trailHalfLife > 0)
      ? Math.exp(-Math.LN2 * Math.max(1e-6, dt) / params.trailHalfLife)
      : THREE.MathUtils.clamp(params.trailPersistence / 100.0, 0.0, 0.9999);
    if (_clearTrailsNext) { damp = 0.0; _clearTrailsNext = false; }
    // clamp slightly below 1.0 so new pixels can enter
    damp = Math.min(damp, 0.9999);
    afterimagePass.uniforms['damp'].value = damp;
    trailCompositePass.uniforms['uDamp'].value = damp;
  }

  // RGB offset animation (brightness preserved in shader)
  let rgbAngle = THREE.MathUtils.degToRad(params.rgbAngle);
  if (params.rgbAnimate && params.rgbSpinHz > 0) {
    rgbAngle += 2.0 * Math.PI * params.rgbSpinHz * (now / 1000);
  }
  const pulse = (params.rgbPulseAmp > 0)
    ? params.rgbPulseAmp * (0.5 + 0.5 * Math.sin(2.0 * Math.PI * params.rgbPulseHz * (now / 1000)))
    : 0.0;
  const rgbAmt = params.rgbAmount + pulse;
  rgbShiftPass.uniforms['angle'].value  = rgbAngle;
  rgbShiftPass.uniforms['amount'].value = rgbAmt;

  controls.update();
  composer.render();
}
animate();

// ---------- resize ----------
function onResize() {
  const w = container.clientWidth;
  const h = container.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  composer.setSize(w, h);
  bloomPass.setSize(w, h);
  saveBaseRT.setSize(w, h);

  if (trailCompositePass && trailCompositePass.uniforms && trailCompositePass.uniforms.uInvResolution) {
    trailCompositePass.uniforms.uInvResolution.value.set(1 / w, 1 / h);
  }
}
window.addEventListener('resize', onResize);
new ResizeObserver(onResize).observe(container);
onResize();
