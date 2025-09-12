// Origata — Brownian polygon surface + echo trails + luma‑preserving RGB + Presets
// Reflection overhaul: full PBR controls + PMREM video/image loader + flat/smooth reflection shading
// Built atop your latest code; overlay/glow/trails preserved.  :contentReference[oaicite:4]{index=4}

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { AfterimagePass } from 'three/addons/postprocessing/AfterimagePass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { SavePass } from 'three/addons/postprocessing/SavePass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

import GUI from 'https://cdn.jsdelivr.net/npm/lil-gui@0.20/+esm';

// ---------- scene bootstrap ----------
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

// Environment (default RoomEnvironment → replaced by reflection file if present)
const pmrem = new THREE.PMREMGenerator(renderer);
const defaultEnvRT = pmrem.fromScene(new RoomEnvironment(), 0.04);
scene.environment = defaultEnvRT.texture;
let activeEnvRT = defaultEnvRT;     // track active PMREM to dispose when replaced
let reflectionVideo = null;         // HTMLVideoElement if using mp4
let reflectionTex = null;           // THREE.VideoTexture or THREE.Texture
let lastVideoPmrem = 0;

// ---------- overlay shader (your original look preserved) ----------
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

// ---------------- Brownian polygon surface (shared vertices; faces only) ----------------
class BrownianSurface {
  constructor(opts) {
    this.params = { ...opts };

    this.surfaceType   = opts.surfaceType || 'sphere';  // 'sphere'|'cube'|'torus'
    this.surfaceJitter = opts.surfaceJitter ?? 0.12;

    this.N = 0;
    this.pos = null;   // Float32Array (N*3)
    this.vel = null;   // Float32Array (N*3)
    this.anc = null;   // Float32Array (N*3)
    this.state = null; // Uint8Array (N*N) symmetric

    this.tris = [];    // Array<[i,j,k]> referencing vertex indices in [0..N)

    // Material (physical with iridescence); reflection props will be synced from GUI
    this.material = new THREE.MeshPhysicalMaterial({
      color: 0x151515,
      roughness: 0.26,
      metalness: 0.0,
      envMapIntensity: 1.2,
      ior: 1.3,                   // dielectric IOR
      specularIntensity: 1.0,     // dielectric specular
      specularColor: new THREE.Color(0xffffff),
      clearcoat: 0.0,
      clearcoatRoughness: 0.2,
      flatShading: true,          // “origami” by default
      side: THREE.DoubleSide
    });
    injectOverlay(this.material);

    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
    this.geometry.setAttribute('normal',   new THREE.BufferAttribute(new Float32Array(0), 3));
    this.geometry.setIndex([]);

    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.group = new THREE.Group();
    this.group.add(this.mesh);
    scene.add(this.group);

    this.setCount(this.params.count);
  }

  setSurfaceType(type) { this.surfaceType = type; this.setCount(this.N); }
  setSurfaceJitter(j)  { this.surfaceJitter = j;  this.setCount(this.N); }
  setFlatShading(on)   { this.material.flatShading = !!on; this.material.needsUpdate = true; }

  setCount(n) {
    n = Math.max(3, Math.floor(n));
    this.N = n;
    this.pos = new Float32Array(n * 3);
    this.vel = new Float32Array(n * 3);
    this.anc = new Float32Array(n * 3);
    this.state = new Uint8Array(n * n);

    this._initAnchorsFromSurface(this.surfaceType, this.surfaceJitter);
    this._rebuildGeometry([]); // no triangles yet
  }

  _initAnchorsFromSurface(type, jitter) {
    const R = 1.45;             // presentation scale
    const half = 1.0;           // cube half-size
    const Rt = 1.10, rt = 0.45; // torus radii

    const v = new THREE.Vector3(), n = new THREE.Vector3();
    for (let i = 0; i < this.N; i++) {
      if (type === 'cube') {
        const f = Math.floor(Math.random()*6);
        const u = (Math.random()*2 - 1) * half;
        const w = (Math.random()*2 - 1) * half;
        switch (f) {
          case 0: v.set( half, u, w); n.set( 1, 0, 0); break;
          case 1: v.set(-half, u, w); n.set(-1, 0, 0); break;
          case 2: v.set(u,  half, w); n.set(0,  1, 0); break;
          case 3: v.set(u, -half, w); n.set(0, -1, 0); break;
          case 4: v.set(u, w,  half); n.set(0, 0,  1); break;
          default:v.set(u, w, -half); n.set(0, 0, -1); break;
        }
        v.multiplyScalar(R / (Math.sqrt(3)*half));
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

  _rebuildGeometry(tris) {
    this.tris = tris;

    // Shared-vertex layout: position len = N, indices = [i,j,k] per triangle
    const positions = new Float32Array(this.N * 3);
    const normals   = new Float32Array(this.N * 3);
    const indices   = new Uint32Array(tris.length * 3);

    for (let i = 0; i < this.N; i++) {
      const o = i * 3;
      positions[o+0] = this.pos[o+0];
      positions[o+1] = this.pos[o+1];
      positions[o+2] = this.pos[o+2];
    }
    for (let t = 0; t < tris.length; t++) {
      const [a,b,c] = tris[t];
      const o = t * 3;
      indices[o+0] = a;
      indices[o+1] = b;
      indices[o+2] = c;
    }

    this.geometry.dispose();
    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    this.geometry.setAttribute('normal',   new THREE.BufferAttribute(normals,   3));
    this.geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    this.geometry.computeBoundingSphere();

    this.mesh.geometry = this.geometry;
  }

  _updatePositionsAndNormals() {
    const posAttr = this.geometry.getAttribute('position');
    if (!posAttr) return;

    // write positions for all vertices (shared)
    for (let i = 0; i < this.N; i++) {
      const pi = i * 3;
      posAttr.setXYZ(i, this.pos[pi+0], this.pos[pi+1], this.pos[pi+2]);
    }
    posAttr.needsUpdate = true;

    // Smooth normals if needed (for reflection spread across faces)
    if (!this.material.flatShading) {
      this.geometry.computeVertexNormals();
      const nrmAttr = this.geometry.getAttribute('normal');
      if (nrmAttr) nrmAttr.needsUpdate = true;
    }
  }

  update(dt, now) {
    // Motion (OU-like around anchors)
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
    const S = this.state.fill(0);

    for (let i = 0; i < N; i++) {
      const ix = i * 3; const pix = this.pos[ix], piy = this.pos[ix+1], piz = this.pos[ix+2];
      for (let j = i + 1; j < N; j++) {
        const jx = j * 3; const pjx = this.pos[jx], pjy = this.pos[jx+1], pjz = this.pos[jx+2];
        const dx = pjx - pix, dy = pjy - piy, dz = pjz - piz; const d = Math.sqrt(dx*dx + dy*dy + dz*dz);
        if (d <= onR) { S[i*N+j] = S[j*N+i] = 1; }
      }
    }

    // Triangles = 3‑cliques of S
    const tris = this._computeTriangles();

    // Rebuild index buffer if tri count changed
    if (tris.length !== this.tris.length) this._rebuildGeometry(tris);
    else this.tris = tris;

    if (this.tris.length > 0) this._updatePositionsAndNormals();

    // Drive overlay time + presentation rotation
    const uniforms = this.material.userData._overlayUniforms;
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
const composer = new EffectComposer(renderer);
const renderPass = new RenderPass(scene, camera);
composer.addPass(renderPass);

// Save current frame (for trail composition)
const saveBaseRT = new THREE.WebGLRenderTarget(container.clientWidth, container.clientHeight, { depthBuffer: false, stencilBuffer: false });
const saveBasePass = new SavePass(saveBaseRT);
saveBasePass.enabled = false;
composer.addPass(saveBasePass);

// Afterimage (temporal accumulator)
const afterimagePass = new AfterimagePass();
afterimagePass.enabled = false;
afterimagePass.uniforms['damp'].value = 0.96;
composer.addPass(afterimagePass);

// TrailComposite shader (history only) + base preserved crisp
const TrailCompositeShader = {
  uniforms: {
    tDiffuse:      { value: null },
    tBase:         { value: null },
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
    uniform sampler2D tBase;      // current saved base frame (crisp)
    uniform float uDamp;

    uniform float uGain, uGamma, uSaturation;
    uniform float uBlurPx, uBlurSigma;
    uniform vec2  uInvResolution;
    varying vec2 vUv;

    vec3 blur9(sampler2D tex, vec2 uv){
      float sigma = max(uBlurSigma, 1.0e-4);
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
      vec3 base     = texture2D(tBase,    vUv).rgb;
      vec3 afterImg = (uBlurPx > 0.0001) ? blur9(tDiffuse, vUv)
                                         : texture2D(tDiffuse, vUv).rgb;

      vec3 history = max(afterImg - (1.0 - uDamp) * base, vec3(0.0));

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
trailCompositePass.uniforms.tBase.value = saveBaseRT.texture;
composer.addPass(trailCompositePass);

// Bloom (kept)
const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(container.clientWidth, container.clientHeight),
  0.0, 0.2, 0.9
);
bloomPass.enabled = false;
composer.addPass(bloomPass);

// Luma‑preserving RGB shift
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
composer.addPass(rgbShiftPass);

// Always-on final output
const outputPass = new OutputPass();
composer.addPass(outputPass);

// ---------- GUI + parameters ----------
const uiHost = document.getElementById('ui');
const gui = new GUI({ title: 'Controls', width: 320 });
uiHost.appendChild(gui.domElement);

const params = {
  // Brownian surface
  primitive:     'sphere',  // 'sphere' | 'cube' | 'torus'
  surfaceJitter: 0.12,
  count:         36,
  amp:           0.85,
  freq:          0.9,
  connectDist:   0.78,
  breakDist:     0.98,

  // Animation
  play:       true,
  timeScale:  1.0,

  // Bloom
  bloomStrength: 0.0,
  bloomThreshold:0.9,
  bloomRadius:   0.2,

  // Trails (Echoes)
  trailEnabled:     false,
  trailPersistence: 96.0,
  trailHalfLife:    0.0,
  trailBlurPx:      0.75,
  trailBlurSigma:   1.0,
  trailGain:        1.0,
  trailGamma:       1.0,
  trailSaturation:  1.0,
  trailClear:       () => { _clearTrailsNext = true; },

  // RGB offset (luma‑neutral)
  rgbAmount:     0.0,
  rgbAngle:      0.0,
  rgbAnimate:    false,
  rgbSpinHz:     0.0,
  rgbPulseAmp:   0.0,
  rgbPulseHz:    0.5,

  // Reflection material
  reflMode: 'dielectric',        // 'dielectric' | 'metal'
  reflFlatShading: true,         // keep origami by default
  reflEnvIntensity: 1.2,         // envMapIntensity
  reflRoughness: 0.26,           // “spread”
  reflMetalness: 0.0,            // only used in metal mode (0..1)
  reflIOR: 1.3,                  // dielectric index of refraction
  reflSpecularIntensity: 1.0,    // dielectric specular (0..1)
  reflSpecularColor: '#ffffff',  // dielectric specular tint
  reflMetalColor: '#ffffff',     // metal base color (chrome=white)
  reflClearcoat: 0.0,
  reflClearcoatRoughness: 0.2,
  reflVideoUpdateHz: 0.0,        // 0 = static; >0 = re-PMREM video N times/sec

  // View
  exposure: renderer.toneMappingExposure,
  resetCamera: () => {
    camera.position.set(3.2, 1.8, 4.8);
    controls.target.set(0,0,0);
    controls.update();
  }
};

function syncMaterialFromParams() {
  const mat = surface.material;
  mat.envMapIntensity    = params.reflEnvIntensity;
  mat.roughness          = params.reflRoughness;
  mat.clearcoat          = params.reflClearcoat;
  mat.clearcoatRoughness = params.reflClearcoatRoughness;
  mat.flatShading        = params.reflFlatShading;
  mat.ior                = params.reflIOR;                  // dielectric reflectance control
  mat.specularIntensity  = params.reflSpecularIntensity;    // dielectric “mirror” strength
  mat.specularColor.set(params.reflSpecularColor);

  if (params.reflMode === 'metal') {
    mat.metalness = params.reflMetalness;
    mat.color.set(params.reflMetalColor);
  } else {
    mat.metalness = 0.0;
    mat.color.set(0x151515);
  }
  mat.needsUpdate = true;
  surface.setFlatShading(params.reflFlatShading);
}

// Build surface
const surface = new BrownianSurface({
  surfaceType: params.primitive,
  surfaceJitter: params.surfaceJitter,
  count: params.count,
  amp: params.amp,
  freq: params.freq,
  connectDist: params.connectDist,
  breakDist: params.breakDist
});
syncMaterialFromParams();

// --- Folders ---
const fSrc = gui.addFolder('Surface Source');
fSrc.add(params, 'primitive', { Sphere: 'sphere', Cube: 'cube', Torus: 'torus' })
   .name('Primitive').onChange(v => surface.setSurfaceType(v));
fSrc.add(params, 'surfaceJitter', 0.0, 0.6, 0.001).name('± Offset')
    .onFinishChange(v => surface.setSurfaceJitter(v));

const fSys = gui.addFolder('Brownian Surface');
const breakCtrl = fSys.add(params, 'breakDist', 0.05, 2.0, 0.001).name('Break (≥)').onChange(v => {
  surface.params.breakDist = Math.max(v, params.connectDist);
  params.breakDist = surface.params.breakDist;
  breakCtrl.updateDisplay();
});
fSys.add(params, 'count', 4, 200, 1).name('Vertices').onFinishChange(v => surface.setCount(v));
fSys.add(params, 'amp', 0.0, 2.0, 0.001).name('Amplitude').onChange(v => surface.params.amp = v);
fSys.add(params, 'freq', 0.0, 3.0, 0.001).name('Frequency').onChange(v => surface.params.freq = v);
fSys.add(params, 'connectDist', 0.05, 2.0, 0.001).name('Connect (≤)').onChange(v => {
  surface.params.connectDist = v;
  if (params.breakDist < v) { params.breakDist = v; surface.params.breakDist = v; breakCtrl.updateDisplay(); }
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

const fTrail = gui.addFolder('Trails (Echoes)');
fTrail.add(params, 'trailEnabled').name('Enable').onChange(updateTrailEnabled);
fTrail.add(params, 'trailPersistence', 0.0, 99.9, 0.1).name('Persistence (%)');
fTrail.add(params, 'trailHalfLife', 0.0, 6.0, 0.01).name('Half-life (s)');
fTrail.add(params, 'trailBlurPx', 0.0, 3.0, 0.01).name('Blur (px)').onChange(updateTrailUniforms);
fTrail.add(params, 'trailBlurSigma', 0.5, 3.0, 0.01).name('Blur Sigma').onChange(updateTrailUniforms);
fTrail.add(params, 'trailGain', 0.1, 4.0, 0.01).name('Echo Gain').onChange(updateTrailUniforms);
fTrail.add(params, 'trailGamma', 0.5, 2.5, 0.01).name('Echo Gamma').onChange(updateTrailUniforms);
fTrail.add(params, 'trailSaturation', 0.0, 2.0, 0.01).name('Echo Saturation').onChange(updateTrailUniforms);
fTrail.add(params, 'trailClear').name('Clear Trails');

const fRGB = gui.addFolder('RGB Offset (Luma‑neutral)');
fRGB.add(params, 'rgbAmount', 0.0, 0.10, 0.0001).name('Amount').onChange(updateRGBEnabled);
fRGB.add(params, 'rgbAngle', 0.0, 360.0, 0.1).name('Angle (°)')
    .onChange(v => rgbShiftPass.uniforms['angle'].value = THREE.MathUtils.degToRad(v));
fRGB.add(params, 'rgbAnimate').name('Spin Angle').onChange(updateRGBEnabled);
fRGB.add(params, 'rgbSpinHz', 0.0, 3.0, 0.001).name('Spin (Hz)');
fRGB.add(params, 'rgbPulseAmp', 0.0, 0.10, 0.0001).name('Pulse Amp').onChange(updateRGBEnabled);
fRGB.add(params, 'rgbPulseHz', 0.0, 5.0, 0.001).name('Pulse (Hz)');

// === Reflection (new) ===
const fRefl = gui.addFolder('Reflection (PBR)');
fRefl.add(params, 'reflMode', { Dielectric: 'dielectric', Metal: 'metal' }).name('Workflow').onChange(syncMaterialFromParams);
fRefl.add(params, 'reflFlatShading').name('Origami (Flat)').onChange(syncMaterialFromParams);
fRefl.add(params, 'reflEnvIntensity', 0.0, 3.0, 0.01).name('Env Intensity').onChange(syncMaterialFromParams);
fRefl.add(params, 'reflRoughness', 0.0, 1.0, 0.001).name('Spread (Roughness)').onChange(syncMaterialFromParams);
const metalCtrl = fRefl.add(params, 'reflMetalness', 0.0, 1.0, 0.001).name('Metalness').onChange(syncMaterialFromParams);
const metalCol  = fRefl.addColor(params, 'reflMetalColor').name('Metal Color').onChange(syncMaterialFromParams);
const iorCtrl   = fRefl.add(params, 'reflIOR', 1.0, 2.333, 0.001).name('IOR (dielectric)').onChange(syncMaterialFromParams);
const specCtrl  = fRefl.add(params, 'reflSpecularIntensity', 0.0, 1.0, 0.001).name('Specular Intensity').onChange(syncMaterialFromParams);
const specCol   = fRefl.addColor(params, 'reflSpecularColor').name('Specular Color').onChange(syncMaterialFromParams);
fRefl.add(params, 'reflClearcoat', 0.0, 1.0, 0.001).name('Clearcoat').onChange(syncMaterialFromParams);
fRefl.add(params, 'reflClearcoatRoughness', 0.0, 1.0, 0.001).name('Coat Roughness').onChange(syncMaterialFromParams);
fRefl.add(params, 'reflVideoUpdateHz', 0.0, 12.0, 0.1).name('Video PMREM Hz');

function toggleReflUI() {
  const isMetal = (params.reflMode === 'metal');
  metalCtrl.domElement.parentElement.style.display = isMetal ? '' : 'none';
  metalCol.domElement.parentElement.style.display  = isMetal ? '' : 'none';
  iorCtrl.domElement.parentElement.style.display   = !isMetal ? '' : 'none';
  specCtrl.domElement.parentElement.style.display  = !isMetal ? '' : 'none';
  specCol.domElement.parentElement.style.display   = !isMetal ? '' : 'none';
}
toggleReflUI();
fRefl.controllers.forEach(c => {
  if (c.property === 'reflMode') c.onChange(() => { toggleReflUI(); });
});

// View
const fView = gui.addFolder('View');
fView.add(params, 'exposure', 0.6, 1.8, 0.01).name('Exposure').onChange(v => renderer.toneMappingExposure = v);
fView.add(params, 'resetCamera').name('Reset Camera');

// ---------- helpers ----------
function updateTrailUniforms() {
  trailCompositePass.uniforms.uGain.value        = params.trailGain;
  trailCompositePass.uniforms.uGamma.value       = params.trailGamma;
  trailCompositePass.uniforms.uSaturation.value  = params.trailSaturation;
  trailCompositePass.uniforms.uBlurPx.value      = params.trailBlurPx;
  trailCompositePass.uniforms.uBlurSigma.value   = params.trailBlurSigma;
}
function updateTrailEnabled() {
  const enabled = params.trailEnabled;
  saveBasePass.enabled        = enabled;
  afterimagePass.enabled      = enabled;
  trailCompositePass.enabled  = enabled;
  updateTrailUniforms();
}
function updateRGBEnabled() {
  const active = (params.rgbAmount + params.rgbPulseAmp) > 0 || params.rgbAnimate;
  rgbShiftPass.enabled = active;
}
updateTrailEnabled();
updateRGBEnabled();

// ---------- Reflection map loader (mp4 or image) with PMREM ----------
async function exists(url) {
  try {
    const r = await fetch(url, { method: 'HEAD', cache: 'no-store' });
    return r.ok;
  } catch { return false; }
}
function disposeActiveEnvRT(keepDefault = true) {
  if (activeEnvRT && activeEnvRT !== defaultEnvRT) {
    activeEnvRT.dispose();
  } else if (!keepDefault && activeEnvRT === defaultEnvRT) {
    activeEnvRT.dispose();
  }
}
async function applyPMREMFromTexture(tex) {
  // Ensure equirect mapping
  tex.mapping = THREE.EquirectangularReflectionMapping;
  const rt = pmrem.fromEquirectangular(tex);
  scene.environment = rt.texture;
  disposeActiveEnvRT();
  activeEnvRT = rt;
}
async function applyReflectionFromImage(url) {
  const tex = await new Promise((resolve, reject) => {
    const loader = new THREE.TextureLoader();
    loader.load(url, resolve, undefined, reject);
  });
  tex.colorSpace = THREE.SRGBColorSpace;
  await applyPMREMFromTexture(tex); // PMREM for proper IBL. :contentReference[oaicite:5]{index=5}
  if (reflectionTex && reflectionTex.isTexture) reflectionTex.dispose();
  reflectionTex = tex;
}
async function applyReflectionFromVideo(url) {
  const video = document.createElement('video');
  video.src = url;
  video.crossOrigin = 'anonymous';
  video.loop = true;
  video.muted = true;
  video.playsInline = true;
  video.preload = 'auto';
  await new Promise((resolve, reject) => {
    const onErr = () => reject(new Error('Video load error'));
    video.addEventListener('error', onErr, { once: true });
    video.addEventListener('canplay', () => resolve(), { once: true });
  });
  await video.play().catch(() => {});
  const vtex = new THREE.VideoTexture(video);
  vtex.colorSpace = THREE.SRGBColorSpace;
  await applyPMREMFromTexture(vtex);      // initial PMREM; we’ll refresh at Hz if requested. :contentReference[oaicite:6]{index=6}
  if (reflectionTex && reflectionTex.isTexture) reflectionTex.dispose();
  reflectionTex = vtex;
  reflectionVideo = video;
}
async function loadReflectionAuto(logOn = false) {
  try {
    if (await exists('reflection.mp4')) {
      await applyReflectionFromVideo('reflection.mp4');
      if (logOn) console.info('[reflection] Loaded reflection.mp4');
      return;
    }
    const candidates = ['reflection.jpg', 'reflection.jpeg', 'reflection.png', 'reflection.webp'];
    for (const name of candidates) {
      if (await exists(name)) {
        await applyReflectionFromImage(name);
        if (logOn) console.info(`[reflection] Loaded ${name}`);
        return;
      }
    }
    if (logOn) console.info('[reflection] No reflection.* found, using RoomEnvironment.');
  } catch (e) {
    if (logOn) console.warn('[reflection] Load error:', e);
  }
}
loadReflectionAuto(true);

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

  // Trails damp
  if (params.trailEnabled) {
    let damp = (params.trailHalfLife > 0)
      ? Math.exp(-Math.LN2 * Math.max(1e-6, dt) / params.trailHalfLife)
      : THREE.MathUtils.clamp(params.trailPersistence / 100.0, 0.0, 0.9999);
    if (_clearTrailsNext) { damp = 0.0; _clearTrailsNext = false; }
    damp = Math.min(damp, 0.9999);
    afterimagePass.uniforms['damp'].value = damp;
    trailCompositePass.uniforms['uDamp'].value = damp;
  }

  // RGB offset animation
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

  // Optional PMREM refresh for video env‑map (so reflections update)
  if (reflectionVideo && params.reflVideoUpdateHz > 0) {
    const sec = now / 1000;
    if (sec - lastVideoPmrem >= (1 / params.reflVideoUpdateHz)) {
      lastVideoPmrem = sec;
      // regenerate PMREM from current video frame (costly; keep Hz modest)
      applyPMREMFromTexture(reflectionTex);
    }
  }

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
