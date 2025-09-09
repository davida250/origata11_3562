// Origata — mesh folds + facet thin‑film + TD-style feedback/rgbaDelay
// Robust across GPUs (no manual #extension). Derivatives are enabled via material.extensions.
// Sources used: TouchDesigner Feedback TOP & rgbaDelay, tri-planar mapping, thin-film iridescence (see notes at bottom).

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import GUI from 'https://cdn.jsdelivr.net/npm/lil-gui@0.19/+esm';

/* ───────────────────────── Renderer / Scene / Camera ───────────────────────── */
const canvas = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
renderer.setClearColor(0x000000, 1);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 50);
camera.position.set(0, 0, 5.5);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.08; controls.rotateSpeed = 0.6;

/* ───────────────────────── Parameters & GUI ───────────────────────── */
const P = {
  // timeline
  cycleSeconds: 16, speed: 1.0, autoRotate: true, spin: 0.15, postBypass: false,

  // folding
  twist: 2.0, foldP1: 1.10, foldP2: 1.35, foldI1: 1.10, foldI2: 1.25,

  // facet texture
  stripeFreq: 22.0, stripeWarp: 0.48, filmBase: 420.0, filmAmp: 360.0,
  vibrance: 2.2, pastel: 0.18, edgeWarm: 0.30, envTint: 0.6,

  // feedback chain
  fbDecay: 0.92, fbGain: 0.50, currGain: 0.85,
  warpRotate: 0.010, warpZoom: 0.0015, warpJitter: 0.0010,
  blurRadius: 1.3, maskHardness: 1.08, dispersion: 0.0016,

  sheetSize: 320
};

const gui = new GUI({ title: 'Controls', width: 300 });
const fT = gui.addFolder('Timeline');
fT.add(P, 'cycleSeconds', 8, 32, 1).name('Cycle (s)');
fT.add(P, 'speed', 0.25, 3.0, 0.01);
fT.add(P, 'autoRotate'); fT.add(P, 'spin', 0.0, 1.0, 0.01);
fT.add(P, 'postBypass').name('Bypass Post');

const fS = gui.addFolder('Shape');
fS.add(P, 'twist', 0.0, 4.0, 0.01);
fS.add(P, 'foldP1', 0.6, 2.0, 0.01);
fS.add(P, 'foldP2', 0.6, 2.0, 0.01);
fS.add(P, 'foldI1', 0.5, 2.0, 0.01);
fS.add(P, 'foldI2', 0.5, 2.0, 0.01);

const fI = gui.addFolder('Facet Texture');
fI.add(P, 'stripeFreq', 6, 36, 0.1);
fI.add(P, 'stripeWarp', 0.0, 1.0, 0.01);
fI.add(P, 'filmBase', 300, 520, 1);
fI.add(P, 'filmAmp', 100, 520, 1);
fI.add(P, 'vibrance', 0.2, 3.0, 0.01);
fI.add(P, 'pastel', 0.0, 1.0, 0.01);
fI.add(P, 'edgeWarm', 0.0, 1.0, 0.01);

const fF = gui.addFolder('Feedback');
fF.add(P, 'fbDecay', 0.80, 0.995, 0.001).name('Decay');
fF.add(P, 'fbGain', 0.0, 1.0, 0.01).name('FB Gain');
fF.add(P, 'currGain', 0.0, 1.5, 0.01).name('Current Gain');
fF.add(P, 'warpRotate', -0.05, 0.05, 0.0005).name('Warp Rot');
fF.add(P, 'warpZoom', -0.02, 0.02, 0.0001).name('Warp Zoom');
fF.add(P, 'warpJitter', 0.0, 0.01, 0.0001).name('Warp Jitter');
fF.add(P, 'blurRadius', 0.0, 4.0, 0.01).name('Blur');
fF.add(P, 'maskHardness', 0.5, 1.5, 0.01).name('Mask Hardness');
fF.add(P, 'dispersion', 0.0, 0.01, 0.0001).name('Dispersion');

/* ───────────────────────── Mesh: dense cube, continuous folds ───────────────────────── */
const geo = new THREE.BoxGeometry(2, 2, 2, 120, 120, 120);

const foldMat = new THREE.ShaderMaterial({
  uniforms: {
    uTime: { value: 0.0 },

    // folds
    uTwist: { value: P.twist },
    uFoldP1: { value: P.foldP1 }, uFoldP2: { value: P.foldP2 },
    uFoldI1: { value: P.foldI1 }, uFoldI2: { value: P.foldI2 },

    // facet texture
    uStripeFreq: { value: P.stripeFreq }, uStripeWarp: { value: P.stripeWarp },
    uFilmBase: { value: P.filmBase }, uFilmAmp: { value: P.filmAmp },
    uVibrance: { value: P.vibrance }, uPastel: { value: P.pastel },
    uEdgeWarm: { value: P.edgeWarm }, uEnvTint: { value: 0.6 }
  },
  vertexShader: /* glsl */`
    precision highp float;

    uniform float uTime;
    uniform float uTwist, uFoldP1, uFoldP2, uFoldI1, uFoldI2;
    varying vec3 vWorldPos;

    vec3 foldTriWave(vec3 p, vec3 n, float period, float intensity){
      float d   = dot(p, n);
      float tri = abs(mod(d + period, 2.0*period) - period) - 0.5*period; // [-p/2, p/2]
      float delta = (tri - d) * intensity;
      return p + n * delta;
    }

    void main(){
      vec3 p = position;

      // mild twist depending on height (keeps silhouettes angular but one solid)
      float a = uTwist * p.y * 0.5;
      p.xz = mat2(cos(a), -sin(a), sin(a), cos(a)) * p.xz;

      // two continuous plane folds — “origata” style
      p = foldTriWave(p, normalize(vec3(1.0, 0.0, 0.0)), uFoldP1, uFoldI1);
      p = foldTriWave(p, normalize(vec3(0.0, 1.0, 0.0)), uFoldP2, uFoldI2);

      // subtle rocking so the look evolves
      float r = 0.14 * sin(uTime * 0.18);
      p.xz = mat2(cos(r), -sin(r), sin(r), cos(r)) * p.xz;

      vec4 wp = modelMatrix * vec4(p, 1.0);
      vWorldPos = wp.xyz;
      gl_Position = projectionMatrix * viewMatrix * wp;
    }
  `,
  fragmentShader: /* glsl */`
    precision highp float;

    uniform float uTime;
    uniform float uStripeFreq, uStripeWarp, uFilmBase, uFilmAmp, uVibrance, uPastel, uEdgeWarm, uEnvTint;
    varying vec3 vWorldPos;

    // Facet normal from deformed geometry (requires derivatives; engine enables them if needed)
    vec3 geomNormal(){
      vec3 dx = dFdx(vWorldPos);
      vec3 dy = dFdy(vWorldPos);
      return normalize(cross(dx, dy));
    }

    // Tri‑planar stripes (facet‑anchored)
    vec3 triWeights(vec3 n){ vec3 w = pow(abs(n), vec3(8.0)); return w/(w.x+w.y+w.z + 1e-5); }
    float stripe2D(vec2 uv, float f){
      float s   = sin(uv.x*f)*0.7 + sin((uv.x+uv.y)*f*0.33)*0.3;
      float saw = fract(uv.x*f/6.2831853) - 0.5;   // crisp banding like the reference
      return s + 0.35*saw;
    }
    float triStripeField(vec3 p, vec3 n, float freq, float warp, float t){
      vec3 w = triWeights(n);
      vec2 Ux = p.yz + vec2(0.25*sin(t*0.7), 0.25*cos(t*0.6));
      vec2 Uy = p.xz + vec2(0.30*cos(t*0.5), 0.20*sin(t*0.8));
      vec2 Uz = p.xy + vec2(0.22*sin(t*0.9), 0.18*cos(t*0.4));
      float sx = stripe2D(Ux, freq*(1.0+0.2*warp));
      float sy = stripe2D(Uy, freq*(1.0+0.1*warp));
      float sz = stripe2D(Uz, freq);
      float s  = dot(vec3(sx,sy,sz), w);
      float band = smoothstep(-0.25,0.25,s) - smoothstep(0.25,0.75,s);
      return 0.5 + 0.5*(s + 0.45*band);
    }

    // Thin‑film interference (soap‑film colors)
    vec3 thinFilm(float ndv, float d_nm, float n2){
      float n1=1.0, n3=1.50;
      float sin2 = max(0.0, 1.0 - ndv*ndv);
      float cos2 = sqrt(max(0.0, 1.0 - sin2/(n2*n2)));
      float R12 = pow((n1-n2)/(n1+n2), 2.0), R23 = pow((n2-n3)/(n2+n3), 2.0);
      float A = 2.0*sqrt(R12*R23);
      float lR=650.0, lG=510.0, lB=440.0;
      float phiR=12.5663706*n2*d_nm*cos2/lR;
      float phiG=12.5663706*n2*d_nm*cos2/lG;
      float phiB=12.5663706*n2*d_nm*cos2/lB;
      return clamp(vec3(R12+R23+A*cos(phiR),
                        R12+R23+A*cos(phiG),
                        R12+R23+A*cos(phiB)), 0.0, 1.0);
    }

    void main(){
      vec3 N   = geomNormal();
      vec3 V   = normalize(cameraPosition - vWorldPos);
      float ndv = clamp(dot(N, V), 0.0, 1.0);

      float t = uTime;
      float F = triStripeField(vWorldPos*0.9 + N*0.25, N, uStripeFreq, uStripeWarp, t);
      float d_nm = uFilmBase + uFilmAmp * (F - 0.5) * 2.0;

      vec3 filmRGB = thinFilm(ndv, d_nm, 1.35);
      filmRGB = mix(filmRGB, vec3(1.0), clamp(uPastel,0.0,1.0)) * uVibrance;

      // warm rim and a tiny diffuse base (to mimic soft shading in the reference)
      float rim = pow(1.0 - ndv, 3.0);
      vec3 warm = vec3(1.0, 0.56, 0.35) * uEdgeWarm * rim;
      float diff = 0.35 + 0.65*max(dot(N, normalize(vec3(0.4,0.8,0.2))), 0.0);
      vec3 base = vec3(0.05) * diff * uEnvTint;

      vec3 col = clamp(base + filmRGB + warm, 0.0, 1.0);
      gl_FragColor = vec4(col, 1.0); // alpha=1 for mask extraction
    }
  `,
  depthTest: true, depthWrite: true, transparent: false
});
// IMPORTANT: let Three.js manage OES_standard_derivatives (no manual #extension in GLSL)
foldMat.extensions = { derivatives: true };

const mesh = new THREE.Mesh(geo, foldMat);
scene.add(mesh);

/* ───────────────────────── Post: Feedback TOP + rgbaDelay ───────────────────────── */
let W=2, H=2;
const makeRT = () => new THREE.WebGLRenderTarget(W, H, {
  minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, depthBuffer:false, stencilBuffer:false
});
let rtCurr=makeRT(), rtMask=makeRT(), rtWarp=makeRT(), rtComp=makeRT(), rtTemp=makeRT();

const HISTORY=6, history=new Array(HISTORY).fill(0).map(()=>makeRT()); let histIndex=0;

// Post scene/quad
const postScene = new THREE.Scene();
const postCam   = new THREE.OrthographicCamera(-1,1,1,-1,0,1);
const postQuad  = new THREE.Mesh(new THREE.PlaneGeometry(2,2), new THREE.MeshBasicMaterial({color:0xffffff}));
postScene.add(postQuad);

const fsVS = /* glsl */`varying vec2 vUv; void main(){ vUv=uv; gl_Position=vec4(position.xy,0.,1.);} `;
const noiseGLSL = /* glsl */`
  float hash(vec2 p){ p=fract(p*vec2(123.34,345.45)); p+=dot(p,p+34.345); return fract(p.x*p.y); }
  float noise(vec2 p){ vec2 i=floor(p), f=fract(p);
    float a=hash(i), b=hash(i+vec2(1,0)), c=hash(i+vec2(0,1)), d=hash(i+vec2(1,1));
    vec2 u=f*f*(3.0-2.0*f); return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
  }
`;

// Alpha → mask
const alphaCopyMat = new THREE.ShaderMaterial({
  vertexShader: fsVS,
  fragmentShader: /* glsl */`varying vec2 vUv; uniform sampler2D tInput; void main(){ float a=texture2D(tInput,vUv).a; gl_FragColor=vec4(a,a,a,1.0);} `,
  uniforms: { tInput:{value:null} }
});

// Warp previous composite (Feedback TOP)
const warpMat = new THREE.ShaderMaterial({
  vertexShader: fsVS,
  fragmentShader: /* glsl */`
    varying vec2 vUv; uniform sampler2D tPrev; uniform float uRot,uZoom,uJitter,uTime;
    ${noiseGLSL}
    void main(){
      vec2 uv=vUv-0.5; float c=cos(uRot), s=sin(uRot); uv=mat2(c,-s,s,c)*uv; uv*=(1.0+uZoom); uv+=0.5;
      vec2 jitter=(vec2(noise(vUv*8.0+uTime*0.20), noise(vUv*8.0-uTime*0.22))-0.5)*uJitter;
      gl_FragColor=texture2D(tPrev, uv + jitter);
    }
  `,
  uniforms: { tPrev:{value:null}, uRot:{value:0}, uZoom:{value:0}, uJitter:{value:0}, uTime:{value:0} }
});

// Composite current + warped history, masked
const compMat = new THREE.ShaderMaterial({
  vertexShader: fsVS,
  fragmentShader: /* glsl */`
    varying vec2 vUv; uniform sampler2D tWarped,tCurr,tMask; uniform float uDecay,uFbGain,uCurrGain,uHard;
    void main(){
      vec4 fb = texture2D(tWarped, vUv) * uDecay * uFbGain;
      vec4 cur= texture2D(tCurr,   vUv) * uCurrGain;
      float m  = pow(texture2D(tMask, vUv).r, uHard);
      gl_FragColor = clamp(m*fb + cur, 0.0, 1.0);
    }
  `,
  uniforms: { tWarped:{value:null}, tCurr:{value:null}, tMask:{value:null},
              uDecay:{value:P.fbDecay}, uFbGain:{value:P.fbGain}, uCurrGain:{value:P.currGain}, uHard:{value:P.maskHardness} }
});

// Blur (soft glow)
const blurMat = new THREE.ShaderMaterial({
  vertexShader: fsVS,
  fragmentShader: /* glsl */`
    varying vec2 vUv; uniform sampler2D tInput; uniform vec2 uTexel,uDir; uniform float uRadius;
    void main(){
      vec2 stepv=uDir*uTexel; float r=uRadius; vec4 acc=vec4(0.0); float wsum=0.0;
      for(int i=-4;i<=4;i++){ float x=float(i); float w=exp(-0.5*(x*x)/(r*r+1e-5));
        acc+=texture2D(tInput, vUv + x*stepv)*w; wsum+=w; }
      gl_FragColor=acc/max(wsum,1e-5);
    }
  `,
  uniforms: { tInput:{value:null}, uTexel:{value:new THREE.Vector2(1,1)}, uDir:{value:new THREE.Vector2(1,0)}, uRadius:{value:P.blurRadius} }
});

// Copy (for history ring)
const copyMat = new THREE.ShaderMaterial({
  vertexShader: fsVS, fragmentShader: /* glsl */`varying vec2 vUv; uniform sampler2D tInput; void main(){ gl_FragColor=texture2D(tInput,vUv);} `,
  uniforms: { tInput:{value:null} }
});

// rgbaDelay final (integer delays, avoids array-uniform pitfalls)
const finalMat = new THREE.ShaderMaterial({
  vertexShader: fsVS,
  fragmentShader: /* glsl */`
    varying vec2 vUv;
    uniform sampler2D t0,t1,t2,t3,t4,t5;
    uniform vec2 uTexel; uniform float uDispersion;
    uniform int uRDelay, uGDelay, uBDelay;

    vec4 T(int i, vec2 uv){
      if(i==0) return texture2D(t0,uv);
      if(i==1) return texture2D(t1,uv);
      if(i==2) return texture2D(t2,uv);
      if(i==3) return texture2D(t3,uv);
      if(i==4) return texture2D(t4,uv);
      return texture2D(t5,uv);
    }
    void main(){
      vec2 ur=vUv+vec2( uDispersion,-uDispersion)*uTexel;
      vec2 ug=vUv;
      vec2 ub=vUv+vec2(-uDispersion, uDispersion)*uTexel;
      float r=T(uRDelay, ur).r;
      float g=T(uGDelay, ug).g;
      float b=T(uBDelay, ub).b;
      gl_FragColor=vec4(r,g,b,1.0);
    }
  `,
  uniforms: {
    t0:{value:null}, t1:{value:null}, t2:{value:null}, t3:{value:null}, t4:{value:null}, t5:{value:null},
    uTexel:{value:new THREE.Vector2(1,1)}, uDispersion:{value:P.dispersion},
    uRDelay:{value:1}, uGDelay:{value:3}, uBDelay:{value:5}
  }
});

/* ───────────────────────── Poses (16 stances → seamless loop) ───────────────────────── */
const POSES = [
  {tw:2.1,p1:1.30,i1:1.30,p2:0.95,i2:1.05,sf:22.0,sw:0.48,base:420,amp:360, dR:1,dG:3,dB:5},
  {tw:2.4,p1:1.15,i1:1.45,p2:1.05,i2:0.90,sf:19.6,sw:0.50,base:418,amp:350, dR:0,dG:2,dB:4},
  {tw:2.8,p1:1.05,i1:1.60,p2:0.90,i2:1.10,sf:23.0,sw:0.44,base:440,amp:330, dR:0,dG:3,dB:5},
  {tw:2.2,p1:1.45,i1:1.10,p2:1.10,i2:1.25,sf:18.0,sw:0.56,base:446,amp:320, dR:1,dG:2,dB:5},
  {tw:2.0,p1:1.22,i1:1.35,p2:0.85,i2:0.95,sf:18.4,sw:0.62,base:425,amp:360, dR:0,dG:1,dB:4},
  {tw:2.6,p1:1.00,i1:1.45,p2:1.25,i2:0.85,sf:20.0,sw:0.50,base:435,amp:340, dR:2,dG:4,dB:5},
  {tw:2.3,p1:0.95,i1:1.55,p2:1.30,i2:1.00,sf:16.0,sw:0.64,base:445,amp:330, dR:1,dG:4,dB:5},
  {tw:2.1,p1:1.35,i1:1.15,p2:1.10,i2:1.25,sf:18.2,sw:0.56,base:420,amp:360, dR:0,dG:5,dB:5},
  {tw:2.1,p1:1.30,i1:1.30,p2:0.95,i2:1.05,sf:22.0,sw:0.48,base:420,amp:360, dR:1,dG:3,dB:5},
  {tw:2.4,p1:1.15,i1:1.45,p2:1.05,i2:0.90,sf:19.2,sw:0.50,base:415,amp:350, dR:0,dG:2,dB:4},
  {tw:2.7,p1:1.05,i1:1.60,p2:0.90,i2:1.10,sf:22.2,sw:0.44,base:438,amp:328, dR:0,dG:3,dB:5},
  {tw:2.2,p1:1.43,i1:1.10,p2:1.10,i2:1.25,sf:17.8,sw:0.58,base:445,amp:320, dR:1,dG:2,dB:5},
  {tw:2.0,p1:1.22,i1:1.35,p2:0.86,i2:0.95,sf:18.0,sw:0.62,base:425,amp:360, dR:0,dG:1,dB:4},
  {tw:2.5,p1:1.00,i1:1.45,p2:1.24,i2:0.85,sf:19.6,sw:0.50,base:435,amp:340, dR:2,dG:4,dB:5},
  {tw:2.2,p1:0.95,i1:1.55,p2:1.28,i2:0.98,sf:15.4,sw:0.64,base:445,amp:330, dR:1,dG:4,dB:5},
  {tw:2.1,p1:1.30,i1:1.30,p2:0.95,i2:1.05,sf:22.0,sw:0.48,base:420,amp:360, dR:1,dG:3,dB:5}
];

function applyPose(p){
  foldMat.uniforms.uTwist.value = p.tw;
  foldMat.uniforms.uFoldP1.value = p.p1; foldMat.uniforms.uFoldI1.value = p.i1;
  foldMat.uniforms.uFoldP2.value = p.p2; foldMat.uniforms.uFoldI2.value = p.i2;
  foldMat.uniforms.uStripeFreq.value = p.sf; foldMat.uniforms.uStripeWarp.value = p.sw;
  foldMat.uniforms.uFilmBase.value = p.base; foldMat.uniforms.uFilmAmp.value = p.amp;
  finalMat.uniforms.uRDelay.value = p.dR;
  finalMat.uniforms.uGDelay.value = p.dG;
  finalMat.uniforms.uBDelay.value = p.dB;
}

/* ───────────────────────── Resize ───────────────────────── */
function panelWidth(){ const el=document.querySelector('.lil-gui.root'); return el?Math.ceil(el.getBoundingClientRect().width):300; }
function onResize(){
  const w=Math.max(1, window.innerWidth - panelWidth()), h=Math.max(1, window.innerHeight);
  renderer.setSize(w, h, false); camera.aspect = w/h; camera.updateProjectionMatrix();
  W=w; H=h; [rtCurr, rtMask, rtWarp, rtComp, rtTemp, ...history].forEach(rt=>rt.setSize(W,H));
  blurMat.uniforms.uTexel.value.set(1/W,1/H); finalMat.uniforms.uTexel.value.set(1/W,1/H);
}
window.addEventListener('resize', onResize); onResize();

/* ───────────────────────── Main Loop ───────────────────────── */
let simTime=0; const clock=new THREE.Clock(); let bootFilled=false;

function drawSceneTo(rt){
  // Offscreen render with alpha=0 background → clean mask
  renderer.setClearColor(0x000000, 0); // alpha 0
  renderer.setRenderTarget(rt); renderer.clear(); renderer.render(scene, camera);
  renderer.setRenderTarget(null);
  renderer.setClearColor(0x000000, 1); // restore
}

function step(dt){
  simTime += dt * P.speed;

  // Seamless 16‑second loop (no ping‑pong)
  const T=P.cycleSeconds, t=simTime%T, i=Math.floor(t), u=t-i;
  const s=u*u*(3.0-2.0*u), A=POSES[i], B=POSES[(i+1)%POSES.length];
  const L=(a,b)=>a+(b-a)*s;
  applyPose({ tw:L(A.tw,B.tw), p1:L(A.p1,B.p1), i1:L(A.i1,B.i1), p2:L(A.p2,B.p2), i2:L(A.i2,B.i2),
              sf:L(A.sf,B.sf), sw:L(A.sw,B.sw), base:L(A.base,B.base), amp:L(A.amp,B.amp),
              dR:(u<0.5?A.dR:B.dR), dG:(u<0.5?A.dG:B.dG), dB:(u<0.5?A.dB:B.dB) });

  if(P.autoRotate){ camera.rotation.y += P.spin*dt; camera.updateMatrixWorld(); }
  foldMat.uniforms.uTime.value = simTime;
  controls.update();

  if(P.postBypass){ renderer.render(scene, camera); return; }

  // 1) Scene → rtCurr
  drawSceneTo(rtCurr);

  // 2) Alpha → mask
  alphaCopyMat.uniforms.tInput.value = rtCurr.texture; postQuad.material = alphaCopyMat;
  renderer.setRenderTarget(rtMask); renderer.render(postScene, postCam);

  // 3) Warp previous composite (feedback)
  warpMat.uniforms.tPrev.value = history[(histIndex-1+HISTORY)%HISTORY].texture;
  warpMat.uniforms.uRot.value = P.warpRotate; warpMat.uniforms.uZoom.value = P.warpZoom;
  warpMat.uniforms.uJitter.value = P.warpJitter; warpMat.uniforms.uTime.value = simTime;
  postQuad.material = warpMat;
  renderer.setRenderTarget(rtWarp); renderer.render(postScene, postCam);

  // 4) Composite current + warped history, masked
  compMat.uniforms.tWarped.value = rtWarp.texture; compMat.uniforms.tCurr.value = rtCurr.texture; compMat.uniforms.tMask.value = rtMask.texture;
  compMat.uniforms.uDecay.value = P.fbDecay; compMat.uniforms.uFbGain.value = P.fbGain; compMat.uniforms.uCurrGain.value = P.currGain; compMat.uniforms.uHard.value = P.maskHardness;
  postQuad.material = compMat;
  renderer.setRenderTarget(rtComp); renderer.render(postScene, postCam);

  // 5) Blur glow
  if(P.blurRadius>0.001){
    blurMat.uniforms.uRadius.value=P.blurRadius; blurMat.uniforms.tInput.value=rtComp.texture; blurMat.uniforms.uDir.value.set(1,0);
    postQuad.material=blurMat; renderer.setRenderTarget(rtTemp); renderer.render(postScene, postCam);
    blurMat.uniforms.tInput.value=rtTemp.texture; blurMat.uniforms.uDir.value.set(0,1);
    postQuad.material=blurMat; renderer.setRenderTarget(rtComp); renderer.render(postScene, postCam);
  }

  // 6) Boot‑fill history (prevents initial black)
  if(!bootFilled){
    copyMat.uniforms.tInput.value = rtComp.texture; postQuad.material = copyMat;
    for(let k=0;k<HISTORY;k++){ renderer.setRenderTarget(history[k]); renderer.render(postScene, postCam); }
    renderer.setRenderTarget(null); bootFilled=true;
  }

  // 7) Push current composite into ring
  copyMat.uniforms.tInput.value = rtComp.texture; postQuad.material = copyMat;
  renderer.setRenderTarget(history[histIndex]); renderer.render(postScene, postCam);
  renderer.setRenderTarget(null); histIndex=(histIndex+1)%HISTORY;

  // 8) Final rgbaDelay to screen
  for(let k=0;k<HISTORY;k++){ const slot=(histIndex-1-k+HISTORY)%HISTORY; finalMat.uniforms['t'+k].value = history[slot].texture; }
  finalMat.uniforms.uDispersion.value = P.dispersion;
  postQuad.material = finalMat; renderer.setRenderTarget(null); renderer.render(postScene, postCam);
}
function animate(){ const dt=clock.getDelta(); step(dt); requestAnimationFrame(animate); }
requestAnimationFrame(animate);

/* ───────────────────────── Contact Sheet (press C) ───────────────────────── */
window.addEventListener('keydown', async (e)=>{
  if(e.key.toLowerCase()!=='c') return;
  const rows=4, cols=4, pad=8, cw=P.sheetSize, ch=P.sheetSize;
  const W2=cols*cw+(cols+1)*pad, H2=rows*ch+(rows+1)*pad;
  const off=document.createElement('canvas'); off.width=W2; off.height=H2;
  const ctx=off.getContext('2d'); ctx.imageSmoothingEnabled=true; ctx.fillStyle='#000'; ctx.fillRect(0,0,W2,H2);

  const secondsPerSample=1.0, substeps=60, shots=[];
  for(let n=0;n<rows*cols;n++){
    for(let s=0;s<substeps;s++) step(secondsPerSample/substeps);
    shots.push(await createImageBitmap(renderer.domElement));
  }
  let idx=0;
  for(let r=0;r<rows;r++) for(let c=0;c<cols;c++){
    const x=pad+c*(cw+pad), y=pad+r*(ch+pad);
    ctx.drawImage(shots[idx++], 0,0,renderer.domElement.width,renderer.domElement.height, x,y,cw,ch);
  }
  const a=document.createElement('a'); a.download='contact_sheet.png'; a.href=off.toDataURL('image/png'); a.click();
});

/* ───────────────────────── Initial Layout ───────────────────────── */
function panelW(){ const el=document.querySelector('.lil-gui.root'); return el?Math.ceil(el.getBoundingClientRect().width):300; }
function size(){ const w=Math.max(1, window.innerWidth-panelW()); const h=Math.max(1, window.innerHeight);
  renderer.setSize(w,h,false); camera.aspect=w/h; camera.updateProjectionMatrix();
  W=w; H=h; [rtCurr, rtMask, rtWarp, rtComp, rtTemp, ...history].forEach(rt=>rt.setSize(W,H));
  blurMat.uniforms.uTexel.value.set(1/W,1/H); finalMat.uniforms.uTexel.value.set(1/W,1/H);
}
window.addEventListener('resize', size); size();

/* ───────────────────────── Notes / Sources ─────────────────────────
  • TouchDesigner Feedback TOP & rgbaDelay behavior informed the post chain (feedback + decay + blur + per-channel taps). Docs: Derivative Feedback TOP and Palette: rgbaDelay. :contentReference[oaicite:0]{index=0}
  • Facet‑anchored stripes use tri‑planar mapping. :contentReference[oaicite:1]{index=1}
  • Thin‑film iridescence (soap‑film colors) approach adapted from Three.js community references. :contentReference[oaicite:2]{index=2}
  • Derivatives are enabled via Three.js material extensions — no manual #extension lines are needed; the engine inserts or omits it correctly. :contentReference[oaicite:3]{index=3}
*/ 
