// Origata — SDF folding + thin-film facet texture + Feedback/RGBA delay
// Frustum-corner ray construction (robust for raymarchers) + typed-array uniforms.
// References: TouchDesigner Feedback TOP & rgbaDelay, tri-planar mapping, thin-film iridescence, raymarching frustum corners.
// TD Feedback: https://docs.derivative.ca/Feedback_TOP
// TD rgbaDelay: https://docs.derivative.ca/Palette%3ArgbaDelay
// Tri-planar mapping: https://catlikecoding.com/unity/tutorials/advanced-rendering/triplanar-mapping/
// Thin film iridescence (DerSchmale): https://github.com/DerSchmale/threejs-thin-film-iridescence
// Frustum corner rays: https://adrianb.io/2016/10/01/raymarching.html

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import GUI from 'https://cdn.jsdelivr.net/npm/lil-gui@0.19/+esm';

/* ────────────────────────── Renderer / Cameras ────────────────────────── */
const canvas = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
renderer.setClearColor(0x000000, 1);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;

// Fullscreen scene (all passes)
const fsScene = new THREE.Scene();
const fsCam   = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

// View camera used to define rays (and orbit)
const viewCam = new THREE.PerspectiveCamera(30, 1, 0.1, 20);
viewCam.position.set(0, 0, 4.0);
const controls = new OrbitControls(viewCam, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.08; controls.rotateSpeed = 0.6;

/* ───────────────────────────── Controls ───────────────────────────── */
const P = {
  cycleSeconds: 16, speed: 1.0, autoRotate: true, spin: 0.15,

  // folding
  twist: 2.2, fold1: 1.30, fold2: 1.10, foldInt1: 1.25, foldInt2: 1.10,

  // facet texture
  stripeFreq: 20.0, stripeWarp: 0.55, filmBase: 420.0, filmAmp: 360.0,
  vibrance: 2.2, pastel: 0.15, edgeWarm: 0.26, envTint: 0.6,

  // feedback
  fbDecay: 0.92, fbGain: 0.48, currGain: 0.78,
  warpRotate: 0.010, warpZoom: 0.0015, warpJitter: 0.0010,
  blurRadius: 1.2, maskHardness: 1.06, dispersion: 0.0014,

  sheetSize: 320
};

const gui = new GUI({ title: 'Controls', width: 300 });
const fT = gui.addFolder('Timeline');
fT.add(P, 'cycleSeconds', 8, 32, 1).name('Cycle (s)');
fT.add(P, 'speed', 0.25, 3.0, 0.01);
fT.add(P, 'autoRotate'); fT.add(P, 'spin', 0.0, 1.0, 0.01);
const fS = gui.addFolder('Shape');
fS.add(P, 'twist', 0.0, 4.0, 0.01);
fS.add(P, 'fold1', 0.6, 2.0, 0.01).name('Fold P1');
fS.add(P, 'fold2', 0.6, 2.0, 0.01).name('Fold P2');
fS.add(P, 'foldInt1', 0.5, 2.0, 0.01).name('Fold I1');
fS.add(P, 'foldInt2', 0.5, 2.0, 0.01).name('Fold I2');
const fI = gui.addFolder('Facet Texture');
fI.add(P, 'stripeFreq', 6, 36, 0.1).name('Stripe Freq');
fI.add(P, 'stripeWarp', 0.0, 1.0, 0.01).name('Stripe Warp');
fI.add(P, 'filmBase', 300, 520, 1).name('Film Base (nm)');
fI.add(P, 'filmAmp', 100, 520, 1).name('Film Amp (nm)');
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

/* ───────────────────────────── SDF Pass ───────────────────────────── */
const fsGeo = new THREE.PlaneGeometry(2, 2);
const sdfMat = new THREE.ShaderMaterial({
  uniforms: {
    uTime: { value: 0 },
    uResolution: { value: new THREE.Vector2(1,1) },

    // frustum-corner rays (world-space)
    uCamPos: { value: new THREE.Vector3() },
    uCornerTL: { value: new THREE.Vector3() },
    uCornerTR: { value: new THREE.Vector3() },
    uCornerBL: { value: new THREE.Vector3() },
    uCornerBR: { value: new THREE.Vector3() },

    // folding controls
    uTwist:  { value: P.twist },
    uFoldP1: { value: P.fold1 }, uFoldP2: { value: P.fold2 },
    uFoldI1: { value: P.foldInt1 }, uFoldI2: { value: P.foldInt2 },

    // facet texture
    uStripeFreq: { value: P.stripeFreq }, uStripeWarp: { value: P.stripeWarp },
    uFilmBase: { value: P.filmBase },   uFilmAmp: { value: P.filmAmp },
    uVibrance: { value: P.vibrance },   uPastel: { value: P.pastel },
    uEdgeWarm: { value: P.edgeWarm },   uEnvTint: { value: P.envTint }
  },
  vertexShader: /* glsl */`
    varying vec2 vUv;
    void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }
  `,
  fragmentShader: /* glsl */`
    precision highp float;
    varying vec2 vUv;

    uniform float uTime;
    uniform vec2  uResolution;
    uniform vec3  uCamPos, uCornerTL, uCornerTR, uCornerBL, uCornerBR;

    uniform float uTwist, uFoldP1, uFoldP2, uFoldI1, uFoldI2;
    uniform float uStripeFreq, uStripeWarp, uFilmBase, uFilmAmp, uVibrance, uPastel, uEdgeWarm;
    uniform float uEnvTint;

    vec3 rotY(vec3 p, float a){ float c=cos(a), s=sin(a); return vec3(c*p.x - s*p.z, p.y, s*p.x + c*p.z); }
    vec3 rotX(vec3 p, float a){ float c=cos(a), s=sin(a); return vec3(p.x, c*p.y - s*p.z, s*p.y + c*p.z); }
    float sdBox(vec3 p, vec3 b){ vec3 q=abs(p)-b; return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0); }

    // Continuous triangular-wave plane fold
    vec3 foldTriWave(vec3 p, vec3 n, float period, float intensity){
      float d = dot(p, n);
      float tri = abs(mod(d + period, 2.0*period) - period) - 0.5*period;
      float delta = (tri - d) * intensity;
      return p + n * delta;
    }

    // Scene SDF: box + continuous plane folds (angular silhouettes, no separation)
    float sdScene(vec3 p){
      // subtle motion + twist
      p = rotX(p, 0.2*sin(uTime*0.21));
      float a = uTwist * p.y;
      p.xz = mat2(cos(a),-sin(a), sin(a),cos(a)) * p.xz;

      // two independent folds
      p = foldTriWave(p, normalize(vec3(1,0,0)), uFoldP1, uFoldI1);
      p = foldTriWave(p, normalize(vec3(0,1,0)), uFoldP2, uFoldI2);

      // base primitive (box)
      float d = sdBox(p, vec3(1.0));
      return d;
    }

    vec2 raymarch(vec3 ro, vec3 rd){
      float t = 0.0; float hit = -1.0;
      for(int i=0;i<128;i++){
        vec3 pos = ro + rd*t;
        float d = sdScene(pos);
        if(d < 0.001){ hit = 1.0; break; }
        t += d;
        if(t>20.0) break;
      }
      return vec2(t, hit);
    }

    vec3 calcNormal(vec3 p){
      const float e=0.0015;
      vec2 h=vec2(1.0,-1.0)*0.5773;
      return normalize( h.xyy*sdScene(p+h.xyy*e) +
                        h.yyx*sdScene(p+h.yyx*e) +
                        h.yxy*sdScene(p+h.yxy*e) +
                        h.xxx*sdScene(p+h.xxx*e) );
    }

    // Tri-planar stripe field (facet-aligned)
    vec3 triWeights(vec3 n){ vec3 w = pow(abs(n), vec3(8.0)); return w/(w.x+w.y+w.z+1e-5); }
    float stripe2D(vec2 uv, float f){
      float s = sin(uv.x*f)*0.7 + sin((uv.x+uv.y)*f*0.33)*0.3;
      float saw = fract(uv.x*f/6.2831853) - 0.5;
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
      float s = dot(vec3(sx,sy,sz), w);
      float band = smoothstep(-0.25,0.25,s) - smoothstep(0.25,0.75,s);
      return 0.5 + 0.5*(s + 0.45*band);
    }

    // Thin-film interference (RGB phase)
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

    // Build a ray from frustum corners (world-space) — robust for raymarching
    void makeRay(out vec3 ro, out vec3 rd){
      vec3 dx0 = mix(uCornerBL, uCornerBR, vUv.x);
      vec3 dx1 = mix(uCornerTL, uCornerTR, vUv.x);
      vec3 dir = normalize(mix(dx0, dx1, vUv.y));
      ro = uCamPos; rd = dir;
    }

    void main(){
      vec3 ro, rd; makeRay(ro, rd);

      // tiny global rocking to match reference vibe
      float t = uTime;
      rd = rotY(rd, 0.12*sin(t*0.17));

      vec2 hit = raymarch(ro, rd);
      if(hit.y < 0.0){ gl_FragColor = vec4(0.0); return; }

      vec3 pos = ro + rd*hit.x;
      vec3 N   = calcNormal(pos);
      vec3 V   = normalize(ro - pos);
      float ndv = clamp(dot(N,V), 0.0, 1.0);

      // facet texture
      float F = triStripeField(pos*0.8 + N*0.25, N, uStripeFreq, uStripeWarp, t);
      float d_nm = uFilmBase + uFilmAmp * (F - 0.5) * 2.0;
      vec3 filmRGB = thinFilm(ndv, d_nm, 1.35);

      // pastelized emissive, warm rim, and a tiny diffuse tint
      filmRGB = mix(filmRGB, vec3(1.0), clamp(uPastel, 0.0, 1.0)) * uVibrance;
      float rim = pow(1.0 - ndv, 3.0);
      vec3 warm = vec3(1.0, 0.56, 0.35) * uEdgeWarm * rim;
      float diff = 0.35 + 0.65*max(dot(N, normalize(vec3(0.4,0.8,0.2))), 0.0);
      vec3 base = vec3(0.05) * diff * uEnvTint;

      vec3 col = base + filmRGB + warm;
      gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0); // alpha used as mask
    }
  `,
  depthTest: false, depthWrite: false, transparent: false
});
const quad = new THREE.Mesh(fsGeo, sdfMat); fsScene.add(quad);

/* ─────────────────────── Post: Feedback + RGBA Delay ─────────────────────── */
let W = 2, H = 2;
const makeRT = () => new THREE.WebGLRenderTarget(W, H, {
  minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter,
  depthBuffer: false, stencilBuffer: false
});
let rtScene = makeRT(), rtA = makeRT(), rtB = makeRT(), rtTemp = makeRT(), rtMask = makeRT();

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
  fragmentShader: /* glsl */`varying vec2 vUv; uniform sampler2D tInput;
    void main(){ float a=texture2D(tInput,vUv).a; gl_FragColor=vec4(a,a,a,1.0); }`,
  uniforms: { tInput: { value: null } }
});

// Warp previous frame
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

// Composite current + warped feedback (masked)
const compositeMat = new THREE.ShaderMaterial({
  vertexShader: fsVS,
  fragmentShader: /* glsl */`
    varying vec2 vUv; uniform sampler2D tWarped,tCurr,tMask;
    uniform float uDecay,uFbGain,uCurrGain,uHard;
    void main(){
      vec4 fb = texture2D(tWarped, vUv) * uDecay * uFbGain;
      vec4 cur= texture2D(tCurr,   vUv) * uCurrGain;
      float m = pow(texture2D(tMask, vUv).r, uHard);
      gl_FragColor = clamp(m * fb + cur, 0.0, 1.0);
    }
  `,
  uniforms: {
    tWarped:{value:null}, tCurr:{value:null}, tMask:{value:null},
    uDecay:{value:P.fbDecay}, uFbGain:{value:P.fbGain}, uCurrGain:{value:P.currGain}, uHard:{value:P.maskHardness}
  }
});

// Separable blur
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

// History ring (RGBA delay) — use TYPED ARRAYS for weights so uniforms always update
const HISTORY = 8;
const history = new Array(HISTORY).fill(0).map(()=>makeRT()); let histIndex = 0;
const copyMat = new THREE.ShaderMaterial({
  vertexShader: fsVS,
  fragmentShader: /* glsl */`varying vec2 vUv; uniform sampler2D tInput; void main(){ gl_FragColor=texture2D(tInput,vUv);} `,
  uniforms: { tInput:{value:null} }
});

// IMPORTANT: use Float32Array for uniform arrays
const wR = new Float32Array(8), wG = new Float32Array(8), wB = new Float32Array(8);
wR[0]=wG[0]=wB[0]=1; // ensure non-black first frame
const finalMat = new THREE.ShaderMaterial({
  vertexShader: fsVS,
  fragmentShader: /* glsl */`
    varying vec2 vUv;
    uniform sampler2D t0; uniform sampler2D t1; uniform sampler2D t2; uniform sampler2D t3;
    uniform sampler2D t4; uniform sampler2D t5; uniform sampler2D t6; uniform sampler2D t7;
    uniform float wR[8]; uniform float wG[8]; uniform float wB[8];
    uniform vec2 uTexel; uniform float uDispersion;
    void main(){
      vec2 ur=vUv+vec2( uDispersion,-uDispersion)*uTexel;
      vec2 ug=vUv;
      vec2 ub=vUv+vec2(-uDispersion, uDispersion)*uTexel;
      float r=texture2D(t0,ur).r*wR[0]+texture2D(t1,ur).r*wR[1]+texture2D(t2,ur).r*wR[2]+texture2D(t3,ur).r*wR[3]+
              texture2D(t4,ur).r*wR[4]+texture2D(t5,ur).r*wR[5]+texture2D(t6,ur).r*wR[6]+texture2D(t7,ur).r*wR[7];
      float g=texture2D(t0,ug).g*wG[0]+texture2D(t1,ug).g*wG[1]+texture2D(t2,ug).g*wG[2]+texture2D(t3,ug).g*wG[3]+
              texture2D(t4,ug).g*wG[4]+texture2D(t5,ug).g*wG[5]+texture2D(t6,ug).g*wG[6]+texture2D(t7,ug).g*wG[7];
      float b=texture2D(t0,ub).b*wB[0]+texture2D(t1,ub).b*wB[1]+texture2D(t2,ub).b?wB[2]+texture2D(t3,ub).b*wB[3]+
              texture2D(t4,ub).b*wB[4]+texture2D(t5,ub).b*wB[5]+texture2D(t6,ub).b*wB[6]+texture2D(t7,ub).b*wB[7];
      gl_FragColor=vec4(r,g,b,1.0);
    }
  `,
  uniforms: {
    t0:{value:null}, t1:{value:null}, t2:{value:null}, t3:{value:null},
    t4:{value:null}, t5:{value:null}, t6:{value:null}, t7:{value:null},
    wR:{value:wR}, wG:{value:wG}, wB:{value:wB},
    uTexel:{value:new THREE.Vector2(1,1)}, uDispersion:{value:P.dispersion}
  }
});

// helper to mutate typed-array weights (don’t replace the array!)
function setDelays(r,g,b){
  wR.fill(0); wG.fill(0); wB.fill(0);
  wR[Math.max(0,Math.min(7,r))] = 1.0;
  wG[Math.max(0,Math.min(7,g))] = 1.0;
  wB[Math.max(0,Math.min(7,b))] = 1.0;
}

/* ─────────────────── Resize + frustum corners (rays) ─────────────────── */
function panelWidth(){ const el=document.querySelector('.lil-gui.root'); return el?Math.ceil(el.getBoundingClientRect().width):300; }
function updateFrustumCorners(){
  const invProj = new THREE.Matrix4().copy(viewCam.projectionMatrix).invert();
  const mWorld  = viewCam.matrixWorld;
  const ndc = [
    new THREE.Vector3(-1,  1, 1), // TL
    new THREE.Vector3( 1,  1, 1), // TR
    new THREE.Vector3(-1, -1, 1), // BL
    new THREE.Vector3( 1, -1, 1)  // BR
  ];
  const toDir = (v)=>{
    const p4 = new THREE.Vector4(v.x, v.y, v.z, 1).applyMatrix4(invProj);
    p4.divideScalar(p4.w).applyMatrix4(mWorld);
    return new THREE.Vector3(p4.x,p4.y,p4.z).sub(viewCam.position).normalize();
  };
  sdfMat.uniforms.uCamPos.value.copy(viewCam.position);
  sdfMat.uniforms.uCornerTL.value.copy(toDir(ndc[0]));
  sdfMat.uniforms.uCornerTR.value.copy(toDir(ndc[1]));
  sdfMat.uniforms.uCornerBL.value.copy(toDir(ndc[2]));
  sdfMat.uniforms.uCornerBR.value.copy(toDir(ndc[3]));
}
function onResize(){
  const w=Math.max(1, window.innerWidth - panelWidth());
  const h=Math.max(1, window.innerHeight);
  renderer.setSize(w, h, false);
  viewCam.aspect = w/h; viewCam.updateProjectionMatrix();
  W=w; H=h;
  [rtScene, rtA, rtB, rtTemp, rtMask, ...history].forEach(rt=>rt.setSize(W,H));
  blurMat.uniforms.uTexel.value.set(1/W,1/H);
  finalMat.uniforms.uTexel.value.set(1/W,1/H);
  sdfMat.uniforms.uResolution.value.set(W,H);
  updateFrustumCorners();
}
window.addEventListener('resize', onResize); onResize();

/* ───────────────────────── 16 stances → seamless loop ───────────────────────── */
const POSES = [
  {tw:2.1, p1:1.30, i1:1.30, p2:0.95, i2:1.05, sf:20.0, sw:0.55, base:420, amp:360, d:[1,3,5]},
  {tw:2.5, p1:1.15, i1:1.45, p2:1.05, i2:0.90, sf:18.0, sw:0.51, base:415, amp:350, d:[0,2,4]},
  {tw:2.9, p1:1.05, i1:1.60, p2:0.90, i2:1.10, sf:21.0, sw:0.46, base:440, amp:330, d:[0,3,6]},
  {tw:2.3, p1:1.45, i1:1.10, p2:1.10, i2:1.25, sf:16.8, sw:0.56, base:445, amp:320, d:[1,2,5]},
  {tw:2.0, p1:1.22, i1:1.35, p2:0.85, i2:0.95, sf:17.0, sw:0.62, base:425, amp:360, d:[0,1,4]},
  {tw:2.7, p1:1.00, i1:1.45, p2:1.25, i2:0.85, sf:18.5, sw:0.50, base:435, amp:340, d:[2,4,6]},
  {tw:2.4, p1:0.95, i1:1.55, p2:1.30, i2:1.00, sf:15.0, sw:0.64, base:445, amp:330, d:[1,4,7]},
  {tw:2.2, p1:1.35, i1:1.15, p2:1.10, i2:1.25, sf:17.2, sw:0.56, base:420, amp:360, d:[0,5,7]},
  {tw:2.1, p1:1.30, i1:1.30, p2:0.95, i2:1.05, sf:20.0, sw:0.53, base:422, amp:355, d:[1,3,5]},
  {tw:2.5, p1:1.15, i1:1.45, p2:1.05, i2:0.90, sf:17.6, sw:0.50, base:413, amp:350, d:[0,2,4]},
  {tw:2.8, p1:1.05, i1:1.60, p2:0.90, i2:1.10, sf:20.2, sw:0.46, base:438, amp:328, d:[0,3,6]},
  {tw:2.3, p1:1.43, i1:1.10, p2:1.10, i2:1.25, sf:16.5, sw:0.58, base:445, amp:320, d:[1,2,5]},
  {tw:2.0, p1:1.22, i1:1.35, p2:0.86, i2:0.95, sf:16.8, sw:0.62, base:425, amp:360, d:[0,1,4]},
  {tw:2.6, p1:1.00, i1:1.45, p2:1.24, i2:0.85, sf:18.0, sw:0.50, base:435, amp:340, d:[2,4,6]},
  {tw:2.3, p1:0.95, i1:1.55, p2:1.28, i2:0.98, sf:14.6, sw:0.64, base:445, amp:330, d:[1,4,7]},
  {tw:2.1, p1:1.30, i1:1.30, p2:0.95, i2:1.05, sf:20.0, sw:0.55, base:420, amp:360, d:[1,3,5]} // wrap
];

function applyPose(p){
  sdfMat.uniforms.uTwist.value = p.tw;
  sdfMat.uniforms.uFoldP1.value = p.p1; sdfMat.uniforms.uFoldI1.value = p.i1;
  sdfMat.uniforms.uFoldP2.value = p.p2; sdfMat.uniforms.uFoldI2.value = p.i2;
  sdfMat.uniforms.uStripeFreq.value = p.sf; sdfMat.uniforms.uStripeWarp.value = p.sw;
  sdfMat.uniforms.uFilmBase.value = p.base; sdfMat.uniforms.uFilmAmp.value = p.amp;
  setDelays(p.d[0], p.d[1], p.d[2]);
}

/* ─────────────────────────── Main loop ─────────────────────────── */
let simTime = 0; const clock = new THREE.Clock();
let bootFilled = false;

function step(dt){
  simTime += dt * P.speed;

  // seamless loop (no ping-pong)
  const T=P.cycleSeconds, t=simTime%T, i=Math.floor(t), u=t-i;
  const s=u*u*(3.0-2.0*u), A=POSES[i], B=POSES[(i+1)%POSES.length];
  const L=(a,b)=>a+(b-a)*s;
  applyPose({
    tw:L(A.tw,B.tw),
    p1:L(A.p1,B.p1), i1:L(A.i1,B.i1),
    p2:L(A.p2,B.p2), i2:L(A.i2,B.i2),
    sf:L(A.sf,B.sf), sw:L(A.sw,B.sw), base:L(A.base,B.base), amp:L(A.amp,B.amp),
    d:(u<0.5?A.d:B.d)
  });

  if(P.autoRotate){ viewCam.rotation.y += P.spin * dt; viewCam.updateMatrixWorld(); }
  updateFrustumCorners();

  // 1) SDF → rtScene (alpha as mask)
  sdfMat.uniforms.uTime.value = simTime;
  renderer.setRenderTarget(rtScene); renderer.render(fsScene, fsCam); renderer.setRenderTarget(null);

  // 2) Mask
  alphaCopyMat.uniforms.tInput.value = rtScene.texture;
  postQuad.material = alphaCopyMat;
  renderer.setRenderTarget(rtMask); renderer.render(postScene, postCam); renderer.setRenderTarget(null);

  // 3) Feedback warp
  warpMat.uniforms.tPrev.value = rtA.texture;
  warpMat.uniforms.uRot.value  = P.warpRotate;
  warpMat.uniforms.uZoom.value = P.warpZoom;
  warpMat.uniforms.uJitter.value = P.warpJitter;
  warpMat.uniforms.uTime.value = simTime;
  postQuad.material = warpMat;
  renderer.setRenderTarget(rtTemp); renderer.render(postScene, postCam);

  // 4) Composite current + warped feedback
  compositeMat.uniforms.tWarped.value = rtTemp.texture;
  compositeMat.uniforms.tCurr.value   = rtScene.texture;
  compositeMat.uniforms.tMask.value   = rtMask.texture;
  compositeMat.uniforms.uDecay.value  = P.fbDecay;
  compositeMat.uniforms.uFbGain.value = P.fbGain;
  compositeMat.uniforms.uCurrGain.value = P.currGain;
  compositeMat.uniforms.uHard.value   = P.maskHardness;
  postQuad.material = compositeMat;
  renderer.setRenderTarget(rtB); renderer.render(postScene, postCam);

  // 5) Blur
  if(P.blurRadius>0.001){
    blurMat.uniforms.uRadius.value=P.blurRadius; blurMat.uniforms.tInput.value=rtB.texture; blurMat.uniforms.uDir.value.set(1,0);
    postQuad.material=blurMat; renderer.setRenderTarget(rtTemp); renderer.render(postScene, postCam);
    blurMat.uniforms.tInput.value=rtTemp.texture; blurMat.uniforms.uDir.value.set(0,1);
    postQuad.material=blurMat; renderer.setRenderTarget(rtB); renderer.render(postScene, postCam);
  }

  // 6) Boot: prefill history so rgbaDelay has data (prevents startup black)
  if(!bootFilled){
    copyMat.uniforms.tInput.value = rtB.texture; postQuad.material = copyMat;
    for(let k=0;k<HISTORY;k++){ renderer.setRenderTarget(history[k]); renderer.render(postScene, postCam); }
    renderer.setRenderTarget(null); bootFilled = true;
  }

  // 7) Push to history ring
  copyMat.uniforms.tInput.value = rtB.texture; postQuad.material = copyMat;
  renderer.setRenderTarget(history[histIndex]); renderer.render(postScene, postCam);
  renderer.setRenderTarget(null); histIndex = (histIndex+1)%HISTORY;

  for(let k=0;k<HISTORY;k++){ const slot=(histIndex-1-k+HISTORY)%HISTORY; finalMat.uniforms['t'+k].value = history[slot].texture; }
  finalMat.uniforms.uDispersion.value = P.dispersion;

  // 8) Final to screen
  postQuad.material = finalMat;
  renderer.setRenderTarget(null); renderer.render(postScene, postCam);

  // swap feedback buffers
  const tmp=rtA; rtA=rtB; rtB=tmp;

  controls.update();
}
function animate(){ const dt=clock.getDelta(); step(dt); requestAnimationFrame(animate); }
requestAnimationFrame(animate);

/* ───────────────────────── Contact sheet (C) ───────────────────────── */
window.addEventListener('keydown', async (e)=>{
  if(e.key.toLowerCase()!=='c') return;
  const rows=4, cols=4, pad=8, cw=P.sheetSize, ch=P.sheetSize;
  const W2=cols*cw+(cols+1)*pad, H2=rows*ch+(rows+1)*pad;
  const off=document.createElement('canvas'); off.width=W2; off.height=H2;
  const ctx=off.getContext('2d'); ctx.imageSmoothingEnabled=true; ctx.fillStyle='#000'; ctx.fillRect(0,0,W2,H2);

  const secondsPerSample=1.0, substeps=60;
  const shots=[];
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

/* ───────────────────────── Initial layout ───────────────────────── */
function panelW(){ const el=document.querySelector('.lil-gui.root'); return el?Math.ceil(el.getBoundingClientRect().width):300; }
function size(){ const w=Math.max(1, window.innerWidth-panelW()); const h=Math.max(1, window.innerHeight);
  renderer.setSize(w,h,false); viewCam.aspect=w/h; viewCam.updateProjectionMatrix();
  W=w; H=h; [rtScene, rtA, rtB, rtTemp, rtMask, ...history].forEach(rt=>rt.setSize(W,H));
  blurMat.uniforms.uTexel.value.set(1/W,1/H); finalMat.uniforms.uTexel.value.set(1/W,1/H);
  sdfMat.uniforms.uResolution.value.set(W,H);
  updateFrustumCorners();
}
window.addEventListener('resize', size); size();
