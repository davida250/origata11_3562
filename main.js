// Origata — Folding iridescent cube, 16 poses over 16 seconds.
// - Continuous plane-fold + mild twist in the vertex shader (no cracks)
// - Facet-aligned thin-film bands (emissive) + PBR glass (MeshPhysicalMaterial)
// - Feedback + blur + RGBA delay (TouchDesigner recipe) confined to the silhouette
// - Press 'C' to export a 4×4 contact sheet (1 s per tile, with substeps)

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import GUI from 'https://cdn.jsdelivr.net/npm/lil-gui@0.19/+esm';

/* ─────────── Renderer / Scene / Camera ─────────── */
const canvas = /** @type {HTMLCanvasElement} */ (document.getElementById('scene'));
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
renderer.setClearColor(0x000000, 1);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.physicallyCorrectLights = true;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

// Clean PMREM for transmission/clearcoat highlights
const pmrem = new THREE.PMREMGenerator(renderer);
const envRT = pmrem.fromScene(new RoomEnvironment(renderer), 0.04);
scene.environment = envRT.texture;

const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
camera.position.set(0, 0, 4.9);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.08; controls.rotateSpeed = 0.6;

/* ─────────── Parameters ─────────── */
const P = {
  cycleSeconds: 16,     // 16 frames / 16 seconds
  speed: 1.0,
  autoRotate: true,
  spin: 0.18,           // subtle; silhouettes change mostly due to folds

  // Optics (keep under control to avoid white blowout)
  roughness: 0.08, transmission: 0.72, thickness: 0.85, ior: 1.36,
  clearcoat: 1.0, clearcoatRoughness: 0.14, iridescence: 1.0, envIntensity: 1.25,

  // Feedback chain (TouchDesigner-style)
  fbDecay: 0.90, fbGain: 0.68, currGain: 0.90, // tuned to avoid “white” start
  warpRotate: 0.010, warpZoom: 0.0020, warpJitter: 0.0012,
  blurRadius: 1.6, maskHardness: 1.05,
  dispersion: 0.0018,             // slight chromatic UV shift in final

  sheetSize: 256
};

/* ─────────── Timeline poses (hand-tuned ranges to emulate your grid) ───────────
   Each pose defines:
   - twist: radians per unit height (small; most shape change is from folds)
   - two plane folds (period, intensity, axis)
   - thin-film bands (stripe freq/warp + film thickness base/amp)
   - RGBA channel delays (like TouchDesigner Palette: rgbaDelay)
   These 16 poses loop 1/second, producing your 4×4 progression.             */

const V = (x,y,z) => new THREE.Vector3(x,y,z).normalize();
const POSES = [
  {twist:2.2,  f1:{p:1.30,i:1.20,n:V(1,0,0)},   f2:{p:0.95,i:0.85,n:V(0,1,0)},  sf:12.5, sw:0.54, base:425, amp:260, d:[1,3,5]},
  {twist:2.6,  f1:{p:1.15,i:1.35,n:V(1,1,0)},   f2:{p:1.05,i:0.70,n:V(0,1,1)},  sf:14.0, sw:0.50, base:410, amp:300, d:[0,2,4]},
  {twist:2.9,  f1:{p:1.10,i:1.45,n:V(1,0,1)},   f2:{p:0.90,i:0.90,n:V(1,-1,0)}, sf:15.5, sw:0.46, base:435, amp:250, d:[0,3,6]},
  {twist:2.3,  f1:{p:1.40,i:0.95,n:V(0,1,0)},   f2:{p:1.10,i:1.10,n:V(1,0,-1)}, sf:11.8, sw:0.58, base:445, amp:280, d:[1,2,5]},
  {twist:2.0,  f1:{p:1.25,i:1.25,n:V(1,1,0)},   f2:{p:0.85,i:0.80,n:V(0,1,-1)}, sf:12.0, sw:0.60, base:420, amp:320, d:[0,1,4]},
  {twist:2.7,  f1:{p:1.05,i:1.35,n:V(1,0,0)},   f2:{p:1.20,i:0.70,n:V(0,0,1)},  sf:13.8, sw:0.50, base:430, amp:260, d:[2,4,6]},
  {twist:2.4,  f1:{p:0.95,i:1.40,n:V(0,1,1)},   f2:{p:1.30,i:0.80,n:V(1,-1,0)}, sf:10.8, sw:0.62, base:440, amp:280, d:[1,4,7]},
  {twist:2.1,  f1:{p:1.35,i:1.05,n:V(0,1,0)},   f2:{p:1.10,i:1.10,n:V(1,0,1)},  sf:12.6, sw:0.56, base:415, amp:300, d:[0,5,7]},
  {twist:2.2,  f1:{p:1.25,i:1.15,n:V(1,0,-1)},  f2:{p:0.95,i:0.85,n:V(0,1,0)},  sf:13.2, sw:0.52, base:425, amp:260, d:[1,3,5]},
  {twist:2.5,  f1:{p:1.10,i:1.40,n:V(1,1,0)},   f2:{p:1.05,i:0.75,n:V(0,1,1)},  sf:14.4, sw:0.48, base:410, amp:300, d:[0,2,4]},
  {twist:2.8,  f1:{p:1.05,i:1.45,n:V(1,0,1)},   f2:{p:0.90,i:0.95,n:V(1,-1,0)}, sf:15.2, sw:0.44, base:435, amp:250, d:[0,3,6]},
  {twist:2.3,  f1:{p:1.35,i:1.00,n:V(0,1,0)},   f2:{p:1.10,i:1.10,n:V(1,0,-1)}, sf:11.5, sw:0.58, base:445, amp:280, d:[1,2,5]},
  {twist:2.0,  f1:{p:1.20,i:1.25,n:V(1,1,0)},   f2:{p:0.85,i:0.80,n:V(0,1,-1)}, sf:11.8, sw:0.60, base:420, amp:320, d:[0,1,4]},
  {twist:2.6,  f1:{p:1.05,i:1.35,n:V(1,0,0)},   f2:{p:1.20,i:0.70,n:V(0,0,1)},  sf:13.6, sw:0.50, base:430, amp:260, d:[2,4,6]},
  {twist:2.3,  f1:{p:0.95,i:1.40,n:V(0,1,1)},   f2:{p:1.25,i:0.80,n:V(1,-1,0)}, sf:10.6, sw:0.62, base:440, amp:280, d:[1,4,7]},
  {twist:2.1,  f1:{p:1.35,i:1.05,n:V(0,1,0)},   f2:{p:1.10,i:1.10,n:V(1,0,1)},  sf:12.4, sw:0.56, base:415, amp:300, d:[0,5,7]}
];

/* ─────────── Mesh + material ─────────── */
const geo = new THREE.BoxGeometry(2.2, 2.2, 2.2, 96, 144, 96);
const material = new THREE.MeshPhysicalMaterial({
  color: 0xffffff,
  metalness: 0.0,
  roughness: P.roughness,
  transmission: P.transmission,
  thickness: P.thickness,
  ior: P.ior,
  clearcoat: P.clearcoat,
  clearcoatRoughness: P.clearcoatRoughness,
  iridescence: P.iridescence,
  envMapIntensity: P.envIntensity,
  side: THREE.DoubleSide,
  flatShading: true
});
const cube = new THREE.Mesh(geo, material);
scene.add(cube);

// Mask mesh (same deformation) to confine feedback to silhouette
const maskMat = new THREE.ShaderMaterial({
  uniforms: {
    uTwist:{value:2.2},
    uF1P:{value:1.2}, uF1I:{value:1.0}, uF1N:{value:new THREE.Vector3(1,0,0)},
    uF2P:{value:1.0}, uF2I:{value:0.8}, uF2N:{value:new THREE.Vector3(0,1,0)}
  },
  vertexShader: /* glsl */`
    uniform float uTwist;
    uniform float uF1P, uF1I; uniform vec3 uF1N;
    uniform float uF2P, uF2I; uniform vec3 uF2N;

    // Triangular-wave plane fold: keeps surface continuous (no cracks)
    vec3 foldSpace(vec3 p, vec3 n, float period, float intensity){
      float d = dot(p, n);
      float tri = abs(mod(d + period, 2.0*period) - period) - 0.5*period;
      float delta = (tri - d) * intensity;
      return p + n * delta;
    }

    void main(){
      vec3 p = position;
      // Mild twist about Y
      float a = uTwist * p.y;
      float s = sin(a), c = cos(a);
      p = vec3(c*p.x - s*p.z, p.y, s*p.x + c*p.z);
      // Two folding planes
      p = foldSpace(p, normalize(uF1N), uF1P, uF1I);
      p = foldSpace(p, normalize(uF2N), uF2P, uF2I);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
    }
  `,
  fragmentShader: /* glsl */` void main(){ gl_FragColor = vec4(1.0); } `,
  depthTest: true, depthWrite: true
});
const cubeMask = new THREE.Mesh(geo, maskMat);
const maskScene = new THREE.Scene(); maskScene.add(cubeMask);

// Inject folds + thin-film into Physical material (safe patches)
material.onBeforeCompile = (sh) => {
  Object.assign(sh.uniforms, {
    uTwist:{value:2.2},
    uF1P:{value:1.2}, uF1I:{value:1.0}, uF1N:{value:new THREE.Vector3(1,0,0)},
    uF2P:{value:1.0}, uF2I:{value:0.8}, uF2N:{value:new THREE.Vector3(0,1,0)},
    uStripeFreq:{value:12.5}, uStripeWarp:{value:0.5},
    uFilmBase:{value:430.0}, uFilmAmp:{value:280.0},
    uVibrance:{value:1.30}, uPastel:{value:0.22}, uEdgeWarm:{value:0.25}
  });

  // Deform (twist + folds)
  sh.vertexShader = `
    uniform float uTwist;
    uniform float uF1P, uF1I; uniform vec3 uF1N;
    uniform float uF2P, uF2I; uniform vec3 uF2N;
    varying vec3 vWorldPos; varying vec3 vNormalW;

    vec3 foldSpace(vec3 p, vec3 n, float period, float intensity){
      float d = dot(p, n);
      float tri = abs(mod(d + period, 2.0*period) - period) - 0.5*period;
      float delta = (tri - d) * intensity;
      return p + n * delta;
    }
  ` + sh.vertexShader;

  sh.vertexShader = sh.vertexShader.replace(
    '#include <begin_vertex>',
    `
      #include <begin_vertex>
      {
        vec3 p = transformed;

        // Mild twist
        float a = uTwist * p.y;
        float s = sin(a), c = cos(a);
        p = vec3(c*p.x - s*p.z, p.y, s*p.x + c*p.z);

        // Two plane folds
        p = foldSpace(p, normalize(uF1N), uF1P, uF1I);
        p = foldSpace(p, normalize(uF2N), uF2P, uF2I);

        transformed = p;
      }
    `
  );

  sh.vertexShader = sh.vertexShader.replace(
    '#include <project_vertex>',
    `
      #include <project_vertex>
      vWorldPos = (modelMatrix * vec4(transformed,1.0)).xyz;
      vNormalW = normalize(mat3(modelMatrix) * normal);
    `
  );

  // Facet-aligned thin-film + a slight warm rim
  sh.fragmentShader = `
    uniform float uStripeFreq, uStripeWarp, uFilmBase, uFilmAmp, uVibrance, uPastel, uEdgeWarm;
    varying vec3 vWorldPos; varying vec3 vNormalW;

    vec2 rot2(vec2 p, float a){ float c=cos(a), s=sin(a); return mat2(c,-s,s,c)*p; }
    float stripeField(vec3 p, float freq, float warp){
      vec2 q = p.xy + vec2(0.6*p.z, 0.35*p.x);
      q = rot2(q, 0.785398); // 45°
      float s1 = sin(q.x*freq + 0.7*q.y);
      float s2 = sin((q.x+q.y)*freq*0.35 + 1.7*sin(q.x*0.6));
      return 0.5 + 0.5 * mix(s1, s2, clamp(warp,0.0,1.0));
    }
    vec3 thinFilm(float ndv, float d_nm, float n2){
      float n1=1.0, n3=1.50;
      float sin2=max(0.0, 1.0-ndv*ndv);
      float cos2=sqrt(max(0.0, 1.0 - sin2/(n2*n2)));
      float R12=pow((n1-n2)/(n1+n2),2.0), R23=pow((n2-n3)/(n2+n3),2.0);
      float A=2.0*sqrt(R12*R23);
      float lR=650.0, lG=510.0, lB=440.0;
      float phiR=12.5663706*n2*d_nm*cos2/lR;
      float phiG=12.5663706*n2*d_nm*cos2/lG;
      float phiB=12.5663706*n2*d_nm*cos2/lB;
      return clamp(vec3(R12+R23+A*cos(phiR),
                        R12+R23+A*cos(phiG),
                        R12+R23+A*cos(phiB)),0.0,1.0);
    }
  ` + sh.fragmentShader;

  sh.fragmentShader = sh.fragmentShader.replace(
    '#include <emissivemap_fragment>',
    `
      #include <emissivemap_fragment>
      {
        vec3 N = normalize(vNormalW);
        vec3 V = normalize(cameraPosition - vWorldPos);
        float ndv = clamp(dot(N,V), 0.0, 1.0);

        float f = stripeField(vWorldPos*0.7 + N*0.25, uStripeFreq, uStripeWarp);
        float d_nm = uFilmBase + uFilmAmp * (f - 0.5) * 2.0;
        vec3 filmRGB = thinFilm(ndv, d_nm, 1.35);
        filmRGB = mix(filmRGB, vec3(1.0), clamp(uPastel, 0.0, 1.0));

        float rim = pow(1.0 - ndv, 3.0);
        vec3 warm = vec3(1.0, 0.56, 0.35) * uEdgeWarm * rim;

        totalEmissiveRadiance += (filmRGB * uVibrance + warm);
      }
    `
  );

  material.userData.shader = sh;
};
material.needsUpdate = true;

/* ─────────── Post chain (Feedback + Blur + RGBA Delay) ───────────
   Mirrors TouchDesigner’s Feedback TOP + rgbaDelay; confined to the shape.  */
const postScene = new THREE.Scene();
const postCam   = new THREE.OrthographicCamera(-1,1,1,-1,0,1);
const quad      = new THREE.Mesh(new THREE.PlaneGeometry(2,2), new THREE.MeshBasicMaterial({color:0xffffff}));
postScene.add(quad);

let W=2, H=2;
const makeRT = () => new THREE.WebGLRenderTarget(W,H,{minFilter:THREE.LinearFilter,magFilter:THREE.LinearFilter,depthBuffer:false,stencilBuffer:false});
let rtScene = makeRT(), rtMask = makeRT(), rtA = makeRT(), rtB = makeRT(), rtTemp = makeRT();
const HISTORY = 8, history = new Array(HISTORY).fill(0).map(()=>makeRT()); let histIndex = 0;

const fsVS = /* glsl */`varying vec2 vUv; void main(){ vUv=uv; gl_Position=vec4(position.xy,0.,1.);} `;
const noiseGLSL = /* glsl */`
  float hash(vec2 p){ p=fract(p*vec2(123.34,345.45)); p+=dot(p,p+34.345); return fract(p.x*p.y); }
  float noise(vec2 p){ vec2 i=floor(p), f=fract(p);
    float a=hash(i), b=hash(i+vec2(1,0)), c=hash(i+vec2(0,1)), d=hash(i+vec2(1,1));
    vec2 u=f*f*(3.0-2.0*f); return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
  }
`;

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

// Composite with current frame (masked), gently (avoid white blowout)
const compositeMat = new THREE.ShaderMaterial({
  vertexShader: fsVS,
  fragmentShader: /* glsl */`
    varying vec2 vUv; uniform sampler2D tWarped,tCurr,tMask;
    uniform float uDecay,uFbGain,uCurrGain,uHard;
    void main(){
      vec4 fb = texture2D(tWarped, vUv) * uDecay * uFbGain;
      vec4 cur= texture2D(tCurr,   vUv) * uCurrGain;
      float m = pow(texture2D(tMask, vUv).r, uHard);
      // Blend like TD feedback: add but damped; clamp to prevent blowout
      gl_FragColor = clamp(m * fb + cur, 0.0, 1.0);
    }
  `,
  uniforms: {
    tWarped:{value:null}, tCurr:{value:null}, tMask:{value:null},
    uDecay:{value:P.fbDecay}, uFbGain:{value:P.fbGain}, uCurrGain:{value:P.currGain}, uHard:{value:P.maskHardness}
  }
});

// Separable blur (H/V) — WebGL1-safe (texel size provided)
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
  uniforms: { tInput:{value:null}, uTexel:{value:new THREE.Vector2(1/Math.max(2,W),1/Math.max(2,H))}, uDir:{value:new THREE.Vector2(1,0)}, uRadius:{value:P.blurRadius} }
});

// Copy to history slot
const copyMat = new THREE.ShaderMaterial({
  vertexShader: fsVS, fragmentShader: /* glsl */`varying vec2 vUv; uniform sampler2D tInput; void main(){ gl_FragColor=texture2D(tInput,vUv);} `,
  uniforms:{ tInput:{value:null} }
});

// Final RGBA delay (per channel) + slight dispersion
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

      float b=texture2D(t0,ub).b*wB[0]+texture2D(t1,ub).b*wB[1]+texture2D(t2,ub).b+wB[2]+texture2D(t3,ub).b*wB[3]+
              texture2D(t4,ub).b*wB[4]+texture2D(t5,ub).b*wB[5]+texture2D(t6,ub).b*wB[6]+texture2D(t7,ub).b*wB[7];

      gl_FragColor=vec4(r,g,b,1.0);
    }
  `,
  uniforms: {
    t0:{value:null}, t1:{value:null}, t2:{value:null}, t3:{value:null},
    t4:{value:null}, t5:{value:null}, t6:{value:null}, t7:{value:null},
    wR:{value:new Array(8).fill(0)}, wG:{value:new Array(8).fill(0)}, wB:{value:new Array(8).fill(0)},
    uTexel:{value:new THREE.Vector2(1/Math.max(2,W),1/Math.max(2,H))},
    uDispersion:{value:P.dispersion}
  }
});

/* ─────────── Resize ─────────── */
function panelWidth(){ const el=document.querySelector('.lil-gui.root'); return el?Math.ceil(el.getBoundingClientRect().width):300; }
function resize(){
  const w=Math.max(1, window.innerWidth-panelWidth());
  const h=Math.max(1, window.innerHeight);
  renderer.setSize(w,h,false);
  renderer.domElement.style.width=`${w}px`;
  renderer.domElement.style.height=`${h}px`;
  camera.aspect=w/h; camera.updateProjectionMatrix();
  W=w; H=h;
  [rtScene,rtMask,rtA,rtB,rtTemp, ...history].forEach(rt=>rt.setSize(W,H));
  blurMat.uniforms.uTexel.value.set(1/W,1/H);
  finalMat.uniforms.uTexel.value.set(1/W,1/H);
}
window.addEventListener('resize', resize); resize();

/* ─────────── Minimal Controls ─────────── */
const gui = new GUI({ title: 'Controls', width: 300 });
const fTime = gui.addFolder('Timeline');
fTime.add(P, 'cycleSeconds', 4, 32, 1).name('Cycle (s)');
fTime.add(P, 'speed', 0.1, 3.0, 0.01).name('Speed');
fTime.add(P, 'autoRotate').name('Auto Rotate');
fTime.add(P, 'spin', 0.0, 1.0, 0.01).name('Spin');

const fOpt = gui.addFolder('Optics');
fOpt.add(P, 'roughness', 0.0, 1.0, 0.001).name('Roughness').onChange(()=>material.roughness=P.roughness);
fOpt.add(P, 'transmission', 0.0, 1.0, 0.001).name('Transmission').onChange(()=>material.transmission=P.transmission);
fOpt.add(P, 'thickness', 0.0, 2.0, 0.01).name('Thickness').onChange(()=>material.thickness=P.thickness);
fOpt.add(P, 'ior', 1.0, 2.0, 0.001).name('IOR').onChange(()=>material.ior=P.ior);
fOpt.add(P, 'clearcoat', 0.0, 1.0, 0.001).name('Clearcoat').onChange(()=>material.clearcoat=P.clearcoat);
fOpt.add(P, 'clearcoatRoughness', 0.0, 1.0, 0.001).name('Clearcoat Rough.').onChange(()=>material.clearcoatRoughness=P.clearcoatRoughness);
fOpt.add(P, 'iridescence', 0.0, 1.0, 0.001).name('PBR Iridescence').onChange(()=>material.iridescence=P.iridescence);
fOpt.add(P, 'envIntensity', 0.0, 3.0, 0.01).name('Env Intensity').onChange(()=>material.envMapIntensity=P.envIntensity);

const fFb = gui.addFolder('Feedback');
fFb.add(P, 'fbDecay', 0.80, 0.995, 0.001).name('Decay');
fFb.add(P, 'fbGain', 0.0, 1.0, 0.01).name('FB Gain');
fFb.add(P, 'currGain', 0.0, 1.5, 0.01).name('Current Gain');
fFb.add(P, 'warpRotate', -0.05, 0.05, 0.0005).name('Warp Rot');
fFb.add(P, 'warpZoom', -0.02, 0.02, 0.0001).name('Warp Zoom');
fFb.add(P, 'warpJitter', 0.0, 0.01, 0.0001).name('Warp Jitter');
fFb.add(P, 'blurRadius', 0.0, 4.0, 0.01).name('Blur');
fFb.add(P, 'maskHardness', 0.5, 1.5, 0.01).name('Mask Hardness');
fFb.add(P, 'dispersion', 0.0, 0.01, 0.0001).name('Dispersion');

/* ─────────── Timeline & updates ─────────── */
function setDelays(r,g,b){
  const R=new Array(8).fill(0), G=new Array(8).fill(0), B=new Array(8).fill(0);
  R[r]=1; G[g]=1; B[b]=1;
  finalMat.uniforms.wR.value=R; finalMat.uniforms.wG.value=G; finalMat.uniforms.wB.value=B;
}
function applyPose(p){
  const sh=material.userData.shader; if(!sh) return;
  sh.uniforms.uTwist.value=p.twist;
  sh.uniforms.uF1P.value=p.f1.p; sh.uniforms.uF1I.value=p.f1.i; sh.uniforms.uF1N.value.copy(p.f1.n);
  sh.uniforms.uF2P.value=p.f2.p; sh.uniforms.uF2I.value=p.f2.i; sh.uniforms.uF2N.value.copy(p.f2.n);
  sh.uniforms.uStripeFreq.value=p.sf; sh.uniforms.uStripeWarp.value=p.sw; sh.uniforms.uFilmBase.value=p.base; sh.uniforms.uFilmAmp.value=p.amp;
  maskMat.uniforms.uTwist.value=p.twist;
  maskMat.uniforms.uF1P.value=p.f1.p; maskMat.uniforms.uF1I.value=p.f1.i; maskMat.uniforms.uF1N.value.copy(p.f1.n);
  maskMat.uniforms.uF2P.value=p.f2.p; maskMat.uniforms.uF2I.value=p.f2.i; maskMat.uniforms.uF2N.value.copy(p.f2.n);
  setDelays(p.d[0], p.d[1], p.d[2]);
  // Gentle auto-rotate; silhouettes mainly from folds
  cube.rotation.y += P.autoRotate ? P.spin * 0.016 : 0.0;
}

/* ─────────── Simulation / Feedback ─────────── */
let simTime=0;
function step(dt){
  simTime += dt * P.speed;

  const T=P.cycleSeconds, t=simTime%T, i=Math.floor(t), u=t-i;
  // Smoothstep for smooth morph between consecutive poses
  const s=u*u*(3.0-2.0*u);
  const A=POSES[i], B=POSES[(i+1)%POSES.length];
  const lerp=(a,b)=>a+(b-a)*s;
  const axisLerp=(a,b)=>new THREE.Vector3().copy(a).normalize().lerp(b.clone().normalize(), s).normalize();
  const poseNow={
    twist:lerp(A.twist,B.twist),
    f1:{p:lerp(A.f1.p,B.f1.p), i:lerp(A.f1.i,B.f1.i), n:axisLerp(A.f1.n,B.f1.n)},
    f2:{p:lerp(A.f2.p,B.f2.p), i:lerp(A.f2.i,B.f2.i), n:axisLerp(A.f2.n,B.f2.n)},
    sf:lerp(A.sf,B.sf), sw:lerp(A.sw,B.sw), base:lerp(A.base,B.base), amp:lerp(A.amp,B.amp),
    d:(u<0.5?A.d:B.d)
  };
  applyPose(poseNow);

  // Render 3D scene
  renderer.setRenderTarget(rtScene);
  renderer.render(scene, camera);
  renderer.setRenderTarget(null);

  // Render mask (same deformations)
  cubeMask.rotation.copy(cube.rotation);
  renderer.setRenderTarget(rtMask);
  renderer.clear();
  renderer.render(maskScene, camera);
  renderer.setRenderTarget(null);

  // Feedback warp → composite → blur → history → RGBA delay compose
  warpMat.uniforms.tPrev.value=rtA.texture; warpMat.uniforms.uRot.value=P.warpRotate; warpMat.uniforms.uZoom.value=P.warpZoom;
  warpMat.uniforms.uJitter.value=P.warpJitter; warpMat.uniforms.uTime.value=simTime;
  quad.material=warpMat; renderer.setRenderTarget(rtTemp); renderer.render(postScene, postCam);

  compositeMat.uniforms.tWarped.value=rtTemp.texture; compositeMat.uniforms.tCurr.value=rtScene.texture;
  compositeMat.uniforms.tMask.value=rtMask.texture; compositeMat.uniforms.uDecay.value=P.fbDecay;
  compositeMat.uniforms.uFbGain.value=P.fbGain; compositeMat.uniforms.uCurrGain.value=P.currGain;
  quad.material=compositeMat; renderer.setRenderTarget(rtB); renderer.render(postScene, postCam);

  if(P.blurRadius>0.001){
    blurMat.uniforms.uRadius.value=P.blurRadius; blurMat.uniforms.tInput.value=rtB.texture; blurMat.uniforms.uDir.value.set(1,0);
    quad.material=blurMat; renderer.setRenderTarget(rtTemp); renderer.render(postScene, postCam);
    blurMat.uniforms.tInput.value=rtTemp.texture; blurMat.uniforms.uDir.value.set(0,1);
    quad.material=blurMat; renderer.setRenderTarget(rtB); renderer.render(postScene, postCam);
  }

  // Update history ring
  copyMat.uniforms.tInput.value=rtB.texture;
  quad.material=copyMat; renderer.setRenderTarget(history[histIndex]); renderer.render(postScene, postCam);
  renderer.setRenderTarget(null); histIndex=(histIndex+1)%HISTORY;

  // Bind newest→oldest to t0..t7
  for(let k=0;k<HISTORY;k++){ const slot=(histIndex-1-k+HISTORY)%HISTORY; finalMat.uniforms['t'+k].value=history[slot].texture; }
  finalMat.uniforms.uDispersion.value=P.dispersion;

  // Final composite to screen
  quad.material=finalMat;
  renderer.setRenderTarget(null); renderer.render(postScene, postCam);

  // Ping-pong
  const tmp=rtA; rtA=rtB; rtB=tmp;

  controls.update();
}
const clock=new THREE.Clock();
function animate(){ const dt=clock.getDelta(); step(dt); requestAnimationFrame(animate); }
requestAnimationFrame(animate);

/* ─────────── Contact sheet (press 'C') ───────────
   Captures exactly 16 frames, stepping the simulation by 1 s each (with
   substeps for feedback). Produces a 4×4 PNG similar to your reference.     */
window.addEventListener('keydown', async (e)=>{
  if(e.key.toLowerCase()!=='c') return;
  const rows=4, cols=4, pad=8, cw=P.sheetSize, ch=P.sheetSize;
  const W2=cols*cw+(cols+1)*pad, H2=rows*ch+(rows+1)*pad;
  const off=document.createElement('canvas'); off.width=W2; off.height=H2;
  const ctx=off.getContext('2d'); ctx.imageSmoothingEnabled=true; ctx.fillStyle='#000'; ctx.fillRect(0,0,W2,H2);

  const secondsPerSample=1.0, substeps=60; // ~60 fps worth of evolution per second
  const bitmaps=[];
  for(let n=0;n<rows*cols;n++){
    for(let s=0;s<substeps;s++) step(secondsPerSample/substeps);
    const bmp=await createImageBitmap(renderer.domElement);
    bitmaps.push(bmp);
  }
  let idx=0;
  for(let r=0;r<rows;r++){
    for(let c=0;c<cols;c++){
      const x=pad+c*(cw+pad), y=pad+r*(ch+pad);
      ctx.drawImage(bitmaps[idx++], 0,0,renderer.domElement.width,renderer.domElement.height, x,y,cw,ch);
    }
  }
  const link=document.createElement('a');
  link.download='contact_sheet.png'; link.href=off.toDataURL('image/png'); link.click();
});

/* ─────────── GUI theme polish ─────────── */
document.querySelector('.lil-gui.root')?.style.setProperty('backdrop-filter','blur(6px)');
