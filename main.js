// ----------------------------------------------
// Folding Iridescent Shape (SDF ray-march) in Three.js
// ----------------------------------------------
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import GUI from 'https://unpkg.com/lil-gui@0.19/dist/lil-gui.esm.min.js';

// Scene setup
const app = document.getElementById("app");
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
app.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.01, 100);
camera.position.set(0.0, 0.0, 3.5);
camera.lookAt(0, 0, 0);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.enablePan = false;

// Fullscreen plane to run the shader on
const quadGeo = new THREE.PlaneGeometry(2, 2);
const uniforms = {
  uTime: { value: 0 },
  uSpeed: { value: 0.9 },
  uRotSpeed: { value: 0.35 },
  uFoldAmp: { value: 0.85 },
  uEdgeSoften: { value: 0.0025 },
  uStripeFreq: { value: 8.0 },
  uStripeWarp: { value: 0.7 },
  uIriStrength: { value: 1.0 },
  uFilmThickness: { value: 420.0 }, // in nm
  uColorDelay: { value: 0.35 },
  uExposure: { value: 1.15 },

  uResolution: { value: new THREE.Vector2(renderer.domElement.width, renderer.domElement.height) },
  uCamPos: { value: new THREE.Vector3() },
  uCamRight: { value: new THREE.Vector3() },
  uCamUp: { value: new THREE.Vector3() },
  uCamForward: { value: new THREE.Vector3() },
  uFovY: { value: THREE.MathUtils.degToRad(camera.fov) },

  uTintA: { value: new THREE.Color(0.85, 1.05, 1.20) },
  uTintB: { value: new THREE.Color(1.12, 0.9, 1.2) },
};

const material = new THREE.ShaderMaterial({
  uniforms,
  vertexShader: /* glsl */`
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = vec4(position.xy, 0.0, 1.0);
    }
  `,
  fragmentShader: /* glsl */`#ifdef GL_ES
precision highp float;
#endif

uniform float uTime;
uniform float uSpeed;
uniform float uRotSpeed;
uniform float uFoldAmp;
uniform float uEdgeSoften;
uniform float uStripeFreq;
uniform float uStripeWarp;
uniform float uIriStrength;
uniform float uFilmThickness; // nm
uniform float uColorDelay;
uniform float uExposure;

uniform vec2  uResolution;
uniform vec3  uCamPos;
uniform vec3  uCamRight;
uniform vec3  uCamUp;
uniform vec3  uCamForward;
uniform float uFovY;

uniform vec3 uTintA;
uniform vec3 uTintB;

varying vec2 vUv;

const float PI = 3.141592653589793;
const int   MAX_STEPS = 160;
const float MAX_DIST = 20.0;
const float SURF_EPS = 0.0007;

float hash1(float n){ return fract(sin(n)*43758.5453123); }
vec3  hash3(float n){ return fract(sin(vec3(n, n+1.0, n+2.0))*vec3(43758.5453, 22578.1459, 19642.3490)); }

vec3 rotateAxisAngle(vec3 v, vec3 axis, float angle){
  float c = cos(angle), s = sin(angle);
  return v*c + cross(axis, v)*s + axis*dot(axis, v)*(1.0-c);
}

float sdPlane(vec3 p, vec4 pl){ return dot(p, pl.xyz) - pl.w; }

const int NUM_PLANES = 12;

vec3 baseN(int i){
  if(i==0) return normalize(vec3( 1.0,  0.25,  0.10));
  if(i==1) return normalize(vec3(-1.0,  0.30,  0.15));
  if(i==2) return normalize(vec3( 0.20,  1.0,   0.35));
  if(i==3) return normalize(vec3( 0.20, -1.0,   0.05));
  if(i==4) return normalize(vec3( 0.05,  0.25,  1.0));
  if(i==5) return normalize(vec3( 0.25,  0.10, -1.0));
  if(i==6) return normalize(vec3(-0.6,  -0.1,   0.8));
  if(i==7) return normalize(vec3( 0.7,  -0.2,  -0.6));
  if(i==8) return normalize(vec3( 0.5,   0.7,  -0.15));
  if(i==9) return normalize(vec3(-0.25,  0.75,  0.35));
  if(i==10) return normalize(vec3( 0.15, -0.65,  0.7));
  return               normalize(vec3(-0.7, -0.6, -0.2));
}

vec3 planeAxis(int i){
  vec3 h = hash3(float(i)*17.123+9.73)*2.0 - 1.0;
  return normalize(mix(h, vec3(0.0, 1.0, 0.0), 0.25));
}

float baseH(int i){
  float b = 0.85 + 0.25 * float(i%3==0 ? 1 : 0);
  if(i==0 || i==2) b += 0.10;
  return b;
}

vec4 getPlane(int i, float t){
  vec3 n  = baseN(i);
  vec3 ax = planeAxis(i);
  float phase = float(i)*1.618;
  float ang   = uFoldAmp * 0.9 * sin(t*uRotSpeed*(0.7+0.1*float(i%5)) + phase);
  n = rotateAxisAngle(n, ax, ang);

  float h = baseH(i) + 0.12 * sin(t * (0.5 + 0.17*float(i%4)) + phase*1.37);
  return vec4(n, h);
}

float sdConvexPoly(vec3 p, float t){
  float d = -1e6;
  float k = uEdgeSoften;
  for(int i=0;i<NUM_PLANES;i++){
    vec4 pl = getPlane(i, t);
    float di = sdPlane(p, pl);
    if(k>0.0){
      float h = clamp(0.5 + 0.5*(di - d)/k, 0.0, 1.0);
      d = mix(di, d, h) + k*h*(1.0-h);
    } else {
      d = max(d, di);
    }
  }
  return d;
}

mat3 rotY(float a){
  float c=cos(a), s=sin(a);
  return mat3(c,0.,-s,  0.,1.,0.,  s,0.,c);
}
mat3 rotX(float a){
  float c=cos(a), s=sin(a);
  return mat3(1.,0.,0.,  0.,c,-s,  0.,s,c);
}

float map(vec3 p, float t){
  p *= rotY(t*0.1);
  p *= rotX(t*0.07);
  return sdConvexPoly(p, t);
}

vec3 calcNormal(vec3 p, float t){
  const vec2 e = vec2(1.0, -1.0)*0.0015;
  return normalize(e.xyy*map(p+e.xyy,t) + e.yyx*map(p+e.yyx,t) + e.yxy*map(p+e.yxy,t) + e.xxx*map(p+e.xxx,t));
}

float fresnelSchlick(float cosTheta, float F0){
  return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

vec3 thinFilmRGB(float cosThetaV, float thickness_nm, float n1, float n2, float n3){
  float sinThetaV = sqrt(max(0.0, 1.0 - cosThetaV*cosThetaV));
  float sinThetaT = clamp(n1/n2 * sinThetaV, 0.0, 0.9999);
  float cosThetaT = sqrt(max(0.0, 1.0 - sinThetaT*sinThetaT));

  float Rs12 = pow((n1*cosThetaV - n2*cosThetaT) / (n1*cosThetaV + n2*cosThetaT), 2.0);
  float Rp12 = pow((n1*cosThetaT - n2*cosThetaV) / (n1*cosThetaT + n2*cosThetaV), 2.0);
  float R12 = 0.5*(Rs12 + Rp12);

  float Rs23 = pow((n2*cosThetaT - n3*cosThetaT) / (n2*cosThetaT + n3*cosThetaT), 2.0);
  float Rp23 = pow((n2*cosThetaT - n3*cosThetaT) / (n2*cosThetaT + n3*cosThetaT), 2.0);
  float R23 = 0.5*(Rs23 + Rp23);

  float opd = 2.0 * n2 * (thickness_nm * 1e-9) * cosThetaT;

  float phaseShift = 0.0;
  if ((n1 < n2) != (n2 < n3)) phaseShift = 0.5;

  vec3 lambda = vec3(680.0e-9, 540.0e-9, 450.0e-9);
  vec3 phase = 2.0*3.141592653589793 * (opd / lambda + phaseShift);
  vec3 cosTerm = 0.5 + 0.5*cos(phase);
  vec3 airy = (R12 + R23 + 2.0*sqrt(R12*R23)*cos(phase));
  airy = clamp(airy, 0.0, 1.0);
  vec3 rgb = mix(cosTerm, airy, 0.65);
  rgb = pow(clamp(rgb, 0.0, 1.0), vec3(0.9));
  return rgb;
}

float filmThicknessField(vec3 wp, float tColor, float base_nm){
  vec3 axis = normalize(vec3(0.85, 0.05, 0.52));
  float s   = dot(wp, axis);
  float f   = uStripeFreq;
  float warp = uStripeWarp;

  float bands = sin(s*f + 0.7*sin(s*(f*0.23) + tColor*0.7) + warp*0.9*sin(s*1.7 - tColor*0.6));
  bands = 0.5 + 0.5*bands;
  bands = smoothstep(0.15, 0.85, bands);

  return base_nm * (0.75 + 0.5*bands);
}

vec3 shade(vec3 p, vec3 n, vec3 rd, float tGeo, float tCol){
  vec3 L1 = normalize(vec3( 0.7,  0.6,  0.4));
  vec3 L2 = normalize(vec3(-0.4, -0.2, -0.8));
  vec3 cL1 = vec3(0.85, 1.10, 1.00);
  vec3 cL2 = vec3(1.10, 0.86, 1.15);

  float ndl1 = max(dot(n, L1), 0.0);
  float ndl2 = max(dot(n, L2), 0.0);
  float ndv  = max(dot(n, -rd), 0.0);

  float thickness = filmThicknessField(p, tCol, uFilmThickness);
  vec3 film = thinFilmRGB(ndv, thickness, 1.0, 1.52, 1.0);

  float F = fresnelSchlick(1.0 - ndv, 0.04);
  vec3 spec = (cL1*ndl1 + cL2*ndl2) * (film * (0.35 + 0.65*F));

  vec3 base = mix(uTintA, uTintB, 0.5 + 0.5*sin(0.7*p.x + 1.3*p.z));
  base *= 0.18 + 0.82*pow(ndl1+ndl2, 0.8);

  vec3 col = base + spec * uIriStrength;

  vec3 ambTop = vec3(0.02, 0.02, 0.03);
  vec3 ambBot = vec3(0.00, 0.00, 0.00);
  float h = clamp(p.y*0.25 + 0.5, 0.0, 1.0);
  col += mix(ambBot, ambTop, h);

  return col;
}

vec3 rayDirection(vec2 uv){
  float vy = tan(0.5 * uFovY);
  float vx = vy * (uResolution.x / uResolution.y);
  vec3 dir = normalize(uv.x * vx * uCamRight + uv.y * vy * uCamUp + 1.0 * uCamForward);
  return dir;
}

void main(){
  vec2 p = (vUv*2.0 - 1.0);
  p.x *= uResolution.x / uResolution.y;

  float t = uTime * uSpeed;
  float tCol = t - uColorDelay;

  vec3 ro = uCamPos;
  vec3 rd = rayDirection(p);

  float tMarch = 0.0;
  bool hit = false;
  for(int i=0;i<MAX_STEPS;i++){
    vec3 pos = ro + rd*tMarch;
    float d = map(pos, t);
    if(d < SURF_EPS){ hit = true; break; }
    tMarch += d;
    if(tMarch > MAX_DIST) break;
  }

  vec3 color;
  if(hit){
    vec3 pos = ro + rd*tMarch;
    vec3 nor = calcNormal(pos, t);
    color = shade(pos, nor, rd, t, tCol);
    float rim = pow(1.0 - max(dot(nor, -rd), 0.0), 3.0);
    color += rim * 0.12 * vec3(1.2, 1.0, 1.35);
  } else {
    float v = pow(1.0 - clamp(length(p), 0.0, 1.0), 3.0);
    color = mix(vec3(0.0), vec3(0.01, 0.008, 0.012), v);
  }

  color = 1.0 - exp(-color * uExposure);
  color = pow(color, vec3(1.0/2.2));
  gl_FragColor = vec4(color, 1.0);
}
  `,
  depthWrite: false,
  depthTest: false,
  transparent: false,
});

const quad = new THREE.Mesh(quadGeo, material);
scene.add(quad);

// GUI
const gui = new GUI({ title: "Controls" });
gui.add(uniforms.uSpeed, "value", 0.0, 3.0, 0.01).name("Speed");
gui.add(uniforms.uRotSpeed, "value", 0.0, 2.0, 0.01).name("Rotation");
gui.add(uniforms.uFoldAmp, "value", 0.0, 1.5, 0.001).name("Fold Intensity");
gui.add(uniforms.uEdgeSoften, "value", 0.0, 0.01, 0.0001).name("Edge Soften");
gui.add(uniforms.uStripeFreq, "value", 1.0, 20.0, 0.1).name("Stripe Freq");
gui.add(uniforms.uStripeWarp, "value", 0.0, 2.5, 0.01).name("Stripe Warp");
gui.add(uniforms.uIriStrength, "value", 0.0, 2.0, 0.01).name("Iridescence");
gui.add(uniforms.uFilmThickness, "value", 200.0, 800.0, 1).name("Film Thickness (nm)");
gui.add(uniforms.uColorDelay, "value", 0.0, 1.2, 0.01).name("Color Delay");
gui.add(uniforms.uExposure, "value", 0.5, 2.0, 0.01).name("Exposure");

// Resize
function onResize(){
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  uniforms.uResolution.value.set(renderer.domElement.width, renderer.domElement.height);
  uniforms.uFovY.value = THREE.MathUtils.degToRad(camera.fov);
}
window.addEventListener("resize", onResize);

// Per-frame camera basis -> shader
function updateCameraBasis(){
  const m = camera.matrixWorld;
  const right = new THREE.Vector3();
  const up    = new THREE.Vector3();
  const fwd   = new THREE.Vector3();
  m.extractBasis(right, up, fwd);
  uniforms.uCamRight.value.copy(right);
  uniforms.uCamUp.value.copy(up);
  uniforms.uCamForward.value.copy(fwd);
  uniforms.uCamPos.value.copy(camera.position);
}

// Render loop
const clock = new THREE.Clock();
function render(){
  uniforms.uTime.value = clock.getElapsedTime();
  controls.update();
  updateCameraBasis();
  renderer.render(scene, camera);
  requestAnimationFrame(render);
}
render();
