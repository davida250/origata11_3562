import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// --- Scene Setup ---
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 3;

const renderer = new THREE.WebGLRenderer({
    canvas: document.querySelector('#webgl'),
    antialias: true
});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

// --- Controls ---
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.5;

// --- Load Texture ---
// This texture provides the base iridescent color and linear patterns.
const textureLoader = new THREE.TextureLoader();
const holographicTexture = textureLoader.load(
    'https://upload.wikimedia.org/wikipedia/commons/8/82/Holographic_texture_02.jpg',
    (texture) => {
        texture.wrapS = THREE.RepeatWrapping;
        texture.wrapT = THREE.RepeatWrapping;
    }
);

// --- UI Sliders ---
const ui = {
    speed: document.getElementById('speed'),
    distortion: document.getElementById('distortion'),
    fresnel: document.getElementById('fresnel'),
};

// --- Shaders ---

const vertexShader = `
    uniform float uTime;
    uniform float uSpeed;
    uniform float uDistortion;

    varying vec2 vUv;
    varying vec3 vNormal;
    varying vec3 vViewPosition;

    // Simplex 3D Noise - used for organic, smooth procedural values
    // Author: Ashima Arts
    vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
    vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
    vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
    vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

    float snoise(vec3 v) {
        const vec2 C = vec2(1.0/6.0, 1.0/3.0);
        const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
        vec3 i  = floor(v + dot(v, C.yyy));
        vec3 x0 = v - i + dot(i, C.xxx);
        vec3 g = step(x0.yzx, x0.xyz);
        vec3 l = 1.0 - g;
        vec3 i1 = min(g.xyz, l.zxy);
        vec3 i2 = max(g.xyz, l.zxy);
        vec3 x1 = x0 - i1 + C.xxx;
        vec3 x2 = x0 - i2 + C.yyy;
        vec3 x3 = x0 - D.yyy;
        i = mod289(i);
        vec4 p = permute(permute(permute(
            i.z + vec4(0.0, i1.z, i2.z, 1.0))
            + i.y + vec4(0.0, i1.y, i2.y, 1.0))
            + i.x + vec4(0.0, i1.x, i2.x, 1.0));
        float n_ = 0.142857142857;
        vec3 ns = n_ * D.wyz - D.xzx;
        vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
        vec4 x_ = floor(j * ns.z);
        vec4 y_ = floor(j - 7.0 * x_);
        vec4 x = x_ * ns.x + ns.yyyy;
        vec4 y = y_ * ns.x + ns.yyyy;
        vec4 h = 1.0 - abs(x) - abs(y);
        vec4 b0 = vec4(x.xy, y.xy);
        vec4 b1 = vec4(x.zw, y.zw);
        vec4 s0 = floor(b0)*2.0 + 1.0;
        vec4 s1 = floor(b1)*2.0 + 1.0;
        vec4 sh = -step(h, vec4(0.0));
        vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
        vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
        vec3 p0 = vec3(a0.xy,h.x);
        vec3 p1 = vec3(a0.zw,h.y);
        vec3 p2 = vec3(a1.xy,h.z);
        vec3 p3 = vec3(a1.zw,h.w);
        vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
        p0 *= norm.x;
        p1 *= norm.y;
        p2 *= norm.z;
        p3 *= norm.w;
        vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
        m = m * m;
        return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
    }

    void main() {
        vUv = uv;
        vNormal = normal;
        
        // --- Vertex Displacement ---
        // This is the core logic for the folding effect.
        // We use two layers of noise ("octaves") for more complex folding.
        // The first layer creates the large, slow folds.
        float noiseTime = uTime * uSpeed;
        float largeFolds = snoise(position * 0.5 + noiseTime);
        
        // The second layer adds smaller, faster details on top.
        float smallFolds = snoise(position * 2.5 + noiseTime * 1.5);

        // Combine the noise and apply it along the vertex normal to push/pull the vertex.
        float displacement = (largeFolds * 0.7 + smallFolds * 0.3) * uDistortion;
        vec3 newPosition = position + normal * displacement;

        vec4 modelViewPosition = modelViewMatrix * vec4(newPosition, 1.0);
        vViewPosition = -modelViewPosition.xyz;
        gl_Position = projectionMatrix * modelViewPosition;
    }
`;

const fragmentShader = `
    uniform sampler2D uTexture;
    uniform float uFresnelPower;

    varying vec2 vUv;
    varying vec3 vNormal;
    varying vec3 vViewPosition;

    void main() {
        // --- Base Color from Texture ---
        // We tile the texture (vUv * 2.0) to make the pattern finer.
        vec4 textureColor = texture2D(uTexture, vUv * 2.0);
        
        // --- Fresnel Effect for iridescent highlights ---
        // This calculates how much a surface is facing the camera.
        // Edges will have a higher fresnel value, creating a rim-light effect.
        vec3 viewDirection = normalize(vViewPosition);
        float fresnel = 1.0 - dot(normalize(vNormal), viewDirection);
        fresnel = pow(fresnel, uFresnelPower); // Power controls the sharpness of the glow

        // --- Final Color ---
        // We add the bright fresnel highlight to the texture color.
        vec3 finalColor = textureColor.rgb + fresnel;

        gl_FragColor = vec4(finalColor, 1.0);
    }
`;


// --- Geometry and Material ---
// An Icosahedron gives us a complex starting shape.
// Higher detail (the second parameter, 15) gives a smoother deformation.
const geometry = new THREE.IcosahedronGeometry(1.2, 15);
const material = new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    uniforms: {
        uTime: { value: 0 },
        uSpeed: { value: parseFloat(ui.speed.value) },
        uDistortion: { value: parseFloat(ui.distortion.value) },
        uFresnelPower: { value: parseFloat(ui.fresnel.value) },
        uTexture: { value: holographicTexture }
    },
});

const mesh = new THREE.Mesh(geometry, material);
scene.add(mesh);

// --- UI Listeners ---
ui.speed.addEventListener('input', (event) => {
    material.uniforms.uSpeed.value = parseFloat(event.target.value);
});

ui.distortion.addEventListener('input', (event) => {
    material.uniforms.uDistortion.value = parseFloat(event.target.value);
});

ui.fresnel.addEventListener('input', (event) => {
    material.uniforms.uFresnelPower.value = parseFloat(event.target.value);
});

// --- Handle Window Resize ---
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
});

// --- Animation Loop ---
const clock = new THREE.Clock();

function animate() {
    const elapsedTime = clock.getElapsedTime();

    // Update shader time uniform for animation
    material.uniforms.uTime.value = elapsedTime;

    // Update orbit controls
    controls.update();

    // Render the scene
    renderer.render(scene, camera);

    // Call animate again on the next frame
    requestAnimationFrame(animate);
}

animate();
