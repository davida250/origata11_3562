import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import GUI from 'lil-gui';

// --- Scene Setup ---
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 0, 3.5);

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
controls.autoRotateSpeed = 0.3;

// --- Load Texture ---
const textureLoader = new THREE.TextureLoader();
const holographicTexture = textureLoader.load(
    'https://upload.wikimedia.org/wikipedia/commons/8/82/Holographic_texture_02.jpg',
    (texture) => {
        texture.wrapS = THREE.RepeatWrapping;
        texture.wrapT = THREE.RepeatWrapping;
    }
);

// --- Settings and GUI ---
const gui = new GUI();
const settings = {
    speed: 0.15,
    distortion: 1.2,
    density: 1.5,
    strength: 0.6,
    shininess: 80.0,
    lightIntensity: 1.5,
};

// --- Shaders ---

const vertexShader = `
    uniform float uTime;
    uniform float uSpeed;
    uniform float uDensity;
    uniform float uStrength;

    varying vec3 vNormal;
    varying vec3 vPosition;
    varying vec2 vUv;

    // 4D Simplex Noise for smooth, time-varying noise
    // Author: Ian McEwan, Ashima Arts
    vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
    vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}
    float snoise(vec4 v){
        const vec2 C = vec2(0.138196601125010504, 0.309016994374947451);
        vec4 i  = floor(v + dot(v, C.yyyy) );
        vec4 x0 = v -   i + dot(i, C.xxxx);
        vec4 i0;
        vec3 isX = step( x0.yzw, x0.xxx );
        vec3 isYZ = step( x0.zww, x0.yyz );
        i0.x = isX.x + isX.y + isX.z;
        i0.yzw = 1.0 - isX;
        i0.y += isYZ.x;
        i0.z += isYZ.y;
        i0.w += isYZ.z;
        vec4 i3 = clamp( i0, 0.0, 1.0 );
        vec4 i2 = clamp( i0-1.0, 0.0, 1.0 );
        vec4 i1 = clamp( i0-2.0, 0.0, 1.0 );
        vec4 x1 = x0 - i1 + C.xxxx;
        vec4 x2 = x0 - i2 + C.yyyy;
        vec4 x3 = x0 - i3 + C.zzzz;
        vec4 x4 = x0 - 1.0 + C.wwww;
        i = mod(i, 289.0);
        float j0 = permute( permute( permute( permute(i.w) + i.z) + i.y) + i.x).x;
        vec4 j1 = permute( permute( permute( permute (
                    i.w + vec4(i1.w, i2.w, i3.w, 1.0 ))
                + i.z + vec4(i1.z, i2.z, i3.z, 1.0 ))
                + i.y + vec4(i1.y, i2.y, i3.y, 1.0 ))
                + i.x + vec4(i1.x, i2.x, i3.x, 1.0 ));
        vec4 ip = vec4(1.0/294.0, 1.0/49.0, 1.0/7.0, 0.0) ;
        vec4 p0 = sin(j0 * ip.x);
        vec4 p1 = sin(j1 * ip.x);
        vec4 p = vec4(p0.x, p1.xyz);
        p = p * 2.0 - 1.0;
        vec4 h = p * p * (3.0 - 2.0 * abs(p));
        vec4 m = dot(p, p) * -0.5 + 0.5;
        vec4 r = h * m;
        r.x *= dot(p, vec4(0.0, 1.0, 1.0, 1.0));
        vec4 grad = r.x * p + h * vec4(r.y, r.z, r.w, r.x);
        return 130.0 * dot(grad, vec4(x0.x, x1.x, x2.x, x3.x));
    }
    
    void main() {
        vUv = uv;
        float t = uTime * uSpeed;
        
        // --- Domain Warping for complex folding ---
        vec3 pos = position * uDensity;
        vec4 warpNoiseInput = vec4(pos * 0.5, t);
        float warpNoise = snoise(warpNoiseInput);
        
        // Use the warped coordinates to get the final displacement noise
        vec4 finalNoiseInput = vec4(pos + warpNoise, t);
        float finalNoise = snoise(finalNoiseInput);
        
        // Displace the vertex
        vec3 newPosition = position + normal * finalNoise * uStrength;
        
        // Recalculate normals for correct lighting
        // This is a common technique for procedurally displaced meshes
        float offset = 0.01;
        vec3 tangent = normalize(cross(normal, vec3(0.0, 1.0, 0.0)) + cross(normal, vec3(-1.0, 0.0, 0.0)));
        vec3 bitangent = normalize(cross(normal, tangent));
        vec3 neighbour1 = position + tangent * offset;
        vec3 neighbour2 = position + bitangent * offset;

        vec3 warpedN1 = neighbour1 + normal * snoise(vec4(neighbour1 * uDensity + warpNoise, t)) * uStrength;
        vec3 warpedN2 = neighbour2 + normal * snoise(vec4(neighbour2 * uDensity + warpNoise, t)) * uStrength;

        vNormal = normalize(cross(warpedN2 - newPosition, warpedN1 - newPosition));
        
        vec4 modelPosition = modelMatrix * vec4(newPosition, 1.0);
        vPosition = modelPosition.xyz;

        gl_Position = projectionMatrix * viewMatrix * modelPosition;
    }
`;

const fragmentShader = `
    uniform sampler2D uTexture;
    uniform float uShininess;
    uniform float uLightIntensity;
    
    varying vec3 vNormal;
    varying vec3 vPosition;
    varying vec2 vUv;
    
    void main() {
        // --- Lighting ---
        vec3 lightPosition = vec3(2.0, 2.0, 4.0);
        vec3 lightDirection = normalize(lightPosition - vPosition);
        vec3 viewDirection = normalize(cameraPosition - vPosition);
        
        // Diffuse
        float diffuse = max(0.0, dot(vNormal, lightDirection));
        
        // Specular (Blinn-Phong)
        vec3 halfwayDir = normalize(lightDirection + viewDirection);
        float spec = pow(max(dot(vNormal, halfwayDir), 0.0), uShininess);
        vec3 specular = vec3(1.0) * spec * uLightIntensity;

        // --- Iridescent Color ---
        float fresnel = dot(viewDirection, vNormal) + 0.1;
        vec3 color1 = vec3(0.1, 0.2, 0.9); // Blue/Purple
        vec3 color2 = vec3(0.2, 0.9, 0.4); // Green/Cyan
        vec3 iridescentColor = mix(color1, color2, fresnel);

        // --- Texture and Chromatic Aberration ---
        float chromaticAberration = pow(fresnel, 2.0) * 0.01; // Stronger at edges
        vec2 uvR = vUv + vec2(chromaticAberration, 0.0);
        vec2 uvG = vUv;
        vec2 uvB = vUv - vec2(chromaticAberration, 0.0);
        
        float texR = texture2D(uTexture, uvR * 2.0).r;
        float texG = texture2D(uTexture, uvG * 2.0).g;
        float texB = texture2D(uTexture, uvB * 2.0).b;
        vec3 textureColor = vec3(texR, texG, texB);
        
        // --- Final Combination ---
        vec3 finalColor = (iridescentColor * 0.5 + textureColor * 0.8) * diffuse + specular;
        gl_FragColor = vec4(finalColor, 1.0);
    }
`;


// --- Geometry and Material ---
// Use a LOW-POLY geometry to keep the shape angular and sharp.
const geometry = new THREE.IcosahedronGeometry(1.5, 1);
const material = new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    uniforms: {
        uTime: { value: 0 },
        uSpeed: { value: settings.speed },
        uDensity: { value: settings.density },
        uStrength: { value: settings.strength },
        uShininess: { value: settings.shininess },
        uLightIntensity: { value: settings.lightIntensity },
        uTexture: { value: holographicTexture }
    },
});

const mesh = new THREE.Mesh(geometry, material);
scene.add(mesh);

// --- UI Listeners ---
gui.add(settings, 'speed', 0, 1, 0.01).name('Animation Speed').onChange(v => material.uniforms.uSpeed.value = v);
gui.add(settings, 'distortion', 0, 3, 0.01).name('Distortion').onChange(v => {
    // Control both density and strength for a more intuitive "Distortion" slider
    material.uniforms.uDensity.value = v * 1.25;
    material.uniforms.uStrength.value = v * 0.5;
});
gui.add(settings, 'shininess', 1, 200, 1).name('Shininess').onChange(v => material.uniforms.uShininess.value = v);
gui.add(settings, 'lightIntensity', 0, 5, 0.1).name('Light Intensity').onChange(v => material.uniforms.uLightIntensity.value = v);

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
    material.uniforms.uTime.value = elapsedTime;
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
}

animate();
