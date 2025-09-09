import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import GUI from 'lil-gui';
import { SimplexNoise } from 'three/addons/math/SimplexNoise.js';

class CrystalGrid {
    constructor() {
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 100);
        this.camera.position.z = 10;
        this.renderer = new THREE.WebGLRenderer({
            canvas: document.getElementById('webgl-canvas'),
            antialias: true,
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.clock = new THREE.Clock();
        this.simplex = new SimplexNoise();
        this.mouse = new THREE.Vector2();
        this.meshes = [];

        this.init();
    }

    init() {
        this.setupControls();
        this.createGrid();
        this.setupPostProcessing();
        this.addEventListeners();
        this.animate();
    }
    
    setupControls() {
        const guiContainer = document.getElementById('gui-container');
        this.gui = new GUI({ container: guiContainer });

        this.settings = {
            animationSpeed: 0.15,
            deformation: 0.5,
            lineFrequency: 8.0,
            lineFlicker: 0.7,
            fresnelPower: 2.5,
            colorCycleSpeed: 0.2,
            glowStrength: 0.4,
            glowRadius: 0.6,
        };
        
        const shapeFolder = this.gui.addFolder('Shape Controls');
        shapeFolder.add(this.settings, 'animationSpeed', 0.0, 1.0).name('Animation Speed');
        shapeFolder.add(this.settings, 'deformation', 0.0, 2.0).name('Deformation');

        const textureFolder = this.gui.addFolder('Texture & Color');
        textureFolder.add(this.settings, 'lineFrequency', 1.0, 20.0).name('Line Density');
        textureFolder.add(this.settings, 'lineFlicker', 0.0, 2.0).name('Line Flicker');
        textureFolder.add(this.settings, 'fresnelPower', 0.1, 5.0).name('Iridescence');
        textureFolder.add(this.settings, 'colorCycleSpeed', 0.0, 1.0).name('Color Speed');

        const glowFolder = this.gui.addFolder('Glow Effect');
        glowFolder.add(this.settings, 'glowStrength', 0.0, 2.0).onChange(val => {
            if (this.bloomPass) this.bloomPass.strength = val;
        });
        glowFolder.add(this.settings, 'glowRadius', 0.0, 2.0).onChange(val => {
            if (this.bloomPass) this.bloomPass.radius = val;
        });
    }

    createGrid() {
        const grid_size = 4;
        const spacing = 4.5;
        
        this.sharedMaterial = new THREE.ShaderMaterial({
            uniforms: {
                u_time: { value: 0.0 },
                u_lineFrequency: { value: this.settings.lineFrequency },
                u_lineFlicker: { value: this.settings.lineFlicker },
                u_fresnelPower: { value: this.settings.fresnelPower },
                u_colorCycleSpeed: { value: this.settings.colorCycleSpeed },
            },
            vertexShader: `
                varying vec3 v_normal;
                varying vec3 v_viewDirection;
                
                void main() {
                    vec4 modelViewPosition = modelViewMatrix * vec4(position, 1.0);
                    v_viewDirection = -modelViewPosition.xyz;
                    v_normal = normalize(normalMatrix * normal);
                    gl_Position = projectionMatrix * modelViewPosition;
                }
            `,
            fragmentShader: `
                uniform float u_time;
                uniform float u_lineFrequency;
                uniform float u_lineFlicker;
                uniform float u_fresnelPower;
                uniform float u_colorCycleSpeed;

                varying vec3 v_normal;
                varying vec3 v_viewDirection;

                float noise(vec2 st) {
                    return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
                }

                vec3 getPalette(float t) {
                    vec3 color1 = vec3(1.0, 0.4, 0.7); // Pink
                    vec3 color2 = vec3(0.4, 1.0, 0.7); // Green
                    vec3 color3 = vec3(0.4, 0.7, 1.0); // Blue
                    
                    vec3 finalColor = mix(color1, color2, smoothstep(0.0, 0.5, t));
                    finalColor = mix(finalColor, color3, smoothstep(0.5, 1.0, t));
                    
                    return finalColor;
                }

                void main() {
                    vec3 viewDir = normalize(v_viewDirection);
                    float fresnel = 1.0 - dot(viewDir, v_normal);
                    fresnel = pow(fresnel, u_fresnelPower);

                    float colorTime = u_time * u_colorCycleSpeed * 0.2;
                    vec3 baseColor = getPalette(fract(fresnel + colorTime));

                    float lineNoise = noise(gl_FragCoord.xy * 0.1 + u_time);
                    float linePos = gl_FragCoord.y + lineNoise * u_lineFlicker * 20.0;
                    float linePattern = sin(linePos * u_lineFrequency * 0.1);
                    linePattern = smoothstep(0.4, 0.6, linePattern);
                    
                    vec3 finalColor = baseColor * (0.5 + 0.5 * linePattern);
                    
                    gl_FragColor = vec4(finalColor, 1.0);
                }
            `,
            side: THREE.DoubleSide
        });

        for (let i = 0; i < grid_size; i++) {
            for (let j = 0; j < grid_size; j++) {
                const geometry = new THREE.IcosahedronGeometry(1.2, 3);
                geometry.userData.originalPositions = geometry.attributes.position.clone();
                const mesh = new THREE.Mesh(geometry, this.sharedMaterial);
                mesh.position.x = (i - (grid_size - 1) / 2) * spacing;
                mesh.position.y = (j - (grid_size - 1) / 2) * spacing;
                mesh.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, Math.random() * Math.PI);
                this.scene.add(mesh);
                this.meshes.push(mesh);
            }
        }
    }

    setupPostProcessing() {
        this.composer = new EffectComposer(this.renderer);
        const renderPass = new RenderPass(this.scene, this.camera);
        this.composer.addPass(renderPass);
        this.bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 1.5, 0.4, 0.85);
        this.bloomPass.threshold = 0;
        this.bloomPass.strength = this.settings.glowStrength;
        this.bloomPass.radius = this.settings.glowRadius;
        this.composer.addPass(this.bloomPass);
    }
    
    updateMeshes(elapsedTime) {
        this.meshes.forEach((mesh, index) => {
            const geometry = mesh.geometry;
            const originalPositions = geometry.userData.originalPositions.array;
            const positions = geometry.attributes.position.array;
            const time = elapsedTime * this.settings.animationSpeed;
            const deformation = this.settings.deformation;
            for (let i = 0; i < positions.length; i += 3) {
                const ox = originalPositions[i];
                const oy = originalPositions[i + 1];
                const oz = originalPositions[i + 2];
                const noise1 = this.simplex.noise3d(ox * 0.8 + time + index, oy * 0.8 + time, oz * 0.8 + time);
                const noise2 = this.simplex.noise3d(ox * 2.5 + time, oy * 2.5 + time, oz * 2.5 + time + index);
                const totalNoise = noise1 + (noise2 * 0.3);
                positions[i] = ox + totalNoise * deformation;
                positions[i + 1] = oy + totalNoise * deformation;
                positions[i + 2] = oz + totalNoise * deformation;
            }
            geometry.attributes.position.needsUpdate = true;
            geometry.computeVertexNormals();
        });
    }

    addEventListeners() {
        window.addEventListener('resize', () => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
            this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            this.composer.setSize(window.innerWidth, window.innerHeight);
        });
        
        window.addEventListener('mousemove', (event) => {
            this.mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            this.mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
        });
    }
    
    animate() {
        const elapsedTime = this.clock.getElapsedTime();
        this.sharedMaterial.uniforms.u_time.value = elapsedTime;
        this.sharedMaterial.uniforms.u_lineFrequency.value = this.settings.lineFrequency;
        this.sharedMaterial.uniforms.u_lineFlicker.value = this.settings.lineFlicker;
        this.sharedMaterial.uniforms.u_fresnelPower.value = this.settings.fresnelPower;
        this.sharedMaterial.uniforms.u_colorCycleSpeed.value = this.settings.colorCycleSpeed;
        this.updateMeshes(elapsedTime);
        this.camera.position.x += (this.mouse.x * 2 - this.camera.position.x) * 0.02;
        this.camera.position.y += (this.mouse.y * 2 - this.camera.position.y) * 0.02;
        this.camera.lookAt(this.scene.position);
        this.composer.render();
        requestAnimationFrame(() => this.animate());
    }
}

window.addEventListener('DOMContentLoaded', () => {
    new CrystalGrid();
});
