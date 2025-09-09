import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import GUI from 'lil-gui';
import { SimplexNoise } from 'three/addons/math/SimplexNoise.js';

/**
 * Main application class
 */
class CrystalGrid {
    constructor() {
        // Basic Scene Setup
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
    
    /**
     * GUI Controls using lil-gui
     */
    setupControls() {
        this.gui = new GUI();
        this.settings = {
            animationSpeed: 0.2,
            deformationAmount: 0.4,
            lineFrequency: 4.0,
            lineThickness: 0.8,
            glowStrength: 0.4,
            glowRadius: 0.6,
        };

        this.gui.add(this.settings, 'animationSpeed', 0.0, 1.0, 0.01).name('Animation Speed');
        this.gui.add(this.settings, 'deformationAmount', 0.0, 2.0, 0.01).name('Deformation');
        this.gui.add(this.settings, 'lineFrequency', 1.0, 20.0, 0.1).name('Texture Lines');
        this.gui.add(this.settings, 'lineThickness', 0.1, 1.0, 0.01).name('Line Thickness');
        this.gui.add(this.settings, 'glowStrength', 0.0, 2.0, 0.01).name('Glow Strength').onChange(val => {
            if (this.bloomPass) this.bloomPass.strength = val;
        });
        this.gui.add(this.settings, 'glowRadius', 0.0, 2.0, 0.01).name('Glow Radius').onChange(val => {
            if (this.bloomPass) this.bloomPass.radius = val;
        });
    }

    /**
     * Creates the 4x4 grid of crystal meshes
     */
    createGrid() {
        const grid_size = 4;
        const spacing = 4;
        
        // Define our custom shader material
        const material = new THREE.ShaderMaterial({
            uniforms: {
                u_time: { value: 0.0 },
                u_lineFrequency: { value: this.settings.lineFrequency },
                u_lineThickness: { value: this.settings.lineThickness },
            },
            vertexShader: `
                varying vec3 v_normal;
                varying vec3 v_position;
                
                void main() {
                    v_normal = normalize(normalMatrix * normal);
                    v_position = position;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform float u_time;
                uniform float u_lineFrequency;
                uniform float u_lineThickness;

                varying vec3 v_normal;
                varying vec3 v_position;

                // Color palette function
                vec3 palette(float t) {
                    vec3 a = vec3(0.5, 0.5, 0.5);
                    vec3 b = vec3(0.5, 0.5, 0.5);
                    vec3 c = vec3(1.0, 1.0, 1.0);
                    vec3 d = vec3(0.263, 0.416, 0.557);
                    return a + b * cos(6.28318 * (c * t + d));
                }

                void main() {
                    // --- Iridescent/Holographic Effect ---
                    // Use the angle between the camera view and the surface normal (Fresnel effect)
                    vec3 view_dir = normalize(cameraPosition - (modelMatrix * vec4(v_position, 1.0)).xyz);
                    float fresnel = 1.0 - dot(view_dir, v_normal);
                    fresnel = pow(fresnel, 2.0);

                    // --- Glitchy Line Texture ---
                    // Create horizontal lines based on screen position (gl_FragCoord)
                    // and animate their offset with time
                    float line_pattern = sin(gl_FragCoord.y * u_lineFrequency * 0.1 + u_time * 5.0);
                    line_pattern = step(u_lineThickness, line_pattern); // Create hard lines
                    
                    // Combine effects
                    vec3 base_color = palette(fresnel + u_time * 0.1);
                    vec3 final_color = base_color * (0.6 + 0.4 * line_pattern);
                    
                    gl_FragColor = vec4(final_color, 1.0);
                }
            `,
            side: THREE.DoubleSide
        });

        // Create 16 meshes
        for (let i = 0; i < grid_size; i++) {
            for (let j = 0; j < grid_size; j++) {
                // Icosahedron geometry gives a nice crystalline base shape
                const geometry = new THREE.IcosahedronGeometry(1, 1);
                
                // Store original positions for deformation calculation
                geometry.userData.originalPositions = geometry.attributes.position.clone();
                
                const mesh = new THREE.Mesh(geometry, material);
                
                // Position the mesh in the grid
                mesh.position.x = (i - (grid_size - 1) / 2) * spacing;
                mesh.position.y = (j - (grid_size - 1) / 2) * spacing;
                
                // Give each mesh a random rotation offset for variation
                mesh.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, Math.random() * Math.PI);
                
                this.scene.add(mesh);
                this.meshes.push(mesh);
            }
        }
    }

    /**
     * Post-processing for the glow effect
     */
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
    
    /**
     * Update geometry vertices to create the folding/morphing effect
     */
    updateMeshes(elapsedTime) {
        this.meshes.forEach((mesh, index) => {
            const geometry = mesh.geometry;
            const originalPositions = geometry.userData.originalPositions.array;
            const positions = geometry.attributes.position.array;

            const time = elapsedTime * this.settings.animationSpeed;
            const deformation = this.settings.deformationAmount;

            // Loop through each vertex
            for (let i = 0; i < positions.length; i += 3) {
                const ox = originalPositions[i];
                const oy = originalPositions[i + 1];
                const oz = originalPositions[i + 2];
                
                // Use simplex noise to create organic, flowing deformation
                // We use the vertex's original position and a time component to get a noise value
                const noise = this.simplex.noise3d(
                    ox * 0.5 + time + index, // Add index to vary noise per mesh
                    oy * 0.5 + time,
                    oz * 0.5 + time
                );

                // Apply the noise to the vertex position
                positions[i] = ox + noise * deformation;
                positions[i + 1] = oy + noise * deformation;
                positions[i + 2] = oz + noise * deformation;
            }

            // Important: Tell Three.js that the vertices have been updated
            geometry.attributes.position.needsUpdate = true;
            geometry.computeVertexNormals(); // Recalculate normals for correct lighting
        });
    }

    /**
     * Event listeners for window resize and mouse movement
     */
    addEventListeners() {
        window.addEventListener('resize', () => {
            // Update camera
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();

            // Update renderer
            this.renderer.setSize(window.innerWidth, window.innerHeight);
            this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

            // Update post-processing composer
            this.composer.setSize(window.innerWidth, window.innerHeight);
        });
        
        window.addEventListener('mousemove', (event) => {
            // Normalize mouse position (-1 to +1)
            this.mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            this.mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
        });
    }
    
    /**
     * Animation loop
     */
    animate() {
        const elapsedTime = this.clock.getElapsedTime();

        // Update uniforms for all meshes (since they share the material)
        const sharedMaterial = this.meshes[0].material;
        sharedMaterial.uniforms.u_time.value = elapsedTime;
        sharedMaterial.uniforms.u_lineFrequency.value = this.settings.lineFrequency;
        sharedMaterial.uniforms.u_lineThickness.value = this.settings.lineThickness;

        // Update the geometry of each mesh
        this.updateMeshes(elapsedTime);

        // Add subtle camera movement based on mouse position for a parallax effect
        this.camera.position.x += (this.mouse.x * 2 - this.camera.position.x) * 0.02;
        this.camera.position.y += (this.mouse.y * 2 - this.camera.position.y) * 0.02;
        this.camera.lookAt(this.scene.position);

        // Use the composer to render the scene with post-processing effects
        this.composer.render();

        // Request next frame
        requestAnimationFrame(() => this.animate());
    }
}

// Start the application
new CrystalGrid();
