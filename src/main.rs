use std::collections::HashMap;

use bevy_ecs::prelude::*;
use bevy_ecs::world::World;
use dreamfield_macros::{preprocess_shader_vf, preprocess_shader_vtf};
use dreamfield_renderer::camera::{Camera, FpsCamera};
use dreamfield_renderer::gl_backend::*;
use glfw::{Key, Action, Context};
use cgmath::{vec2, vec3, perspective, Deg, SquareMatrix, Matrix4};

/// Whether wireframe mode is enabled
const WIREFRAME_MODE: bool = false;

/// The camera look speed
const CAM_LOOK_SPEED: f32 = 0.001;

/// The camera fly speed
const CAM_FLY_SPEED: f32 = 0.1;

/// The width of the window
const WINDOW_WIDTH: u32 = 1024 * 2;

/// The height of the window
const WINDOW_HEIGHT: u32 = 768 * 2;

/// The fixed update frequency
const FIXED_UPDATE: i32 = 30;

/// The fixed update target time
const FIXED_UPDATE_TIME: f64 = 1.0 / (FIXED_UPDATE as f64);

/// The ps1 shader
const PS1_SHADER_SOURCE: (&str, &str, &str, &str) = preprocess_shader_vtf!(include_bytes!("../resources/shaders/ps1.glsl"));

/// The blit shader
const BLIT_SHADER_SOURCE: (&str, &str) = preprocess_shader_vf!(include_bytes!("../resources/shaders/blit.glsl"));

/// The render width
const RENDER_WIDTH: i32 = 320;

/// The render height
const RENDER_HEIGHT: i32 = 240;

/// The render aspect ratio
const RENDER_ASPECT: f32 = 4.0 / 3.0;

/// Position component
#[derive(Component)]
struct Position { x: f32, y: f32, z: f32 }

/// Velocity component
#[derive(Component)]
struct Velocity { x: f32, y: f32, z: f32 }


/// PitchYaw component
#[derive(Component)]
struct PitchYaw { pitch: f32, yaw: f32 }

/// Model component
#[derive(Component)]
struct Model { name: String }

/// Camera component
#[derive(Component)]
struct CameraComponent { camera: FpsCamera }

/// Time resource
#[derive(Default)]
struct Time {
    time: f64,
    time_delta: f64
}

// RenderParams resource
struct RenderParams {
    ubo_global: UniformBuffer<GlobalParams>,
    ps1_shader_program: ShaderProgram,
    blit_shader_program: ShaderProgram,
    framebuffer: Framebuffer,
    full_screen_rect: Mesh
}

impl RenderParams {
    pub fn new() -> Self {
        // Create ubo_global
        let mut ubo_global = UniformBuffer::new();

        ubo_global.set_fog_color(&vec3(0.05, 0.05, 0.05));
        ubo_global.set_fog_dist(&vec2(7.5, 15.0));

        ubo_global.set_target_aspect(&RENDER_ASPECT);
        ubo_global.set_render_res(&vec2(RENDER_WIDTH as f32, RENDER_HEIGHT as f32));

        ubo_global.set_window_aspect(&(WINDOW_WIDTH as f32 / WINDOW_HEIGHT as f32));

        ubo_global.set_mat_proj(&perspective(Deg(60.0), RENDER_ASPECT, 0.01, 20.0));

        ubo_global.bind(bindings::UniformBlockBinding::GlobalParams);

        // Load ps1 shaders
        let ps1_shader_program = ShaderProgram::new_from_vtf(PS1_SHADER_SOURCE);
        let blit_shader_program = ShaderProgram::new_from_vf(BLIT_SHADER_SOURCE);

        // Create framebuffer
        let framebuffer = Framebuffer::new(RENDER_WIDTH, RENDER_HEIGHT);

        // Create full screen rect
        let full_screen_rect = Mesh::new_indexed(
            &vec![
                 1.0,  1.0, 0.0, 1.0, 1.0,  // top right
                 1.0, -1.0, 0.0, 1.0, 0.0,  // bottom right
                -1.0, -1.0, 0.0, 0.0, 0.0,  // bottom left
                -1.0,  1.0, 0.0, 0.0, 1.0,  // top left
            ],
            &vec![
                0, 1, 3,
                1, 2, 3,
            ],
            &vec![
                VertexAttrib { index: 0, size: 3, attrib_type: gl::FLOAT },
                VertexAttrib { index: 1, size: 2, attrib_type: gl::FLOAT },
            ]);

        RenderParams {
            ubo_global,
            ps1_shader_program,
            blit_shader_program,
            framebuffer,
            full_screen_rect
        }
    }
}

// ModelManager resource
struct ModelManager {
    loaded_models: HashMap<String, GltfModel>
}

impl ModelManager {
    pub fn new() -> Self {
        ModelManager {
            loaded_models: HashMap::new()
        }
    }

    pub fn with_model<F: FnMut(&mut GltfModel) -> ()>(&mut self, name: &str, mut f: F) {
        match self.loaded_models.get_mut(name) {
            Some(model) => {
                f(model);
            },
            _ => {
                let mut model = GltfModel::from_file(name)
                    .unwrap_or_else(|_| panic!("Failed to load model {name}"));
                f(&mut model);
                self.loaded_models.insert(name.to_string(), model);
            }
        }
    }
}

/// Camera system
fn camera(mut query: Query<(&Position, &PitchYaw, &mut CameraComponent)>) {
    let (pos, pitch_yaw, mut camera) = query.get_single_mut().expect("Expected one camera");

    camera.camera.set_pos(&vec3(pos.x, pos.y, pos.z));
    camera.camera.set_pitch_yaw(pitch_yaw.pitch, pitch_yaw.yaw);
    camera.camera.update();
}

/// Movement system
fn movement(time: Res<Time>, mut query: Query<(&mut Position, &Velocity)>) {
    for (mut position, velocity) in query.iter_mut() {
        position.x += velocity.x * time.time_delta as f32;
        position.y += velocity.y * time.time_delta as f32;
        position.z += velocity.z * time.time_delta as f32;
    }
}

/// Render system
fn render(mut render_params: ResMut<RenderParams>, mut model_manager: ResMut<ModelManager>, time: Res<Time>,
    mut cam_query: Query<&mut CameraComponent>, model_query: Query<(&Position, &Model)>)
{
    // Update matrices
    let camera = cam_query.get_single_mut().expect("Expected one camera");

    // Update ubo_global
    render_params.ubo_global.set_sim_time(&(time.time as f32));
    render_params.ubo_global.set_mat_view_derive(&camera.camera.get_view_matrix());
    render_params.ubo_global.upload_changed();

    // Bind FBO
    render_params.framebuffer.bind_draw();

    // Clear screen
    unsafe {
        gl::ClearColor(0.0, 0.0, 0.0, 1.0);
        gl::Viewport(0, 0, RENDER_WIDTH, RENDER_HEIGHT);
        gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
        gl::Enable(gl::DEPTH_TEST);

        if WIREFRAME_MODE {
            gl::PolygonMode(gl::FRONT_AND_BACK, gl::LINE);
        }
    }

    // Render objects
    render_params.ps1_shader_program.use_program();

    for (position, model) in model_query.iter() {
        render_params.ubo_global.set_mat_model_derive(&SquareMatrix::identity());
        render_params.ubo_global.upload_changed();
        model_manager.with_model(&model.name, |model: &mut GltfModel| {
            model.set_transform(&Matrix4::from_translation(vec3(position.x, position.y, position.z)));
            model.render(&mut render_params.ubo_global, true)
        });
    }

    // Blit to window
    unsafe {
        gl::Disable(gl::DEPTH_TEST);
        gl::Viewport(0, 0, WINDOW_WIDTH as i32, WINDOW_HEIGHT as i32);
        gl::PolygonMode(gl::FRONT_AND_BACK, gl::FILL);
    }
    render_params.framebuffer.unbind();
    render_params.framebuffer.bind_color_tex(bindings::TextureSlot::BaseColor);
    render_params.blit_shader_program.use_program();
    render_params.full_screen_rect.draw_indexed(gl::TRIANGLES, 6);
}

/// Main
fn main() {
    // Create window
    let mut window = Window::new_with_context(WINDOW_WIDTH, WINDOW_HEIGHT, "Dreamfield", gl::DEBUG_SEVERITY_LOW - 500);

    // Create bevy world
    let mut world = World::default();

    // Spawn camera entity
    let camera_id = world.spawn()
        .insert(Position { x: 0.0, y: 3.0, z: 10.0 })
        .insert(PitchYaw { pitch: -0.1, yaw: 0.0 })
        .insert(CameraComponent { camera: FpsCamera::new() })
        .id();

    // Spawn world entity
    world.spawn()
        .insert(Position { x: 0.0, y: 0.0, z: 0.0 })
        .insert(Velocity { x: 0.0, y: 0.0, z: 0.0 })
        .insert(Model { name: "resources/models/demo_scene.glb".to_string() });

    // Spawn ball entities
    world.spawn()
        .insert(Position { x: 0.0, y: 0.0, z: 0.0 })
        .insert(Velocity { x: 0.0, y: 0.5, z: 0.0 })
        .insert(Model { name: "resources/models/fire_orb.glb".to_string() });

    world.spawn()
        .insert(Position { x: 0.0, y: 0.0, z: 20.0 })
        .insert(Velocity { x: 0.0, y: 0.5, z: 0.0 })
        .insert(Model { name: "resources/models/fire_orb.glb".to_string() });

    world.spawn()
        .insert(Position { x: -10.0, y: 0.0, z: 10.0 })
        .insert(Velocity { x: 0.0, y: 0.5, z: 0.0 })
        .insert(Model { name: "resources/models/fire_orb.glb".to_string() });

    world.spawn()
        .insert(Position { x: 10.0, y: 0.0, z: 10.0 })
        .insert(Velocity { x: 0.0, y: 0.5, z: 0.0 })
        .insert(Model { name: "resources/models/fire_orb.glb".to_string() });

    // Add resources
    world.insert_resource(Time::default());
    world.insert_resource(RenderParams::new());
    world.insert_resource(ModelManager::new());

    // Create schedule
    let mut update_schedule = Schedule::default();
    update_schedule.add_stage("update", SystemStage::parallel()
        .with_system(movement)
        .with_system(camera)
    );

    let mut render_schedule = Schedule::default();
    render_schedule.add_stage("render", SystemStage::single_threaded()
        .with_system(render)
    );

    // Fixed timestep - https://gafferongames.com/post/fix_your_timestep/
    let mut current_time = window.glfw.get_time();
    let mut sim_time = 0.0;
    let mut accumulator = 0.0;

    // Current mouse pos
    let (mut mouse_x, mut mouse_y) = window.window.get_cursor_pos();

    // Camera input
    let mut forward_held: bool = false;

    // TODO: might be worth doing this on click, and adding a release button
    window.set_mouse_captured(true);

    // Main loop
    while !window.window.should_close() {
        // Handle events
        for event in window.poll_events() {
            match event {
                glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
                    window.window.set_should_close(true)
                },
                glfw::WindowEvent::Key(Key::W, _, Action::Press, _) => {
                    forward_held = true;
                },
                glfw::WindowEvent::Key(Key::W, _, Action::Release, _) => {
                    forward_held = false;
                },
                _ => {}
            }
        }

        // Update camera
        // TODO: I doubt this is the best way to get input into the system tbh
        let (old_mouse_x, old_mouse_y) = (mouse_x, mouse_y);
        (mouse_x, mouse_y) = window.window.get_cursor_pos();
        let (mouse_dx, mouse_dy) = (mouse_x - old_mouse_x, mouse_y - old_mouse_y);

        let mut camera_mut = world.entity_mut(camera_id);
        let mut camera_pitch_yaw = camera_mut.get_mut::<PitchYaw>().unwrap();

        camera_pitch_yaw.pitch -= mouse_dy as f32 * CAM_LOOK_SPEED;
        camera_pitch_yaw.yaw -= mouse_dx as f32 * CAM_LOOK_SPEED;

        if forward_held {
            let cam_forward = *camera_mut.get::<CameraComponent>().unwrap().camera.forward();
            let mut camera_pos = camera_mut.get_mut::<Position>().unwrap();
            let new_pos = vec3(camera_pos.x, camera_pos.y, camera_pos.z) + cam_forward * CAM_FLY_SPEED;
            camera_pos.x = new_pos.x;
            camera_pos.y = new_pos.y;
            camera_pos.z = new_pos.z;
        }

        // Fixed timestep
        let new_time = window.glfw.get_time();
        let frame_time = new_time - current_time;

        current_time = new_time;
        accumulator += frame_time;

        while accumulator >= FIXED_UPDATE_TIME {
            // Update time
            let mut time = world.resource_mut::<Time>();
            time.time = sim_time;
            time.time_delta = FIXED_UPDATE_TIME;

            // Run updates
            update_schedule.run(&mut world);

            // Consume accumulated time
            accumulator -= FIXED_UPDATE_TIME;
            sim_time += FIXED_UPDATE_TIME;
        }

        // Render
        render_schedule.run(&mut world);
        window.window.swap_buffers();
    }
}
