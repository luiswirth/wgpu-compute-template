use std::fs::{read_to_string, write};

fn main() {
    let shader_path = "src/shader.comp";
    let shader_src = read_to_string(shader_path).expect("unable to read shader");
    let mut compiler = shaderc::Compiler::new().expect("Unable to create shader compiler");
    let shader_spirv = compiler
        .compile_into_spirv(
            &shader_src,
            shaderc::ShaderKind::Compute,
            shader_path,
            "main",
            None,
        )
        .expect("Failed to compile");
    write(format!("{}.spv", shader_path), shader_spirv.as_binary_u8())
        .expect("Failed to write shader");
}
