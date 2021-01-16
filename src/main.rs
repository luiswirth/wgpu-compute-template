use std::{convert::TryInto, mem};
use wgpu::util::DeviceExt;

async fn run() {
    let input = vec![0; 1024];
    let output = execute_gpu(input).await;
    println!("{:#?}", output);
}

type DataType = u32;
type Data = Vec<DataType>;

async fn execute_gpu(input: Data) -> Data {
    // wgpu instance
    let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);

    // general gpu connection
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();

    // feature specific gpu connection
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
                shader_validation: true,
            },
            None,
        )
        .await
        .unwrap();

    // load shader
    let cs_module = device.create_shader_module(wgpu::include_spirv!("shader.comp.spv"));

    let buffer_size = (input.len() * mem::size_of::<DataType>()) as wgpu::BufferAddress;

    // buffer to upload
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buffer_size,
        usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
        mapped_at_creation: false,
    });

    // buffer for computation (storage buffer)
    let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Storage Buffer"),
        contents: bytemuck::cast_slice(&input),
        usage: wgpu::BufferUsage::STORAGE
            | wgpu::BufferUsage::COPY_DST
            | wgpu::BufferUsage::COPY_SRC,
    });

    // A bind group defines how buffers are accessed by shaders.
    // wgpu: "binding group" | vulkan: "descriptor set"
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStage::COMPUTE,
            ty: wgpu::BindingType::StorageBuffer {
                dynamic: false,
                min_binding_size: wgpu::BufferSize::new(4), // TODO: is 4 correct?
                readonly: false,
            },
            count: None,
        }],
    });

    // Instantiates the bind group, once again specifying the binding of buffers.
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(storage_buffer.slice(..)),
        }],
    });

    // A pipeline specifices the operation of a shader

    // Here we specifiy the layout of the pipeline.
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // Instantiates the pipeline.
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        compute_stage: wgpu::ProgrammableStageDescriptor {
            module: &cs_module,
            entry_point: "main",
        },
    });

    // wgpu: "command encoder" | vulkan: "command buffer"
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass();
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.insert_debug_marker("compute");
        cpass.dispatch(input.len().try_into().unwrap(), 1, 1); // thread chunk size
    }
    // Sets adds copy operation to command encoder.
    // Will copy data from storage buffer on GPU to staging buffer on CPU.
    encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, buffer_size);

    // Submits command encoder for processing
    queue.submit(Some(encoder.finish()));

    // Note that we're not calling `.await` here.
    let buffer_slice = staging_buffer.slice(..);
    // Gets the future representing when `staging_buffer` can be read from
    let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    device.poll(wgpu::Maintain::Wait);

    // Awaits until `buffer_future` can be read from
    if let Ok(()) = buffer_future.await {
        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();
        // Since contents are got in bytes, this converts these bytes back to u32
        let result = data
            .chunks_exact(mem::size_of::<DataType>() / mem::size_of::<u8>())
            .map(|b| DataType::from_ne_bytes(b.try_into().unwrap()))
            .collect();

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(data);
        staging_buffer.unmap(); // free buffer memory

        // Returns data from buffer
        result
    } else {
        panic!("failed to run compute on gpu!")
    }
}

fn main() {
    wgpu_subscriber::initialize_default_subscriber(None);
    pollster::block_on(run());
}
