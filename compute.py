import torch
import gc
from fvcore.nn import FlopCountAnalysis, parameter_count
from diffusers import UNetSpatioTemporalConditionModel, UNet2DConditionModel, UNet3DConditionModel

# ==========================================
# CONFIGURATION
# ==========================================
DIFFUSION_STEPS = 50 
TDP_WATTS = 300       
TEST_SET_SIZE = 899 

# ==========================================
# 1. MODEL FACTORY FUNCTIONS
# ==========================================
# These functions instantiate the models ONLY when called.

def geodes_2d():
    return UNet2DConditionModel(
        sample_size        = 32,
        in_channels        = 5, 
        out_channels       = 5,
        block_out_channels = [512, 1024, 2048], # Fixed the missing brackets here
        layers_per_block   = 2,
        down_block_types   = ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types     = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim= 768,
        # variant="bf16", # Keep commented out unless you want to force benchmark in bf16
    ) 

def geodes_3d():
    return UNet3DConditionModel(
        sample_size        = 32,
        in_channels        = 5,
        out_channels       = 5,
        block_out_channels = [512, 1024, 2048],
        layers_per_block   = 2,
        attention_head_dim = 8, 
        down_block_types   = ("CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "DownBlock3D"),
        up_block_types     = ("UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D"),
        cross_attention_dim= 768,
    ) 

def svd():
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        'stabilityai/stable-video-diffusion-img2vid',
        subfolder="unet",
        low_cpu_mem_usage=True,
        # variant="fp16", # Keep commented to benchmark in fp32, or uncomment if running low on VRAM
    )
    unet.config["sample_size"] = 32 
    
    old_conv_in = unet.conv_in
    new_conv_in = torch.nn.Conv2d(
        5, #channels
        old_conv_in.out_channels, 
        old_conv_in.kernel_size, 
        old_conv_in.stride, 
        old_conv_in.padding,
        device=unet.device,
        dtype=unet.dtype
    )
    unet.conv_in = new_conv_in
    unet.config["in_channels"] = 5   
        
    old_conv_out = unet.conv_out
    new_conv_out = torch.nn.Conv2d(
        old_conv_out.in_channels, 
        5,  #channels
        old_conv_out.kernel_size, 
        old_conv_out.stride, 
        old_conv_out.padding,
        device=unet.device,
        dtype=unet.dtype
    )
    unet.conv_out = new_conv_out
    unet.config["out_channels"] = 5
    return unet
    

# ==========================================
# 2. WRAPPERS (Required for fvcore tracing)
# ==========================================
# fvcore needs a standard forward pass. Diffusers models return special Output objects.
class UNetStandardWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
    def forward(self, sample, timestep, encoder_hidden_states):
        return self.unet(sample, timestep, encoder_hidden_states).sample

class SVDWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
    def forward(self, sample, timestep, encoder_hidden_states, added_time_ids):
        return self.unet(sample, timestep, encoder_hidden_states, added_time_ids=added_time_ids).sample

# ==========================================
# 3. BENCHMARK RUNNER
# ==========================================

def get_dummy_inputs(name, device, dtype):
    """Generates the exact tensor shapes required for each specific UNet."""
    batch = 1
    timestep = torch.tensor([1], dtype=dtype, device=device)

    if name == 'geodes_2d':
        sample = torch.randn(batch, 5, 32, 32, device=device, dtype=dtype)
        ctx = torch.randn(batch, 1, 768, device=device, dtype=dtype)
        return (sample, timestep, ctx)
        
    elif name == 'geodes_3d':
        frames = 8 # Adjust if your geodes 3D uses a different sequence length
        sample = torch.randn(batch, 5, frames, 32, 32, device=device, dtype=dtype)
        ctx = torch.randn(batch, 1, 768, device=device, dtype=dtype)
        return (sample, timestep, ctx)
        
    elif name == 'svd':
        frames = 8
        sample = torch.randn(batch, frames, 5, 32, 32, device=device, dtype=dtype)
        ctx = torch.randn(batch, 1, 1024, device=device, dtype=dtype)
        added_time_ids = torch.randn(batch, 3, device=device, dtype=dtype)
        return (sample, timestep, ctx, added_time_ids)

def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("❌ CUDA is not available. Please run on a GPU.")
        return

    print(f"Target Device: {torch.cuda.get_device_name(0)}")
    print(f"{'Model':<12} | {'Params (M)':<11} | {'FLOPs (G)':<11} | {'Latency (ms)':<12} | {'VRAM (GB)':<10} | {'Test Energy (Wh)':<16}")
    print("-" * 82)

    # Note: These are FUNCTIONS, not instantiated models
    model_factories = {
        'svd': svd,
        'geodes_2d': geodes_2d,
        'geodes_3d': geodes_3d,
    }

    for name, factory_func in model_factories.items():
        # 1. Initialize ONE model
        raw_model = factory_func()
        
        # Wrap it for fvcore
        if name == 'svd':
            model = SVDWrapper(raw_model).to(device)
        else:
            model = UNetStandardWrapper(raw_model).to(device)
            
        model.eval()
        
        # Automatically detect dtype (handles your bf16/fp16 variants)
        try:
            dtype = next(raw_model.parameters()).dtype
        except StopIteration:
            print(f"⚠️ Could not detect dtype for {name}, falling back to float32.")
            dtype = torch.float32
        dummy_inputs = get_dummy_inputs(name, device, dtype)

        # Clear memory completely before profiling
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # --- Metric 1 & 2: Parameters and FLOPs ---
        try:
            # FIX 1: Move model and inputs to CPU temporarily to bypass JIT tracer device bugs
            model.cpu()
            cpu_inputs = tuple(t.cpu() for t in dummy_inputs)
            
            flops_analyser = FlopCountAnalysis(model, cpu_inputs)
            
            # FIX 2: Silence the harmless "Unsupported operator" spam
            flops_analyser.unsupported_ops_warnings(False)
            flops_analyser.uncalled_modules_warnings(False)
            
            single_pass_flops = flops_analyser.total()
            params = parameter_count(model)['']
            
            # Move model back to GPU for the Latency and VRAM tests
            model.to(device)
            
            # Scale FLOPs by the number of diffusion steps
            total_flops = single_pass_flops * DIFFUSION_STEPS 
        except Exception as e:
            print(f"FLOP calc failed for {name}: {e}")
            total_flops, params = 0, parameter_count(model)['']
            model.to(device) # Ensure it's back on GPU even if it fails
            
        # --- Metric 3: Precise Latency ---
        with torch.no_grad():
            for _ in range(3): # Warmup
                _ = model(*dummy_inputs)
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            # Simulate the full diffusion generation loop
            for _ in range(DIFFUSION_STEPS):
                _ = model(*dummy_inputs)
        end_event.record()
        
        torch.cuda.synchronize()
        latency_ms = start_event.elapsed_time(end_event)

        # --- Metric 4: Peak VRAM ---
        peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

        # --- Metric 5: Energy Calculation ---
        latency_s = latency_ms / 1000.0
        energy_per_sample_wh = (TDP_WATTS * latency_s) / 3600.0
        total_test_energy_wh = energy_per_sample_wh * TEST_SET_SIZE

        print(f"{name:<12} | {params/1e6:<11.2f} | {total_flops/1e9:<11.2f} | {latency_ms:<12.2f} | {peak_vram_gb:<10.3f} | {total_test_energy_wh:<16.2f}")
        
        # ==========================================
        # CRITICAL DELETION STEP FOR MEMORY
        # ==========================================
        model.cpu()
        del model
        del raw_model
        del dummy_inputs
        gc.collect()
        torch.cuda.empty_cache()
        
    print("-" * 82)

if __name__ == "__main__":
    run_benchmark()
    
    