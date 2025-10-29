# Single GPU Feasibility Test

## Memory Optimizations Applied
- ✅ Gradient checkpointing enabled
- ✅ Flash Attention 2 enabled
- ✅ Skip merge_and_unload for inference
- ✅ 8-bit quantization

## Estimated Memory Usage

### Training (Single A100-80GB)
- Model (8-bit): ~70GB
- LoRA adapters: ~2-3GB
- Activations (with grad checkpoint): ~10-14GB
- **Total: ~82-87GB** ⚠️ (TIGHT - may need adjustments)

### Inference/Evaluation (Single A100-80GB)
- Model: ~70GB
- Activations: ~2-5GB
- **Total: ~72-75GB** ✅ (Should work)

## Testing Steps

### Step 1: Test Inference First (Lower Risk)
```python
# In eval_wmdp.py - this should work on single GPU
GPU_CONFIG = modal.gpu.A100(count=1, size="80GB")
```

Run evaluation on baseline or existing checkpoint to confirm memory usage.

### Step 2: Test Training with Reduced Batch Size
```python
# In config.py - adjust if needed
@dataclass
class TeacherTrainingConfig:
    batch_size: int = 2  # Reduced from 4
    gradient_accumulation_steps: int = 16  # Increased from 8
    # Effective batch size remains 32 (2 * 16)
```

### Step 3: Monitor Memory During Training
Add memory monitoring:
```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f}GB")
print(f"Max allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
```

### Step 4: Fallback Options if OOM

If you hit OOM errors:

**Option A: Further reduce batch size**
```python
batch_size: int = 1
gradient_accumulation_steps: int = 32
```

**Option B: Reduce max_length**
```python
max_length: int = 1536  # Down from 2048
```

**Option C: Use 2x A100-40GB instead**
- Same 160GB total capacity
- Potentially lower cost than 2x A100-80GB
- More headroom for safety

**Option D: Keep 2x A100-80GB for training**
- Use single GPU only for inference/evaluation
- This is the safest approach

## Cost Analysis

### GPU Configurations (Modal/RunPod approximate pricing)
- 2x A100-80GB: ~$6-8/hour
- 1x A100-80GB: ~$3-4/hour (**50% cost savings**)
- 2x A100-40GB: ~$4-6/hour (25-33% savings)

For 16-hour training run:
- Current (2x A100-80GB): ~$96-128
- Single GPU: ~$48-64 (**saves $48-64**)
- 2x A100-40GB: ~$64-96 (saves $32-48)

## Recommendation

**Conservative Approach:**
1. ✅ Switch to **single A100-80GB for all evaluation/inference** (safe bet)
2. ⚠️ **Test single GPU for training** with batch_size=2
3. 🔄 Keep 2x A100-40GB as middle-ground option
4. 📊 Monitor and adjust based on actual memory usage

**Aggressive Approach (if budget is priority):**
1. Test single A100-80GB for everything
2. Adjust batch size dynamically if needed
3. Accept slightly longer training time if using smaller batches

## Expected Outcome

With our efficiency improvements, **single GPU training is theoretically possible** but will require careful memory management.

**Inference should definitely work on single GPU** and save you ~50% of compute costs for the evaluation phase.

## Next Steps

Would you like me to:
1. Create a single-GPU test configuration?
2. Add automatic memory monitoring code?
3. Modify the Modal configuration to test single GPU?
