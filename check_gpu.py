import os
import tensorflow as tf

# 1. Set environment variables BEFORE importing TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warnings, 3=errors
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # Better GPU mem management

# 2. Import TF and immediately configure logging
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)  # 0=all, 1=warnings, 2=errors, 3=none

# 3. Disable device placement logging (critical for your issue)
tf.debugging.set_log_device_placement(False)  # ← THIS IS WHAT YOU NEED

# 4. GPU configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✔ Using {len(gpus)} GPU(s) with memory growth enabled")
    except RuntimeError as e:
        print("GPU configuration error:", e)
else:
    print("✘ No GPUs found - using CPU")
