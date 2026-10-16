import os

# not sure if good to have, put easiest way to suppress the warning
# that's printed every time (don't know why the warn when it's NOT on)
# obviously only when using with ROCM
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"