from llava.train.train import train

if __name__ == "__main__":
    # Use SDPA (scaled dot product attention) instead of flash_attention_2
    # flash-attn doesn't have kernels compiled for Blackwell GPUs yet
    train(attn_implementation="sdpa")
