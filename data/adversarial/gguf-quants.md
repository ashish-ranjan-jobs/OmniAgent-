# GGUF Quantization Formats

GGUF files carry model weights in integer-quantized blocks: Q4_K_M, Q5_K_S, Q8_0 and friends mix bit widths per tensor with per-block scales. Smaller formats shrink files and resident memory at mild perplexity cost, and run on CPU runtimes like llama.cpp. Format choice is a deployment knob, matched against available RAM and target tokens per second, with perplexity deltas reported per release.
