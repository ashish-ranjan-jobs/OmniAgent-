# Model Quantization

Quantization stores weights and activations in lower precision: 8-bit or
4-bit integers instead of 32-bit floats. This cuts model memory roughly in
proportion to the bit width, at a small accuracy cost. Post-training
quantization converts a trained model directly; quantization-aware training
simulates rounding during fine-tuning to recover accuracy. Serve quantized
weights when memory bandwidth is the bottleneck.
