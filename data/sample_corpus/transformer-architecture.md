# Transformer Architecture

The transformer replaced recurrence with stacked self-attention blocks.
Each block pairs multi-head attention with a position-wise feed-forward
network and residual connections with layer normalization. Encoders build
bidirectional representations; decoders attend over encoder outputs while
masking future positions. Depth, width, and head count trade compute against
representational capacity.
