# Prefix Tuning and Lightweight PEFT

Besides adapter layers, parameter-efficient fine-tuning includes prefix tuning (learned virtual key-value prefixes prepended to each attention layer) and IA3 (three learned rescaling vectors per layer). All freeze the backbone and train a small auxiliary parameter set, keeping optimizer state tiny. Prefix methods influence behavior through attention context rather than weight edits, which changes how edits compose and how far they transfer.
