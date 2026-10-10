# RAGAS Evaluation Metrics

RAGAS scores RAG pipelines along four reference-free axes: faithfulness (are answer statements entailed by retrieved context), answer relevancy (does the answer address the prompt), context precision (are retrieved chunks ranked usefully), and context recall (is the needed information retrieved at all). Decomposition-based scoring turns each axis into judgeable sub-questions. The axes localize failures: low context metrics point at retrieval, low faithfulness at generation.
