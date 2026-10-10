# The BGE Model Family

BAAI's BGE family ships base and large English encoders using 768 and 1024 dimensions respectively, a multilingual variant, and reranker heads trained for passage scoring. The small variant targets CPU-friendly deployment. Each release pairs with its own retrieval recipe: the models are trained contrastively on large query-passage pair corpora, so follow the recipe shipped with the checkpoint rather than mixing components across variants.
