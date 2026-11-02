# Retrieval modes

Lexical retrieval uses TF-IDF with unigrams and bigrams. Dense retrieval computes sentence embeddings with a pretrained transformer. Hybrid retrieval uses reciprocal rank fusion (RRF) to combine ranked results from lexical and dense retrieval. Hybrid RRF scores are not probabilities.
