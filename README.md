# Overview
- Graph Attention Networks (GAT) represents a given graph leveraging masked self-attention layer. This project aims to implement pytorch dependency-graph-attention-networks in order to represent the dependency relations of each word from given text utilizing masked self-attention. The output of the dependency-graph-attention-networks is the token-level representation of the sum of the token and its dependency.

# Brief description
- DepGCN.py
> Output format
> - output: List of tensor of the sum of token embedding itself and dependency relation that is connected to the governor. (list)
- text_processing.py
> Output format
> - input_tk_list (list): Tokens of given text based on the selected nlp pipeline.
> - input_dep_list (list): Dependency triple of given text based on the selected nlp pipeline.

# Prerequisites
- argparse
- torch
- stanza
- spacy

# Parameters
- nlp_pipeline(str, defaults to "stanza"): NLP preprocessing pipeline.
- alpha(float, defaults to 0.01): Controls the angle of the negative slope of LeakyReLU.

# References
- Attention: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.
- Graph Attention Networks (GAT): Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2017). Graph attention networks. arXiv preprint arXiv:1710.10903.
