import torch
import argparse

from text_processing import preprocessing
from DepGCN import Dependency_GCN

def tk2onehot(_tk_list):
    tk_dim = len(_tk_list)
    tk2onehot = []
    for idx,_ in enumerate(_tk_list):
        temp = torch.zeros(tk_dim)
        temp[idx] = 1
        tk2onehot.append(temp)
    return tk2onehot

def main(args):
    sample_text = "My dog likes eating sausage"
    input_tk_list, input_dep_list = preprocessing(sample_text, args.nlp_pipeline)

    # Simple One-hot encoding is applied. This can be replaced based on the choice of embedding language model.
    input_rep = tk2onehot(input_tk_list)
    
    dependency_list = [x[1] for x in input_dep_list]

    # input_dim: word embedding dimension
    model = Dependency_GAT(in_dim=len(input_tk_list), out_dim=len(input_tk_list), alpha=args.alpha, dependency_list=dependency_list)
    """
    print(model)
    ->GAT(
      (weight): Linear(in_features=7, out_features=7, bias=False)
      (attn_weight): Linear(in_features=14, out_features=1, bias=False)
      (softmax): Softmax(dim=1)
      (leakyrelu): LeakyReLU(negative_slope=0.1)
    )
    """
    output = model(input_rep, input_dep_list)

    print(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--nlp_pipeline", default="stanza", type=str, help="NLP preprocessing pipeline.")
    parser.add_argument("--alpha", default=0.01, type=float, help="Negative slope that controls the angle of the negative slope of LeakyReLU")

    args = parser.parse_args()

    main(args)
