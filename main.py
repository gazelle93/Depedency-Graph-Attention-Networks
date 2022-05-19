import torch
import argparse

from text_processing import preprocessing
from DepGAT import Dependency_GAT

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

    in_dim = len(input_tk_list)
    out_dim = len(input_rep[0])

    # input_dim: word embedding dimension
    model = Dependency_GAT(in_dim=in_dim, out_dim=out_dim, alpha=args.alpha, num_layers=args.num_layers)
    
    output = model(input_rep, input_dep_list)

    print(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--nlp_pipeline", default="stanza", type=str, help="NLP preprocessing pipeline.")
    parser.add_argument("--num_layers", default=1, type=int, help="The number of hidden layers of GCN.")
    parser.add_argument("--alpha", default=0.01, type=float, help="Negative slope that controls the angle of the negative slope of LeakyReLU")

    args = parser.parse_args()

    main(args)
