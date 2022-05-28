import torch
import torch.nn as nn
import torch.nn.functional as F

class Dependency_GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, dropout_rate=0.1):
        super(Dependency_GATLayer, self).__init__()
        # in_dim: number of tokens
        # out_dim: dimension of word embedding
        # dependency_list: the entire dependency types
        # reverse_case (default=True): Considering not only the result of dependency representation but also the reversed dependency representation
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Linear(out_dim, out_dim, bias=False)
        self.attn_weight = nn.Linear(out_dim*2, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def self_loop(self, _input, dependency_triples):
        self_loop_dict = {0:torch.zeros(self.out_dim)}
        h_dict = {0:torch.zeros(self.out_dim)}
        
        for dep_triple in dependency_triples:
            cur_governor = dep_triple[2]
            cur_dependent = dep_triple[0]
            
            self_loop_dict[cur_dependent] = self.weight(_input[cur_governor].T)
            h_dict[cur_dependent] = self.weight(_input[cur_governor].T)
            
        return self_loop_dict, h_dict

    def self_attn_mechanism(self, _input, dependency_triples):
        e_tensor = torch.zeros(len(_input),len(_input))
        
        # egde attention
        for dep_triple in dependency_triples:
            cur_governor = dep_triple[2]
            cur_dependent = dep_triple[0]
            
            e_governor_dependent = self.attn_weight(torch.cat((self.weight(_input[cur_governor].T), self.weight(_input[cur_dependent].T)), -1))
            e_tensor[cur_governor, cur_dependent] = e_governor_dependent

        # Normalize edge attention
        for dep_triple in dependency_triples:
            cur_governor = dep_triple[2]
            cur_dependent = dep_triple[0]
            
            # masked attention
            zero_attn_mask = -1e18*torch.ones_like(e_tensor[cur_governor])
            masked_e = torch.where(e_tensor[cur_governor] > 0, e_tensor[cur_governor], zero_attn_mask)
            e_tensor[cur_governor] = self.softmax(masked_e.view(1,len(masked_e)))
        
        return e_tensor
        

    def forward(self, _input, dependency_triples, is_dropout=True):
        # self loop of each token
        self_loop_dict, h_dict = self.self_loop(_input, dependency_triples)

        # normalized attention score of each token
        attn_score_tensor = self.self_attn_mechanism(_input, dependency_triples)

        # Weighted sum based on the final attention weight
        for dep_triple in dependency_triples:
            cur_governor = dep_triple[2]
            cur_dependent = dep_triple[0]

            cur_attn = attn_score_tensor[cur_governor, cur_dependent] * self.weight(_input[cur_dependent])
            h_dict[cur_governor] += cur_attn
        
        output_list = list(h_dict.values())
        
        if is_dropout:
            return self.dropout(self.leakyrelu(torch.stack(output_list)))
        
        return self.leakyrelu(torch.stack(output_list))
    
class Dependency_GAT(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, num_layers=1):
        super(Dependency_GAT, self).__init__()
        # in_dim: number of tokens
        # out_dim: dimension of word embedding
        # dependency_list: the entire dependency types
        # reverse_case (default=True): Considering not only the result of dependency representation but also the reversed dependency representation
        
        self.num_layers = num_layers
        self.gat_layer = []
        for i in range(num_layers):
            self.gat_layer.append(Dependency_GATLayer(in_dim, out_dim, alpha))
        

    def forward(self, _input, dependency_triples):
        output = self.gat_layer[0](_input, dependency_triples)
        if self.num_layers > 1:
            for i in range(self.num_layers-2):
                output = self.gat_layer[i+1](output, dependency_triples)
            output = self.gat_layer[self.num_layers-1](output, dependency_triples, False)
        
        return output
