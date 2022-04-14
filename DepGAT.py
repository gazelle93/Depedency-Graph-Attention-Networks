import torch
import torch.nn as nn
import torch.nn.functional as F

class Dependency_GAT(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, dependency_list):
        super(Dependency_GAT, self).__init__()
        # dim: dimension of dependency weight
        # dependency_list: the entire dependency types
        # reverse_case (default=True): Considering not only the result of dependency representation but also the reversed dependency representation
        
        """
        - Text:
            My dog likes eating sausage.
            
        - Universal dependencies: 
            nmod:poss(dog-2, My-1)
            nsubj(likes-3, dog-2)
            root(ROOT-0, likes-3)
            xcomp(likes-3, eating-4)
            obj(eating-4, sausage-5)
            
        * Dependency can be presented as a directed graph
                  likes
                 /     \
           (nsubj)     (xcomp)
            |             |
            dog         eating
            |             |
           (nmod:poss)  (obj)
            |             |
            My          sausage
        """
        self.weight = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_weight = nn.Linear(out_dim*2, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.leakyrelu = nn.LeakyReLU(alpha)
        
    def self_loop(self, _input, dependency_triples):
        self_loop_dict = {0:torch.zeros(len(_input))}
        h_dict = {0:torch.zeros(len(_input))}
        
        for dep_triple in dependency_triples:
            cur_governor = dep_triple[2]
            cur_dependency = dep_triple[1]
            cur_dependent = dep_triple[0]
            
            self_loop_dict[cur_dependent] = self.weight(_input[cur_governor])
            h_dict[cur_dependent] = self.weight(_input[cur_governor])
            
        return self_loop_dict, h_dict

    def self_attn_mechanism(self, _input, dependency_triples):
        e_tensor = torch.zeros(len(_input),len(_input))
        
        # egde attention
        for dep_triple in dependency_triples:
            cur_governor = dep_triple[2]
            cur_dependent = dep_triple[0]
            
            e_governor_dependent = self.attn_weight(torch.cat((self.weight(_input[cur_governor]), self.weight(_input[cur_dependent])), -1))
            e_tensor[cur_governor, cur_dependent] = e_governor_dependent

        # Normalize egde attention
        for dep_triple in dependency_triples:
            cur_governor = dep_triple[2]
            cur_dependent = dep_triple[0]
            
            # masked attention
            zero_attn_mask = -1e18*torch.ones_like(e_tensor[cur_governor])
            masked_e = torch.where(e_tensor[cur_governor] > 0, e_tensor[cur_governor], zero_attn_mask)
            e_tensor[cur_governor] = self.softmax(masked_e.view(1,len(masked_e)))
        
        return e_tensor

    def forward(self, _input, dependency_triples):
        # _input: tokenized input text representation in vector space
        # dependency_triples: (dependent index, dependency, governor index)
        # * dependent and governor index follows the index of _input
        
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
        output_list = self.leakyrelu(torch.stack(output_list))
        
        return output_list
