import stanza
import spacy

def preprocessing(_input_text, _nlp_pipeline):
    input_tk_list = ["ROOT"]
    input_dep_list = []

    if _nlp_pipeline == "stanza":
        nlp = stanza.Pipeline('en')
        text = nlp(_input_text)

        for sen in text.sentences:
            for tk in sen.tokens:
                tk_infor_dict = tk.to_dict()[0]
                cur_tk = tk_infor_dict["text"]

                cur_id = tk_infor_dict['id']
                cur_head = tk_infor_dict['head']
                cur_dep = tk_infor_dict["deprel"]

                cur_dep_triple = (cur_id, cur_dep, cur_head)
                input_tk_list.append(cur_tk)
                input_dep_list.append(cur_dep_triple)
        """
        print(input_tk_list)
        -> ['ROOT', 'My', 'dog', 'likes', 'eating', 'sausage']
    
        print(input_dep_list)
        -> [(1, 'nmod:poss', 2), (2, 'nsubj', 3), (3, 'root', 0), (4, 'xcomp', 3), (5, 'obj', 4)]
        """

    elif _nlp_pipeline == "spacy":
        nlp = spacy.load("en_core_web_sm")
        text = nlp(_input_text)

        for tk_idx, tk in enumerate(text):
            cur_tk = tk.text

            cur_id = tk_idx+1
            cur_dep = tk.dep_

            if cur_dep == "ROOT":
                cur_head = 0
            else:
                cur_head = tk.head.i+1

            cur_dep_triple = (cur_id, cur_dep, cur_head)
            input_tk_list.append(cur_tk)
            input_dep_list.append(cur_dep_triple)
        """
        print(input_tk_list)
        -> ['ROOT', 'My', 'dog', 'likes', 'eating', 'sausage']
    
        print(input_dep_list)
        -> [(1, 'poss', 2), (2, 'nsubj', 3), (3, 'ROOT', 0), (4, 'xcomp', 3), (5, 'dobj', 4)]
        """
    return input_tk_list, input_dep_list
