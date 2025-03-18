#from transformers import T5Tokenizer, T5ForConditionalGeneration
import rdkit.Chem as Chem
from dockstring import load_target
import re
from tqdm import trange
import json
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

with open('multitext_out_greedy.json', 'r') as o:
    smiles_dict = json.load(o)

print("loaded smiles dict, starting run now..")

#class StructureOptimizationError(Exception):
#    """Error raised during structure optimization"""
#    pass

def get_docking_scores(out_dict):

    keys = [k for k in out_dict.keys()]
    scores = {}
    for i in trange(len(keys)):
        k = keys[i]
        target = load_target(k)
        print('loaded target')
        for o in trange(len(out_dict[k])):
            RDLogger.DisableLog('rdApp.*')
            score, aux = target.dock(out_dict[k][o])
            scores[k] = (o, score)
        with open('scores_dict_maxiter1000.json', 'w') as m:
            json.dump(scores, m)
    return scores

scores = get_docking_scores(smiles_dict)

with open('scores2_maxiter1000.json', 'w') as m:
    json.dump(scores, m)

print("done")
