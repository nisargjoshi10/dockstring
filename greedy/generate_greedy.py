from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import rdkit.Chem as Chem
from dockstring import load_target
import re
from tqdm import trange
import json
import torch

def get_prompts(target_file):

    with open(target_file, 'r') as  d:
        target_lists = [l.rstrip('\n') for l in d.readlines()]
        
    prompts = []
    for t in target_lists:
        txt = 'This molecule is a peptide. This molecule binds to ' + t + ' protein.'
        prompts.append(txt)
    return prompts

prompts = get_prompts('/home/prg/Nisarg_data/moljet/molt5/dockstring_proteins.txt')
#device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_mols(prompts):
    """
    Currently the model outputs two molecules with num_beams=2, later we change the number of outputs and other parameters
    in model.generate()
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = AutoModelForSeq2SeqLM.from_pretrained("GT4SD/multitask-text-and-chemistry-t5-base-augm")
    tokenizer = AutoTokenizer.from_pretrained("GT4SD/multitask-text-and-chemistry-t5-base-augm")
#    model.to(device)
#    model.eval()
    smiles, log = {}, {}
    for p in trange(len(prompts)):
        # mols = []
        instance = prompts[p]
        input_text = f"Write in SMILES the described molecule: {instance}"
        text = tokenizer(input_text, return_tensors="pt")
#        input_id = tokenizer(input_text, return_tensors="pt").to(device).input_ids
        target = re.search(r'\b([A-Za-z0-9]+)\s+protein\b', prompts[p]).group(1)
        # outputs = model.generate(input_id, num_beams=100, num_return_sequences=100,  max_length=512, min_length=64, early_stopping=True, do_sample=True, temperature=0.5)#, top_k=50, no_repeat_ngram_size=2)
        outputs = model.generate(input_ids=text["input_ids"], min_length=64, max_length=512, num_beams=1, return_dict_in_generate=True, output_scores=True)
        mols = [tokenizer.decode(i, skip_special_tokens=True) for i in outputs['sequences']]
        smiles[target] = mols
        
        transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)# ,outputs.beam_indices)\
                                                                                                            # use beam_indices if using beam search
        log_p = []
        for score in transition_scores:
            mask = torch.isfinite(score)
            log_p.append(str(score[mask].sum().numpy()))

        log[target] = log_p  
        print(len(mols) == len(set(mols)))
#        torch.cuda.empty_cache()
    return smiles, log

output = get_mols(prompts)

with open('multitext_out_greedy.json', 'w') as m:
    json.dump(output[0], m)

with open('multitext_log_greedy.json', 'w') as m:
    json.dump(output[1], m)

print("Done")
