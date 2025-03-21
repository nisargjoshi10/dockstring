import rdkit.Chem as Chem
from dockstring import load_target
from tqdm import trange
import json
from rdkit import RDLogger
import multiprocessing

RDLogger.DisableLog('rdApp.*')

def get_docking_scores(sub_dict, queue, process_id):
    scores = {}
    for k in trange(len(sub_dict), desc=f"Process {process_id}"):
        target_name = list(sub_dict.keys())[k]
        target = load_target(target_name)
        for smile in trange(len(sub_dict[target_name])):
            score, aux = target.dock(sub_dict[target_name][smile], num_cpus=16)
            scores.setdefault(target_name, []).append(score)

    queue.put((process_id, scores))  # return result via queue


if __name__ == "__main__":
    with open('multitext_out_beamsearch_temp1.json', 'r') as o:
        smiles_dict = json.load(o)

    print("loaded smiles dict, starting run now..")

    keys = list(smiles_dict.keys())
    chunks = [keys[i:i + 16] for i in range(0, len(keys), 16)]
    sub_dicts = [{k: smiles_dict[k] for k in chunk} for chunk in chunks]

    queue = multiprocessing.Queue()
    processes = []

    for idx, sub_dict in enumerate(sub_dicts):
        p = multiprocessing.Process(target=get_docking_scores, args=(sub_dict, queue, idx))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Gather results
    all_scores = {}
    while not queue.empty():
        proc_id, scores = queue.get()
        all_scores.update(scores)
        with open(f'scores_proc_{proc_id}.json', 'w') as f:
            json.dump(scores, f)

    with open('all_scores_combined.json', 'w') as f:
        json.dump(all_scores, f)

    print("done")

