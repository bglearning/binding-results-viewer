import json
import sys

from pathlib import Path

results_dir = Path(sys.argv[1])

out_file = Path(sys.argv[2])

to_replace = '_test-in'
if 'test-out' in sys.argv[2]:
    to_replace = '_test-out'

wandb_results = {}

for result_file in results_dir.glob("*.json"):
    if 'attr-recog-swap-binary' in result_file.name:
        wandb_name = result_file.stem.replace('attr-recog-swap-binary__', '').replace(to_replace, '')
        print(wandb_name)
        wandb_results[wandb_name] = {}
        with open(result_file) as f:
            d = json.load(f)
            for attr in d:
                rank_tuples = list(d[attr]['original_index_to_rank'].values())
                n_recog = len([t for t in rank_tuples if 0 in t])
                n_swap_cond = len([t for t in rank_tuples if ((0 in t) and (t[0] < t[1]))])
                n_swap = len([t for t in rank_tuples if (t[0] < t[1])])
                wandb_results[wandb_name][attr] = {
                    'recog_acc': n_recog / len(rank_tuples),
                    'swap_acc_cond': n_swap_cond / n_recog,
                    'swap_acc_uncond': n_swap / len(rank_tuples),
                }

with open(out_file, 'w') as f:
    json.dump(wandb_results, f)
