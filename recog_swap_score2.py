import json
import sys

import pandas as pd

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
            obj_result_vals = list(d['object']['original_index_to_rank'].values())
            obj_df = pd.DataFrame(
                obj_result_vals, 
                index=list(d['object']['original_index_to_rank'].keys()),
                columns=[
                    'rank_org', 'rank_sw', 'rank_org1', 'rank_org2',
                    'same_attr', 'same_obj'
                ]
            )
            obj_df['obj_detected'] = (
                (obj_df['rank_org1'] == 0) & (obj_df['rank_org2'] == 0)
            )

            for attr in d:
                result_vals = list(d[attr]['original_index_to_rank'].values())
                df = pd.DataFrame(
                    result_vals, 
                    index=list(d[attr]['original_index_to_rank'].keys()),
                    columns=[
                        'rank_org', 'rank_sw', 'rank_org1', 'rank_org2',
                        'same_attr', 'same_obj'
                    ]
                )
                df['obj_detected'] = obj_df['obj_detected']
                recog_acc = (
                    (df['rank_org1'] == 0).sum() + 
                    (df['rank_org2'] == 0).sum()
                ) / (2 * len(df))
                base_cond = ((df['same_attr'] != 1) & (df['same_obj'] != 1))

                swap_cond_df = df[
                    (df['rank_org1'] == 0)
                    & (df['rank_org2'] == 0)
                    & base_cond
                ].copy()

                swap_obj_cond_df = df[
                    (df['rank_org1'] == 0)
                    & (df['rank_org2'] == 0)
                    & (df['obj_detected'])
                    & base_cond
                ].copy()

                swap_df = df[
                    base_cond
                ].copy()

                wandb_results[wandb_name][attr] = {
                    'recog_acc': recog_acc,
                    'swap_acc_cond': (swap_cond_df['rank_org'] < swap_cond_df['rank_sw']).sum() / len(swap_cond_df),
                    'swap_acc_cond_obj': (swap_obj_cond_df['rank_org'] < swap_obj_cond_df['rank_sw']).sum() / len(swap_obj_cond_df),
                    'swap_acc_uncond': (swap_df['rank_org'] < swap_df['rank_sw']).sum() / len(swap_df),
                }

with open(out_file, 'w') as f:
    json.dump(wandb_results, f)
