import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set page title
st.set_page_config(page_title="Clip Binding Results Explorer", layout="wide")

# Create a header
# st.title("Data Explorer with Segmented Controls")

# Create sample data if needed
# @st.cache_data
def all_runs():
    with open("wandb_results_05_19_full.json") as f:
        wandb_results = json.load(f)
    return wandb_results

def all_runs_acc():
    with open("recog-swap-accuracies_05_19_full.json") as f:
        recog_swap_accuracies = json.load(f)
    with open("recog-swap-accuracies-test-out_05_19_full.json") as f:
        recog_swap_accuracies_out = json.load(f)
    return recog_swap_accuracies, recog_swap_accuracies_out

wandb_results = all_runs()
recog_swap_accuracies, recog_swap_accuracies_out = all_runs_acc()

# Sidebar for controls
st.sidebar.header("Filter Controls")

# VARS_OF_INTEREST_ = ['TRAIN_CAPTION_MODE', 'CAPTION_OBJECT_INCLUSION_PROB', 'IMG_OBJECT_INCLUSION_PROB', 'SALIENCY_PROB']
VARS_OF_INTEREST_ = ['TRAIN_CAPTION_MODE', 'IMG_OBJECT_INCLUSION_PROB', 'SALIENCY_PROB']
BATCH_SIZES_ = [16, 32, 64, 128, 256]
EMBED_DIMS_ = [32, 64, 128, 256]
TEST_DISTS_NUM_ATTRS = ['always_three_four', 'skewed_to_one']
MAIN_METRICS_OPTIONS = ['swap_conditional', 'swap_unconditional', 'always_three_four', 'skewed_to_one']

# Extract unique values for each parameter
variables = sorted(VARS_OF_INTEREST_)
batch_sizes = sorted(BATCH_SIZES_)
embed_dims = sorted(EMBED_DIMS_)

with st.sidebar:
    TAKE_ATTRIBUTE_AVERAGE = st.checkbox("Aggregate Attrs")
    DISPLAY_ONLY_TEST_IN = st.checkbox("Only test-in")
    RECOG_ACC_LIMIT = st.number_input("Recognition Acc Cutoff", min_value=0.1, max_value=1.0, value=0.2, step=0.05)
    VAR_OF_INTEREST = st.radio(
        "Variable of Interest:",
        options=variables,
        key="var_selector"
    )

    batch_size = st.radio(
        "Batch Size:",
        options=batch_sizes,
        key="batch_selector"
    )

    embed_dim = st.radio(
        "Embedding Dimension:",
        options=embed_dims,
        key="embed_selector"
    )

    main_metric = st.radio(
        "Main Metric:",
        options=MAIN_METRICS_OPTIONS,
        key="main_metric"
    )

SWAP_METRIC = {
    'swap_conditional': 'recog-swap-cond',
    'swap_unconditional': 'recog-swap-uncond',
    'always_three_four': 'swap-mid-acc',
    'skewed_to_one': 'swap-skt1-acc',
}[main_metric]

RECOG_METRIC = {
    'recog-swap-cond': 'recog-both',
    'recog-swap-uncond': 'recog-both',
    'swap-mid-acc': 'recog-multi',
    'swap-skt1-acc': 'recog-multi',
}[SWAP_METRIC]

# Display metrics
# st.header(f"Metrics for {VAR_OF_INTEREST} (Batch Size: {batch_size}, Embed Dim: {embed_dim})")

VAR_VALUES = {
    # 'TRAIN_CAPTION_MODE': ['skewed_to_zero', 'skewed_to_one', 'high_two', 'high_two_five', 'high_five', 'skewed_to_full'],
    'TRAIN_CAPTION_MODE': ['skewed_to_zero', 'skewed_to_one', 'high_two', 'high_five', 'skewed_to_full'],
    # 'CAPTION_OBJECT_INCLUSION_PROB': [0.1, 0.25, 0.5, 0.75, 0.9, 1.],
    'CAPTION_OBJECT_INCLUSION_PROB': [0.1, 0.25, 0.5, 0.75, 1.],
    # 'IMG_OBJECT_INCLUSION_PROB': [0.1, 0.25, 0.5, 0.75, 0.9, 1.],
    'IMG_OBJECT_INCLUSION_PROB': [0.1, 0.25, 0.5, 0.75, 1.],
    'SALIENCY_PROB': [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0],
}

WANDB_TAG = {
    'TRAIN_CAPTION_MODE': 'train-caption-mode-core',
    'CAPTION_OBJECT_INCLUSION_PROB': 'cap-obj-incl-prob',
    'IMG_OBJECT_INCLUSION_PROB': 'img-obj-incl-prob',
    'SALIENCY_PROB': 'saliency',
}[VAR_OF_INTEREST]

X_LIMS = {
    'TRAIN_CAPTION_MODE': (0, 6.),
    'CAPTION_OBJECT_INCLUSION_PROB': (0, 1.),
    'IMG_OBJECT_INCLUSION_PROB': (0, 1.),
    'SALIENCY_PROB': (0, 1.),
}[VAR_OF_INTEREST]

X_LABEL = {
    'TRAIN_CAPTION_MODE': 'Expected-Num-Attributes',
    'CAPTION_OBJECT_INCLUSION_PROB': 'Caption Two-object Probability',
    'IMG_OBJECT_INCLUSION_PROB': 'Image Two-object Probability',
    'SALIENCY_PROB': 'Saliency Probability'
}[VAR_OF_INTEREST]

sel_runs = []
for r in wandb_results[WANDB_TAG]:
    r_batch_size = r['config'].get('effective_batch_size')
    if r_batch_size is None:
        r_batch_size = r['config']['batch_size']
    r_embed_dim = r['config']['clip_embed_dim']
    if r_batch_size == batch_size and r_embed_dim == embed_dim:
        sel_runs.append(r)

attrs = ['color', 'object', 'scaling', 'fracture', 'rotation', 'swelling', 'thick_thinning']

n_rows, n_cols = len(VAR_VALUES[VAR_OF_INTEREST]), len(attrs)
full_results = {
    'test-in': {
        'swap-mid-acc': pd.DataFrame([[[] for _ in range(n_cols)] for _ in range(n_rows)], columns=attrs, index=VAR_VALUES[VAR_OF_INTEREST]),
        'swap-skt1-acc': pd.DataFrame([[[] for _ in range(n_cols)] for _ in range(n_rows)], columns=attrs, index=VAR_VALUES[VAR_OF_INTEREST]),
        'recog-multi': pd.DataFrame([[[] for _ in range(n_cols)] for _ in range(n_rows)], columns=attrs, index=VAR_VALUES[VAR_OF_INTEREST]),
        'recog-both': pd.DataFrame([[[] for _ in range(n_cols)] for _ in range(n_rows)], columns=attrs, index=VAR_VALUES[VAR_OF_INTEREST]),
        'recog-swap-cond': pd.DataFrame([[[] for _ in range(n_cols)] for _ in range(n_rows)], columns=attrs, index=VAR_VALUES[VAR_OF_INTEREST]),
        'recog-swap-uncond': pd.DataFrame([[[] for _ in range(n_cols)] for _ in range(n_rows)], columns=attrs, index=VAR_VALUES[VAR_OF_INTEREST]),
    },
    'test-out': {
        'swap-mid-acc': pd.DataFrame([[[] for _ in range(n_cols)] for _ in range(n_rows)], columns=attrs, index=VAR_VALUES[VAR_OF_INTEREST]),
        'swap-skt1-acc': pd.DataFrame([[[] for _ in range(n_cols)] for _ in range(n_rows)], columns=attrs, index=VAR_VALUES[VAR_OF_INTEREST]),
        'recog-multi': pd.DataFrame([[[] for _ in range(n_cols)] for _ in range(n_rows)], columns=attrs, index=VAR_VALUES[VAR_OF_INTEREST]),
        'recog-both': pd.DataFrame([[[] for _ in range(n_cols)] for _ in range(n_rows)], columns=attrs, index=VAR_VALUES[VAR_OF_INTEREST]),
        'recog-swap-cond': pd.DataFrame([[[] for _ in range(n_cols)] for _ in range(n_rows)], columns=attrs, index=VAR_VALUES[VAR_OF_INTEREST]),
        'recog-swap-uncond': pd.DataFrame([[[] for _ in range(n_cols)] for _ in range(n_rows)], columns=attrs, index=VAR_VALUES[VAR_OF_INTEREST]),
    }
}

CAPTION_MODE_NUM_ATTR_DISTS = {
    'uniform': [1. / 7] * 7, # Uniform distribution over [0, 6]
    'full': [0.] * 6 + [1.], # Include all attributes
    'skewed_partial': [x / 120 for x in [2, 40, 32, 22, 16, 6, 2]],
    'skewed_to_zero': [x / 120 for x in [80., 24, 10, 3, 1, 1, 1]],
    'skewed_to_one': [x / 120 for x in [0., 60, 40, 8, 8, 3, 1]],
    'skewed_to_two': [x / 120 for x in [0., 40, 60, 8, 8, 3, 1]],
    'high_two': [x / 120 for x in [0., 20, 60, 8, 8, 20, 4]],
    'high_two_five': [x / 120 for x in [0., 12, 40, 8, 8, 40, 12]],
    'high_five': [x / 120 for x in [0., 4, 20, 8, 8, 60, 20]],
    'skewed_to_five': [x / 120 for x in [0., 1, 3, 8, 8, 60, 40]],
    'skewed_to_full': [x / 120 for x in [0., 1, 3, 8, 8, 40, 60]],
    'always_three_four': [0., 0., 0., 0.5, 0.5, 0., 0.],
}
EXPECTED_NUM_ATTRS = {}
for dist_name, dist in CAPTION_MODE_NUM_ATTR_DISTS.items():
    expected_num = sum(i * p for i, p in enumerate(dist))
    EXPECTED_NUM_ATTRS[dist_name] = expected_num

def get_attr_metric(recog_swap_accuracies_, wandb_name, attr, metric):
    if wandb_name not in recog_swap_accuracies_:
        return np.nan
    if attr not in recog_swap_accuracies_[wandb_name]:
        return np.nan
    return recog_swap_accuracies_[wandb_name][attr].get(metric, np.nan)


for test_dist in ('test-in', 'test-out'):
    for r in sel_runs:
        for attr in attrs:
            var_of_interest_val = r['config'][VAR_OF_INTEREST]
            if var_of_interest_val not in VAR_VALUES[VAR_OF_INTEREST]:
                continue
            full_results[test_dist]['recog-multi'].loc[r['config'][VAR_OF_INTEREST], attr].append(float(r['summary'].get(f'{test_dist}_acc@1-{attr}', np.nan)))
            # the test-out is marked with no tag on the wandb key
            dist_tag = '-test-in' if test_dist == 'test-in' else ''
            full_results[test_dist]['swap-mid-acc'].loc[r['config'][VAR_OF_INTEREST], attr].append(float(r['summary'].get(f'binary-swap-mid{dist_tag}_{attr}', np.nan)))
            full_results[test_dist]['swap-skt1-acc'].loc[r['config'][VAR_OF_INTEREST], attr].append(float(r['summary'].get(f'binary-swap-skt1{dist_tag}_{attr}', np.nan)))
            if test_dist == 'test-in':
                full_results[test_dist]['recog-both'].loc[r['config'][VAR_OF_INTEREST], attr].append(float(get_attr_metric(recog_swap_accuracies, r['name'], attr, 'recog_acc')))
                full_results[test_dist]['recog-swap-cond'].loc[r['config'][VAR_OF_INTEREST], attr].append(float(get_attr_metric(recog_swap_accuracies, r['name'], attr, 'swap_acc_cond')))
                full_results[test_dist]['recog-swap-uncond'].loc[r['config'][VAR_OF_INTEREST], attr].append(float(get_attr_metric(recog_swap_accuracies, r['name'], attr, 'swap_acc_uncond')))
            else:
                full_results[test_dist]['recog-both'].loc[r['config'][VAR_OF_INTEREST], attr].append(float(get_attr_metric(recog_swap_accuracies_out, r['name'], attr, 'recog_acc')))
                full_results[test_dist]['recog-swap-cond'].loc[r['config'][VAR_OF_INTEREST], attr].append(float(get_attr_metric(recog_swap_accuracies_out, r['name'], attr, 'swap_acc_cond')))
                full_results[test_dist]['recog-swap-uncond'].loc[r['config'][VAR_OF_INTEREST], attr].append(float(get_attr_metric(recog_swap_accuracies_out, r['name'], attr, 'swap_acc_uncond')))
            # swap_skt1_acc_df.loc[r.config[VAR_OF_INTEREST], attr] = float(r.summary[f'binary-swap-skt1-test-in_{attr}'])

    if VAR_OF_INTEREST == 'TRAIN_CAPTION_MODE':
        for df_ in full_results[test_dist].values():
            df_.index = df_.index.map(EXPECTED_NUM_ATTRS).round(2)


def process_df(df_, distr):
    df_ = df_.dropna(axis=0, how='all')
    df_ = df_.dropna(axis=1, how='all')
    df_ = df_[(df_ != 0).any(axis=1)]
    df_ = df_.loc[:, (df_ != 0).any(axis=0)]
    if df_.empty:
        return df_
    if TAKE_ATTRIBUTE_AVERAGE and distr == 'test-in':
        df_new = pd.DataFrame(
            {
                'Attributes_Avg': df_[[col for col in df_.columns if col != 'object']].mean(axis=1),
                'object': df_['object']
            }
        )
        df_ = df_new
    return df_


def process_distr_df(df_, distr):
    mean_df = process_df(df_.map(lambda x: np.mean(x) if isinstance(x, list) else x), distr)
    std_df = process_df(df_.map(lambda x: np.std(x) if isinstance(x, list) else x), distr)
    return mean_df, std_df


for run in sel_runs:
    print(run['name'])

underlying_df = (
   full_results['test-in'][SWAP_METRIC].map(lambda x: ','.join([f'{v:.3f}' for v in x])).T
)
underlying_recog_df = (
   full_results['test-in'][RECOG_METRIC].map(lambda x: ','.join([f'{v:.3f}' for v in x])).T
)

print(underlying_recog_df)
print(underlying_df)

for test_dist in ('test-in', 'test-out'):
    for metric in ('swap-mid-acc', 'swap-skt1-acc', 'recog-multi', 'recog-both', 'recog-swap-cond', 'recog-swap-uncond'):
        full_results[test_dist][metric] = process_distr_df(full_results[test_dist][metric], test_dist)

mean_df = full_results['test-in'][SWAP_METRIC][0]
std_df = full_results['test-in'][SWAP_METRIC][1]

# Create and display plot
# st.header("Data Visualization")

LABEL_SIZES = 14

def plot_fig(data_df, ax, xlabel, ylabel, title, xlims, ylims, xtick=True, dist='test-in', static_at=None, errs_df=None):
    data_df = data_df.sort_index()
    if dist == 'test-out':
        data_df = data_df.copy()
        if errs_df is not None:
            errs_df = errs_df.copy()
        for col in data_df.columns:
            if col not in ('object', 'color', 'scaling'):
                data_df[col] = None
                if errs_df is not None:
                    errs_df[col] = None

    data_df.plot(ax=ax, yerr=errs_df)
    ax.set_title(title, fontsize=LABEL_SIZES, pad=15)
    if static_at:
        ax.hlines(static_at, xmin=xlims[0], xmax=xlims[1], linestyles='dashed')
    if xtick:
        # ticks = [(x if x != 0.95 else None) for x in data_df.index.unique().tolist()]
        ticks = [x for x in data_df.index.unique().tolist()]
        ax.set_xticks(ticks)
        # ax.xaxis.get_major_ticks()[-2].draw = lambda *args:None
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.legend(loc='lower left')
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZES)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZES, labelpad=15)
    # ax.grid()

if len(sel_runs) > 0:
    dists_to_plot = ['test-in'] if DISPLAY_ONLY_TEST_IN else ['test-in', 'test-out']
    # if SWAP_METRIC in ('recog-swap-cond', 'recog-swap-uncond'):
        # dists_to_plot = ['test-in']

    fig, axes = plt.subplots(2, len(dists_to_plot), figsize=(8 * len(dists_to_plot), 10))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for i, dist in enumerate(dists_to_plot):
        dist_label = f"[{dist.split('-')[-1]}]"
        mdf_recog, std_df_recog = full_results[dist][RECOG_METRIC]
        ax_ = axes[0, i] if len(dists_to_plot) > 1 else axes[0]
        plot_fig(
            mdf_recog,
            xlabel='',
            ylabel="Recognition (multi-object-scene) " if i == 0 else '',
            ax=ax_,
            title=f'{dist_label}-distribution',
            xlims=X_LIMS,
            ylims=(0, 1.01),
            xtick=True,
            dist=dist,
            errs_df=std_df_recog,
        )
        mdf, std_df = full_results[dist][SWAP_METRIC]
        mdf_ = pd.DataFrame(np.where(mdf_recog.loc[mdf.index, mdf.columns] > RECOG_ACC_LIMIT, mdf.values, np.nan), columns=mdf.columns, index=mdf.index)
        std_df_ = pd.DataFrame(np.where(mdf_recog.loc[std_df.index, std_df.columns] > RECOG_ACC_LIMIT, std_df.values, np.nan), columns=std_df.columns, index=std_df.index)
        mdf_ = mdf_.dropna(axis=0, how='all')
        mdf_ = mdf_.dropna(axis=1, how='all')
        std_df_ = std_df_.dropna(axis=0, how='all')
        std_df_ = std_df_.dropna(axis=1, how='all')
        ax_ = axes[1, i] if len(dists_to_plot) > 1 else axes[1]
        plot_fig(
            mdf_,
            # full_results[dist]['swap-mid-acc'],
            xlabel=X_LABEL,
            ylabel="Binary swap accuracy" if i == 0 else '',
            ax=ax_,
            title='',
            xlims=X_LIMS,
            ylims=(0.2, 1.0),
            xtick=True,
            dist=dist,
            static_at=0.5,
            errs_df=std_df_,
        )
    plt.suptitle(f'Effect of {X_LABEL} on Binding and Recognition\nb{batch_size} e{embed_dim}', fontsize=18)
    # plt.savefig(f'{X_LABEL}-{WANDB_TAG}_b256-E128.png', bbox_inches='tight', dpi=300)
    # plt.show()
    st.pyplot(fig)
else:
    st.write("No data to plot")

st.write("### [test-in] Underlying recog-accuracy values:")
st.dataframe(underlying_recog_df)

st.write("### [test-in] Underlying swap-accuracy values:")
st.dataframe(underlying_df)

st.write("## [test-in] Mean Values (swap-accuracy):")
st.dataframe(mean_df)

st.write("## [test-in] Std Values (swap-accuracy):")
st.dataframe(std_df)
