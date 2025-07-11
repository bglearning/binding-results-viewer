import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


ATTRIBUTES_ALL = [
    'color', 'scaling', 'fracture', 'rotation', 'thick_thinning',
    'skip1', 'skip2', 'object', 'swelling', 
]
COLORS = sns.color_palette("bright", len(ATTRIBUTES_ALL))
ATTR_TO_COLOR = {a: c for a, c in zip(ATTRIBUTES_ALL, COLORS)}

def plot_fig_new(
    df,
    ax,
    xlabel, ylabel, title,
    xlims, ylims,
    xtick=True,
    dist='test-in',
    chance_level=None,
    second_avg_df=None,
    avg_labels=None
):

    attributes = [a for a in df['attribute'].unique() if a != 'Average']

    df = df.sort_values(by='x')

    # Plot each attribute
    for i, attr in enumerate(attributes):
        attr_data = df[df['attribute'] == attr].copy()

        attr_data['y'] = np.where(attr_data['low_recognition'], np.nan, attr_data['y'])
        
        ax.plot(attr_data['x'], attr_data['y'], 
                color=ATTR_TO_COLOR[attr], linewidth=2.5, linestyle='-', 
                marker='o', markersize=7, alpha=0.6,
                markeredgewidth=2,
                label=attr)

        ax.fill_between(attr_data['x'], 
                        attr_data['y'] - 1.96 * attr_data['error'], 
                        attr_data['y'] + 1.96 * attr_data['error'],
                        color=ATTR_TO_COLOR[attr], alpha=0.1)

    avg_data = df[df['attribute'] == 'Average']

    avg_label = 'Average' if avg_labels is None else avg_labels[0]

    ax.plot(avg_data['x'], avg_data['y'], 
            color='black', linewidth=6, linestyle='-', 
            marker='D', markersize=10, markerfacecolor='white', 
            markeredgecolor='black', markeredgewidth=2,
            label=avg_label, zorder=10)
    ax.fill_between(avg_data['x'], 
                    avg_data['y'] - 1.96 * avg_data['error'], 
                    avg_data['y'] + 1.96 * avg_data['error'],
                    color='black', alpha=0.1, zorder=5)

    if second_avg_df is not None:
        second_avg_df = second_avg_df.sort_values(by='x')
        second_avg_data = second_avg_df[second_avg_df['attribute'] == 'Average']
        second_avg_label = 'Average-2' if avg_labels is None else avg_labels[1]
        ax.plot(second_avg_data['x'], second_avg_data['y'], 
                color='indigo', linewidth=6, linestyle='--', 
                marker='D', markersize=10, markerfacecolor='white', 
                markeredgecolor='indigo', markeredgewidth=2,
                label=second_avg_label, zorder=10)
        ax.fill_between(second_avg_data['x'], 
                        second_avg_data['y'] - 1.96 * second_avg_data['error'], 
                        second_avg_data['y'] + 1.96 * second_avg_data['error'],
                        color='black', alpha=0.1, zorder=5)


    # Add chance level line (no text label, will be in legend)
    if chance_level is not None:
        ax.axhline(y=chance_level, color='gray', linestyle='--', linewidth=2, alpha=0.8, zorder=1)

    # Customize the plot
    ax.set_xlabel(xlabel, fontsize=26, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=26, fontweight='bold')
    ax.set_title(title, fontsize=30, fontweight='bold', pad=20)

    # Set axis limits and ticks
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.spines[['right', 'top']].set_visible(False)
    # ax.set_xticks([0.1, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis='both', which='major', labelsize=24)
    if xtick:
        # ticks = [(x if x != 0.95 else None) for x in data_df.index.unique().tolist()]
        ticks = [round(x, 2) for x in df['x'].unique().tolist()]
        tick_labels = [(f'{x:.2f}' if x != 0.95 else '') for x in ticks]
        ax.set_xticks(ticks, labels=tick_labels)
    # ax.set_yticks(np.arange(0.5, 1.1, 0.1))

    # Improve grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Create legend with recognition caveat explanation
    legend_elements = []
    # Add attribute lines to legend (all with circle markers, different colors)
    for i, attr in enumerate(attributes):
        legend_elements.append(plt.Line2D([0], [0], color=ATTR_TO_COLOR[attr], 
                                        linestyle='-', marker='o',
                                        markersize=8, linewidth=3,
                                        label=attr.replace('_', ' ').title()))

    # Add average line (diamond marker)
    legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='-', 
                                    marker='D', markersize=10, linewidth=8,
                                    markerfacecolor='white', markeredgecolor='black',
                                    label=avg_label))
    if second_avg_df is not None:
        second_avg_label = 'Average-2' if avg_labels is None else avg_labels[1]
        legend_elements.append(plt.Line2D([0], [0], color='indigo', linestyle='--', 
                                        marker='D', markersize=10, linewidth=8,
                                        markerfacecolor='white', markeredgecolor='indigo',
                                        label=second_avg_label))

    # Add chance level (square marker)
    if chance_level is not None:
        legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle='--', 
                                        marker=None, markersize=8, linewidth=3,
                                        markerfacecolor='gray', markeredgecolor='gray',
                                        label='Chance Level'))

    # Add recognition caveat indicators
    # legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle=':', 
    #                                 marker='o', markersize=8, linewidth=1.5,
    #                                 markerfacecolor='white', markeredgecolor='gray',
    #                                 markeredgewidth=2, alpha=0.6,
    #                                 label='Low Recognition (<0.5)'))

    # Create two-column legend
    # legend1 = ax.legend(handles=legend_elements[:-1], loc='upper left', 
    legend1 = ax.legend(handles=legend_elements, loc='lower left', 
                    bbox_to_anchor=(0.02, 0.01), frameon=True, 
                    fancybox=True, shadow=True, ncol=1, fontsize=20)
    legend1.get_frame().set_facecolor('white')
    legend1.get_frame().set_alpha(0.9)

    # Add recognition caveat legend separately
    # legend2 = ax.legend(handles=[legend_elements[-1]], loc='lower right', 
    #                 bbox_to_anchor=(0.98, 0.02), frameon=True,
    #                 fancybox=True, shadow=True, fontsize=11)
    # legend2.get_frame().set_facecolor('lightyellow')
    # legend2.get_frame().set_alpha(0.9)

    # Add both legends to the plot
    ax.add_artist(legend1)

    # Add statistical annotation box
    # textstr = '\n'.join([
    #     'Error bars: ±1 SEM',
    #     'n = 100 per condition',
    #     'Hollow markers: recognition < 0.6'
    # ])
    # props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    # ax.text(0.75, 0.25, textstr, transform=ax.transAxes, fontsize=10,
    #         verticalalignment='top', bbox=props)

    # Improve overall appearance
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    # Add subtle background color
    # ax.set_facecolor('#fafafa')