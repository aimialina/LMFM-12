# Visualization code of the Lightweight and Complex Models
# Comment/Uncomment for any of it

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def load_model_data(csv_path, model_name=None):
    df = pd.read_csv(csv_path)

    if model_name is None:
        parts = csv_path.split("/")
        for part in parts:
            if any(model in part for model in ['ShuffleNetV2', 'MobileNetV2', 'DenseNet121']):
            #if any(model in part for model in ["EfficientNetV2", "ResNeXt50", "ConvNext_Base", "VGG19"]):
                model_name = part
                break
        if model_name is None:
            model_name = os.path.basename(os.path.dirname(csv_path))

    df["model"] = model_name
    return df

def get_top_two_layers(df, model_name):
    # for Lightweight Models
    top_layers_map = {
        'ShuffleNetV2': ['stage4.0', 'conv5'],
        'MobileNetV2': ['blocks.5', 'conv_head'],  
        'DenseNet121': ['features.denseblock3', 'features.denseblock4']
    }
    
    # for Complex Models
    #top_layers_map = {
    #    "EfficientNetV2": ["features.5", "features.7"],
    #    "ResNeXt50"     : ["layer3.0", "layer4.0"],
    #    "ConvNext_Base" : ["stage.2", "stage.3"],
    #    "VGG19"         : ["features.30", "features.43"],
    #}

    model_type = None
    for key in top_layers_map:
        if key in model_name:
            model_type = key
            break

    if model_type is None:
        layers = df['layer'].unique()
        layers = sorted(layers, key=lambda x: int(x.split('.')[-1]) if x.split('.')[-1].isdigit() else 0)
        return layers[-2:] if len(layers) >= 2 else layers

    existing_top_layers = [layer for layer in top_layers_map[model_type] if layer in df['layer'].unique()]
    
    if len(existing_top_layers) < 2:
        all_layers = df['layer'].unique()
        additional_layers = [l for l in all_layers if l not in existing_top_layers]
        additional_layers.sort(key=lambda x: int(x.split('.')[-1]) if x.split('.')[-1].isdigit() else 0)
        existing_top_layers.extend(additional_layers[-(2-len(existing_top_layers)):])
    
    return existing_top_layers

def generate_stats_text(model_paths, model_top_layers, plot_data, comparisons, labels=None):
    if comparison_labels is None:
        comparison_labels = {
            "baseline": "Baseline",
            "ft_vs_rd": "FT vs Random",
            "fb_vs_rd": "FB vs Random",
            "ft_vs_fb": "FT vs FB",
            "pretrained_vs_random": "Pretrained vs Random",
        }

    stat_text = "Top 2 Layers Used:\n"
    for name, layers in model_top_layers.items():
        stat_text += f"{name}: {', '.join(layers)}\n"

    text += "\nStatistical Summary:\n"

    if isinstance(plot_data, list) and 'pretrained_vs_random' in plot_data[0]:
        for model_data in plot_data:
            model = model_data['model']
            baseline_mean = np.mean(model_data['baseline'])
            baseline_std = np.std(model_data['baseline'])
            pretrained_mean = np.mean(model_data['pretrained_vs_random'])
            pretrained_std = np.std(model_data['pretrained_vs_random'])
            
            stat_text += f"{model}:\n"
            stat_text += f"  Baseline: mean={baseline_mean:.3f}, std={baseline_std:.3f}, n={len(model_data['baseline'])}\n"
            stat_text += f"  Pretrained vs Random: mean={pretrained_mean:.3f}, std={pretrained_std:.3f}, n={len(model_data['pretrained_vs_random'])}\n"
    
    else:
        for model_name in model_paths.keys():
            stat_text += f"{model_name}:\n"
            
            for comparison in comparisons:
                if model_name in plot_data and comparison in plot_data[model_name]:
                    data = plot_data[model_name][comparison]
                    if len(data) > 0:
                        mean = np.mean(data)
                        std = np.std(data)
                        stat_text += f"  {comparison_labels[comparison]}: mean={mean:.3f}, std={std:.3f}, n={len(data)}\n"
    
    return stat_text

def create_detailed_comparison(model_paths, output_dir, include_ft_vs_fb=False):
    comparisons = ['baseline', 'ft_vs_rd', 'fe_vs_rd']
    if include_ft_vs_fb:
        comparisons.append('ft_vs_fb')

    comparison_labels = {
        'baseline': 'Baseline',
        'ft_vs_rd': 'FT vs Random',
        'fb_vs_rd': 'FB vs Random',
        'ft_vs_fb': 'FT vs FB'
    }
    
    colors = {
        'baseline': '#d62728',     
        'ft_vs_rd': '#1f77b4',     
        'fb_vs_rd': '#2ca02c',     
        'ft_vs_fb': '#ff7f0e', 
    }
    
    all_data = []
    model_top_layers = {}
    
    for model_name, path in model_paths.items():
        df = load_model_data(path, model_name)
        top_layers = get_top_two_layers(df, model_name)
        model_top_layers[model_name] = top_layers
        print(f"Model: {model_name}, Top layers: {top_layers}")
        df_top = df[df['layer'].isin(top_layers)]
        all_data.append(df_top)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    num_models = len(model_paths)
    num_comparisons = len(comparisons)
    model_width = 0.75  
    bar_width = model_width / num_comparisons 
    
    x_positions = np.arange(num_models)  
    
    plot_data = {}
    for model_name in model_paths.keys():
        plot_data[model_name] = {}
        df_model = combined_df[combined_df['model'] == model_name]
        
        for comparison in comparisons:
            plot_data[model_name][comparison] = df_model[df_model['comparison'] == comparison]['score'].values
    
    for i, model_name in enumerate(model_paths.keys()):
        for j, comparison in enumerate(comparisons):
            pos = x_positions[i] + (j - num_comparisons/2 + 0.5) * bar_width
            bp = ax.boxplot([plot_data[model_name][comparison]], 
                           positions=[pos], 
                           widths=[bar_width * 0.8],
                           patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(colors[comparison])
                patch.set_alpha(0.7)
                patch.set_edgecolor('none')   
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(['ShuffleNet-V2', 'MobileNet-V2', 'DenseNet-121'])
    # ax.set_xticklabels(['EfficientNet-V2', 'ResNeXt-50', 'ConvNeXt-Base', 'VGG-19'])
    ax.set_ylabel('SVCCA Similarity Score', fontsize=18, fontweight='bold', labelpad=16)
    ax.set_xlabel('Lightweight Model', fontsize=20, fontweight='bold', labelpad=20)
    # ax.set_xlabel('Complex Model', fontsize=20, fontweight='bold', labelpad=20)
    ax.tick_params(axis="x", labelsize=17)  
    ax.tick_params(axis="y", labelsize=17)   

    ax.set_ylim(0, 0.6)
    
    legend_elements = [
        Patch(facecolor=colors[comp], alpha=0.7, label=comparison_labels[comp])
        for comp in comparisons
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=17)
    
    stat_text = generate_stats_text(model_paths, model_top_layers, plot_data, comparisons, comparison_labels)
    
    os.makedirs(output_dir, exist_ok=True)
    
    stats_filename = "l_detailed_comparison_stats.txt"
    if include_ft_vs_fb:
        stats_filename = "l_full_comparison_stats.txt"
    stats_path = os.path.join(output_dir, stats_filename)
    
    with open(stats_path, 'w') as f:
        f.write(stat_text)
    print(f"Statistics saved to {stats_path}")
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    
    # For lightweight
    fig_filename = "svcca_alightweight_top2layers.png" # rename for complex models
    if include_ft_vs_fb:
        fig_filename = "lightweight_full_comparison.png"
    fig_path = os.path.join(output_dir, fig_filename)
    
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {fig_path}")
    
    return fig


if __name__ == "__main__":
    model_paths = {
        'ShuffleNetV2': '/home/aimi/pub_7_models/finalized/part_b_svcca/svcca_fig2/new/ShuffleNet/svcca_individual_scores_ShuffleNet.csv',
        'MobileNetV2': '/home/aimi/pub_7_models/finalized/part_b_svcca/svcca_fig2/new/MobileNet/svcca_individual_scores_MobileNet.csv',
        'DenseNet121': '/home/aimi/pub_7_models/finalized/part_b_svcca/svcca_fig2/new/DenseNet-121/svcca_individual_scores_DenseNet-121.csv',  
    }

    #model_paths = {
    #    "EfficientNet-V2": "/path/to/EfficientNet/svcca_individual_scores_EfficientNet.csv",
    #    "ResNeXt-50": "/path/to/ResNet50/svcca_individual_scores_ResNet50.csv",
    #    "ConvNeXt": "/path/to/ConvNext/svcca_individual_scores_ConvNext.csv",
    #    "VGG-19": "/path/to/VGG19/svcca_individual_scores_VGG19.csv",
    #}

    output_dir = "/path/to/output/dir"
    os.makedirs(output_dir, exist_ok=True)

    create_detailed_comparison(model_paths, output_dir)
    
    print("All visualizations completed successfully!")