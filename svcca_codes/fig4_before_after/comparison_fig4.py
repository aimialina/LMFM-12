import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
from collections import OrderedDict
from torch.utils.data import DataLoader
import scipy.linalg as la
from torchvision import transforms
from torchvision.datasets import ImageFolder

model_csvs = {
    "svcca_a_ShuffleNet-V2"            : "/ShuffleNet-V2/figure4_aligned_results.csv",
    "svcca_b_MobileNet-V2"             : "/MobileNet-V2/figure4_aligned_results.csv",
    "svcca_c_DenseNet-121"             : "/DenseNet-121/figure4_aligned_results.csv",
    "svcca_d_EfficientNet-V2 (Small)"  : "/EfficientNet-V2 (Small)/figure4_aligned_results.csv",
    "svcca_e_ResNeXt-50"               : "/ResNeXt-50/figure4_aligned_results.csv",
    "svcca_f_ConvNeXt-Base"            : "/ConvNext-Base/figure4_aligned_results.csv",
    "svcca_g_VGG-19"                   : "/VGG-19/figure4_aligned_results.csv"
}

output_dir        = '/path/to/output/directory/'
os.makedirs(output_dir, exist_ok=True)


def save_legend(ax, output_path):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        print("⚠️ No legend handles found.")
        return
    fig_legend = plt.figure(figsize=(10, 6)) 
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis("off")

    #ax_legend.set_title("Legend", fontsize=17, fontweight='bold')

    legend = ax_legend.legend(
        handles,
        labels,
        loc="center",
        frameon=False,
        fontsize=16,
        ncol=1, 
    )

    fig_legend.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig_legend)
    print(f"Legend saved: {output_path}")


# Loop models 
for model_name, csv_path in model_csvs.items():
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(df['layer']))

    ax.plot(x, df['ft'], 'o-', label='FT (Fully fine-tuning)', linewidth=2.5, markersize=8, color='#1f77b4')
    ax.plot(x, df['fe'], 's-', label='FB (Frozen Backbone)', linewidth=2.5, markersize=8, color='#2ca02c')
    ax.plot(x, df['rd'], '^-', label='RD (Random)', linewidth=2.5, markersize=8, color='#d62728')

    ax.plot(x, df['ft_minus_rd'], ':', label='FT - RD', linewidth=2, color='grey')
    ax.plot(x, df['fe_minus_rd'], '--', label='FB - RD', linewidth=2, color='darkgrey')

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel("Layers", fontsize=18, fontweight='bold')
    ax.set_ylabel("SVCCA Similarity Score", fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['layer'], ha="center") 
    ax.tick_params(axis="x", labelsize=17)  
    ax.tick_params(axis="y", labelsize=17) 

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)

    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{model_name.replace(' ', '_')}_svcca.png")
    plt.savefig(save_path, dpi=300)

    legend_path = os.path.join(output_dir, "svcca_h_legend.png")
    save_legend(ax, legend_path)

    print(f"Saved: {save_path}")