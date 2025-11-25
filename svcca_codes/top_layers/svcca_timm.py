# SVCCA for top 2 layers (Fig 3a)
# applied for timm models (densenet121 and etc)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import timm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib
import seaborn as sns
import os
from collections import OrderedDict
from torch.utils.data import DataLoader
import scipy.linalg as la
from torchvision import transforms
from torchvision.datasets import ImageFolder

model_name_script = 'DenseNet-121'

dataset_path = "path/to/dataset"
target_dir = "path/to/output/directory"

# Provide paths of the weights accordingly
model_groups = {
    'full': {
        'ft1': '/weights_1/densenet121_timm_ft_1/best_model_densenet121_ft.pth',
        'ft2': '/weights_2/densenet121_timm_ft_2/best_model_densenet121_ft.pth',
        'ft3': '/weights_3/densenet121_timm_ft_3/best_model_densenet121_ft.pth'
    },
    'frozen': {
        'fb1': '/weights_1/densenet121_timm_fb_1/best_model_densenet121_fb.pth',
        'fb2': '/weights_2/densenet121_timm_fb_2/best_model_densenet121_fb.pth',
        'fb3': '/weights_3/densenet121_timm_fb_3/best_model_densenet121_fb.pth'
    },
    'random': {
        'rd1': '/weights_1/densenet121_timm_rd_1/best_model_densenet121_rd.pth',
        'rd2': '/weights_2/densenet121_timm_rd_2/best_model_densenet121_rd.pth',
        'rd3': '/weights_3/densenet121_timm_rd_3/best_model_densenet121_rd.pth'
    },
    'random_b': {
        'rdb1': '/weights_b/densenet121_timm_rd_1/best_model_densenet121_rd.pth',
        'rdb2': '/weights_b/densenet121_timm_rd_2/best_model_densenet121_rd.pth',
        'rdb3': '/weights_b/densenet121_timm_rd_3/best_model_densenet121_rd.pth'
    }
}


#-------------- DenseNet --------------
def load_model(path, device, num_classes):
    model = timm.create_model('densenet121', pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    return model

def get_layers():
    return [
        'features.conv0',       
        'features.denseblock1', 
        'features.denseblock2', 
        'features.denseblock3', 
        'features.denseblock4', 
    ]
    
#-------------- MobileNet --------------
#def load_model(path, device, num_classes):
#    model = timm.create_model("mobilenetv2_100", pretrained=False)
#    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
#    model.load_state_dict(torch.load(path, map_location=device))
#    model.to(device).eval()
#    return model
#
#def get_layers():
#    return [
#        'conv_stem',  
#        'blocks.1',   
#        'blocks.3',   
#        'blocks.5',   
#        'conv_head'    
#    ]

#-------------- ResNeXt --------------
#def load_model(path, device, num_classes):
#    model = timm.create_model("resnext50_32x4d", pretrained=False)
#    model.fc = nn.Linear(model.fc.in_features, num_classes)
#    model.load_state_dict(torch.load(path, map_location=device))
#    model.to(device).eval()
#    return model
#
#def get_layers():
#    return [
#        'conv1',       
#        'layer1.0',
#        'layer2.0',    
#        'layer3.0',    
#        'layer4.0',    
#    ]

#-------------- ConvNeXt --------------
#def load_model(path, device, num_classes):
#    model = timm.create_model('convnext_base', pretrained=False)
#    model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
#    model.load_state_dict(torch.load(path, map_location=device))
#    model.to(device).eval()
#    return model
#
#def get_layers():
#    return [
#        'stem',       
#        'stages.0',   
#        'stages.1',   
#        'stages.2',   
#        'stages.3',   
#    ]

#-------------------------- SVCCA --------------------------
class ModelHook:
    def __init__(self, model, layers_of_interest):
        self.model = model
        self.layers_of_interest = layers_of_interest
        self.activations = OrderedDict()
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        for name, layer in self.model.named_modules():
            if name in self.layers_of_interest:
                self.hooks.append(
                    layer.register_forward_hook(
                        lambda module, input, output, layer_name=name: 
                        self._hook_fn(module, output, layer_name)
                    )
                )
                
    def _hook_fn(self, module, output, layer_name):
        if isinstance(output, torch.Tensor):
            # Ensure proper dimensionality (flatten spatial dimensions)
            if len(output.shape) == 4:  # [batch, channels, height, width]
                output = output.permute(0, 2, 3, 1).reshape(-1, output.size(1))
            self.activations[layer_name] = output.detach().cpu().numpy()
        
    def get_activations(self, dataloader, num_batches=None):
        self.model.eval()
        collected_activations = {layer: [] for layer in self.layers_of_interest}
        
        with torch.no_grad():
            for i, (inputs, _) in enumerate(dataloader):
                if num_batches is not None and i >= num_batches:
                    break
                    
                self.model(inputs.cuda() if torch.cuda.is_available() else inputs)
                for layer_name, activation in self.activations.items():
                    collected_activations[layer_name].append(activation)
        for layer_name in self.layers_of_interest:
            if collected_activations[layer_name]:
                collected_activations[layer_name] = np.concatenate(
                    collected_activations[layer_name], axis=0
                )
        
        return collected_activations
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def compute_svcca(activations1, activations2, epsilon=1e-6):
    if len(activations1.shape) > 2:
        activations1 = activations1.reshape(activations1.shape[0], -1)
    if len(activations2.shape) > 2:
        activations2 = activations2.reshape(activations2.shape[0], -1)
    
    act1_centered = activations1 - np.mean(activations1, axis=0)
    act2_centered = activations2 - np.mean(activations2, axis=0)
    
    # 1: SVD for dimensionality reduction
    try:
        u1, s1, vh1 = la.svd(act1_centered.T, full_matrices=False)
        u2, s2, vh2 = la.svd(act2_centered.T, full_matrices=False)
    except np.linalg.LinAlgError:
        print("SVD failed")
        return 0.0
    
    variance_explained1 = np.cumsum(s1**2) / np.sum(s1**2)
    variance_explained2 = np.cumsum(s2**2) / np.sum(s2**2)
    k1 = np.where(variance_explained1 >= 0.99)[0][0] + 1 if np.any(variance_explained1 >= 0.99) else len(s1)
    k2 = np.where(variance_explained2 >= 0.99)[0][0] + 1 if np.any(variance_explained2 >= 0.99) else len(s2)
    k = min(k1, k2, u1.shape[1], u2.shape[1])
    # Project to reduced subspace - using vh (right singular vectors)
    svd_acts1 = act1_centered @ u1[:k].T  # [samples, k]
    svd_acts2 = act2_centered @ u2[:k].T  # [samples, k]
    
    # 2: CCA on reduced representations
    try:
        # Compute covariance matrices
        cov11 = np.cov(svd_acts1.T) + epsilon * np.eye(k)
        cov22 = np.cov(svd_acts2.T) + epsilon * np.eye(k)
        cov12 = np.cov(svd_acts1.T, svd_acts2.T)[:k, k:]
        # CCA via eigenvalue decomposition
        cov11_sqrt_inv = la.inv(la.sqrtm(cov11))
        cov22_sqrt_inv = la.inv(la.sqrtm(cov22))
        # Compute canonical correlations
        T = cov11_sqrt_inv @ cov12 @ cov22_sqrt_inv
        canonical_correlations = la.svd(T, compute_uv=False)
        # Ensure correlations are in valid range [0, 1]
        canonical_correlations = np.minimum(canonical_correlations, 1.0)
        canonical_correlations = np.maximum(canonical_correlations, 0.0)
        # SVCCA similarity is mean of canonical correlations
        svcca_similarity = np.mean(canonical_correlations)
        
    except np.linalg.LinAlgError:
        print("CCA computation failed, using fallback")
        # Fallback: simple correlation coefficient
        svcca_similarity = np.abs(np.corrcoef(
            svd_acts1.flatten(), 
            svd_acts2.flatten()
        )[0, 1])
    
    return svcca_similarity

def get_activations(model, layer, dataloader):
    """Extract activations for a specific layer"""
    hook = ModelHook(model, [layer])
    activations = hook.get_activations(dataloader, num_batches=64)
    hook.remove_hooks()
    return activations[layer]

#-------------------------- Visualization --------------------------
def calculate_baseline_comparison(model_groups, dataloader, device, output_dir, num_classes=12):
    layers = get_layers()
    layer_results = {layer: {} for layer in layers}

    comparisons = {
        'baseline': ('random', 'random_b', 'rd', 'rdb'),
        'ft_vs_rd': ('full', 'random', 'ft', 'rd'),
        'fb_vs_rd': ('frozen', 'random', 'fb', 'rd'),
        'ft_vs_fb': ('full', 'frozen', 'ft', 'fb')
    }

    for layer in layers:
        print(f'\nProcessing layer: {layer}')

        for each_comp, (group1, group2, prefix1, prefix2) in comparisons.items():
            scores = []

            for i in range(1, 4):
                model1_path = model_groups[group1][f'{prefix1}{i}']
                model2_path = model_groups[group2][f'{prefix2}{i}']

                model1 = load_model(model1_path, device, num_classes)
                model2 = load_model(model2_path, device, num_classes)

                hook1 = ModelHook(model1, [layer])
                activation1 = hook1.get_activations(dataloader, num_batches=64)
                hook1.remove_hooks()

                hook2 = ModelHook(model2, [layer])
                activation2 = hook2.get_activations(dataloader, num_batches=64)
                hook2.remove_hooks()

                similarity = compute_svcca(activation1[layer], activation2[layer])
                scores.append(similarity)

            layer_results[layer][each_comp] = {
                'scores': scores,
                'mean': np.mean(scores)
            }

            clean_scores = [float(s) for s in scores]
            print(f"{each_comp}: {clean_scores}, mean={np.mean(scores):.3f}")
    
    plot_figure2_boxplot_detailed(layer_results, output_dir)
    plot_line_all_comparison(layer_results, output_dir)
    
    return layer_results

# Boxplot 
def plot_figure2_boxplot_detailed(results, save_dir):
    deep_layers = ['stage4.0', 'conv5']
    
    baseline_scores = []
    ft_vs_random_scores = []
    fb_vs_random_scores = []
    
    for layer in deep_layers:
        if layer in results:
            baseline_scores.extend(results[layer]['baseline']['scores'])
            ft_vs_random_scores.extend(results[layer]['ft_vs_rd']['scores'])
            fb_vs_random_scores.extend(results[layer]['fb_vs_rd']['scores'])
    
    pretrained_vs_random_scores = ft_vs_random_scores + fb_vs_random_scores
    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Right: Combined pretrained vs random
    data = [baseline_scores, pretrained_vs_random_scores]
    bp1 = ax1.boxplot(data, 
                      patch_artist=True,
                      labels=['Baseline\n(Random vs Random)', 
                             'Pretrained vs Random'],
                      widths=0.3,
                      showfliers=True
                      )
    
    colors = ['#aec6cf', '#ffb347']
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('CCA Similarity Score')
    ax1.set_title(f'CCA Similarity in 2 Top Layers (Pretrained vs Random) -- {model_name_script}')
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)
    
    # Right: Separated
    data_separated = [baseline_scores, ft_vs_random_scores, fb_vs_random_scores]
    bp2 = ax2.boxplot(data_separated,
                      patch_artist=True,
                      labels=['Baseline', 'FT vs Random', 'FB vs Random'],
                      widths=0.3,
                      showfliers=True)
    
    colors_separated = ['blue', 'red', 'green']
    for patch, color in zip(bp2['boxes'], colors_separated):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('CCA Similarity Score')
    ax2.set_title(f'CCA Similarity in 2 Top Layers (FT vs FB vs RD) -- {model_name_script}')
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Line plot
def plot_line_all_comparison(results, save_dir):
    layers = list(results.keys())
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    layer_indices = range(len(layers))
    
    baselines_mean = [results[layer]['baseline']['mean'] for layer in layers]
    ft_vs_rd_mean = [results[layer]['ft_vs_rd']['mean'] for layer in layers]
    fb_vs_rd_mean = [results[layer]['fb_vs_rd']['mean'] for layer in layers]
    ft_vs_fb_mean = [results[layer]['ft_vs_fb']['mean'] for layer in layers]
    
    # Calculate standard errors (std / sqrt(n))
    baselines_se = [np.std(results[layer]['baseline']['scores']) / np.sqrt(len(results[layer]['baseline']['scores'])) for layer in layers]
    ft_vs_rd_se = [np.std(results[layer]['ft_vs_rd']['scores']) / np.sqrt(len(results[layer]['ft_vs_rd']['scores'])) for layer in layers]
    fb_vs_rd_se = [np.std(results[layer]['fb_vs_rd']['scores']) / np.sqrt(len(results[layer]['fb_vs_rd']['scores'])) for layer in layers]
    ft_vs_fb_se = [np.std(results[layer]['ft_vs_fb']['scores']) / np.sqrt(len(results[layer]['ft_vs_fb']['scores'])) for layer in layers]
    
    ax.errorbar(layer_indices, baselines_mean, yerr=baselines_se, 
                fmt='o--', label='Baseline', color='gray', linewidth=2, markersize=8, 
                capsize=5, capthick=1.5, elinewidth=1.5)
    
    ax.errorbar(layer_indices, ft_vs_rd_mean, yerr=ft_vs_rd_se, 
                fmt='s-', label='FT vs rd', color='blue', linewidth=2, markersize=8,
                capsize=5, capthick=1.5, elinewidth=1.5)
    
    ax.errorbar(layer_indices, fb_vs_rd_mean, yerr=fb_vs_rd_se, 
                fmt='^-', label='FB vs rd', color='green', linewidth=2, markersize=8,
                capsize=5, capthick=1.5, elinewidth=1.5)
    
    ax.errorbar(layer_indices, ft_vs_fb_mean, yerr=ft_vs_fb_se, 
                fmt='D-', label='FT vs FB', color='red', linewidth=2, markersize=8,
                capsize=5, capthick=1.5, elinewidth=1.5)
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('SVCCA Similarity')
    ax.set_title(f'SVCCA Similarity Across Layers ({model_name_script})')
    ax.set_xticks(layer_indices)
    ax.set_xticklabels(layers)
    ax.set_ylim(0,1.05)
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'line_all_layers_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_svcca_results_to_csv(layer_results, save_dir):
    print(f"\nSaving SVCCA results to CSV files...")
    
    # 1. Create group_score.csv with mean scores
    group_data = []
    for layer, layer_data in layer_results.items():
        row = {'layer': layer}
        for comp_type in ['baseline', 'ft_vs_rd', 'fb_vs_rd', 'ft_vs_fb']:
            if comp_type in layer_data:
                row[comp_type] = float(layer_data[comp_type]['mean'])
        group_data.append(row)
    
    group_df = pd.DataFrame(group_data)
    group_csv_path = os.path.join(save_dir, f'svcca_group_scores_{model_name_script}.csv')
    group_df.to_csv(group_csv_path, index=False)
    
    # 2. Create individual_scores.csv with all individual scores
    individual_data = []
    for layer, layer_data in layer_results.items():
        for comp_type, comp_data in layer_data.items():
            for i, score in enumerate(comp_data['scores'], 1):
                individual_data.append({
                    'layer': layer,
                    'comparison': comp_type,
                    'run': i,
                    'score': float(score) 
                })
    
    individual_df = pd.DataFrame(individual_data)
    individual_csv_path = os.path.join(save_dir, f'svcca_individual_scores_{model_name_script}.csv')
    individual_df.to_csv(individual_csv_path, index=False)
    print(f"SVCCA group scores saved to: {group_csv_path}")
    print(f"SVCCA individual scores saved to: {individual_csv_path}")

#-------------------------- Main --------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_target_dir = os.path.join(target_dir, f'{model_name_script}')
    os.makedirs(output_target_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = ImageFolder(root=os.path.join(dataset_path, "test"), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    class_names = test_dataset.classes
    
    print("\n\nRunning SVCCA analysis...")
    results = calculate_baseline_comparison(
        model_groups, 
        test_loader, 
        device, 
        output_target_dir
    )
    
    save_svcca_results_to_csv(results, output_target_dir)
    print(f"\nAnalysis complete! Results saved in {output_target_dir}")

if __name__ == "__main__":
    main()