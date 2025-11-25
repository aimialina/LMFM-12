# SVCCA for top 2 layers (Fig 3a)
# applied for timm models (vgg19)

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
from tqdm import tqdm

#import timm 
#model = timm.create_model('vgg19_bn', pretrained=False)
#print(model)

model_name_script = 'VGG19'

dataset_path = "path/to/dataset"
target_dir = "path/to/output/directory"

# Provide paths of the weights accordingly
model_groups = {
    'frozen': {
        'fb1': '/weights_1/vgg19_bn_timm_fb_1/best_model_vgg19_bn_fb.pth',
        'fb2': '/weights_2/vgg19_bn_timm_fb_2/best_model_vgg19_bn_fb.pth',
        'fb3': '/weights_3/vgg19_bn_timm_fb_3/best_model_vgg19_bn_fb.pth'
    },
    'random': {
        'rd1': '/weights_1/vgg19_bn_timm_rd_1/best_model_vgg19_bn_rd.pth',
        'rd2': '/weights_2/vgg19_bn_timm_rd_2/best_model_vgg19_bn_rd.pth',
        'rd3': '/weights_3/vgg19_bn_timm_rd_3/best_model_vgg19_bn_rd.pth'
    },
    'full': {
        'ft1': '/weights_1/vgg19_bn_timm_ft_1/best_model_vgg19_bn_ft.pth',
        'ft2': '/weights_2/vgg19_bn_timm_ft_2/best_model_vgg19_bn_ft.pth',
        'ft3': '/weights_3/vgg19_bn_timm_ft_3/best_model_vgg19_bn_ft.pth'
    },
    'random_b': {
        'rdb1': '/weights_b/vgg19_bn_timm_rd_1/best_model_vgg19_bn_rd.pth',
        'rdb2': '/weights_b/vgg19_bn_timm_rd_2/best_model_vgg19_bn_rd.pth',
        'rdb3': '/weights_b/vgg19_bn_timm_rd_3/best_model_vgg19_bn_rd.pth'
    }
}


#-------------------------- VGG19 --------------------------
def load_model(path, device, num_classes):
    model = timm.create_model('vgg19_bn', pretrained=False)
    model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    return model

def get_layers():
    return [
        'features.0',   
        'features.10',  
        'features.20',  
        'features.30',  
        'features.43',  
    ]

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


def compute_svcca_robust(activations1, activations2, epsilon=1e-6, layer_name=None):
    # Ensure 2D shape
    if len(activations1.shape) > 2:
        activations1 = activations1.reshape(activations1.shape[0], -1)
    if len(activations2.shape) > 2:
        activations2 = activations2.reshape(activations2.shape[0], -1)
    
    # Layer-specific preprocessing with much more aggressive settings
    if layer_name in ['features.0', 'features.3', 'features.7']:
        print(f"Applying aggressive preprocessing for {layer_name}")
        
        # More aggressive subsampling for the first layer specifically
        if layer_name == 'features.0':
            max_features = 200  # Ultra aggressive for first layer
            if activations1.shape[0] > 32:
                row_indices = np.random.choice(activations1.shape[0], 32, replace=False)
                activations1 = activations1[row_indices, :]
                activations2 = activations2[row_indices, :]
        elif layer_name == 'features.3':
            max_features = 1000  # Very aggressive for second layer
        else:
            max_features = 2000  # Still aggressive but less so for later layers
            
        # Apply feature subsampling
        if activations1.shape[1] > max_features:
            indices = np.random.choice(activations1.shape[1], max_features, replace=False)
            activations1 = activations1[:, indices]
        if activations2.shape[1] > max_features:
            indices = np.random.choice(activations2.shape[1], max_features, replace=False)
            activations2 = activations2[:, indices]
            
        print(f"After subsampling, matrices are of shape {activations1.shape} and {activations2.shape}")
    
    act1_centered = activations1 - np.mean(activations1, axis=0)
    act2_centered = activations2 - np.mean(activations2, axis=0)
    
    del activations1, activations2
    import gc
    gc.collect()
    
    # 1: SVD for dimensionality reduction
    try:
        print(f"Computing SVD on matrices of shape {act1_centered.T.shape} and {act2_centered.T.shape}")
        
        u1, s1, vh1 = np.linalg.svd(act1_centered.T, full_matrices=False)
        del vh1
        gc.collect()
        
        u2, s2, vh2 = np.linalg.svd(act2_centered.T, full_matrices=False)
        del vh2
        gc.collect()
        
        variance_explained1 = np.cumsum(s1**2) / np.sum(s1**2)
        variance_explained2 = np.cumsum(s2**2) / np.sum(s2**2)
        k1 = np.where(variance_explained1 >= 0.99)[0][0] + 1 if np.any(variance_explained1 >= 0.99) else len(s1)
        k2 = np.where(variance_explained2 >= 0.99)[0][0] + 1 if np.any(variance_explained2 >= 0.99) else len(s2)
        k = min(k1, k2, u1.shape[1], u2.shape[1], 50)  
        print(f"Using {k} dimensions for CCA (99% variance threshold)")
        
        # Project to reduced subspace - using the right vectors
        svd_acts1 = act1_centered @ u1[:, :k]
        svd_acts2 = act2_centered @ u2[:, :k]
        del act1_centered, act2_centered
        gc.collect()
        
        # 2: CCA on reduced representations with more robust approach
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
    
    except np.linalg.LinAlgError as e:
        print(f"SVD failed with error: {e}")
        print("Using fallback correlation of means")
        try:
            mean1 = np.mean(act1_centered, axis=1)
            mean2 = np.mean(act2_centered, axis=1)
            svcca_similarity = np.abs(np.corrcoef(mean1, mean2)[0, 1])
        except Exception as e:
            print(f"Mean correlation also failed: {e}")
            svcca_similarity = 0.0
    
    return svcca_similarity

def get_activations(model, layer, dataloader):
    """Extract activations for a specific layer"""
    hook = ModelHook(model, [layer])
    activations = hook.get_activations(dataloader, num_batches=64)
    hook.remove_hooks()
    return activations[layer]

#-------------------------- Visualization --------------------------
def calculate_baseline_comparison_fixed(model_groups, dataloader, device, output_dir, num_classes=12):
    layers = get_layers()
    
    layer_results = {layer: {} for layer in layers}

    comparisons = {
        'baseline': ('random', 'random_b', 'rd', 'rdb'),
        'ft_vs_rd': ('full', 'random', 'ft', 'rd'),
        'fb_vs_rd': ('frozen', 'random', 'fb', 'rd'),
        'ft_vs_fb': ('full', 'frozen', 'ft', 'fb')
    }
    batch_limits = {
        'features.0': 64,    
        'features.10': 64,   
        'features.23': 64,   
        'features.36': 64,   
        'features.49': 64    
    }

    for layer in layers:
        print(f'\nProcessing layer: {layer}')
        # Use layer-specific batch limit
        batch_limit = batch_limits.get(layer, 64)
        print(f"Using batch limit of {batch_limit} for {layer}")

        for each_comp, (group1, group2, prefix1, prefix2) in comparisons.items():
            scores = []

            for i in range(1, 4):
                model1_path = model_groups[group1][f'{prefix1}{i}']
                model2_path = model_groups[group2][f'{prefix2}{i}']

                # Load and setup models
                model1 = load_model(model1_path, device, num_classes)
                model2 = load_model(model2_path, device, num_classes)

                # Extract activations with appropriate batch limit
                hook1 = ModelHook(model1, [layer])
                activation1 = hook1.get_activations(dataloader, num_batches=batch_limit)
                hook1.remove_hooks()

                hook2 = ModelHook(model2, [layer])
                activation2 = hook2.get_activations(dataloader, num_batches=batch_limit)
                hook2.remove_hooks()

                # Calculate similarity with robust implementation
                similarity = compute_svcca_robust(activation1[layer], activation2[layer], layer_name=layer)
                scores.append(similarity)

            layer_results[layer][each_comp] = {
                'scores': scores,
                'mean': np.mean(scores)
            }
            
            clean_scores = [float(s) for s in scores]
            print(f"{each_comp}: {clean_scores}, mean={np.mean(scores):.3f}")
    
    results_df = pd.DataFrame([
        {
            'layer': layer,
            'comparison': comparison_name,
            'run': i + 1,
            'score': float(score)
        }
        for layer, comparisons in layer_results.items()
        for comparison_name, data in comparisons.items()
        for i, score in enumerate(data['scores'])
    ])
    
    csv_path = os.path.join(output_dir, 'individual_scores.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Saved individual scores to {csv_path}")
    
    plot_figure2_boxplot_detailed(layer_results, output_dir)
    plot_line_all_comparison(layer_results, output_dir)
    
    return layer_results

def plot_figure2_boxplot_detailed(results, save_dir):
    """
    Create a more detailed Figure 2 with statistical annotations
    """
    deep_layers = ['features.30', 'features.43']
    
    # Collect scores
    baseline_scores = []
    ft_vs_random_scores = []
    fb_vs_random_scores = []
    
    for layer in deep_layers:
        if layer in results:
            baseline_scores.extend(results[layer]['baseline']['scores'])
            ft_vs_random_scores.extend(results[layer]['ft_vs_rd']['scores'])
            fb_vs_random_scores.extend(results[layer]['fb_vs_rd']['scores'])
    
    # Combine pretrained approaches
    pretrained_vs_random_scores = ft_vs_random_scores + fb_vs_random_scores
    
    # Create figure with subplots for more detail
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Main boxplot
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
    ax1.set_title(f'Figure 2: CCA Similarity in 2 Top Layers (Pretrained vs Random) -- {model_name_script}')
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)
    
    # Right plot: Separated by method
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
    ax2.set_title(f'Figure 2: CCA Similarity in 2 Top Layers (FT vs FB vs RD) -- {model_name_script}')
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'figure2_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_line_all_comparison(results, save_dir):
    layers = list(results.keys())
    
    # Create summary plot across layers
    fig, ax = plt.subplots(figsize=(12, 8))
    
    layer_indices = range(len(layers))
    
    # Extract data for each comparison type - means and standard errors
    baselines_mean = [results[layer]['baseline']['mean'] for layer in layers]
    ft_vs_rd_mean = [results[layer]['ft_vs_rd']['mean'] for layer in layers]
    fb_vs_rd_mean = [results[layer]['fb_vs_rd']['mean'] for layer in layers]
    ft_vs_fb_mean = [results[layer]['ft_vs_fb']['mean'] for layer in layers]
    
    # Calculate standard errors (std / sqrt(n))
    baselines_se = [np.std(results[layer]['baseline']['scores']) / np.sqrt(len(results[layer]['baseline']['scores'])) for layer in layers]
    ft_vs_rd_se = [np.std(results[layer]['ft_vs_rd']['scores']) / np.sqrt(len(results[layer]['ft_vs_rd']['scores'])) for layer in layers]
    fb_vs_rd_se = [np.std(results[layer]['fb_vs_rd']['scores']) / np.sqrt(len(results[layer]['fb_vs_rd']['scores'])) for layer in layers]
    ft_vs_fb_se = [np.std(results[layer]['ft_vs_fb']['scores']) / np.sqrt(len(results[layer]['ft_vs_fb']['scores'])) for layer in layers]
    
    # Create line plots with error bars
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
    results = calculate_baseline_comparison_fixed(
    model_groups, 
    test_loader, 
    device, 
    output_target_dir,
    num_classes=len(class_names)
)
    
    save_svcca_results_to_csv(results, output_target_dir)
    print(f"\nAnalysis complete! Results saved in {output_target_dir}")

if __name__ == "__main__":
    main()



