# SVCCA for before and after training (Fig 4)
# applied for torchvision models (shufflenet and efficientnet)

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


#-------------------------- Config --------------------------
model_name_script = 'ShuffleNet-V2'
dataset_path = "/path/to/dataset"
output_base_dir = "/path/to/output/directory/"

trained_model_weights = {
    'full'   : '/path/to/full_ft.pth',
    'frozen' : '/path/to/fb.pth',
    'random' : '/path/to/rd.pth'
}

#-------------------------- Models --------------------------
# ------- ShuffleNet -------
def load_model(path, device, num_classes):
    model = models.shufflenet_v2_x1_0(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    return model

def get_layers():
    return [
        'conv1',      
        'stage2.0',   
        'stage3.0',   
        'stage4.0',   
        'conv5'
    ]

# ------- EfficientNet -------
#def load_model(path, device, num_classes):
#    """Helper function to load a model"""
#    model = models.efficientnet_v2_s(pretrained=False)
#    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
#    model.load_state_dict(torch.load(path, map_location=device))
#    model.to(device).eval()
#    return model
#
#def get_layers():
#    return [
#        'features.0',  
#        'features.1',  
#        'features.3',  
#        'features.5',  
#        'features.7'
#    ]

#-------------- DenseNet --------------
#def load_model(path, device, num_classes):
#    model = timm.create_model('densenet121', pretrained=False)
#    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
#    model.load_state_dict(torch.load(path, map_location=device))
#    model.to(device).eval()
#    return model
#
#def get_layers():
#    return [
#        'features.conv0',       
#        'features.denseblock1', 
#        'features.denseblock2', 
#        'features.denseblock3', 
#        'features.denseblock4', 
#    ]
    
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

#-------------------------- SVCCA (Before/After) --------------------------
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
            if len(output.shape) == 4: 
                batch, channels, h, w = output.shape
                output = output.permute(0, 2, 3, 1).reshape(-1, channels)
            self.activations[layer_name] = output.detach().cpu().numpy()
        
    def get_activations(self, dataloader, num_batches=None):
        self.model.eval()
        collected_activations = {layer: [] for layer in self.layers_of_interest}
        
        with torch.no_grad():
            for i, (inputs, _) in enumerate(dataloader):
                if num_batches is not None and i >= num_batches:
                    break
                    
                inputs = inputs.cuda() if torch.cuda.is_available() else inputs
                self.model(inputs)
                
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

def get_aligned_activations(device, test_loader, output_dir):
    print("Getting aligned activations for SVCCA comparison...")
    layers = get_layers()

    fixed_batch_size = 64
    fixed_loader = DataLoader(
        test_loader.dataset, 
        batch_size=fixed_batch_size,
        shuffle=False,
        num_workers=4
    )
    
    fixed_batch = next(iter(fixed_loader))
    fixed_inputs, _ = fixed_batch
    fixed_inputs = fixed_inputs.to(device)
    
    activations = {}
    
    for model_type in ['full', 'frozen', 'random']:
        print(f"\nProcessing {model_type} model...")
        activations[model_type] = {'initial': {}, 'final': {}}
        
        if model_type in ['full', 'frozen']:
            initial_model = models.shufflenet_v2_x1_0(pretrained=True)       # Change into other models accordingly
        else:
            initial_model = models.shufflenet_v2_x1_0(pretrained=False)
        
        initial_model.fc = nn.Linear(initial_model.fc.in_features,  12)
        initial_model = initial_model.to(device).eval()
        
        trained_model = load_model(trained_model_weights[model_type], device, num_classes=12)
        
        hook_initial = ModelHook(initial_model, layers)
        hook_final = ModelHook(trained_model, layers)
        
        with torch.no_grad():
            initial_model(fixed_inputs)
            trained_model(fixed_inputs)
            for layer in layers:
                if layer in hook_initial.activations:
                    activations[model_type]['initial'][layer] = hook_initial.activations[layer]
                if layer in hook_final.activations:
                    activations[model_type]['final'][layer] = hook_final.activations[layer]
        
        hook_initial.remove_hooks()
        hook_final.remove_hooks()
    return activations



def compute_svcca_simplified(act1, act2, epsilon=1e-10):
    if isinstance(act1, torch.Tensor):
        act1 = act1.cpu().numpy()
    if isinstance(act2, torch.Tensor):
        act2 = act2.cpu().numpy()
    
    # Reshape 4D activations to 2D
    if len(act1.shape) == 4: 
        batch, channels, h, w = act1.shape
        act1 = act1.transpose(0, 2, 3, 1).reshape(-1, channels)
    if len(act2.shape) == 4:
        batch, channels, h, w = act2.shape
        act2 = act2.transpose(0, 2, 3, 1).reshape(-1, channels)
    
    act1_centered = act1 - np.mean(act1, axis=0)
    act2_centered = act2 - np.mean(act2, axis=0)
    
    std1 = np.std(act1_centered, axis=0)
    std2 = np.std(act2_centered, axis=0)
    
    non_const1 = std1 > epsilon
    non_const2 = std2 > epsilon
    
    common_non_const = non_const1 & non_const2
    
    if np.sum(common_non_const) == 0:
        print("  Warning: No common non-constant features, using all features")
        common_non_const = np.ones_like(non_const1, dtype=bool)
    
    act1_centered = act1_centered[:, common_non_const]
    act2_centered = act2_centered[:, common_non_const]
    
    print(f"  Using {np.sum(common_non_const)} common non-constant features")
    
    try:
        # SVD
        u1, s1, vh1 = np.linalg.svd(act1_centered.T, full_matrices=False)
        u2, s2, vh2 = np.linalg.svd(act2_centered.T, full_matrices=False)
        # Variance thresholding
        variance_threshold = 0.99
        var1 = np.cumsum(s1**2) / np.sum(s1**2)
        var2 = np.cumsum(s2**2) / np.sum(s2**2)
        k1 = np.where(var1 >= variance_threshold)[0][0] + 1 if np.any(var1 >= variance_threshold) else len(s1)
        k2 = np.where(var2 >= variance_threshold)[0][0] + 1 if np.any(var2 >= variance_threshold) else len(s2)
        k = min(k1, k2)
        print(f"  Using {k} dimensions (99% variance)")
        # Project to reduced subspace
        proj1 = act1_centered @ u1[:, :k]
        proj2 = act2_centered @ u2[:, :k]
        # CCA - Cross-covariance 
        cov1 = np.cov(proj1.T) + epsilon * np.eye(k)
        cov2 = np.cov(proj2.T) + epsilon * np.eye(k)
        cov12 = np.cov(proj1.T, proj2.T)[:k, k:]
        # Compute canonical correlations
        invsqrt1 = np.linalg.inv(np.linalg.cholesky(cov1))
        invsqrt2 = np.linalg.inv(np.linalg.cholesky(cov2))
    
        T = invsqrt1.T @ cov12 @ invsqrt2
        
        # SVD of T
        _, s, _ = np.linalg.svd(T)
        mean_correlation = np.mean(np.clip(s, 0, 1))
        
        return mean_correlation
    except np.linalg.LinAlgError as e:
        print(f"  LinAlgError: {e}")
        return 0.0
    except Exception as e:
        print(f"  Unexpected error: {e}")
        return 0.0
    
#-------------------------- Visualization --------------------------
def analyze_figure4(device, test_loader, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    layers = get_layers()
    
    activations = get_aligned_activations(device, test_loader, output_dir)
    
    results = {
        'layer': layers,
        'ft': [],
        'fb': [],
        'rd': [],
        'ft_minus_rd': [],
        'fb_minus_rd': []
    }
    for model_type, type_name in [('full', 'ft'), ('frozen', 'fb'), ('random', 'rd')]:
        print(f"\nComputing SVCCA for {model_type} model...")
        layer_scores = []
        for layer in layers:
            if (layer in activations[model_type]['initial'] and 
                layer in activations[model_type]['final']):
                act_init = activations[model_type]['initial'][layer]
                act_final = activations[model_type]['final'][layer]
                
                print(f"\nLayer {layer}:")
                print(f"  Initial shape: {act_init.shape}")
                print(f"  Final shape: {act_final.shape}")
                
                score = compute_svcca_simplified(act_init, act_final)
                layer_scores.append(score)
                print(f"  SVCCA score: {score:.4f}")
            else:
                layer_scores.append(0.0)
                print(f"\nLayer {layer}: Missing activations")
        
        results[type_name] = layer_scores
    
    for i in range(len(layers)):
        results['ft_minus_rd'].append(results['ft'][i] - results['rd'][i])
        results['fb_minus_rd'].append(results['fb'][i] - results['rd'][i])
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(output_dir, 'figure4_aligned_results.csv'), index=False)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    x = range(len(layers))
    
    ax.plot(x, results['ft'], 'o-', label='Fully Fine-tuning (FT)', linewidth=2.5, markersize=10)
    ax.plot(x, results['fb'], 's-', label='Frozen Backbone (FB)', linewidth=2.5, markersize=10)
    ax.plot(x, results['rd'], '^-', label='Random (RD)', linewidth=2.5, markersize=10)
    
    ax.plot(x, results['ft_minus_rd'], ':', label='ft - rd', linewidth=2, color='grey')
    ax.plot(x, results['fb_minus_rd'], '--', label='fb - rd', linewidth=2, color='darkgrey')
    
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('Layer (Stage)', fontsize=14)
    ax.set_ylabel('CCA Similarity', fontsize=14)
    ax.set_title(f'{model_name_script}', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=0)
    #ax.legend(fontsize=12)
    #ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure4_aligned.png'), dpi=300)
    plt.close()
    
    create_combined_pretrained_visualization(results, output_dir, model_name_script)

    return df_results



def create_combined_pretrained_visualization(results, output_dir, model_name_script=""):
    layers = results['layer']
    x = range(len(layers))
    
    pretrained_avg = [(ft + fe)/2 for ft, fe in zip(results['ft'], results['fb'])]
    
    pretrained_minus_rd = [p - r for p, r in zip(pretrained_avg, results['rd'])]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    ax.plot(x, pretrained_avg, 'o-', color='orange', label='Pretrained (avg)', linewidth=2.5, markersize=10)
    ax.plot(x, results['rd'], '^-', color='blue', label='Random Init (rd)', linewidth=2.5, markersize=10)
    
    ax.plot(x, pretrained_minus_rd, '--', label='Pretrained - Random', linewidth=2, color='gray')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Layer', fontsize=14)
    ax.set_ylabel('CCA Similarity', fontsize=14)
    ax.set_title(f'CCA Similarity Before/After Training {model_name_script}', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=45)
    ax.set_ylim(0, 1.1)
    #ax.legend(fontsize=12)
    #ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'pretrained_vs_random_comparison.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Visualization saved to {output_path}")
    
    combined_results = {
        'layer': layers,
        'pretrained_avg': pretrained_avg,
        'random': results['rd'],
        'pretrained_minus_random': pretrained_minus_rd
    }
    
    df_combined = pd.DataFrame(combined_results)
    csv_path = os.path.join(output_dir, 'pretrained_vs_random_results.csv')
    df_combined.to_csv(csv_path, index=False)
    
    return fig, ax


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    output_dir = os.path.join(output_base_dir, f'{model_name_script}')
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = ImageFolder(root=os.path.join(dataset_path, 'test'), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    print("Analyzing Before/After Training with aligned activations...")
    df_results = analyze_figure4(device, test_loader, output_dir)


    print(f"\nAnalysis complete. Results saved to: {output_dir}")
    print("\nResults summary:")
    print(df_results)


if __name__ == "__main__":
    main()