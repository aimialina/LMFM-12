# Customized testing code for both timm and torchvision models

import os
import torch 
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- Argument Parsing --------------------
parser = argparse.ArgumentParser(description='Evaluate trained models on microalgae test dataset')
parser.add_argument('--test_dir', type=str, default="path/to/test/dataset",
                    help='Directory containing test data')
parser.add_argument('--output_dir', type=str, default="path/to/output",   
                    help='Root directory for saving results')
parser.add_argument('--batch_size', type=int, default=64) 
parser.add_argument('--img_size', type=int, default=224)

parser.add_argument('--baseline_rd', type=str, default=None,
                    help='Path to random initialization baseline model weights')
parser.add_argument('--imagenet_ft', type=str, default=None,
                    help='Path to ImageNet fine-tuned model weights')
parser.add_argument('--imagenet_fb', type=str, default=None,
                    help='Path to ImageNet frozen backbone model weights')
parser.add_argument('--microalgae_specific_ft', type=str, default=None,
                    help='Path to specific microalgae fine-tuned model weights')
parser.add_argument('--microalgae_specific_fb', type=str, default=None,
                    help='Path to specific microalgae frozen backbone model weights')

parser.add_argument('--model_architecture', type=str, required=True,
                    help='Base model architecture name (mobilenetv2_100, efficientnet_b0, shufflenet_v2, efficientnet_v2)')
parser.add_argument('--run_suffix', type=str, default='',
                    help='use to indicate the parameter change')

args = parser.parse_args()


# -------------------- To save into text --------------------
def save_to_txt(filepath, content, mode='w'):
    with open(filepath, mode) as f:
        f.write(content + "\n")

# -------------------- Data preparation --------------------
test_transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(root=args.test_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

print(f"Test samples loaded: {len(test_dataset)}")
print(f"Classes: {test_dataset.classes}")

# -------------------- Model configuration --------------------
model_configs = {
    'baseline': {
        'name': 'baseline_random_init',
        'path': args.baseline_rd,
        'mode': 'rd',
        'description': 'Random Initialization (Baseline)',
        'category': 'baseline'
    },
    'imagenet_ft': {
        'name': 'imagenet_fine_tuned',
        'path': args.imagenet_ft,
        'mode': 'ft',
        'description': 'ImageNet Fine-Tuned',
        'category': 'imagenet'
    },
    'imagenet_fb': {
        'name': 'imagenet_frozen_backbone',
        'path': args.imagenet_fb,
        'mode': 'fb',
        'description': 'ImageNet Frozen Backbone',
        'category': 'imagenet'
    },
    'microalgae_specific_ft': {
        'name': 'microalgae_specific_fine_tuned',
        'path': args.microalgae_specific_ft,
        'mode': 'ft',
        'description': 'Specific Microalgae Fine-Tuned',
        'category': 'microalgae_specific'
    },
    'microalgae_specific_fb': {
        'name': 'microalgae_specific_frozen',
        'path': args.microalgae_specific_fb,
        'mode': 'fb',
        'description': 'Specific Microalgae Frozen Backbone',     
        'category': 'microalgae_specific'
    }
}


# -------------------- Torchvision Models --------------------
def get_torchvision_model(model_name, pretrained, num_classes):
    torchvision_models = {
        'shufflenet_v2': {
            'constructor': models.shufflenet_v2_x1_0,
            'weights': models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None,
            'classifier_path': 'fc',
            'get_in_features': lambda m: m.fc.in_features,
            'set_classifier': lambda m, num_classes: setattr(m, 'fc', nn.Linear(m.fc.in_features, num_classes))
        },
        'efficientnet_v2': {
            'constructor': models.efficientnet_v2_s,
            'weights': models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None,
            'classifier_path': 'classifier.1',
            'get_in_features': lambda m: m.classifier[1].in_features,
            'set_classifier': lambda m, num_classes: setattr(m.classifier, '1', nn.Linear(m.classifier[1].in_features, num_classes))
        }
    }
    
    if model_name not in torchvision_models:
        raise ValueError(f"Torchvision model '{model_name}' not supported. Available: {list(torchvision_models.keys())}")
    
    config = torchvision_models[model_name]
    model = config['constructor'](weights=config['weights'])
    config['set_classifier'](model, num_classes)
    
    return model, config


def detect_model_library(model_name):
    torchvision_models = ['shufflenet_v2', 'efficientnet_v2']
    
    if model_name in torchvision_models:
        return 'torchvision'
    else:
        return 'timm'

# -------------------- Model loading --------------------
def create_model(model_name, num_classes):
    try:
        print(f"Creating model: {model_name} with {num_classes} classes")
        model_library = detect_model_library(model_name)
        print(f"📚 Using library: {model_library}")
        
        if model_library == 'torchvision':
            model, config = get_torchvision_model(model_name, pretrained=False, num_classes=num_classes)
            print(f"✅ Torchvision model '{model_name}' created successfully")
            
            total_params = sum(p.numel() for p in model.parameters())
            print(f"   Parameters: {total_params:,}")
            
            return model
            
        else:
            available_models = timm.list_models()
            if model_name not in available_models:
                print(f"⚠️  '{model_name}' not found in timm models")

            model = timm.create_model(model_name, num_classes=num_classes, pretrained=False)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"✅ Model '{model_name}' created successfully")
            print(f"   Parameters: {total_params:,}")
            
            for name, module in model.named_modules():
                if any(word in name.lower() for word in ['classifier', 'head', 'head.fc']) and hasattr(module, 'out_features'):
                    if module.out_features == num_classes:
                        print(f"   Classifier: {name} -> {module.out_features} classes ✅")
                    else:
                        print(f"   Classifier: {name} -> {module.out_features} classes ❌ (expected {num_classes})")
            
            return model
        
    except Exception as e:
        print(f"❌ Error creating model '{model_name}': {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    

def load_model(model_path, device, model_name=None, num_classes=None, train_mode='ft'):
    try:
        print(f"📦 Loading model from: {os.path.basename(model_path)}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        
        model = create_model(model_name, num_classes)
        if model is None:
            print("❌ Failed to create model architecture")
            return None
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("📦 Using 'state_dict' from checkpoint")
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            print("📦 Using 'model' from checkpoint")
        else:
            state_dict = checkpoint
            print("📦 Using checkpoint directly as state_dict")
        
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            clean_key = key.replace('module.', '') if key.startswith('module.') else key
            cleaned_state_dict[clean_key] = value
        
        model_state_dict = model.state_dict()
        model_keys = set(model_state_dict.keys())
        checkpoint_keys = set(cleaned_state_dict.keys())
        
        common_keys = model_keys & checkpoint_keys
        missing_keys = model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_keys
        
        print(f"Analysis:")
        print(f"  Model parameters: {len(model_keys)}")
        print(f"  Checkpoint parameters: {len(checkpoint_keys)}")
        print(f"  Common parameters: {len(common_keys)}")
        print(f"  Missing parameters: {len(missing_keys)}")
        print(f"  Unexpected parameters: {len(unexpected_keys)}")
        
        # ------------- STRICT LOADING -------------
        print("\n🔒 STRICT MODE: Expecting exact parameter match")
        match_ratio = len(common_keys) / len(model_keys)
        print(f"  Match ratio: {match_ratio:.2%}")
        
        if match_ratio < 0.99:  
            print(f"CRITICAL ERROR: Only {match_ratio:.1%} of model parameters match")
            print(f"Missing keys (first 10): {list(missing_keys)[:10]}")
            print(f"Unexpected keys (first 10): {list(unexpected_keys)[:10]}")
            return None
        try:
            missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=True)
            print("Strict=True (perfect match)")
            
        except RuntimeError as e:
            print(f"Strict loading failed: {str(e)}. All parameters must match exactly")
            return None

        model.to(device)
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model loaded successfully:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        return model
        
    except Exception as e:
        print(f"Error in loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# -------------------- Evaluation --------------------
def evaluate_model_debug(model, data_loader, device, class_names, return_predictions=False):
    print("Starting model evaluation...")
    
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters")
    
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
            if return_predictions:
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Final accuracy: {accuracy:.2f}%")
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    metrics = {
        'accuracy': accuracy,
        'total_samples': total,
        'correct_predictions': correct
    }

    if return_predictions:
        return metrics, {
            'predictions': np.array(all_preds),
            'true_labels': np.array(all_labels),
            'probabilities': np.array(all_probs),
            'class_names': class_names
        }
    return metrics

def plot_confusion_matrix(cm, class_names, model_name, output_dir):
    plt.figure(figsize=(15, 12))
    sns.set(font_scale=1.2)
    
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    cbar=False, square=True,
                    xticklabels=class_names,
                    yticklabels=class_names)
    
    ax.set(xlabel="Predicted Labels", ylabel="True Labels")
    plt.title(f"Confusion Matrix: {model_name}", pad=20, fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    filename = os.path.join(output_dir, f"confusion_matrix_{model_name}.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

# -------------------- Main --------------------
if __name__ == "__main__":
    
    class_names = test_dataset.classes

    # Filter out models that don't have paths
    models_to_evaluate = {k: v for k, v in model_configs.items() if v['path'] is not None}
    skipped_models = {k: v for k, v in model_configs.items() if v['path'] is None}

    print(f"Found {len(models_to_evaluate)} models to evaluate")
    print(f"Skipping {len(skipped_models)} models (no path provided)")
    if skipped_models:
        print("Skipped models:")
        for key, config in skipped_models.items():
            print(f"  - {config['description']}")
    print(f"Model architecture: {args.model_architecture}")

    os.makedirs(args.output_dir, exist_ok=True)
    test_with_suffix = f"{args.model_architecture}{args.run_suffix}"
    test_output_dir = os.path.join(args.output_dir, test_with_suffix)
    os.makedirs(test_output_dir, exist_ok=True)

    # Create summary file
    summary_path = os.path.join(test_output_dir, "evaluation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Microalgae Model Evaluation Summary - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Test Dataset: {args.test_dir}\n")
        f.write(f"Model Architecture: {args.model_architecture}\n")
        f.write(f"Number of Classes: {len(class_names)}\n")
        f.write(f"Classes: {', '.join(class_names)}\n")
        f.write(f"Total Test Samples: {len(test_dataset)}\n")
        f.write(f"Models to Evaluate: {len(models_to_evaluate)}\n")
        f.write(f"Models Skipped: {len(skipped_models)}\n\n")
        
        if skipped_models:
            f.write("Skipped Models (no path provided):\n")
            for key, config in skipped_models.items():
                f.write(f"  - {config['description']}\n")
            f.write("\n")
    
    results_summary = []
    
    for config_key, config in models_to_evaluate.items():
        model_name = config['name']
        model_path = config['path']
        train_mode = config['mode']
        description = config['description']
        category = config['category']

        print(f"\n{'='*60}")
        print(f"Evaluating: {description}")
        print(f"Category: {category}")
        print(f"Mode: {train_mode}")
        print(f"Path: {model_path}")
        print(f"{'='*60}")

        # Create category-specific output directory
        category_output_dir = os.path.join(test_output_dir, category)
        os.makedirs(category_output_dir, exist_ok=True)
        txt_report_path = os.path.join(category_output_dir, f'{model_name}_evaluation_results.txt')

        model = load_model(
            model_path,
            device,
            model_name=args.model_architecture,
            num_classes=len(class_names),
            train_mode=train_mode
        )

        if model is None:
            error_msg = f"Failed to load model: {model_name}"
            print(f"❌ {error_msg}")
            with open(txt_report_path, 'w') as f:
                f.write(f"ERROR: {error_msg}\n")
            continue

        status = "Exact Match Loading"
        header = f"""\
{'='*60}
Model Evaluation Report - {time.strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

Model: {description}
Category: {category}
Architecture: {args.model_architecture}
Training Mode: {train_mode} ({'Frozen Backbone' if train_mode == 'fb' else 
                            'Full Fine-Tuning' if train_mode == 'ft' else 
                            'Random Initialization'})
Loading Mode: {status}
Model Path: {model_path}
Test Data: {args.test_dir}
Classes: {len(class_names)} ({', '.join(class_names)})
"""

        test_metrics, test_predictions = evaluate_model_debug(model, 
                                                              test_loader, 
                                                              device, 
                                                              class_names, 
                                                              return_predictions=True
)
    with open(txt_report_path, 'w') as f:
        f.write(header)

        
        metrics_txt = f"""
Overall Metrics:
- Accuracy: {test_metrics['accuracy']:.2f}%
- Correct: {test_metrics['correct_predictions']}/{test_metrics['total_samples']}
"""
        f.write(metrics_txt)

        results_summary.append({
            'category': category,
            'model': description,
            'mode': train_mode,
            'accuracy': test_metrics['accuracy'],
            'correct_predictions': test_metrics['correct_predictions'],
            'total_samples': test_metrics['total_samples']
        })
        report = classification_report(
            test_predictions['true_labels'],
            test_predictions['predictions'],
            target_names=test_predictions['class_names'],
            digits=4)
        
        f.write("\nClassification Report:\n")
        f.write(report)

        cm = confusion_matrix(test_predictions['true_labels'], test_predictions['predictions'])
        cm_filename = plot_confusion_matrix(cm, class_names, model_name, category_output_dir)

        f.write(f"\nConfusion Matrix Visualization: {cm_filename}\n")
        f.write("\nConfusion Matrix (Counts):\n")
        for i, row in enumerate(cm):
            row_str = " ".join(f"{num:5d}" for num in row)
            species_name = class_names[i]
            f.write(f"{species_name:<30} {row_str}\n")
        
        errors = np.where(test_predictions['predictions'] != test_predictions['true_labels'])[0]
        if len(errors) > 0:
            with open(txt_report_path, 'a') as f:
                f.write("\nTop Misclassifications:\n")
                for i in errors[:10]: 
                    true_class = test_predictions['class_names'][test_predictions['true_labels'][i]]
                    pred_class = test_predictions['class_names'][test_predictions['predictions'][i]]
                    confidence = np.max(test_predictions['probabilities'][i])
                    f.write(f"Sample {i}: True={true_class}, Predicted={pred_class} (Confidence: {confidence:.2%})\n")
        
        # More analysis
        with open(summary_path, 'a') as f:
            f.write("\nResults Summary:\n")
            f.write(f"{'='*80}\n")
            f.write(f"{'Category':<20} {'Model':<30} {'Mode':<6} {'Correct/Total':<15} {'Accuracy':<10}\n")
            f.write(f"{'-'*80}\n")

            expected_random = 100.0 / len(class_names)
            suspicious_count = 0

            for result in results_summary:

                correct_total = f"{result['correct_predictions']}/{result['total_samples']}"
                accuracy_str = f"{result['accuracy']:.2f}%"

                f.write(f"{result['category']:<20} {result['model']:<30} {result['mode']:<6} "
                        f"{correct_total}   {accuracy_str:<10}")

                if abs(result['accuracy'] - expected_random) < 2.0:
                    f.write(" ⚠️ SUSPICIOUS")
                    suspicious_count += 1
                f.write("\n")

        print(f"✅ Evaluation complete for {description}")
        print(f"Results saved to: {txt_report_path}")



