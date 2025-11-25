# Customized training code for both timm and torchvision models
# parameters shown are finalized

import os
import time
import timm # print(timm.list_models())
import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pickle
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler
from tqdm import tqdm
from datetime import datetime, timedelta
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

parser = argparse.ArgumentParser(description='Train a model using timm library')
parser.add_argument('--model', type=str, default='densenet121')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate') 
parser.add_argument('--img_size', type=int, default=224, help='Image size')
parser.add_argument('--root_dir', type=str, default= "path/to/dataset",   
                    help='Directory for train image data')
parser.add_argument('--output_dir', type=str, default= "path/to/output",   
                    help='path to your output') 
parser.add_argument('--train_mode', type=str, default='fb', choices=['fb', 'ft', 'rd'],  
                    help='Training mode: fb (frozen backbone), ft (fine-tuning), rd (random weights)')
parser.add_argument('--run_suffix', type=str, default='',
                    help='Optional suffix to add to the output dir name; just add "2"')

parser.add_argument('--use_custom_weights', action='store_true',
                    help='use this argument to use custom weights LMFM-12 weights instead of ImageNet')
parser.add_argument('--custom_weight_path', type=str, default='',
                    help='Provide the paths of the custom weights of LMFM-12 weights')

parser.add_argument('--early_stopping', action='store_true', default=False, 
                    help='Enable early stopping')
parser.add_argument('--early_stopping_patience', type=int, default=10,
                    help='Number of epochs to wait for improvement before stopping')
parser.add_argument('--early_stopping_min_delta', type=float, default=0.001,
                    help='Minimum change in validation metric to qualify as improvement')
parser.add_argument('--early_stopping_metric', type=str, default='val_loss', 
                    choices=['val_acc', 'val_loss'],
                    help='Metric to monitor for early stopping')

args = parser.parse_args()

# -------------------- Hyperparameters setting --------------------
batch_size = args.batch_size
num_epochs = args.num_epochs
learning_rate = args.learning_rate
img_size = (224, 224)

print(batch_size, num_epochs, learning_rate)

# -------------------- Early Stopping Class --------------------
class EarlyStopping:    
    def __init__(self, patience=10, min_delta=0.001, metric='val_acc', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = 'max' if metric == 'val_acc' else 'min'
        
    def __call__(self, val_metric):
        if self.mode == 'max':
            score = val_metric
            improved = self.best_score is None or score > self.best_score + self.min_delta
        else:  
            score = -val_metric 
            improved = self.best_score is None or score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            if self.verbose:
                metric_name = "accuracy" if self.metric == 'val_acc' else "loss"
                print(f"Validation {metric_name} improved to {val_metric:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                metric_name = "accuracy" if self.metric == 'val_acc' else "loss"
                print(f"No improvement in validation {metric_name} for {self.counter} epochs")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping == No improvement for {self.patience} epochs")
        
        return self.early_stop

# -------------------- Data Preparation --------------------
train_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.RandAugment(num_ops=2, magnitude=9), # <= default setting
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dir = os.path.join(args.root_dir, "train")
val_dir = os.path.join(args.root_dir, "val")
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                          sampler=ImbalancedDatasetSampler(train_dataset), 
                          num_workers=4, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                        num_workers=4, pin_memory=True, persistent_workers=True)

num_classes = len(train_dataset.classes)
print(f"Number of classes: {num_classes}")

# -------------------- For Torchvision Library --------------------
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

# -------------------- Timm & Torchvision Models Setup --------------------
def load_weights_correctly(model_name, weight_path, num_classes, device, model_library='timm'):
    checkpoint = torch.load(weight_path, map_location='cpu')
    original_classes = None
    for key in ['classifier.weight', 'head.fc.weight', 'fc.weight', 'classifier.1.weight']:
        if key in checkpoint:
            original_classes = checkpoint[key].shape[0]
            break
    
    print(f"Loading custom weights: {original_classes} -> {num_classes} classes")
    print("Transferring backbone only (classifier will be reinitialized)")
    
    # 1: Create temporary model using the correct library
    if model_library == 'timm':
        temp_model = timm.create_model(model_name, pretrained=False, num_classes=original_classes)
        target_model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    else: 
        temp_model, _ = get_torchvision_model(model_name, pretrained=False, num_classes=original_classes)
        target_model, _ = get_torchvision_model(model_name, pretrained=False, num_classes=num_classes)

    temp_model.load_state_dict(checkpoint, strict=True)
    
    # 2: Extract backbone weights
    temp_dict = temp_model.state_dict()
    backbone_dict = {}
    classifier_keys_skipped = []
    
    for key, value in temp_dict.items():
        if any(cls_key in key for cls_key in ['classifier', 'head', 'fc']):
            classifier_keys_skipped.append(key)
        else:
            backbone_dict[key] = value
    
    # 3: Load backbone with target class number
    target_model.load_state_dict(backbone_dict, strict=False)
    
    return target_model.to(device)

def initialize_model(model_name, num_classes, train_mode='fb', device='cuda', model_library='auto'):
    print(f"Initializing model: {model_name} in '{train_mode}' mode.")
    
    if model_library == 'auto':
        torchvision_models = ['shufflenet_v2', 'efficientnet_v2']
        if model_name in torchvision_models:
            model_library = 'torchvision'
        else:
            model_library = 'timm'

    print(f"Using library: {model_library}")

    pretrained = train_mode != 'rd'     # Pretrained is True for all modes except 'rd' (random)

    if args.use_custom_weights and pretrained:
        print("Using custom weights instead of ImageNet")
        model = load_weights_correctly(model_name, args.custom_weight_path, num_classes, device, model_library)
        if model_library == 'timm':
            classifier_keywords = ['head', 'classifier', 'fc']
        else:
            _, config = get_torchvision_model(model_name, pretrained=False, num_classes=num_classes)
            classifier_keywords = [config['classifier_path'].split('.')[0]]
    else:
        if model_library == 'timm':
            model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
            if 'vgg' in model_name.lower():
                classifier_keywords = ['head.fc']
                print("VGG: Training head.fc (excluding pre_logits.fc 1 and 2)")
            elif 'convnext_base' in model_name.lower():
                classifier_keywords = ['head']
            elif 'mobilenetv2_100' in model_name.lower():
                classifier_keywords = ['classifier']
            else:
                classifier_keywords = ['head', 'classifier', 'fc']
                print(f"Warning: Using default classifier keywords for {model_name}. "
                      f"Consider adding specific keywords if needed.")
        else:
            model, config = get_torchvision_model(model_name, pretrained=pretrained, num_classes=num_classes)
            classifier_keywords = [config['classifier_path'].split('.')[0]]
        
        model = model.to(device)

    if train_mode == 'fb':
        print("Mode: Frozen backbone, training classifier")
        for name, param in model.named_parameters():
            if not any(cls_key in name for cls_key in classifier_keywords):
                param.requires_grad = False
        
    elif train_mode == 'ft':
        print("Mode: Fine-Tuning. All layers are trainable.")
        pass
        
    elif train_mode == 'rd':
        print("Mode: Random Initialization. Training from scratch.")
        pass

    # Display trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({trainable_params/total_params:.2%})")
    
    return model.to(device) 

# -------------------- Training and Validation --------------------
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, early_stopping=None):
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0
    
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    lr_history = []
    epoch_times = []

    total_start_time = time.time()
    actual_epochs = 0  

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100 * correct / total
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)

        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = 100 * correct / total
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(epoch_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)

        if new_lr != current_lr:
            print(f"Learning rate reduced from {current_lr:.2e} to {new_lr:.2e}")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_val_loss = epoch_val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        epochs_remaining = num_epochs - (epoch + 1)
        estimated_time_remaining = epochs_remaining * avg_epoch_time
        epoch_time_str = str(timedelta(seconds=int(epoch_duration)))
        remaining_time_str = str(timedelta(seconds=int(estimated_time_remaining)))

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%, "
              f"LR: {current_lr:.2e} | "
              f"Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch+1}) | "
              f"Epoch Time: {epoch_time_str} | "
              f"Est. Remaining: {remaining_time_str}")

        # -------------------- Early Stopping Check --------------------
        if early_stopping is not None:
            if early_stopping.metric == 'val_acc':
                monitor_metric = epoch_val_acc
            else: 
                monitor_metric = epoch_val_loss
            
            if early_stopping(monitor_metric):
                print(f"\nXX Early stopping at epoch {epoch+1}")
                print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch+1}")
                actual_epochs = epoch + 1
                break

    total_training_time = time.time() - total_start_time
    time_stats = {
        'total_training_time': total_training_time,
        'average_epoch_time': sum(epoch_times) / len(epoch_times),
        'fastest_epoch': min(epoch_times),
        'slowest_epoch': max(epoch_times),
        'epoch_times': epoch_times,
        'actual_epochs_trained': actual_epochs
    }

    print("\nTraining completed:")
    print(f"Total training time: {str(timedelta(seconds=int(total_training_time)))}")
    print(f"Average epoch time: {str(timedelta(seconds=int(time_stats['average_epoch_time'])))}")
    print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch+1}")

    early_stopping_info = None
    if early_stopping is not None:
        early_stopping_info = {
            'enabled': True,
            'patience': early_stopping.patience,
            'min_delta': early_stopping.min_delta,
            'metric': early_stopping.metric,
            'triggered': early_stopping.early_stop,
            'epochs_without_improvement': early_stopping.counter,
        }

    return best_val_acc, best_val_loss, train_loss, val_loss, train_acc, val_acc, best_model_state, lr_history, best_epoch, time_stats, early_stopping_info



# --------------------------------- Main execution block -------------------------------
if __name__ == "__main__":
    print(f"Intended for {args.output_dir}")

    print(f"Train mode: {args.train_mode}")
    model = initialize_model(
        args.model, 
        num_classes, 
        args.train_mode,
        device=device,
        ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        cooldown=2
    )
    # for vgg19
    conv_trainable = 0
    bn_trainable = 0
    fc_trainable = 0
    other_trainable = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'conv' in name:
                conv_trainable += param.numel()
            elif 'bn' in name:
                bn_trainable += param.numel()
            elif 'classifier' in name or 'fc' in name:
                fc_trainable += param.numel()
            else:
                other_trainable += param.numel()
    
    print("\nTrainable parameters breakdown:")
    print(f"- Convolutional layers: {conv_trainable:,}")
    print(f"- Batch Normalization layers: {bn_trainable:,}")
    print(f"- Fully Connected layers: {fc_trainable:,}")
    print(f"- Other layers: {other_trainable:,}")
    print(f"Total trainable parameters: {conv_trainable + bn_trainable + fc_trainable + other_trainable:,}")

    # -------------------- Initialize Early Stopping --------------------
    early_stopping = None
    if args.early_stopping:
        early_stopping = EarlyStopping(
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            metric=args.early_stopping_metric,
            verbose=True
        )
        print(f"\n🎯 Early stopping enabled:")
        print(f"   Patience: {args.early_stopping_patience} epochs")
        print(f"   Min delta: {args.early_stopping_min_delta}")
        print(f"   Monitoring: {args.early_stopping_metric}")
        print(f"   Mode: {'maximize' if args.early_stopping_metric == 'val_acc' else 'minimize'}")

    start_time = time.time()
    
    # Train and validate
    results = train_and_validate(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer,
        scheduler,
        num_epochs,  
        device,
        early_stopping=early_stopping
    )
    
    best_val_acc, best_val_loss, train_loss, val_loss, train_acc, val_acc, best_model_state, lr_history, best_epoch, time_stats, early_stopping_info = results
    
    # Metrics
    metrics = {
        'train_losses': train_loss,
        'val_losses': val_loss,
        'train_accuracies': train_acc,
        'val_accuracies': val_acc,
        'learning_rates': lr_history,
        'best_validation_accuracy': best_val_acc,
        'best_epoch': best_epoch,
        'total_epochs_trained': num_epochs,
        'actual_epochs_trained': time_stats['actual_epochs_trained'],
        'time_statistics': time_stats,
        'early_stopping': early_stopping_info,
        'hyperparameters': {
            'batch_size': batch_size,
            'initial_learning_rate': learning_rate,
            'scheduler_params': {
                'factor': 0.5,
                'patience': 5,
                'min_lr': 1e-6,
                'cooldown': 2
            }
        }
    }

    # Model-specific output; combine train mode and run suffix
    train_mode_with_suffix = f"{args.train_mode}{args.run_suffix}"
    dir_name = f"{args.model.replace('/', '_')}_timm_{train_mode_with_suffix}"
    model_output_dir = os.path.join(args.output_dir, dir_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Create training summary
    summary_path = os.path.join(model_output_dir, f"training_summary_{args.model.replace('/', '_')}_{args.train_mode}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Microalgae Model Training Summary - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")

        f.write("COMMAND LINE ARGUMENTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Training Mode: {args.train_mode}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Number of Epochs: {args.num_epochs}\n")
        f.write(f"Learning Rate: {args.learning_rate}\n")
        f.write(f"Image Size: {args.img_size}\n")
        f.write(f"Root Directory: {args.root_dir}\n")
        f.write(f"Output Directory: {args.output_dir}\n")
        f.write(f"Run Suffix: {args.run_suffix}\n")
        f.write(f"Use Custom Weights: {args.use_custom_weights}\n")
        f.write(f"Custom Weight Path: {args.custom_weight_path}\n")

        f.write("SYSTEM INFORMATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Device: {device}\n")
        f.write(f"PyTorch Version: {torch.__version__}\n")
        if torch.cuda.is_available():
            f.write(f"CUDA Version: {torch.version.cuda}\n")
            f.write(f"GPU Name: {torch.cuda.get_device_name(0)}\n")
            f.write(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
        f.write("\n")

        f.write("DATASET INFORMATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Number of Classes: {num_classes}\n")
        f.write(f"Class Names: {train_dataset.classes}\n")
        f.write(f"Training Samples: {len(train_dataset)}\n")
        f.write(f"Validation Samples: {len(val_dataset)}\n")
        f.write(f"Train Directory: {train_dir}\n")
        f.write(f"Validation Directory: {val_dir}\n\n")

        f.write("MODEL INFORMATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model Architecture: {args.model}\n")
        f.write(f"Training Mode: {args.train_mode}\n")

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        classifier_params = 0
        backbone_params = 0
        classifier_names = ['head', 'classifier']

        for name, param in model.named_parameters():
            if param.requires_grad:
                if any(cls_name in name for cls_name in classifier_names):
                    classifier_params += param.numel()
                else:
                    backbone_params += param.numel()

        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,} ({trainable_params/total_params:.2%})\n")
        f.write(f"  - Classifier Parameters: {classifier_params:,}\n")
        f.write(f"  - Backbone Parameters: {backbone_params:,}\n\n")

        f.write("TRAINING CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Loss Function: CrossEntropyLoss\n")
        f.write(f"Optimizer: Adam\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Scheduler: ReduceLROnPlateau\n")
        f.write(f"  - Factor: 0.5\n")
        f.write(f"  - Patience: 5\n")
        f.write(f"  - Min LR: 1e-6\n")
        f.write(f"  - Cooldown: 2\n\n")
        
        f.write("TRAINING RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Best Validation Loss: {best_val_loss:.4f}\n")
        f.write(f"Best Epoch: {best_epoch + 1}\n")
        f.write(f"Total Epochs Trained: {num_epochs}\n")
        f.write(f"Final Training Accuracy: {train_acc[-1]:.2f}%\n")
        f.write(f"Final Training Loss: {train_loss[-1]:.4f}\n")
        f.write(f"Final Validation Accuracy: {val_acc[-1]:.2f}%\n")
        f.write(f"Final Validation Loss: {val_loss[-1]:.4f}\n\n")

        f.write("TIMING INFORMATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Training Time: {str(timedelta(seconds=int(time_stats['total_training_time'])))}\n")
        f.write(f"Average Epoch Time: {str(timedelta(seconds=int(time_stats['average_epoch_time'])))}\n")
        f.write(f"Fastest Epoch: {str(timedelta(seconds=int(time_stats['fastest_epoch'])))}\n")
        f.write(f"Slowest Epoch: {str(timedelta(seconds=int(time_stats['slowest_epoch'])))}\n\n")

        f.write("LEARNING RATE HISTORY (First 10 Values)\n")
        f.write("-" * 40 + "\n")
        for i, lr in enumerate(lr_history[:20]):
            f.write(f"Epoch {i+1}: {lr:.2e}\n")
        f.write("\n")

        f.write("EARLY STOPPING TRIGGERED")
        f.write("-" * 40 + "\n")
        f.write(f"Stopped at epoch {time_stats['actual_epochs_trained']}/{num_epochs}")
        f.write(f"Saved {num_epochs - time_stats['actual_epochs_trained']} epochs")

        f.write("OUTPUT FILES\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model Directory: {model_output_dir}\n")
        f.write(f"Best Model: best_model_{args.model.replace('/', '_')}_{args.train_mode}.pth\n")
        f.write(f"Metrics: metrics_{args.model.replace('/', '_')}_{args.train_mode}.pkl\n")
        f.write(f"Accuracy Plot: accuracy_over_epochs.png\n")
        f.write(f"Loss Plot: loss_over_epochs.png\n")
        f.write(f"Summary: training_summary_{args.model.replace('/', '_')}_{args.train_mode}.txt\n\n")

        f.write("="*80 + "\n")
        f.write("Training completed successfully!\n")

    print(f"Training summary saved to '{summary_path}'.")
    print(f"Output dir: {model_output_dir}")

    # Save metrics
    metrics_path = os.path.join(model_output_dir, f"metrics_{args.model.replace('/', '_')}_{args.train_mode}.pkl")
    with open(metrics_path, "wb") as f:
        pickle.dump(metrics, f)
    print(f"Comprehensive metrics saved to '{metrics_path}'.")

    # Save best model
    best_model_path = os.path.join(model_output_dir, f"best_model_{args.model.replace('/', '_')}_{args.train_mode}.pth")
    torch.save(best_model_state, best_model_path)
    print(f"Best model saved to '{best_model_path}'.")

    # Create and save plots
    def save_plot(x_data, y_data, title, xlabel, ylabel, filename, colors=None, markers=None):
        plt.figure(figsize=(14, 10))
        if isinstance(y_data[0], list): 
            for i, data in enumerate(y_data): 
                plt.plot(data, label=colors[i][0], color=colors[i][1], marker=markers[i])
        else:  
            plt.plot(x_data, y_data, color=colors[0][1], marker=markers[0], label=colors[0][0])
        
        plt.title(title, fontsize=24, fontweight='bold')
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        plt.legend(loc='best', fontsize=14)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(model_output_dir, filename))
        plt.close()

    # Plots - accuracy and loss
    save_plot(
        range(num_epochs),
        [train_acc, val_acc],
        f"Accuracy over Epochs - {args.model} - {args.train_mode}",
        'Epoch',
        'Accuracy (%)',
        'accuracy_over_epochs.png',
        colors=[('Train Accuracy', 'blue'), ('Validation Accuracy', 'green')],
        markers=['o', 's']
    )
    save_plot(
        range(num_epochs),
        [train_loss, val_loss],
        f"Loss over Epochs - {args.model} - {args.train_mode}",
        'Epoch',
        'Loss',
        'loss_over_epochs.png',
        colors=[('Train Loss', 'blue'), ('Validation Loss', 'red')],
        markers=['o', 's']
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_hours = int(elapsed_time // 3600)
    elapsed_minutes = int((elapsed_time % 3600) // 60)
    elapsed_seconds = int(elapsed_time % 60)

    if early_stopping_info and early_stopping_info['triggered']:
        print(f"\n📊 Training Summary (Early Stopped):")
        print(f"   • Stopped at epoch {time_stats['actual_epochs_trained']}/{num_epochs}")
        print(f"   • Saved {num_epochs - time_stats['actual_epochs_trained']} epochs")
        print(f"   • Time saved: {str(timedelta(seconds=int((num_epochs - time_stats['actual_epochs_trained']) * time_stats['average_epoch_time'])))}")
    else:
        None

    formatted_end_time = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
    print(f"Ended at {formatted_end_time}!")
