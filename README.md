# LMFM-12

<img width="1920" height="1080" alt="1" src="https://github.com/user-attachments/assets/68f7e7b4-e37a-4144-8e00-ec06082f8173" />


# **A Morphologically Diverse Freshwater Microalgae Dataset for Deep Learning-Based Classification with Transfer Learning Analysis**

Aimi Alina Binti Hussin, Mohd Ibrahim Shapiai, Shaza Eva Mohamad, Koji Iwamoto, Mohd Farizal Kamaroddin, Kazuhiro Takemoto

We introduce the Light Microscopy Freshwater Microalgae (LMFM-12) dataset, comprising 7,555 curated images from 12 species under multiple magnifications, the largest publicly available freshwater microalgae light microscopy dataset to date. Comprehensive evaluation of seven CNN architectures reveals that randomly initialized models achieve accuracies exceeding 98%, approaching the performance of fully fine-tuned ImageNet-pretrained networks. Through the first application of Singular Vector Canonical Correlation Analysis (SVCCA) to microalgae classification, we suggest that random initialization develops different representational strategies that may be more suited to microscopic morphology, contrasting sharply with ImageNet-adapted features. Despite achieving comparable accuracy, these divergent approaches suggest that effective microalgae classification emerges from learning specialized microscopic features rather than adapting generic visual patterns. Cross-domain evaluation reveals that while ImageNet pretraining achieves superior generalization performance, Grad-CAM++ analysis shows distinct attention patterns between ImageNet and LMFM-12 initialization strategies. This positions LMFM-12 as a useful resource for advancing automated microalgae classification research.

#### Keywords: microalgae dataset, transfer learning, datasets comparison, SVCCA, image classification

---

### Sections in this paper:
1) Comparative analysis of model performance across initialization strategies (RD, FT and FB)
2) Analysis of SVCCA hidden representational analysis
3) Effect of transfer learning to other publicly available phytoplankton datasets
4) Attention pattern analysis through Grad-CAM++

---

### Models used are from 2 model libraries:

#### *Timm*:  
- MobileNet V2 (`mobilenetv2_100`)  
- DenseNet 121 (`densenet121`)  
- ResNext 50 (`resnext50_32x4d`)  
- ConvNext Base (`convnext_base`)  
- VGG 19 (`vgg19_bn`)  

#### *Torchvision*:  
- ShuffleNet V2 (`shufflenet_v2_x1_0`)  
- EfficientNet V2 (`efficientnet_v2_s`)

### lmfm_pretrained_weight
This folder contains weights for the three top-performing models on the LMFM-12 dataset, selected based on their overall performance. These models demonstrate strong ability to learn microalgae-specific features from scratch. 
(See Table 2 in the paper for the corresponding results)
   
---

If you find our code/dataset/evaluation useful in your research, please cite:
```
Hussin, A. A., Eva Mohamad, S., Iwamoto, K., & Takemoto, K. (2026). LMFM-12 [Data set]. Zenodo. https://doi.org/10.5281/zenodo.17669912
```
