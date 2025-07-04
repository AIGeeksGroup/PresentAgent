# PresentAgent: Multimodal Agent for Presentation Video Generation
This is the code repository for the paper:
> **PresentAgent: Multimodal Agent for Presentation Video Generation**
>
> Jingwei Shi\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)\*<sup>†</sup>, Biao Wu\*, Yanjie Liang\*, Meng Fang, Ling Chen, and [Yang Zhao](https://yangyangkiki.github.io/)<sup>#</sup>
>
> \*Equal contribution. <sup>†</sup>Project lead. <sup>#</sup>Corresponding author.
>
> **[[arXiv]]()** **[[Paper with Code]]()** **[[HF Paper]]()**

![image](https://github.com/momomoxiaobai/Source/blob/main/Images/arch.png)


## Citation

If you use any content of this repo for your work, please cite the following our paper:
```
xxx
```

## Introduction
We present PresentAgent, a multimodal agent that transforms long-form documents into narrated presentation videos. While existing approaches are limited to generating static slides or text summaries, our method advances beyond these limitations by producing fully synchronized visual and spoken content that closely mimics human-style presentations. To achieve this integration, PresentAgent employs a modular pipeline that systematically segments the input document, plans and renders slide-style visual frames, generates contextual spoken narration with large language models and Text-to-Speech models, and seamlessly composes the final video with precise audio-visual alignment. Given the complexity of evaluating such multimodal outputs, we introduce PresentEval, a unified assessment framework powered by Vision-Language Models that comprehensively scores videos across three critical dimensions: content fidelity, visual clarity, and audience comprehension through prompt-based evaluation. Our experimental validation on a curated dataset of 30 document–presentation pairs demonstrates that PresentAgent approaches human-level quality across all evaluation metrics. These results highlight the significant potential of controllable multimodal agents in transforming static textual materials into dynamic, effective, and accessible presentation formats.
Code will be available at [https://github.com/AIGeeksGroup/PresentAgent](https://github.com/AIGeeksGroup/PresentAgent).

![image](https://github.com/momomoxiaobai/Source/blob/main/Images/first.jpg)


## Resource: PresentAgent Papers



## 🔧 Installation & Setup

```bash
git clone https://github.com/AIGeeksGroup/PresentAgent.git
cd PresentAgent
pip install -r requirements.txt
```

To use on **Google Colab** or **Kaggle**, enable GPU and configure data mounting as required.

---



```

```


### 🧿 Eye Diseases Classification (RGB)

* **URL**: [https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)
* Classes: Cataract, Diabetic Retinopathy, Glaucoma, Normal
* Balanced dataset
* Random split: 80% train / 20% test

<p align="left">
  <img src="https://github.com/AIGeeksGroup/MediAug/blob/main/eye_disease.jpg" width="28%" />
  <img src="https://github.com/AIGeeksGroup/MediAug/blob/main/eye_tsne.jpg" width="33%" />
</p>

The left pie chart shows the class distribution across the four categories, demonstrating good class balance. The right t-SNE plot provides a feature-level visualization of the high-dimensional distribution of eye disease samples after dimensionality reduction.

### 🧠 Brain Tumor MRI Classification (Grayscale)

* **URL**: [https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri/data](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri/data)
* Classes: Glioma, Meningioma, Pituitary, No Tumor
* Imbalanced dataset
* Random split: 80% train / 20% test

<p align="left">
  <img src="https://github.com/AIGeeksGroup/MediAug/blob/main/brain_disease.jpg" width="28%" />
  <img src="https://github.com/AIGeeksGroup/MediAug/blob/main/brain_tsne.jpg" width="33%" />
</p>

The pie chart (left) illustrates the class distribution among four tumor categories. The t-SNE plot (right) visualizes the distribution of brain tumor samples in a two-dimensional space, reflecting their separability and overlap in feature space.


---

## 🏗️ Method Overview

We evaluate six mix-based visual augmentation techniques:

* `MixUp`: Interpolation between image-label pairs
* `YOCO`: Patch-based diverse local/global transforms
* `CropMix`: Multi-scale random crop blending
* `CutMix`: Box-replace image regions + interpolated labels
* `AugMix`: Diverse chained augmentations with consistency
* `SnapMix`: CAM-based semantic-aware mixing

Each method is evaluated on two backbones:

* **ResNet-50** (CNN)
* **ViT-B** (Transformer)

---

## 🧪 Experiments

### ✳️ Comparative Study

| Dataset     | Model     | Best Aug | Accuracy |
| ----------- | --------- | -------- | -------- |
| Brain MRI   | ResNet-50 | MixUp    | 79.19%   |
| Brain MRI   | ViT-B     | SnapMix  | 99.44%   |
| Eye Disease | ResNet-50 | YOCO     | 91.60%   |
| Eye Disease | ViT-B     | CutMix   | 97.94%   |

### 🔬 Ablation Study

Hyperparameter sweep for CutMix (alpha). Best performance at:

* ResNet-50: α = 1.0 → 91.83% Accuracy
* ViT-B: α = 1.0 → 97.94% Accuracy

---

## 💻 Training & Evaluation

To run an experiment with MediAug, follow these steps:

1. **Choose dataset**: `eye` or `brain`
2. **Select model**: `resnet50` or `vit_b`
3. **Pick augmentation method**: one of `mixup`, `cutmix`, `snapmix`, `yoco`, `cropmix`, `augmix`

### Example Commands

Run brain tumor classification with ViT-B and SnapMix:

```bash
python train.py --dataset brain --model vit_b --aug snapmix
```

Run eye disease classification with ResNet-50 and YOCO:

```bash
python train.py --dataset eye --model resnet50 --aug yoco
```

Evaluate a trained model on the test set:

```bash
python evaluate.py --dataset brain --model vit_b --checkpoint ./checkpoints/vit_b_snapmix.pt
```

Visualize augmentation effects (optional):

```bash
python visualize.py --dataset eye --aug mixup --output_dir ./visuals
```

Training details:

* Epochs: 50
* Optimizer: Adam
* Learning Rate: 0.001
* Batch Size: 32
* Image Size: 224×224
* GPU: Tesla T4 or A100 (Google Colab, via mounted Google Drive)
* CPU: Intel Xeon, 80GB RAM

> **Note:** All experiments were conducted on Google Colab. The datasets were uploaded to Google Drive and accessed using standard Colab notebook mounts (e.g., `from google.colab import drive`). Kaggle was not used for runtime.

* Epochs: 50
* Optimizer: Adam
* Learning Rate: 0.001
* Image Size: 224x224
* Hardware: Tesla T4 / A100, Intel Xeon CPU, 80GB RAM

```bash
python train.py --dataset eye --model resnet50 --aug mixup
```

---

## 🧠 Model Zoo

| Model     | Dataset | Aug     | Accuracy |
| --------- | ------- | ------- | -------- |
| ResNet-50 | Eye     | YOCO    | 91.60%   |
| ViT-B     | Brain   | SnapMix | 99.44%   |


---

## 📓 Notebooks

The following notebooks train and evaluate models used in our experiments:

* `resnet50.ipynb`: Trains a ResNet-50 model on the selected dataset with different augmentation strategies. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/Training%20model/resnet50.ipynb)
* `VIT-B.ipynb`: Trains a ViT-B (Vision Transformer) model on the selected dataset and compares augmentation effects. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/Training%20model/VIT-B.ipynb)

The following notebooks apply batch augmentation and visualization on the full **Brain MRI** dataset:

* `AugMix_brain.ipynb`: Applies AugMix to the entire brain dataset and visualizes a batch of augmented images. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/Image%20Augmentation%20Code/Image%20augment%20batch%20processing%20code/brain/AugMix_brain.ipynb)
* `CropMix_brain.ipynb`: Performs CropMix augmentation across the brain dataset with comparative visualization. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/Image%20Augmentation%20Code/Image%20augment%20batch%20processing%20code/brain/CropMix_brain.ipynb)
* `CutMix_brain.ipynb`: Shows CutMix applied to MRI samples in batch for augmentation analysis. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/Image%20Augmentation%20Code/Image%20augment%20batch%20processing%20code/brain/CutMix_brain.ipynb)
* `MixUp_brain.ipynb`: Executes MixUp over MRI images and plots combined outputs. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/Image%20Augmentation%20Code/Image%20augment%20batch%20processing%20code/brain/MixUp_brain.ipynb)
* `SnapMix_brain.ipynb`: Demonstrates CAM-based SnapMix on brain images at dataset level. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/Image%20Augmentation%20Code/Image%20augment%20batch%20processing%20code/brain/SnapMix_brain.ipynb)
* `YOCO_brain.ipynb`: Applies YOCO to a batch of brain samples and shows spatially mixed results. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/Image%20Augmentation%20Code/Image%20augment%20batch%20processing%20code/brain/YOCO_brain.ipynb)

The following notebooks apply batch augmentation and visualization on the full **Eye Disease** dataset:

* `AugMix_eye.ipynb`: Applies AugMix on the entire eye disease dataset with visual comparisons. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/Image%20Augmentation%20Code/Image%20augment%20batch%20processing%20code/eye/AugMix_eye.ipynb)
* `CropMix_eye.ipynb`: Runs CropMix augmentation over eye images and displays batched transformations. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/Image%20Augmentation%20Code/Image%20augment%20batch%20processing%20code/eye/CropMix_eye.ipynb)
* `CutMix_eye.ipynb`: Demonstrates CutMix applied to eye fundus images with batch-level visualization. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/Image%20Augmentation%20Code/Image%20augment%20batch%20processing%20code/eye/CutMix_eye.ipynb)
* `MixUp_eye.ipynb`: Mixes image-label pairs from the eye dataset and renders visual effects. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/Image%20Augmentation%20Code/Image%20augment%20batch%20processing%20code/eye/MixUp_eye.ipynb)
* `SnapMix_eye.ipynb`: Showcases SnapMix on eye disease samples with semantic-preserving augmentation. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/Image%20Augmentation%20Code/Image%20augment%20batch%20processing%20code/eye/SnapMix_eye.ipynb)
* `YOCO_eye.ipynb`: Uses YOCO to enhance eye data samples with region-wise mixed transforms. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/Image%20Augmentation%20Code/Image%20augment%20batch%20processing%20code/eye/YOCO_eye.ipynb)

The following notebooks demonstrate how each augmentation method is applied to a single medical image:

* `AugMix_for_single_picture.ipynb`: Applies AugMix transformations step-by-step to one image and visualizes the results. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/Image%20Augmentation%20Code/AugMix_for_single_picture.ipynb)
* `CropMix_for_single_picture.ipynb`: Demonstrates the CropMix augmentation process with visualization on a single image. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/Image%20Augmentation%20Code/CropMix_for_single_picture.ipynb)
* `CutMix_for_single_picture.ipynb`: Simulates CutMix augmentation by mixing image patches and overlays on one image. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/Image%20Augmentation%20Code/CutMix_for_single_picture.ipynb)
* `MixUp_for_single_picture.ipynb`: Shows how MixUp blends two images and labels, visualized clearly. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/Image%20Augmentation%20Code/MixUp_for_single_picture.ipynb)
* `SnapMix_for_single_picture.ipynb`: Explains SnapMix strategy by combining semantic patches with attention maps. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/Image%20Augmentation%20Code/SnapMix_for_single_picture.ipynb)
* `YOCO_for_single_picture.ipynb`: Visualizes YOCO's patch-wise mixed local augmentations on a single image. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/Image%20Augmentation%20Code/YOCO_for_single_picture.ipynb)

---

For questions, contact [**y.zhao2@latrobe.edu.au**](mailto:y.zhao2@latrobe.edu.au).
