# PresentAgent: Multimodal Agent for Presentation Video Generation
This is the code repository for the paper:
> **PresentAgent: Multimodal Agent for Presentation Video Generation**
>
> Jingwei Shi\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)\*<sup>†</sup>, Biao Wu\*, Yanjie Liang\*, Meng Fang, Ling Chen, and [Yang Zhao](https://yangyangkiki.github.io/)<sup>#</sup>
>
> \*Equal contribution. <sup>†</sup>Project lead. <sup>#</sup>Corresponding author.
>
> **[[arXiv]]()** **[[Paper with Code]]()** **[[HF Paper]]()**

To learn more about PresentAgent, please see the following presentation video, which was generated entirely by PresentAgent without any manual curation.

https://github.com/user-attachments/assets/240d3ae9-61a1-4e5f-98d7-9c20a99f4c2b

---


## Citation

If you use any content of this repo for your work, please cite the following our paper:
```
xxx
```

---

## Introduction
We present PresentAgent, a multimodal agent that transforms long-form documents into narrated presentation videos. While existing approaches are limited to generating static slides or text summaries, our method advances beyond these limitations by producing fully synchronized visual and spoken content that closely mimics human-style presentations. To achieve this integration, PresentAgent employs a modular pipeline that systematically segments the input document, plans and renders slide-style visual frames, generates contextual spoken narration with large language models and Text-to-Speech models, and seamlessly composes the final video with precise audio-visual alignment. Given the complexity of evaluating such multimodal outputs, we introduce PresentEval, a unified assessment framework powered by Vision-Language Models that comprehensively scores videos across three critical dimensions: content fidelity, visual clarity, and audience comprehension through prompt-based evaluation. Our experimental validation on a curated dataset of 30 document–presentation pairs demonstrates that PresentAgent approaches human-level quality across all evaluation metrics. These results highlight the significant potential of controllable multimodal agents in transforming static textual materials into dynamic, effective, and accessible presentation formats.

![image](https://github.com/momomoxiaobai/Source/blob/main/Images/arch.png)


## Resource: PresentAgent Papers



## 🔧 Installation & Setup

### 1. Install & Run PPTAgent

#### Installation Guide

```bash
pip install git+https://github.com/icip-cas/PPTAgent.git
```

### 2. Install megatts3

You can Install megatts3 by the following Web Page: [bytedance/MegaTTS3](https://github.com/bytedance/MegaTTS3)

### 3. Generate Via WebUI

1. **Serve Backend**

   Initialize your models in `presentagent/backend.py`:
   ```python
   language_model = AsyncLLM(
       model="Qwen2.5-72B-Instruct",
       api_base="http://localhost:7812/v1"
   )
   vision_model = AsyncLLM(model="gpt-4o-2024-08-06")
   text_embedder = AsyncLLM(model="text-embedding-3-small")
   ```
   Or use the environment variables:

   ```bash
   export OPENAI_API_KEY="your_key"
   export API_BASE="http://your_service_provider/v1"
   export LANGUAGE_MODEL="Qwen2.5-72B-Instruct-GPTQ-Int4"
   export VISION_MODEL="gpt-4o-2024-08-06"
   export TEXT_MODEL="text-embedding-3-small"
   ```

   ```bash
   python backend.py
   ```

2. **Launch Frontend**

   > Note: The backend API endpoint is configured at `presentagent/vue.config.js`

   ```bash
   cd presentagent
   npm install
   npm run serve
   ```
<p align="left">
  <img src="https://github.com/AIGeeksGroup/PresentAgent/tree/main/presentagent/home.png" width="28%" />
  <img src="https://github.com/AIGeeksGroup/PresentAgent/tree/main/presentagent/ppt2presentation1.png" width="28%" />
  <img src="https://github.com/AIGeeksGroup/PresentAgent/tree/main/presentagent/ppt2presentation2.png" width="28%" />
</p>



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

## Acknowledgement
We thank the authors of [PPTAgent](https://github.com/icip-cas/PPTAgent), [PPT Presenter](https://github.com/chaonan99/ppt_presenter), and [MegaTTS3](https://github.com/bytedance/MegaTTS3) for their open-source code.

