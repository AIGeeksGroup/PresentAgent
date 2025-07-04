# PresentAgent: Multimodal Agent for Presentation Video Generation
This is the code repository for the paper:
> **PresentAgent: Multimodal Agent for Presentation Video Generation**
>
> Jingwei Shi\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)\*<sup>†</sup>, Biao Wu\*, Yanjie Liang\*, Meng Fang, Ling Chen, and [Yang Zhao](https://yangyangkiki.github.io/)<sup>#</sup>
>
> \*Equal contribution. <sup>†</sup>Project lead. <sup>#</sup>Corresponding author.
>
> ### [Paper]() | [HF Demo](https://huggingface.co/spaces/AIGeeksGroup/PresentAgent) | [Papers With Code]() | [HF Paper]()

To learn more about PresentAgent, please see the following presentation video, which was generated entirely by PresentAgent without any manual curation.

https://github.com/user-attachments/assets/240d3ae9-61a1-4e5f-98d7-9c20a99f4c2b

## Citation

If you use any content of this repo for your work, please cite the following our paper:
```
xxx
```

## Todo List

- ✅ code release
- ✅ api version
- **⬜️** local version
- ⬜️ paper release
- ⬜️ gradio demo

## Introduction

We present PresentAgent, a multimodal agent that transforms long-form documents into narrated presentation videos. While existing approaches are limited to generating static slides or text summaries, our method advances beyond these limitations by producing fully synchronized visual and spoken content that closely mimics human-style presentations. To achieve this integration, PresentAgent employs a modular pipeline that systematically segments the input document, plans and renders slide-style visual frames, generates contextual spoken narration with large language models and Text-to-Speech models, and seamlessly composes the final video with precise audio-visual alignment. Given the complexity of evaluating such multimodal outputs, we introduce PresentEval, a unified assessment framework powered by Vision-Language Models that comprehensively scores videos across three critical dimensions: content fidelity, visual clarity, and audience comprehension through prompt-based evaluation. Our experimental validation on a curated dataset of 30 document–presentation pairs demonstrates that PresentAgent approaches human-level quality across all evaluation metrics. These results highlight the significant potential of controllable multimodal agents in transforming static textual materials into dynamic, effective, and accessible presentation formats.

![image](presentagent/arch.png)

## 🔧Run Your PresentAgent

### 1. Install & Requirements

```bash
pip install git+https://github.com/icip-cas/PPTAgent.git
cd presentagent/MegaTTS3
```
**Requirements (for Linux)**

``` sh
pip install -r requirements.txt

# Set the root directory
export PYTHONPATH="/path/to/MegaTTS3:$PYTHONPATH"

# [Optional] Set GPU
export CUDA_VISIBLE_DEVICES=0

# If you encounter bugs with pydantic in inference, you should check if the versions of pydantic and gradio are matched.
# [Note] if you encounter bugs related with httpx, please check that whether your environmental variable "no_proxy" has patterns like "::"
```

**Requirements (for Windows)**
``` sh
pip install -r requirements.txt
conda install -y -c conda-forge pynini==2.1.5
pip install WeTextProcessing==1.0.3

# [Optional] If you want GPU inference, you may need to install specific version of PyTorch for your GPU from https://pytorch.org/.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# [Note] if you encounter bugs related with `ffprobe` or `ffmpeg`, you can install it through `conda install -c conda-forge ffmpeg`

# Set environment variable for root directory
set PYTHONPATH="C:\path\to\MegaTTS3;%PYTHONPATH%" # Windows
$env:PYTHONPATH="C:\path\to\MegaTTS3;%PYTHONPATH%" # Powershell on Windows
conda env config vars set PYTHONPATH="C:\path\to\MegaTTS3;%PYTHONPATH%" # For conda users

# [Optional] Set GPU
set CUDA_VISIBLE_DEVICES=0 # Windows
$env:CUDA_VISIBLE_DEVICES=0 # Powershell on Windows

```

### 2. Generate Via WebUI

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
   ### Usage

   First, you need to upload a PPT template and the document, then click **Generate Slides** to generate and download the PPT. After downloading the PPT, you can modify it in your own way and then click **PPT2Presentation**.
   ![image](presentagent/home.png)
   After uploading the PPT, you can click **Start Conversion** to make a presentation video.
   ![image](presentagent/ppt2presentation1.png)
   Finally, you will get a presentation video and watch it in the page or download it.
   ![image](presentagent/ppt2presentation2.png)

## 📁 Presentation Benchmark

### Doc2Present Benchmark

To support the evaluation of document to presentation video generation, we curate the **Doc2Present Benchmark**, a diverse dataset of document–presentation video pairs spanning multiple domains. As shown in the following figure, our benchmark encompasses four representative document types (academic papers, web pages, technical blogs, and slides) paired with human-authored videos, covering diverse real-world domains like education, research, and business reports.

![image](presentagent/datasets.jpg)

We collect **30 high-quality video samples** from **public platforms**, **educational repositories**, and **professional presentation archives**. Each video follows a structured narration format, combining slide-based visuals with synchronized voiceover. We manually align each video with its source document and ensure the following conditions are met: 

- The content structure of the video follows that of the document. 

- The visuals convey document information in a compact, structured form.
-  The narration and slides are well-aligned temporally.

The average document length is **3,000–8,000 words**, while the corresponding videos range from **1 to 2 minutes** and contain **5-10 slides**. This setting highlights the core challenge of the task: transforming dense, domain-specific documents into effective and digestible multimodal presentations.

### PresentEval

To assess the quality of generated presentation videos, we adopt two complementary evaluation strategies: Objective Quiz Evaluation and Subjective Scoring.

![image](presentagent/eval.jpg)

For each video, we provide the vision-language model with the complete set of slide images and the full narration transcript as a unified input—simulating how a real viewer would experience the presentation. 

- In Objective Quiz Evaluation, the model answers a fixed set of factual questions to determine whether the video accurately conveys the key information from the source content. 
- In Subjective Scoring, the model evaluates the video along three dimensions: the coherence of the narration, the clarity and design of the visuals, and the overall ease of understanding. 
- All evaluations are conducted without ground-truth references and rely entirely on the model’s interpretation of the presented content.

For Objective Quiz Evaluation, to evaluate whether a generated presentation video effectively conveys the core content of its source document, we use a fixed-question comprehension evaluation protocol. Specifically, we manually design five multiple-choice questions for each document, tailored to its content as follows:

|     Prensentation of Web Pages      | What is the main feature highlighted in the iPhone’s promotional webpage? |
| :---------------------------------: | ------------------------------------------------------------ |
|                 A.                  | A more powerful chip for faster performance                  |
|                 B.                  | A brighter and more vibrant display                          |
|                 C.                  | An upgraded camera system with better lenses                 |
|                 D.                  | A longer-lasting and more efficient battery                  |
| **Prensentation of Academic Paper** | What primary research gap did the authors aim to address by introducing the FineGym dataset? |
|                 A.                  | Lack of low-resolution sports footage for compression studies |
|                 B.                  | Need for fine-grained action understanding that goes beyond coarse categories |
|                 C.                  | Absence of synthetic data to replace human annotations       |
|                 D.                  | Shortage of benchmarks for background context recognition    |

For Subjective Scoring, to evaluate the quality of generated presentation videos, we adopt a prompt-based assessment using vision-language models. The prompts are as follows:

|     Video     | Scoring Prompt                                               |
| :-----------: | ------------------------------------------------------------ |
|  Narr. Coh.   | “How coherent is the narration across the video? Are the ideas logically connected and easy to follow?” |
| Visual Appeal | “How would you rate the visual design of the slides in terms of layout, aesthetics, and overall quality?” |
|  Comp. Diff.  | “How easy is it to understand the presentation as a viewer? Were there any confusing or contradictory parts?” |
|   **Audio**   | **Scoring Prompt**                                           |
|  Narr. Coh.   | “How coherent is the narration throughout the audio? Are the ideas logically structured and easy to follow?” |
| Audio Appeal  | “How pleasant and engaging is the narrator’s voice in terms of tone, pacing, and delivery?” |

## 🧪 Experiment

### ✳️ Comparative Study

|    Method    |       Model       | Quiz Accuracy | Video Score(mean) | Audio Score(mean) |
| :----------: | :---------------: | :-----------: | :---------------: | :---------------: |
|    Human     |       Human       |     0.56      |       4.47        |       4.80        |
| PresentAgent | Claude-3.7-sonnet |     0.64      |       4.00        |       4.53        |
| PresentAgent |    Qwen-VL-Max    |     0.52      |       4.47        |       4.60        |
| PresentAgent |  Gemini-2.5-pro   |     0.52      |       4.33        |       4.33        |
| PresentAgent | Gemini-2.5-flash  |     0.52      |       4.33        |       4.40        |
| PresentAgent |    GPT-4o-Mini    |     0.64      |       4.67        |       4.40        |
| PresentAgent |      GPT-4o       |     0.56      |       3.93        |       4.47        |

---

## ⭐ Contribute

We warmly welcome you to contribute to our project by submitting pull requests—your involvement is key to keeping our work at the cutting edge! Specifically, we encourage efforts to expand its compatibility with the **latest visual-language (VL) models** and **text-to-speech (TTS) models**, ensuring the project stays aligned with the most recent advancements in these rapidly evolving fields.

Beyond model updates, we also invite you to explore adding new features that could enhance the project’s functionality, usability, or versatility. Whether it’s optimizing existing workflows, introducing novel tools, or addressing unmet needs in the community, your creative contributions will help make this project more robust and valuable for everyone.

## Acknowledgement
We thank the authors of [PPTAgent](https://github.com/icip-cas/PPTAgent), [PPT Presenter](https://github.com/chaonan99/ppt_presenter), and [MegaTTS3](https://github.com/bytedance/MegaTTS3) for their open-source code.

