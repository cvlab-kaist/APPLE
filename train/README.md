# Training for FLUX

## Table of Contents
- [Training for FLUX](#training-for-flux)
  - [Table of Contents](#table-of-contents)
  - [Environment Setup](#environment-setup)
  - [Dataset Preparation](#dataset-preparation)
  - [Quick Start](#quick-start)
  - [Basic Training](#basic-training)
    - [Tasks from OminiControl](#tasks-from-ominicontrol)
    - [Creating Your Own Task](#creating-your-own-task)
    - [Training Configuration](#training-configuration)
      - [Batch Size](#batch-size)
      - [Optimizer](#optimizer)
      - [LoRA Configuration](#lora-configuration)
      - [Trainable Modules](#trainable-modules)
  - [Advanced Training](#advanced-training)
    - [Multi-condition](#multi-condition)
    - [Efficient Generation (OminiControl2)](#efficient-generation-ominicontrol2)
      - [Feature Reuse (KV-Cache)](#feature-reuse-kv-cache)
      - [Compact Encoding Representation](#compact-encoding-representation)
      - [Token Integration (for Fill task)](#token-integration-for-fill-task)
  - [Citation](#citation)

## Environment Setup

1. Create and activate a new conda environment:
   ```bash
   conda create -n omini python=3.10
   conda activate omini
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

1. Download [Subject200K](https://huggingface.co/datasets/Yuanshi/Subjects200K) dataset for subject-driven generation:
   ```bash
   bash train/script/data_download/data_download1.sh
   ```

2. Download [text-to-image-2M](https://huggingface.co/datasets/jackyhate/text-to-image-2M) dataset for spatial alignment control tasks:
   ```bash
   bash train/script/data_download/data_download2.sh
   ```
   
   **Note:** By default, only a few files will be downloaded. You can edit `data_download2.sh` to download more data, and update the config file accordingly.

## Quick Start

Use these scripts to start training immediately:

1. **Subject-driven generation**:
   ```bash
   bash train/script/train_subject.sh
   ```

2. **Spatial control tasks** (Canny-to-image, colorization, depth map, etc.):
   ```bash
   bash train/script/train_spatial_alignment.sh
   ```

3. **Multi-condition training**:
   ```bash
   bash train/script/train_multi_condition.sh
   ```

4. **Feature reuse** (OminiControl2):
   ```bash
   bash train/script/train_feature_reuse.sh
   ```

5. **Compact token representation** (OminiControl2):
   ```bash
   bash train/script/train_compact_token_representation.sh
   ```

6. **Token integration** (OminiControl2):
   ```bash
   bash train/script/train_token_intergration.sh
   ```

## Basic Training

### Tasks from OminiControl
<a href="https://arxiv.org/abs/2411.15098"><img src="https://img.shields.io/badge/ariXv-2411.15098-A42C25.svg" alt="arXiv"></a>

1. Subject-driven generation:
   ```bash
   bash train/script/train_subject.sh
   ```

2. Spatial control tasks (using canny-to-image as example):
   ```bash
   bash train/script/train_spatial_alignment.sh
   ```

   <details>
   <summary>Supported tasks</summary>

   * Canny edge to image (`canny`)
   * Image colorization (`coloring`)
   * Image deblurring (`deblurring`)
   * Depth map to image (`depth`)
   * Image to depth map (`depth_pred`)
   * Image inpainting (`fill`)
   * Super resolution (`sr`)
   
   🌟 Change the `condition_type` parameter in the config file to switch between tasks.
   </details>

**Note**: Check the **script files** (`train/script/`) and **config files** (`train/configs/`) for WanDB and GPU settings.

### Creating Your Own Task

You can create a custom task by building a new dataset and modifying the test code:

1. **Create a custom dataset:**
   Your custom dataset should follow the format of `Subject200KDataset` in `omini/train_flux/train_subject.py`. Each sample should contain:

   - Image: the target image (`image`)
   - Text: description of the image (`description`)
   - Conditions: image conditions for generation
   - Position delta:
     - Use `position_delta = (0, 0)` to align the condition with the generated image
     - Use `position_delta = (0, -a)` to separate them (a = condition width / 16)

   > **Explanation:**  
   > The model places both the condition and generated image in a shared coordinate system. `position_delta` shifts the condition image in this space.
   > 
   > Each unit equals one patch (16 pixels). For a 512px-wide condition image (32 patches), `position_delta = (0, -32)` moves it fully to the left.
   > 
   > This controls whether conditions and generated images share space or appear side-by-side.

2. **Modify the test code:**
   Define `test_function()` in `train_custom.py`. Refer to the function in `train_subject.py` for examples. Make sure to keep the `position_delta` parameter consistent with your dataset.

### Training Configuration

#### Batch Size
We recommend a batch size of 1 for stable training. And you can set `accumulate_grad_batches` to n to simulate a batch size of n. 

#### Optimizer
The default optimizer is `Prodigy`. To use `AdamW` instead, modify the config file:
```yaml
optimizer:
  type: AdamW
  lr: 1e-4
  weight_decay: 0.001
```

#### LoRA Configuration
Default LoRA rank is 4. Increase it for complex tasks (keep `r` and `lora_alpha` parameters the same):
```yaml
lora_config:
  r: 128
  lora_alpha: 128
```

#### Trainable Modules
The `target_modules` parameter uses regex patterns to specify which modules to train. See [PEFT Documentation](https://huggingface.co/docs/peft/package_reference/lora) for details.

Default configuration trains all modules affecting image tokens:
```yaml
target_modules: "(.*x_embedder|.*(?<!single_)transformer_blocks\\.[0-9]+\\.norm1\\.linear|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_k|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_q|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_v|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_out\\.0|.*(?<!single_)transformer_blocks\\.[0-9]+\\.ff\\.net\\.2|.*single_transformer_blocks\\.[0-9]+\\.norm\\.linear|.*single_transformer_blocks\\.[0-9]+\\.proj_mlp|.*single_transformer_blocks\\.[0-9]+\\.proj_out|.*single_transformer_blocks\\.[0-9]+\\.attn.to_k|.*single_transformer_blocks\\.[0-9]+\\.attn.to_q|.*single_transformer_blocks\\.[0-9]+\\.attn.to_v|.*single_transformer_blocks\\.[0-9]+\\.attn.to_out)"
```

To train only attention components (`to_q`, `to_k`, `to_v`), use:
```yaml
target_modules: "(.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_k|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_q|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_v|.*single_transformer_blocks\\.[0-9]+\\.attn.to_k|.*single_transformer_blocks\\.[0-9]+\\.attn.to_q|.*single_transformer_blocks\\.[0-9]+\\.attn.to_v)"
```

## Advanced Training

### Multi-condition
A basic multi-condition implementation is available in `train_multi_condition.py`:
```bash
bash train/script/train_multi_condition.sh
```

### Efficient Generation (OminiControl2)
<a href="https://arxiv.org/abs/2503.08280"><img src="https://img.shields.io/badge/ariXv-2503.08280-A42C25.svg" alt="arXiv"></a>

[OminiControl2](https://arxiv.org/abs/2503.08280) introduces techniques to improve generation efficiency:

#### Feature Reuse (KV-Cache)
1. Enable `independent_condition` in the config file during training:
   ```yaml
   model:
     independent_condition: true
   ```

2. During inference, set `kv_cache = True` in the `generate` function to speed up generation.

*Example:*
```bash
bash train/script/train_feature_reuse.sh
```

**Note:** Feature reuse speeds up generation but may slightly reduce performance and increase training time.

#### Compact Encoding Representation
Reduce the condition image resolution and use `position_scale` to align it with the output image:

```diff
train:
  dataset:
    condition_size: 
-     - 512
-     - 512
+     - 256
+     - 256
+   position_scale: 2
    target_size: 
      - 512
      - 512
```

*Example:*
```bash
bash train/script/train_compact_token_representation.sh
```

#### Token Integration (for Fill task)
Further reduce tokens by merging condition and generation tokens into a unified sequence. (Refer to [the paper](https://arxiv.org/abs/2503.08280) for details.)

*Example:*
```bash
bash train/script/train_token_intergration.sh
```

## Citation

If you find this code useful, please cite our papers:

```
@article{tan2024ominicontrol,
  title={OminiControl: Minimal and Universal Control for Diffusion Transformer},
  author={Tan, Zhenxiong and Liu, Songhua and Yang, Xingyi and Xue, Qiaochu and Wang, Xinchao},
  journal={arXiv preprint arXiv:2411.15098},
  year={2024}
}

@article{tan2025ominicontrol2,
  title={OminiControl2: Efficient Conditioning for Diffusion Transformers},
  author={Tan, Zhenxiong and Xue, Qiaochu and Yang, Xingyi and Liu, Songhua and Wang, Xinchao},
  journal={arXiv preprint arXiv:2503.08280},
  year={2025}
}
```