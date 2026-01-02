<div align="center">
  <h1>Attribute-Preserving Pseudo-Labeling for Diffusion-Based Face Swapping</h1>

  <p align="center">
    <a href="https://scholar.google.com/citations?user=A2PurdIAAAAJ&hl=en">Jiwon Kang<sup>1</sup></a>
    ·
    Yeji Choi<sup>1</sup>
    ·
    JoungBin Lee<sup>1</sup>
    ·
    <a href="https://scholar.google.com/citations?user=7cyLEQ0AAAAJ&hl=en/">Wooseok Jang<sup>1</sup></a>
    ·
    Jinhyeok Choi<sup>1</sup>
    ·
    Taekeun Kang<sup>2</sup>
    ·
    Yongjae Park<sup>2</sup>
    ·
    Myungin Kim<sup>2</sup>
    ·
    <a href="https://cvlab.kaist.ac.kr/">Seungryong Kim<sup>1</sup></a>
  </p>

  <p align="center">
    <sup>1</sup>KAIST AI &nbsp; <sup>2</sup>SAMSUNG
  </p>

  <h3 align="center"><a href="#">Paper</a> | <a href="https://cvlab-kaist.github.io/APPLE">Project Page</a></h3>

  <div align="center">
    <img src="assets/new_paper_figures/teaser.png" width="100%">
  </div>
</div>

## Abstract
Face swapping aims to transfer the identity of a source face onto a target face while preserving target-specific attributes such as pose, expression, lighting, skin tone, and makeup. However, since real ground truth for face swapping is unavailable, achieving both accurate identity transfer and high-quality attribute preservation remains challenging. Recent diffusion-based approaches attempt to improve visual fidelity through conditional inpainting on masked target images, but the masked condition removes crucial appearance cues, resulting in plausible yet misaligned attributes due to the lack of explicit supervision. To address these limitations, we propose **APPLE** (**A**ttribute-**P**reserving **P**seudo-**L**ab**e**ling), a diffusion-based teacher–student framework that enhances attribute fidelity through attribute-aware pseudo-label supervision. We reformulate face swapping as a conditional deblurring task to more faithfully preserve target-specific attributes such as lighting, skin tone, and makeup. In addition, we introduce an attribute-aware inversion scheme to further improve detailed attribute preservation. Through an elaborate attribute-preserving design for teacher learning, **APPLE** produces high-quality pseudo triplets that explicitly provide the student with direct face-swapping supervision. Overall, **APPLE** achieves state-of-the-art performance in terms of attribute preservation and identity transfer, producing more photorealistic and target-faithful results.
