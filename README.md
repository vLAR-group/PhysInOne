<p align="center">
  <h1 align="center">PhysInOne: Visual Physics Learning and Reasoning in One Suite</h1>
  <p align="center">
    <strong>CVPR 2026</strong>
  </p>
  <p align="center">
    <a href="https://arxiv.org/pdf/2604.09415"><img src="https://img.shields.io/badge/arXiv-2604.09415-b31b1b.svg" alt="arXiv"></a>
    <a href="https://vlar-group.github.io/PhysInOne.html"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
    <a href="https://huggingface.co/datasets/vLAR-group/PhysInOne"><img src="https://img.shields.io/badge/🤗-Dataset-yellow" alt="Dataset"></a>
    <a href="#license"><img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg" alt="License"></a>
  </p>
</p>

<p align="center">
  <img src="teaser.png" alt="PhysInOne Teaser" width="100%">
</p>

## Overview

We present **PhysInOne**, the largest dataset addressing the critical scarcity of physically-grounded training data for AI systems.

### Scale and Diversity
- **2 million videos** generated from **153,810 dynamic 3D scenes**
- Covers **71 fundamental physical phenomena** in everyday environments, spanning four major domains: **Mechanics**, **Optics**, **Fluid Dynamics**, **Magnetism**
- Includes **2,231 common objects** tailored to daily physical interactions
- Enriched with **623 materials** across five categories: plastic, metal, wood, stone, and fabric
- Features **528 diverse 3D backgrounds** to ensure realism and environmental variety

### Scene Characteristics
- Each scene involves **1–3 physical phenomena**, reflecting real-world activities
- Supports **complex multi-object interactions**, with increasing scene complexity
- Average number of objects per scene: **3.9** (single-physics), **6.3** (double-physics), **7.8** (triple-physics)
- Each scene is captured from **13 viewpoints**: 12 static cameras and 1 moving camera

### Rich Annotations
- 3D geometry
- Semantic labels
- Object motion and dynamics
- Physical properties
- Natural-language scene descriptions

### Supported Applications
- Physics-aware video generation
- Short- and long-term future frame prediction
- Physical property estimation
- Motion transfer
- And more...

## Links

| Resource | Link |
|----------|------|
| 📄 Paper | [arXiv](https://arxiv.org/pdf/2604.09415) |
| 🌐 Project Page | [vlar-group.github.io/PhysInOne](https://vlar-group.github.io/PhysInOne.html) |
| 🤗 Dataset | [Hugging Face](https://huggingface.co/datasets/vLAR-group/PhysInOne) |

## Code

🚧 **Coming Soon** 🚧

Data processing and benchmark evaluation code will be released soon. Stay tuned!

## Citation

If you find this work useful, please cite:

```bibtex
@misc{zhou2026physinonevisualphysicslearning,
      title={PhysInOne: Visual Physics Learning and Reasoning in One Suite}, 
      author={Siyuan Zhou and Hejun Wang and Hu Cheng and Jinxi Li and Dongsheng Wang and Junwei Jiang and Yixiao Jin and Jiayue Huang and Shiwei Mao and Shangjia Liu and Yafei Yang and Hongkang Song and Shenxing Wei and Zihui Zhang and Peng Huang and Shijie Liu and Zhengli Hao and Hao Li and Yitian Li and Wenqi Zhou and Zhihan Zhao and Zongqi He and Hongtao Wen and Shouwang Huang and Peng Yun and Bowen Cheng and Pok Kazaf Fu and Wai Kit Lai and Jiahao Chen and Kaiyuan Wang and Zhixuan Sun and Ziqi Li and Haochen Hu and Di Zhang and Chun Ho Yuen and Bing Wang and Zhihua Wang and Chuhang Zou and Bo Yang},
      year={2026},
      eprint={2604.09415},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.09415}, 
}
```

## License

This project is licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

## Acknowledgements

We would like to express our sincere gratitude to all contributors who participated in human evaluations and data collection efforts.
