# 🎨 Diffusion 与多模态：从原理到实战

> 用你已经掌握的 LLM 基础，带你从 Transformer 的世界走进 Diffusion 和多模态的领域。
> 基于 [llm-math-foundations](https://github.com/CCODING04/llm-math-foundations) 的数学基础 + [llm-paper-tutorials](https://github.com/CCODING04/llm-paper-tutorials) 的论文精读方法论。

## 🎯 项目定位

这不是一个从零开始的 AI 入门课程。这是一个**衔接课程**——

**前提**：你已经通过 diy-llm / CS336 掌握了 Transformer、注意力机制、训练流程等 LLM 核心知识。

**目标**：在此基础上，延伸到两个前沿方向：
1. **Diffusion 模型**：图像/视频生成的核心技术
2. **多模态大模型**：让 LLM "看见" 世界

**为什么你能学**：Diffusion 的核心是 Transformer（DiT），多模态的核心是 LLM + 视觉编码器。你已经有了最重要的那块拼图。

---

## 🔗 与已有学习资料的衔接关系

```
你已经掌握的                      本教程要补的
──────────────                    ──────────────
llm-math-foundations              ──→ Diffusion 的数学（随机过程、变分推断）
  Ch01 线性代数                         Score Function、SDE
  Ch02 微积分                           
  Ch03 概率                              多模态对齐的数学
  Ch04 矩阵求导                        （对比学习、KL散度）
  Ch07 注意力数学
  Ch08 高级概率

llm-paper-tutorials               ──→ Diffusion/多模态论文精读
  01 Attention（Transformer）            DDPM、DiT、CLIP、LLaVA
  05 Chinchilla（Scaling Laws）           Diffusion 的 Scaling
  06 InstructGPT（RLHF）                 多模态 RLHF
  07 LLaMA（架构设计）                    多模态架构设计
  08 LoRA（高效微调）                     多模态微调

diy-llm / CS336                   ──→ Diffusion/多模态实践
  Ch2 分词器                              图像分词（VAE Tokenizer）
  Ch4 Transformer                        DiT = Diffusion + Transformer
  Ch7 CUDA                               Diffusion 的加速采样
  Ch10 微调                              多模态指令微调

makemore                          ──→ 生成模型直觉
  Part1-2（统计→神经网络）                从自回归生成到扩散生成
  Part4（反向传播）                       Score Matching 的梯度本质
```

---

## 📖 教程结构：三大模块 × 10 章

### 模块 A：生成模型基础（1章）

#### A1 生成模型全景：从自回归到扩散

**核心内容**：GAN / VAE / Flow / Diffusion 四大生成范式对比；为什么 Diffusion 赢了；生成模型的发展脉络

**衔接已有知识**：
- makemore Part1-2 的语言模型生成（自回归生成）
- llm-paper-tutorials 03-GPT-2（自回归生成大规模应用）

**参考资源**：
| 类型 | 资源 | 对应内容 |
|------|------|---------|
| 🎓 课程 | Stanford CS236: Deep Generative Models (deepgenerativemodels.github.io) | 整个课程就是生成模型全景：VAE/Flow/GAN/Diffusion |
| 🎓 课程 | Stanford CS231n 2025 Lecture 13: Generative Models Part 1 | GAN + VAE 基础（slides: cs231n.stanford.edu） |
| 📄 论文 | NIPS 2016 Tutorial: Generative Adversarial Networks (Goodfellow) | GAN 的经典入门教程论文 |
| 📄 论文 | Auto-Encoding Variational Bayes (Kingma & Welling, 2013) | VAE 奠基论文 |
| 🎬 视频 | DeepLearning.AI: How Diffusion Models Work (deeplearning.ai/courses/diffusion-models) | 前15分钟有生成模型全景对比 |
| 📝 博客 | Lil'Log: Generative Models (lilianweng.github.io) | GAN/VAE/Flow/Diffusion 图文对比，非常清晰 |

---

### 模块 B：Diffusion 模型（5章）

#### B1 Diffusion 直觉：加噪与去噪

**核心内容**：前向过程（逐步加噪）可视化、反向过程（逐步去噪）直觉、核心思想——知道怎么加噪声就能学怎么去掉噪声

**衔接已有知识**：
- llm-math-foundations Ch03 概率分布（高斯分布、条件概率）
- llm-math-foundations Ch06 信息论（熵、信息量）
- makemore 的训练过程直觉（从噪声中学到结构）

**参考资源**：
| 类型 | 资源 | 对应内容 |
|------|------|---------|
| 🎓 短课 | **DeepLearning.AI: How Diffusion Models Work** (deeplearning.ai/courses/diffusion-models) | ⭐ **首选预习**。1小时，Sharon Zhou 主讲，从零构建 Diffusion，含 Jupyter notebook |
| 🎓 短课 | Coursera 同步版 (coursera.org/projects/how-diffusion-models-work-project) | 同上，Coursera 版本 |
| 🎓 课程 | **MIT 6.S183: Practical Diffusion Models** (practical-diffusion.org) | ⭐ **核心课程**。IAP 2026，含 Problem Sets + 采样可视化实验 |
| 🎬 视频 | YouTube: How Diffusion Models Work (DeepLearningAI 频道, 21K views) | 免费预览版，Andrew Ng 介绍 + Sharon Zhou 讲解 |
| 📝 博客 | What are Diffusion Models? (IBM Think) | 直觉级解释，图文并茂 |
| 📄 论文 | **DDPM** (Ho et al., 2020, arXiv:2006.11239) | 奠基论文，本节主要看 Figure 2（加噪过程图）和前两节 |

#### B2 DDPM 数学：从直觉到公式

**核心内容**：前向过程马尔可夫链数学推导、反向过程推导、噪声预测目标、训练目标（ELBO→简化形式）推导

**衔接已有知识**：
- llm-math-foundations Ch04 矩阵求导（梯度形状分析）
- llm-math-foundations Ch08 变分推断（ELBO、KL散度）
- diy-llm Ch3 训练原语（损失函数推导类比）

**参考资源**：
| 类型 | 资源 | 对应内容 |
|------|------|---------|
| 🎓 课程 | **Stanford CS236: Deep Generative Models** (deepgenerativemodels.github.io) | 专门的 Diffusion 课时，含数学推导作业 |
| 🎓 讲座 | **Stanford CS231n 2025 Lecture 14: Generative Models Part 2** | Score Function、SDE 视角、数学推导（slides: cs231n.stanford.edu/slides/2025） |
| 📄 论文 | **DDPM** (Ho et al., 2020) | 数学推导主要参考 Section 3-4 |
| 📄 论文 | **Score-Based Generative Modeling through SDEs** (Song et al., 2021, ICLR) | SDE 统一视角，更优雅的数学框架 |
| 📄 论文 | **Understanding Diffusion Models: A Unified Perspective** (Luo, 2022) | ⭐ 最佳数学综述，ELBO→DDPM 推导链，推荐精读 |
| 📝 博客 | Lil'Log: What are Diffusion Models? (lilianweng.github.io) | 数学推导+直觉，Calvin Luo 的博客 |

#### B3 DDIM 与加速采样

**核心内容**：从随机到确定性采样、跳步采样、ODE 视角、Consistency Models

**衔接已有知识**：
- llm-math-foundations Ch05 优化（梯度下降的类比——迭代优化）
- diy-llm Ch7 CUDA（采样效率的工程意义）

**参考资源**：
| 类型 | 资源 | 对应内容 |
|------|------|---------|
| 📄 论文 | **DDIM** (Song et al., 2020, arXiv:2010.02502) | 核心论文，非马尔可夫推理 |
| 📄 论文 | **DPM-Solver** (Lu et al., 2022) | 更快的 ODE 求解器，工程实践主流 |
| 📄 论文 | **Consistency Models** (Song et al., 2023) | 一步生成，DDIM 的极限加速 |
| 🎓 课程 | MIT 6.S183 Problem Set 2-3 | 采样加速实验 |
| 🎓 讲座 | CS231n 2025 Lecture 14 后半部分 | Diffusion Distillation（蒸馏加速） |

#### B4 Latent Diffusion & 架构设计

**核心内容**：VAE 编码器（像素空间→潜在空间）、UNet vs DiT、条件生成（文本→图像）、Classifier-Free Guidance

**衔接已有知识**：
- diy-llm Ch4 Transformer（DiT 直接用 Transformer 替代 UNet）
- llm-paper-tutorials 07-LLaMA（架构设计的类比——为什么选这个结构）
- llm-math-foundations Ch07 注意力数学（Cross-Attention 用于条件注入）

**参考资源**：
| 类型 | 资源 | 对应内容 |
|------|------|---------|
| 📄 论文 | **High-Resolution Image Synthesis with Latent Diffusion Models** (Rombach et al., 2022) | ⭐ Stable Diffusion 核心，Latent Diffusion 奠基 |
| 📄 论文 | **Classifier-Free Diffusion Guidance** (Ho & Salimans, 2022) | CFG 技术，所有条件 Diffusion 的标配 |
| 🎓 课程 | MIT 6.S183 | 条件生成、图像修复的 Problem Set |
| 🎓 讲座 | CS231n 2025 Lecture 14 | Text-to-Image 架构图、Latent Diffusion 流程 |
| 📄 论文 | **VAE: Auto-Encoding Variational Bayes** (Kingma & Welling, 2013) | VAE 基础，Latent Diffusion 的编码器 |
| 📄 论文 | **VQ-VAE** (van den Oord et al., 2017) | 离散编码，DALL-E 用的分词方式 |
| 🎬 视频 | B站：多模态教程 BV1azRgYbECW 第5-2节 "Image Synthesis with Diffusion Models" | 中文讲解 Stable Diffusion 架构 |

#### B5 DiT 与前沿应用

**核心内容**：DiT 架构详解（Transformer 替代 UNet）、Scaling Laws for Diffusion、Video Diffusion、Sora 技术路线

**衔接已有知识**：
- llm-paper-tutorials 05-Chinchilla（Scaling Laws 直接类比）
- diy-llm Ch13 Scaling Laws（计算最优策略）
- llm-paper-tutorials 01-Attention（Transformer 架构直接复用）

**参考资源**：
| 类型 | 资源 | 对应内容 |
|------|------|---------|
| 📄 论文 | **Scalable Diffusion Models with Transformers (DiT)** (Peebles & Xie, 2022) | ⭐ DiT 核心论文，你的兴趣方向 |
| 📄 论文 | **Scaling Rectified Flow Transformers for High-Resolution Image Synthesis** (2024) | SD3 的 Flow Matching 架构 |
| 📄 论文 | **Sora / Video Generation** 相关技术报告 | Video Diffusion 的最新进展 |
| 🎓 讲座 | CS231n 2025 Lecture 14 最后部分 | Video Diffusion 时间线、模型对比 |
| 📄 论文 | **Scaling Laws for Generative Mixed-Modal Models** | 多模态生成的 Scaling |
| 🎬 视频 | B站：多模态教程相关章节 | SAM2、视觉大模型 |

---

### 模块 C：多模态大模型（4章）

#### C1 视觉基础：Vision Transformer

**核心内容**：图像切 patch、位置编码、ViT 架构、与文本 Transformer 的异同、图像分类到通用视觉编码器

**衔接已有知识**：
- diy-llm Ch4 Transformer（架构完全复用，只是输入从 token 变成 image patch）
- llm-paper-tutorials 01-Attention（自注意力在视觉中的应用）
- llm-math-foundations Ch07 注意力数学（位置编码的数学）

**参考资源**：
| 类型 | 资源 | 对应内容 |
|------|------|---------|
| 📄 论文 | **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)** (Dosovitskiy et al., 2020) | ⭐ ViT 奠基论文 |
| 🎓 讲座 | **Stanford CS231n 2025 Lecture 12: Vision Transformer** | ViT 详解 + 与 CNN 对比 |
| 🎓 课程 | B站：多模态教程 BV1azRgYbECW 第5-1节 "Vision Transformer (ViT)介绍" | 中文讲解，~2小时 |
| 📄 论文 | **Swin Transformer** (Liu et al., 2021) | 层次化 ViT，你的论文复现计划中的目标之一 |
| 📝 博客 | The Illustrated ViT (jalammar.github.io) | 图解 ViT，非常直观 |

#### C2 图文对齐：CLIP

**核心内容**：对比学习（Contrastive Learning）、图文双编码器、Zero-Shot 分类、CLIP 的局限

**衔接已有知识**：
- llm-math-foundations Ch08 对比学习损失（InfoNCE）
- diy-llm Ch10 微调（CLIP 的预训练→下游任务类比）
- llm-paper-tutorials 01-Attention（编码器架构）

**参考资源**：
| 类型 | 资源 | 对应内容 |
|------|------|---------|
| 📄 论文 | **Learning Transferable Visual Models From Natural Language Supervision (CLIP)** (Radford et al., 2021) | ⭐ CLIP 核心，面试高频 |
| 🎓 讲座 | **Stanford CS231n 2025 Lecture 16: Multi-Modal Foundation Models** 前半部分 | CLIP 架构、训练、Zero-Shot 评估 |
| 🎓 课程 | B站：多模态教程 BV1azRgYbECW | "多模态论文精读：CLIP、ViLT、ALBEF、VLMO、BLIP、CoCa" |
| 📄 论文 | **SigLIP** (Zhai et al., 2023) | Sigmoid 替代 Softmax 的对比损失，CLIP 改进 |
| 🎬 视频 | Yannic Kilcher: CLIP Paper Explanation | 英文论文精读视频 |
| 📄 论文 | **BLIP / BLIP-2** (Li et al., 2022/2023) | CLIP 后续，Q-Former 桥接视觉和语言 |

#### C3 多模态架构：从 BLIP-2 到 LLaVA

**核心内容**：Q-Former 桥接机制、LLaVA 架构（CLIP 视觉编码器 + LLM）、指令微调数据构建、多模态幻觉问题

**衔接已有知识**：
- llm-paper-tutorials 06-InstructGPT（指令微调方法论直接复用）
- llm-paper-tutorials 08-LoRA（多模态微调的实际应用）
- diy-llm Ch10 SFT（监督微调流程）
- llm-paper-tutorials 07-LLaMA（LLM backbone 选择）

**参考资源**：
| 类型 | 资源 | 对应内容 |
|------|------|---------|
| 📄 论文 | **Visual Instruction Tuning (LLaVA)** (Liu et al., 2023) | ⭐ 多模态指令微调里程碑 |
| 📄 论文 | **BLIP-2** (Li et al., 2023) | Q-Former 设计，高效桥接 |
| 🎓 讲座 | **CS231n 2025 Lecture 16 后半部分** | LLaVA 架构详解：CLIP→Linear→LLM |
| 🎓 教程 | **CVPR 2024 MLLM Tutorial** (mllm2024.github.io) | 多模态架构设计、幻觉分析、高效学习 |
| 🎓 课程 | **TGLTommy: 多模态大模型前沿算法与实战** (tgltommy.com) | LLaVA/Qwen-VL 全流程实战：推理+微调+部署（付费） |
| 🎬 视频 | B站：多模态教程 相关章节 | LLaVA、LLaVA-NeXT 中文讲解 |
| 📄 论文 | **MiniGPT-4** (Zhu et al., 2023) | 另一种多模态方案，与 LLaVA 对比 |
| 📝 博客 | Chip Huyen: Large Multimodal Models | 多模态系统综述 |

#### C4 多模态前沿

**核心内容**：GPT-4V / Qwen-VL 架构、多模态 RLHF、视频理解（TimeSearch-R 等）、多模态 Agent

**衔接已有知识**：
- llm-paper-tutorials A1-DeepSeek-V3（最新架构设计）
- diy-llm Ch11 RLHF（多模态 RLHF 直接类比）
- llm-paper-tutorials 06-InstructGPT（RLHF 流程）

**参考资源**：
| 类型 | 资源 | 对应内容 |
|------|------|---------|
| 📄 论文 | **Qwen-VL / Qwen2-VL** 技术报告 | 当前最强开源多模态模型之一 |
| 🎓 课程 | TGLTommy: Qwen3-VL 实战（第12章） | 256K 长上下文 + DeepStack + MRoPE，含 SFT 微调全流程 |
| 📄 论文 | **LLaVA-NeXT / LLaVA-OneVision** | LLaVA 系列最新进展 |
| 📄 论文 | **Video-LLaVA / TimeSearch-R** | 视频理解，多模态的下一个前沿 |
| 🎓 教程 | CVPR 2024 MLLM Tutorial 后半部分 | 多模态推理、高效学习 |
| 📄 论文 | **GPT-4V System Card** | OpenAI 官方对多模态能力的分析 |
| 🎓 课程 | B站：多模态教程 第7-5节 "LLM-based Agents" | 多模态 Agent 概念 |

---

## 📚 每章结构模板（参考 llm-math-foundations 风格）

每章包含以下部分：

```
📌 这章讲什么？        — 3句话概述 + 你将学到什么
🔗 前置知识            — 关联到 llm-math-foundations / llm-paper-tutorials 的具体章节
📖 正文               — Step by Step 教学
   直觉解释            — 类比、图示、生活中的例子
   数学推导            — 不跳步，每个符号有含义
   代码验证            — 可运行的 PyTorch 代码 + 输出
🧠 批判性思考          — 面试视角的设计决策分析
🔗 知识网络            — 和你已有知识的关联
❓ 练习题              — 2-3道巩固理解的问题
```

---

## 🗺️ 学习路线图

```
                    你现在在这里
                         │
                    ┌────▼────┐
                    │ A1 生成  │  1天
                    │ 模型全景 │
                    └────┬────┘
                         │
              ┌──────────┴──────────┐
              │                     │
         ┌────▼────┐          ┌─────▼────┐
         │ B1 直觉 │          │ C1 ViT   │  可并行
         └────┬────┘          └─────┬────┘
              │                     │
         ┌────▼────┐          ┌─────▼────┐
         │ B2 DDPM │          │ C2 CLIP  │
         │ 数学    │          │ 图文对齐 │
         └────┬────┘          └─────┬────┘
              │                     │
         ┌────▼────┐                │
         │ B3 DDIM │                │
         │ 加速采样 │                │
         └────┬────┘                │
              │                     │
         ┌────▼────┐          ┌─────▼────┐
         │ B4 架构 │          │ C3 LLaVA │
         │ Latent  │◄────────►│ 多模态   │  ← B4 和 C3 有交叉
         └────┬────┘          └─────┬────┘
              │                     │
         ┌────▼────┐          ┌─────▼────┐
         │ B5 DiT  │          │ C4 前沿  │
         │ 前沿    │          │ 多模态   │
         └────┬────┘          └─────┬────┘
              │                     │
              └───────┬─────────────┘
                      │
                ┌─────▼─────┐
                │ Mini 项目  │  可选：训练一个小 DiT
                └───────────┘
```

**预估学习时间**（每天 2-3 小时）：
- 模块 A：1天
- 模块 B：2-3周（B2 数学较重，需要慢学）
- 模块 C：2周
- 总计：5-6周

---

## 🏗️ 项目目录结构

```
diffusion-multimodal-tutorial/
├── README.md                    ← 本文件（总体规划）
├── WORKFLOW.md                  ← 教程制作流程（参考 llm-paper-tutorials）
├── .gitignore
│
├── module-a-generative-basics/  ← 模块A：生成模型基础
│   └── A1-generative-panorama/
│       ├── README.md            ← 教程正文
│       ├── images/              ← 教程配图
│       └── scripts/             ← 代码验证
│
├── module-b-diffusion/          ← 模块B：Diffusion 模型
│   ├── B1-diffusion-intuition/
│   ├── B2-ddpm-math/
│   ├── B3-ddim-sampling/
│   ├── B4-latent-diffusion/
│   └── B5-dit-frontier/
│       each: README.md + images/ + scripts/
│
├── module-c-multimodal/         ← 模块C：多模态大模型
│   ├── C1-vision-transformer/
│   ├── C2-clip-alignment/
│   ├── C3-llava-multimodal/
│   └── C4-multimodal-frontier/
│       each: README.md + images/ + scripts/
│
└── papers/                      ← 论文精读（参考 llm-paper-tutorials 格式）
    ├── ddpm/
    │   ├── paper.pdf
    │   ├── raw-extract.md
    │   ├── README.md            ← 四层递进法教程
    │   └── merged-tutorial.md
    ├── ddim/
    ├── stable-diffusion/
    ├── dit/
    ├── vit/
    ├── clip/
    └── llava/
```

---

## ⚠️ 设计原则（从 llm-paper-tutorials 和 llm-math-foundations 总结的经验）

1. **不翻译，要教学** — 有提问、类比、引导，不是论文翻译
2. **公式不跳步** — 每个符号有物理含义，每步推导有解释
3. **代码可运行** — 每个概念都有代码验证，带预期输出
4. **关联已有知识** — 明确标注"这个概念你在 llm-math-foundations Ch03 学过"
5. **质量优先，不赶工** — 宁可少一章，不要水一章
6. **教学风格** — "口说无凭""你看""发现了吗？"，避免 AI 味模板词

---

## 📅 制作计划（待后续详细排期）

| 阶段 | 内容 | 预计时间 |
|------|------|---------|
| Phase 1 | A1 生成模型全景 | 2-3天 |
| Phase 2 | B1-B3 Diffusion 核心 | 1-2周 |
| Phase 3 | B4-B5 Diffusion 架构与前沿 | 1周 |
| Phase 4 | C1-C2 视觉基础与图文对齐 | 1周 |
| Phase 5 | C3-C4 多模态架构与前沿 | 1周 |
| Phase 6 | 论文精读（DDPM/DiT/CLIP/LLaVA） | 持续进行 |

> 注意：以上为制作排期，不是学习排期。每章完成后先 commit + push，再继续下一章。
