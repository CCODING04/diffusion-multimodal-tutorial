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

| 资源 | 链接 | 推荐理由 |
|------|------|---------|
| 📝 Lilian Weng: What are Diffusion Models? | https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ | ⭐ AI 博客界最好的 Diffusion 综述，从 VAE/GAN/Flow 到 Diffusion 的完整数学推导链，图文并茂，OpenAI 应用 AI 研究负责人写的 |
| 🎓 Stanford CS236: Deep Generative Models | https://deepgenerativemodels.github.io/ | Stanford 正课，覆盖 VAE/Flow/GAN/Diffusion 全部生成模型，有作业和项目，2023 年秋季版 |
| 📄 Calvin Luo: Understanding Diffusion Models — A Unified Perspective | https://arxiv.org/abs/2208.11970 在线版：https://calvinyluo.com/2022/08/26/diffusion-tutorial.html | ⭐⭐ 被广泛引用的最佳 Diffusion 数学综述，从变分推断和 Score-based 两个视角统一解释，推导非常清晰 |

---

### 模块 B：Diffusion 模型（5章）

#### B1 Diffusion 直觉：加噪与去噪

**核心内容**：前向过程（逐步加噪）可视化、反向过程（逐步去噪）直觉、核心思想——知道怎么加噪声就能学怎么去掉噪声

**衔接已有知识**：
- llm-math-foundations Ch03 概率分布（高斯分布、条件概率）
- llm-math-foundations Ch06 信息论（熵、信息量）

**参考资源**：

| 资源 | 链接 | 推荐理由 |
|------|------|---------|
| 🎓 DeepLearning.AI: How Diffusion Models Work | https://www.deeplearning.ai/courses/diffusion-models/ | ⭐ **首选入门**。Sharon Zhou（现为 AMD AI VP）主讲，1小时从零构建 Diffusion，含 Jupyter notebook 实操，免费 |
| 🎓 MIT 6.S183: A Practical Introduction to Diffusion Models | https://www.practical-diffusion.org/ | ⭐⭐ **核心课程**。MIT IAP 2026 正课，含 Problem Sets（采样可视化、条件生成）+ Mini Project（自定义数据集训练），面向有 PyTorch 基础的学生 |
| 🎓 Hugging Face: The Annotated Diffusion Model | https://huggingface.co/blog/annotated-diffusion | 逐行代码注释版 DDPM 实现，配有 Colab notebook 可直接运行，适合边读边敲 |
| 📄 DDPM 论文 Figure 2 | https://arxiv.org/abs/2006.11239 | 只看 Figure 2（加噪过程示意图）和 Section 1-2（直觉介绍），不用深入数学 |

#### B2 DDPM 数学：从直觉到公式

**核心内容**：前向过程马尔可夫链推导、反向过程推导、噪声预测目标、训练目标（ELBO→简化形式）推导

**衔接已有知识**：
- llm-math-foundations Ch04 矩阵求导
- llm-math-foundations Ch08 变分推断（ELBO、KL散度）
- diy-llm Ch3 训练原语（损失函数推导类比）

**参考资源**：

| 资源 | 链接 | 推荐理由 |
|------|------|---------|
| 📄 Calvin Luo: Understanding Diffusion Models — A Unified Perspective | https://arxiv.org/abs/2208.11970 | ⭐⭐ **本章最重要的参考**。从 ELBO 出发，一步步推导到 DDPM 训练目标，每步有解释，是理解 Diffusion 数学的最佳材料 |
| 📝 Lilian Weng: What are Diffusion Models? | https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ | ⭐ Section "Denoising Diffusion Probabilistic Models" 部分的数学推导非常清晰，配合 Luo 的综述一起看效果最好 |
| 🎓 Stanford CS231n 2025 Lecture 14: Generative Models Part 2 | Slides: https://cs231n.stanford.edu/slides/2025/lecture_14.pdf | ⭐ Stanford 正课 slides，涵盖 Score Function → SDE → DiT 的数学框架，116页高质量内容 |
| 🎓 Stanford CS236: Deep Generative Models | https://deepgenerativemodels.github.io/ | 专门的 Diffusion 课时，含数学推导作业，需要 Stanford 访问权限但 slides 公开 |
| 📄 DDPM 原始论文 | https://arxiv.org/abs/2006.11239 | Section 3-4 是数学推导的原始出处，读完后建议和 Luo 的综述对照 |

#### B3 DDIM 与加速采样

**核心内容**：从随机到确定性采样、跳步采样、ODE 视角、Consistency Models

**衔接已有知识**：
- llm-math-foundations Ch05 优化（梯度下降的类比）
- diy-llm Ch7 CUDA（采样效率的工程意义）

**参考资源**：

| 资源 | 链接 | 推荐理由 |
|------|------|---------|
| 📄 DDIM 原始论文 | https://arxiv.org/abs/2010.02502 | 非马尔可夫推理框架，从随机到确定的数学推导，Song Yang 组的工作 |
| 📄 DPM-Solver | https://arxiv.org/abs/2206.00927 | 当前工程实践中最快的 ODE 求解器，Stable Diffusion WebUI 默认选项 |
| 📄 Consistency Models | https://arxiv.org/abs/2303.01469 | Song Yang 2023 年的工作，一步生成的极限加速，代表加速采样的前沿 |
| 🎓 MIT 6.S183 Problem Sets 2-3 | https://www.practical-diffusion.org/ | 采样加速的动手实验 |

#### B4 Latent Diffusion & 架构设计

**核心内容**：VAE 编码器（像素空间→潜在空间）、UNet vs DiT、条件生成（文本→图像）、Classifier-Free Guidance

**衔接已有知识**：
- diy-llm Ch4 Transformer（DiT 直接用 Transformer 替代 UNet）
- llm-paper-tutorials 07-LLaMA（架构设计类比）
- llm-math-foundations Ch07 注意力数学（Cross-Attention 条件注入）

**参考资源**：

| 资源 | 链接 | 推荐理由 |
|------|------|---------|
| 🎓 Fast.ai Part 2: Deep Learning Foundations to Stable Diffusion | 课程页：https://course.fast.ai/Lessons/part2.html Lesson 9：https://course.fast.ai/Lessons/lesson9.html | ⭐⭐ **强烈推荐**。Jeremy Howard 从零实现 Stable Diffusion，30+ 小时视频，和 Stability.ai + Hugging Face 团队合作制作，代码优先的教学风格 |
| 📄 Latent Diffusion (Stable Diffusion 核心) | https://arxiv.org/abs/2112.10752 | Rombach et al.，Latent Diffusion 奠基论文，解释了为什么在 latent space 做 diffusion 而非 pixel space |
| 📄 Classifier-Free Diffusion Guidance | https://arxiv.org/abs/2207.12598 | Ho & Salimans，CFG 技术，所有条件 Diffusion 模型的标配方法 |
| 📄 VAE 原始论文 | https://arxiv.org/abs/1312.6114 | Kingma & Welling，理解 Latent Diffusion 编码器的基础 |
| 🎓 Hugging Face Diffusion Course | https://huggingface.co/learn/diffusion-course/en/unit1/3 | Hugging Face 官方课程，从 toy model 到完整实现，Colab 可运行 |

#### B5 DiT 与前沿应用

**核心内容**：DiT 架构详解、Scaling Laws for Diffusion、Video Diffusion、Sora 技术路线

**衔接已有知识**：
- llm-paper-tutorials 05-Chinchilla（Scaling Laws 直接类比）
- diy-llm Ch13 Scaling Laws
- llm-paper-tutorials 01-Attention（Transformer 直接复用）

**参考资源**：

| 资源 | 链接 | 推荐理由 |
|------|------|---------|
| 📄 DiT 论文 | https://arxiv.org/abs/2212.09748 | ⭐ Peebles & Xie，用 Transformer 替代 UNet 做 Diffusion，Sora 的基础架构，你的兴趣方向 |
| 🎓 Stanford CS231n 2025 Lecture 14 后半部分 | https://cs231n.stanford.edu/slides/2025/lecture_14.pdf | Video Diffusion 时间线（2024-2025 所有主流视频生成模型对比图），DiT 架构分析 |
| 📄 SD3 / Flow Matching | https://arxiv.org/abs/2403.03206 | Stable Diffusion 3 的 Rectified Flow 架构，Diffusion 的最新演进方向 |
| 📄 Scaling Laws for Neural Visual Generative Models | https://arxiv.org/abs/2412.02560 | Diffusion 模型的 Scaling Laws，直接对应你学过的 Chinchilla |

---

### 模块 C：多模态大模型（4章）

#### C1 视觉基础：Vision Transformer

**核心内容**：图像切 patch、位置编码、ViT 架构、与文本 Transformer 的异同

**衔接已有知识**：
- diy-llm Ch4 Transformer（架构完全复用）
- llm-paper-tutorials 01-Attention
- llm-math-foundations Ch07 注意力数学

**参考资源**：

| 资源 | 链接 | 推荐理由 |
|------|------|---------|
| 📄 ViT 论文 | https://arxiv.org/abs/2010.11929 | Dosovitskiy et al.，奠基论文，核心思想非常简洁——图像就是 16×16 个"词" |
| 🎓 Stanford CS231n 2025 Lecture 12: Vision Transformer | https://cs231n.stanford.edu | Stanford 正课，ViT 详解 + 与 CNN 的对比分析，slides 公开 |
| 📝 Jay Alammar: The Illustrated ViT | https://jalammar.github.io/illustrated-vision-transformer/ | 图解 ViT，和他的 "Illustrated Transformer" 一样清晰直观 |

#### C2 图文对齐：CLIP

**核心内容**：对比学习（Contrastive Learning）、图文双编码器、Zero-Shot 分类、CLIP 的局限

**衔接已有知识**：
- llm-math-foundations Ch08 对比学习损失（InfoNCE）
- diy-llm Ch10 微调

**参考资源**：

| 资源 | 链接 | 推荐理由 |
|------|------|---------|
| 📄 CLIP 论文 | https://arxiv.org/abs/2103.00020 | Radford et al.，面试高频论文，核心方法简洁但影响力巨大 |
| 🎓 Stanford CS231n 2025 Lecture 16 前半部分 | https://cs231n.stanford.edu/slides/2025/lecture_16.pdf | CLIP 架构详解、训练策略、Zero-Shot 评估结果分析 |
| 📝 Chip Huyen: Multimodality and Large Multimodal Models | https://huyenchip.com/2023/10/10/multimodal.html | ⭐⭐ 斯坦福 CS224W 讲师写的多模态全景综述，Part 2 用 CLIP 和 Flamingo 讲清楚多模态基础，写于 2023 年 10 月，内容新且权威 |
| 📄 SigLIP | https://arxiv.org/abs/2303.15343 | Sigmoid 替代 Softmax 的改进，Google 2023，很多新模型在用 |

#### C3 多模态架构：从 BLIP-2 到 LLaVA

**核心内容**：Q-Former 桥接机制、LLaVA 架构（CLIP + LLM + Linear）、指令微调数据构建、多模态幻觉

**衔接已有知识**：
- llm-paper-tutorials 06-InstructGPT（指令微调方法论）
- llm-paper-tutorials 08-LoRA（多模态微调应用）
- diy-llm Ch10 SFT

**参考资源**：

| 资源 | 链接 | 推荐理由 |
|------|------|---------|
| 📄 LLaVA 论文 | https://arxiv.org/abs/2304.08485 | Liu et al.，最经典的开源多模态架构，CLIP + Linear + LLM，简洁优雅 |
| 📄 BLIP-2 论文 | https://arxiv.org/abs/2301.12597 | Q-Former 设计，用轻量级模块桥接冻结的视觉编码器和 LLM，工程上的巧妙设计 |
| 🎓 Stanford CS231n 2025 Lecture 16 后半部分 | https://cs231n.stanford.edu/slides/2025/lecture_16.pdf | LLaVA 架构详解：从 CLIP 到 LLM 的桥接，训练流程，实验分析 |
| 🎓 CVPR 2024 MLLM Tutorial | https://mllm2024.github.io/CVPR2024/ | ⭐ CVPR 顶会教程，覆盖架构设计、幻觉分析、多模态推理、高效学习，slides 公开 |
| 📝 Chip Huyen: Multimodality and LMMs | https://huyenchip.com/2023/10/10/multimodal.html | Part 3-4 详细分析 LLaVA/Flamingo/BLIP-2 等架构的设计选择 |

#### C4 多模态前沿

**核心内容**：Qwen-VL 架构、多模态 RLHF、视频理解、多模态 Agent

**衔接已有知识**：
- llm-paper-tutorials A1-DeepSeek-V3（最新架构）
- diy-llm Ch11 RLHF
- llm-paper-tutorials 06-InstructGPT

**参考资源**：

| 资源 | 链接 | 推荐理由 |
|------|------|---------|
| 📄 Qwen2-VL 技术报告 | https://arxiv.org/abs/2409.12191 | 当前最强开源多模态模型之一，架构设计很有代表性 |
| 📄 LLaVA-OneVision | https://arxiv.org/abs/2408.03326 | LLaVA 系列最新版，单图/多图/视频统一处理 |
| 🎓 CVPR 2024 MLLM Tutorial 后半部分 | https://mllm2024.github.io/CVPR2024/ | 多模态推理和高效学习的前沿方向 |
| 📄 GPT-4V System Card | https://openai.com/index/gpt-4v-system-card/ | OpenAI 官方对多模态能力的评估和局限分析 |

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

## ⚠️ 设计原则

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
