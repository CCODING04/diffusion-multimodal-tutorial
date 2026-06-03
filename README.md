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

## 📖 教程结构：三大模块

### 模块 A：生成模型基础（1章）

> 从你已经知道的"生成"出发，建立通往 Diffusion 的桥梁。

| 章节 | 标题 | 核心内容 | 衔接已有知识 |
|------|------|---------|-------------|
| A1 | 生成模型全景：从自回归到扩散 | GAN/VAE/Flow/Diffusion 四大生成范式对比；为什么 Diffusion 赢了 | makemore 的语言模型生成 + llm-paper-tutorials GPT-2/3 的自回归 |

### 模块 B：Diffusion 模型（5章）

> 从直觉到数学到代码到前沿，完整掌握 Diffusion。

| 章节 | 标题 | 核心内容 | 衔接已有知识 |
|------|------|---------|-------------|
| B1 | Diffusion 直觉：加噪与去噪 | 前向过程（逐步加噪）、反向过程（逐步去噪）、核心思想 | llm-math-foundations Ch03 概率分布、Ch06 信息论 |
| B2 | DDPM 数学：从直觉到公式 | 前向过程的马尔可夫链、反向过程推导、噪声预测、训练目标推导 | llm-math-foundations Ch04 矩阵求导、Ch08 变分推断 |
| B3 | DDIM 与加速采样 | 从随机到确定性采样、跳步采样、ODE 视角 | llm-math-foundations Ch05 优化（梯度下降类比） |
| B4 | Latent Diffusion & 架构 | VAE 编码器、UNet vs DiT、条件生成（文本→图像）、Classifier-Free Guidance | diy-llm Ch4 Transformer + llm-paper-tutorials 07-LLaMA 架构设计 |
| B5 | DiT 与前沿应用 | DiT 架构详解、Video Diffusion、Sora 的技术路线、Scaling | llm-paper-tutorials 05-Chinchilla Scaling Laws + diy-llm Ch13 |

### 模块 C：多模态大模型（4章）

> 让 LLM "看见" 世界——从视觉编码到多模态对话。

| 章节 | 标题 | 核心内容 | 衔接已有知识 |
|------|------|---------|-------------|
| C1 | 视觉基础：ViT | 图像切 patch、位置编码、ViT 架构、与文本 Transformer 的对比 | diy-llm Ch4 Transformer + llm-paper-tutorials 01-Attention |
| C2 | 图文对齐：CLIP | 对比学习、图文编码器、Zero-Shot 分类、CLIP 的局限 | llm-math-foundations Ch08 对比学习损失 + diy-llm Ch10 微调 |
| C3 | 多模态架构：从 BLIP 到 LLaVA | Q-Former 桥接、LLaVA 架构（CLIP + LLM）、指令微调数据构建、多模态幻觉问题 | llm-paper-tutorials 06-InstructGPT 指令微调 + 08-LoRA 高效微调 |
| C4 | 多模态前沿 | GPT-4V / Qwen-VL 架构、多模态 RLHF、视频理解、多模态 Agent | llm-paper-tutorials A1-DeepSeek-V3 + diy-llm Ch11 RLHF |

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

**预估时间**：
- 模块 A：1天
- 模块 B：2-3周（B2 数学较重，需要慢学）
- 模块 C：2周
- 总计：5-6周（每天 2-3 小时）

---

## 📋 推荐学习资源

### Diffusion 模型
| 资源 | 类型 | 说明 | 对应章节 |
|------|------|------|---------|
| DeepLearning.AI: How Diffusion Models Work | 短课(1h) | 从零构建 Diffusion，Sharon Zhou 主讲 | B1-B2 预习 |
| MIT 6.S183: Practical Diffusion Models | 完整课程 | 采样可视化、条件生成、自定义训练 | B1-B4 |
| Stanford CS231n Lecture 14 (2025) | 讲座 | Diffusion 数学：Score、SDE、DiT | B2、B5 |
| Stanford CS236: Deep Generative Models | 课程 | VAE/Flow/Diffusion 深入 | A1、B2 |

### 多模态
| 资源 | 类型 | 说明 | 对应章节 |
|------|------|------|---------|
| Stanford CS231n Lecture 16 (2025) | 讲座 | CLIP→LLaVA→GPT-4V 完整演进 | C1-C4 |
| CVPR 2024 MLLM Tutorial | 教程 | 架构设计、幻觉、推理、高效学习 | C3-C4 |
| B站：多模态+大模型教程 BV1azRgYbECW | 视频 | CLIP、BLIP、DALL-E、SAM 中文精讲 | C1-C3 |

### 核心论文（建议用 llm-paper-tutorials 四层递进法精读）
| 顺序 | 论文 | 年份 | 对应章节 |
|------|------|------|---------|
| 1 | DDPM (Ho et al.) | 2020 | B1-B2 |
| 2 | DDIM (Song et al.) | 2020 | B3 |
| 3 | Stable Diffusion / Latent Diffusion (Rombach et al.) | 2022 | B4 |
| 4 | DiT (Peebles & Xie) | 2022 | B5 |
| 5 | ViT (Dosovitskiy et al.) | 2020 | C1 |
| 6 | CLIP (Radford et al.) | 2021 | C2 |
| 7 | BLIP-2 (Li et al.) | 2023 | C3 |
| 8 | LLaVA (Liu et al.) | 2023 | C3 |
| 9 | Sora / Video Diffusion | 2024 | B5 |

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
    ├── dit/
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
