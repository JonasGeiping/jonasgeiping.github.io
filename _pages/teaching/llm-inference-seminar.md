---
layout: page
permalink: /teaching/llm-inference-seminar/
title: "LLM Inference Optimization"
description: "Seminar, Summer 2026, University of Tübingen"
nav: false
toc:
  sidebar: left
---

## Overview

Large Language Models are quite large. Just to run the trained model and generate text, i.e. to run 'inference', one would think it would cost the companies a lot more money to run them in practice than they seem to have. So what do they do?

This seminar introduces the key technologies in machine learning engineering that make the generation of text from large-scale language models economically feasible at scale. Modern LLM serving systems such as [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang) combine a range of systems-level optimizations that together enable the deployment of models with up to trillions of parameters. Understanding what happens under the hood not only helps you extend inference engines with new ideas, but also how to best use these engines in practice (and what all of the funny settings mean).

We will go through a list of key topics in this seminar. Each topic is presented by one of the participants at the weekly meetings of the seminar (this could be you!). If you do not want to prepare a seminar talk, you can still take part passively, but will not receive a grade. The topic assignment will be finalized in the first week of class. There will likely be a ranked-choice system where you can hope for your favorite topic.

**Instructor:** Jonas Geiping\\
**Module:** ML-4501o\\
**Format:** Seminar (3 CP)\\
**When:** Mondays, 16:00--18:00\\
**Where:** Hörsaal TTR2, Cyber Valley Campus (Maria-von-Linden-Str. 6)\\
**Semester:** Summer 2026 (April 13 -- July 20)\\
**ALMA:** [Course listing](https://alma.uni-tuebingen.de:443/alma/pages/startFlow.xhtml?_flowId=detailView-flow&unitId=98930&periodId=229&navigationPosition=studiesOffered,searchCourses)

### Format

This is a _Seminar_: each participant selects one of the topics below and prepares a presentation for one of the weekly sessions, followed by questions and discussion. There is no separate written report required - but your presentation should be good! You will have about an hour to really go in-depth into your assigned topic. This may include profiling runs or simulations that you have run, ablations or just a very clear explanation of a crunchy piece of systems engineering. Please don't present just a vague collection of high-level ideas that you got from prompting GPT-5.4 "what's going on in PagedAttention".

- **Week 1:** Introduction, Basics, Topic Assignment
- **Week 2:** Office Hour and Q&A - I would suggest starting to read up on your topic a bit, so you can ask questions here.
- **Weeks 4--15:** Student presentations (one topic per week, excluding holidays).
- See the full schedule below.

### Prerequisites

Prior knowledge of machine learning and an understanding of language modeling with transformers are potentially required - but only insofar as this will not be part of the seminar. At a basic level, inference for LLMs is quite simple, so you can also quickly catch up if you're willing to do so. Prior ML engineering or advanced systems knowledge is not required and will (hopefully) be obtained as part of the class.

### Grading

The grade is based on the quality of the presentation, including clarity of explanation, depth of understanding, and ability to answer questions.

---

## Seminar Topics

The following topics are available for presentations. Each topic covers a core technique used in production LLM serving systems. References include both academic papers and open-source implementations --- as presenters you are expected to engage with the actual code in systems like [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang), not only the papers. The references are also not exhaustive, feel free to extend them with your own research, but I hope they give some indication of the planned direction (so we make sure not to repeat concepts in different presentations).

---

### 1. KV Caching and Paged Attention

The key-value (KV) cache avoids redundant computation during autoregressive decoding in transformer models, but its memory footprint grows linearly with sequence length and batch size. PagedAttention, introduced in vLLM, manages KV cache memory using virtual memory and paging concepts from operating systems, eliminating fragmentation and enabling near-optimal memory utilization. Your talk should discuss what this means and how it is handled at scale. If you want, you can also move through the vLLM versions discussing how they have changed the underlying architecture since V0.

When going more in-depth, you could discuss how vLLM's copy-on-write is block-granular: each physical block carries a reference count, and a write to a shared block triggers allocation of a fresh block. You can contrast this with SGLang's RadixAttention, which takes a different approach: instead of per-sequence block tables, it stores all KV cache entries in a radix tree keyed by token sequences with LRU eviction on leaf nodes, enabling automatic prefix sharing without any explicit fork bookkeeping. What are the tradeoffs between these two designs, and when does implicit sharing via the radix tree win over explicit block-table management?

Note: There will be another presentation later, specifically about KV-offloading, so leave that for later.

**Reference Suggestions:**

- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (SOSP 2023)
- Zheng et al., [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104) (2024)
- [vLLM PagedAttention implementation](https://github.com/vllm-project/vllm/tree/main/csrc/attention) and [historic design doc](https://docs.vllm.ai/en/latest/design/paged_attention.html)
- Gao et al., [CachedAttention: Cost-effective Attention Reuse across Multi-turn Conversations](https://arxiv.org/abs/2403.19708) (USENIX ATC 2024)

---

### 2. Request Scheduling and Overlap

All modern serving engines use continuous (iteration-level) batching, where requests are dynamically inserted and retired at each decode step rather than waiting for an entire batch to finish, a concept vLLM made practical alongside PagedAttention. The more interesting question today is what happens on top of this: how do schedulers decide what to run next, how to manage preemption and priorities, and how to hide CPU scheduling overhead. SGLang's overlap scheduler pipelines CPU-side scheduling work behind GPU execution so that scheduling is effectively free. This talk covers the evolution from continuous batching to modern zero-overhead and overlap schedulers.

Going in-depth, you might talk about the CPU-side work being overlapped with batch formation, KV cache memory pre-allocation, and radix tree prefix matching. Without overlap, these operations show up as GPU idle gaps consuming roughly half of wall-clock time in Nsight profiles (you could profile this if you want). The scheduler runs one batch ahead: while the GPU executes iteration _i_, the CPU prepares iteration _i+1_ using CUDA events for synchronization. You can also investigate vLLM's two preemption strategies (swap vs recompute) and discuss in what application scenarios recomputation actually beats swapping, and how this relates to PCI bandwidths.

**Reference Suggestions:**

- Yu et al., [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) (OSDI 2022)
- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (SOSP 2023)
- [SGLang v0.4 Blog: Zero-Overhead Batch Scheduler](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)
- [Mini-SGLang](https://lmsys.org/blog/2025-12-17-minisgl/)
- [vLLM scheduler source](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)
- Engineering discussion: [vLLM #23446](https://github.com/vllm-project/vllm/issues/23446)

---

### 3. Speculative Decoding and Modern Multi-Token-Prediction

Speculative decoding uses a smaller, faster draft model to propose multiple tokens at once, which are then verified in parallel by the target model. This can yield significant speedups without changing model outputs. The EAGLE series replaces the separate draft model with a lightweight prediction head that reuses features from the target model. Multi-token prediction (MTP), as shipped with DeepSeek-V3 or Qwen-Next, takes this further: the model includes dedicated MTP modules that act as native draft heads at inference time, achieving high acceptance rates and increased throughput speedup without an external draft model. Both vLLM and SGLang now support MTP-based speculative decoding in production. Your presentation should walk through the mechanics of speculative decoding, and why it is even possible to accelerate throughput with it, and how it interacts with latency and TTFT.

Your detailed part could discuss how EAGLE's draft head takes the target model's hidden state and the embedding of the actually-sampled token as input and what the reasons for this construction are. Multiple draft candidates are arranged as a tree and verified in a single target-model forward pass using a causal tree attention mask, cf. Medusa. However, at high batch utilization, the target model is already compute-saturated, so verifying a draft tree of _k_ tokens costs nearly as much as generating _k_ tokens normally, while rejected tokens waste compute.

**Reference Suggestions:**

- Leviathan et al., [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (ICML 2023)
- Li et al., [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) (ICML 2024) and [EAGLE-3](https://arxiv.org/abs/2503.01840) (2025) --- [code](https://github.com/SafeAILab/EAGLE)
- Cai et al., [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)
- Gloeckle et al., [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737) (2024), the MTP training objective; DeepSeek-V3 ([report](https://arxiv.org/abs/2412.19437)) ships MTP modules repurposed as draft heads for inference
- [LMSYS Blog: MTP in SGLang](https://lmsys.org/blog/2025-07-17-mtp/)
- [vLLM speculative decoding docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/) and [MTP docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/)
- [SGLang speculative decoding docs](https://docs.sglang.io/advanced_features/speculative_decoding.html)
- Engineering discussion: [SGLang #3582](https://github.com/sgl-project/sglang/pull/3582)

---

### 4. The Properties of Modern GPU Hardware and how to write Software for it

Standard implementations of most operations leave performance on the table because they cannot fuse operations or reason about GPU memory hierarchy. Custom kernels are essential for saturating hardware, a classic one is FlashAttention. This presentation should not necessarily walk the audience through flash attention again (also this could be a good intro), but rather talk about the evolution from FA-1 to FA-4, and how chip advancements and features have required algorithm and kernel pipelining co-design. Explaining this might require explaining some of the Blackwell-generation features first.

Aside from this, you could also discuss FlashInfer, which provides serving-oriented attention kernels optimized for the mixed prefill-decode batches and variable sequence lengths found in production, and Triton which is a 'portable' alternative for writing custom kernels. Ultimately your talk should motivate why co-design and custom kernels are necessary, how they are written and profiled, and how hardware differences (Hopper, Blackwell, MI300X, whatever Cerebras is doing) shape kernel design. For example, FlashAttention-3 pipelines softmax with the next matmul on Hopper, exploiting the fact that Hopper's matmul unit (989 TFLOPS) is ~250x faster than its exponential unit (3.9 TFLOPS), so softmax completes "for free" during the next matmul. The vLLM Triton backend attempts to match this in ~800 lines by using persistent kernels with dynamic work-scheduling and a "parallel tiled softmax" that splits KV-cache traversal across multiple kernel instances for decode. Do they actually succeed?

Note: Leave automated cudagraphs and torch compilation for the next talk.

**Reference Suggestions:**

- Zadouri et al., [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling](https://arxiv.org/abs/2603.05451)
- Shah et al., [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) (2024)
- Ye et al., [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](https://arxiv.org/abs/2501.01005) (MLSys 2025) [source](https://github.com/flashinfer-ai/flashinfer)
- [vLLM Triton attention backend deep dive](https://blog.vllm.ai/2026/03/04/vllm-triton-backend-deep-dive.html)
- [Triton language and compiler](https://github.com/triton-lang/triton)
- Engineering discussion: [SGLang #10054](https://github.com/sgl-project/sglang/issues/10054) --- what is going on here :)

---

### 5. CUDA Graphs and torch.compile for LLM Serving

The alternative to manual kernel design (previous talk) are advanced compilation and graph capture systems. CUDA graphs capture entire sequences of GPU operations and replay them with minimal launch overhead. Piecewise CUDA graph capture splits the computation graph at dynamic attention boundaries, while torch.compile applies graph-level optimizations and kernel fusions. Together, these techniques can substantially reduce per-token latency. Especially the vLLM use of pre-compilation CUDA graph capture is quite noticeable (and a large source of start-up time), the audience should understand what happens there and why it is necessary.

Going into depth, you might discuss how CUDA graphs require all tensor shapes, kernel configurations, and memory addresses to be fixed at capture time, but the KV cache grows by one token every decode step and batch sizes vary as requests arrive and finish. vLLM handles this by pre-capturing separate graphs for a set of padded batch-size buckets (1, 2, 4, ..., max), padding the actual batch up to the next bucket at runtime, which may seem insane at first glance. `torch.compile` also has a quite interesting callstack capturing intermediate Python representations, that is always fun to understand and describe in more detail.

**Reference Suggestions:**

- [NVIDIA CUDA Graphs Documentation](https://developer.nvidia.com/blog/cuda-graphs/)
- Ansel et al., [PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation](https://dl.acm.org/doi/10.1145/3620665.3640366) (ASPLOS 2024)
- [vLLM Blog: torch.compile Integration](https://vllm.ai/blog/torch-compile)
- [vLLM CUDA Graphs design doc](https://docs.vllm.ai/en/stable/design/cuda_graphs/) and [torch.compile design doc](https://docs.vllm.ai/en/latest/design/torch_compile/)
- Engineering discussion: [vLLM #20059](https://github.com/vllm-project/vllm/pull/20059)

---

### 6. Practical Quantization for LLM Inference

Quantization is one of the most impactful practical optimizations: reducing precision allows larger models to fit in fewer GPUs and increases throughput by reducing memory bandwidth requirements. This talk should definitely cover what quantization is and how it works, but it should especially focus on how this is orchestrated in real systems. For example, vLLM serves W4A16 models (quantized via GPTQ or AWQ) through Marlin dequantization kernels. KV cache quantization (FP8 or lower) independently reduces the memory bottleneck for long-context workloads. This talk should cover the quantization methods actually deployed in vLLM and SGLang, their throughput-accuracy tradeoffs, and the kernel mechanics that make them fast.

You could further talk about Marlin's design. Why are W4A16 GEMM kernels entirely memory-bandwidth-bound, and why does this imply that dequantization can be overlapped with memory latency. The kernel uses 4-deep `cp.async` pipelining to prefetch INT4 weight tiles, and while Tensor Cores execute the current tile's matmul, the next tile's 4-bit weights are being unpacked to FP16 in registers and rearranged into the exact layout needed by `mma.sync`. For KV cache quantization, you could investigate why you can quantize values (V) more aggressively than keys (K), what quantization schemes are usable in practice, and how they function.

**Reference Suggestions:**

- Frantar et al., [MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models](https://arxiv.org/abs/2408.11743) (PPoPP 2025)
- [vLLM quantization docs](https://docs.vllm.ai/en/latest/features/quantization/)
- [vLLM Blog: DeepSeek on GB300 with NVFP4](https://blog.vllm.ai/2026/02/13/gb300-deepseek.html)
- [vLLM KV Quant](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)
- [SGLang Quantization](https://docs.sglang.io/advanced_features/quantization.html)

---

### 7. Prefill-Decode Disaggregation and Chunked Prefill

Prefill (processing prompts) is compute-bound, while decoding is memory-bandwidth bound. Disaggregated serving separates these two phases onto different GPU pools, each optimized for its respective workload, as for example described in DistServe, who discuss goodput-optimized placement of prefill and decode instances, or Splitwise who implement phase splitting within datacenter-scale deployments. Chunked prefill, which breaks long prompts into smaller pieces interleaved with decode steps, is a complementary technique that improves latency under mixed workloads.

In such disaggregated serving architectures, KV cache data must be transferred between prefill and decode instances, and long-context workloads can exhaust local GPU memory. Mooncake addresses this by building a distributed cache pool from underutilized CPU DRAM and SSDs across the cluster, connected via RDMA-based transfer engines that achieve near-line-rate bandwidth. Its KVCache-centric scheduler co-optimizes cache placement and request routing to maximize throughput under latency constraints. This talk covers the systems challenges of distributed KV cache storage, transfer, and reuse in production LLM serving.

Taken together, your presentation should discuss the argument for disaggregation strategies, and how they are orchestrated.

You might talk about chunked prefill attention patterns as a function of block size. Smaller chunks reduce head-of-line blocking for concurrent decode requests but increase redundant KV reads per prefill token. For disaggregation, the crossover depends on whether KV transfer time (proportional to seq_len × layers × hidden_dim) is less than the compute the decode instance saves by not doing prefill itself. vLLM offers multiple KV connectors (PyNccl P2P, NixlConnector for RDMA, MooncakeConnector) with very different latency profiles, for example you could contrast MoonCake with LMCache, which operates as a middleware caching layer supporting reuse of text subsequences across serving instances via GPU/CPU/disk/S3 tiers.

**Reference Suggestions:**

- Zhong et al., [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670) (OSDI 2024)
- Patel et al., [Splitwise: Efficient Generative LLM Inference Using Phase Splitting](https://arxiv.org/abs/2311.18677) (ISCA 2024)
- Agrawal et al., [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310) (OSDI 2024)
- [vLLM disaggregated prefill docs](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
- Engineering discussion: [SGLang #5450](https://github.com/sgl-project/sglang/issues/5450)
- Qin et al., [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079) (FAST 2025)
- [Mooncake Transfer Engine](https://github.com/kvcache-ai/Mooncake) --- open-source RDMA/TCP transfer engine and distributed KV store
- Liu et al., [LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference](https://arxiv.org/abs/2510.09665) (2025), [code](https://github.com/LMCache/LMCache)
- Engineering discussion: [SGLang #7211](https://github.com/sgl-project/sglang/pull/7211)

---

### 8. Serving Parallelism: TP, PP, EP, and Other -P Letter Combinations

Deploying large models across multiple GPUs requires choosing how to partition the workload. Tensor parallelism (TP) shards individual layers, pipeline parallelism (PP) splits the model into sequential stages, and expert parallelism (EP) distributes MoE experts across devices. Data parallel attention (DPA), introduced for MLA-based models like DeepSeek-V3, avoids the KV cache duplication inherent in standard TP by giving each data-parallel replica its own cache. This talk should describe these strategies in detail and their interactions in production deployments.

For example you could discuss how EP's all-to-all communication is harder to overlap than TP's all-reduce because it has data-dependent communication volumes as each rank sends a different number of tokens to each expert depending on the gating function, and in DeepEP's normal mode the CPU blocks waiting for a GPU signal reporting per-expert token counts. DeepEP's low-latency mode solves this for decode by switching to pure RDMA with hook-based deferred receives that avoid SM occupation entirely and are CUDA-graph-compatible. SGLang's Two-Batch Overlap (TBO) splits a batch into two micro-batches with "yield points" so that GPU computation for micro-batch B launches before the blocking DeepEP combine call for micro-batch A completes, hiding 27-35% of communication latency, allegedly.

**Reference Suggestions:**

- Shoeybi et al., [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) (2020)
- DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) (2024)
- [DeepEP: An Efficient Expert-Parallel Communication Library](https://github.com/deepseek-ai/DeepEP) (2025)
- [LMSYS Blog: Large-Scale EP for DeepSeek](https://lmsys.org/blog/2025-05-05-large-scale-ep/)
- [LMSYS Blog: Chunked Pipeline Parallelism](https://lmsys.org/blog/2026-01-15-chunked-pipeline/)
- [vLLM parallelism documentation](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/) and [SGLang DP/DPA guide](https://docs.sglang.io/advanced_features/dp_dpa_smg_guide.html)

---

### 9. Long-Context Inference

Serving models at long context lengths (128K to 1M+ tokens) introduces challenges across
memory, compute, and communication. The KV cache for a single 1M-token request can consume tens of  
gigabytes, and the quadratic cost of attention makes prefill latency explode. Context parallelism
addresses this by sharding the sequence across GPUs: Ring Attention forms devices into a logical ring and Striped Attention improves on
this by distributing tokens in an interleaved pattern. On the model side, DeepSeek's Native Sparse Attention
(NSA/DSA) is the first sparse attention method deployed in production at scale.
Snowflake's SwiftKV takes a different approach: it computes later layers' KV cache from an earlier
layer's output, letting prompt tokens skip much of the model during prefill. This talk should cover the
systems and algorithmic techniques that make million-token inference practical today.

You could discuss how Ring Attention achieves exact results via the online softmax merge: each device  
maintains running statistics (maximum logit, sum of exponentials, weighted output) and when a new KV
block arrives, all previous partial results are rescaled by exp(m_old - m_new) to correct for the  
shifting maximum. This is mathematically exact but imposes a strict sequential dependency as no query's
output can be finalized until all KV blocks have rotated through the ring. You could also contrast "lossless" approaches (Ring Attention, context
parallelism) with "lossy" ones (StreamingLLM's attention sinks + sliding window, sparse attention) and
discuss where the quality-efficiency frontier lies.

**Reference Suggestions:**

- Liu et al., [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889) (ICLR 2024), [code](https://github.com/haoliuhl/ringattention)
- Brandon et al., [Striped Attention: Faster Ring Attention for Causal Transformers](https://arxiv.org/abs/2311.09431) (2023)
- Yuan et al., [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089) (DeepSeek, 2025), deployed in DeepSeek-V3.2 with vLLM Day 0 support
- Qiao et al., [SwiftKV: Fast Prefill-Optimized Inference with Knowledge-Preserving Model Transformation](https://arxiv.org/abs/2410.03960), [Arctic Inference plugin for vLLM](https://github.com/snowflakedb/ArcticInference)
- Xiao et al., [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) (ICLR 2024)
- [LMSYS Blog: Chunked Pipeline Parallelism](https://lmsys.org/blog/2026-01-15-chunked-pipeline/)
- [vLLM context parallel deployment docs](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/)
- Engineering discussion: [vLLM #22693](https://github.com/vllm-project/vllm/issues/22693) vs [vLLM #34018](https://github.com/vllm-project/vllm/issues/34018)

---

### 10. Structured Output, Constrained Decoding and Tool Use

Many applications require LLM outputs to conform to a specific format (JSON schemas, SQL, code). Constrained decoding enforces these constraints during generation by masking invalid tokens at each step. XGrammar reduces the overhead of grammar-based constrained decoding by precomputing validity for the majority of the vocabulary offline, and has become the default backend in both vLLM and SGLang. This is especially relevant in tool use, where models need to call tools precisely, and tool latency and reliability need to be taken into account.

For example, for a typical JSON grammar over a 128K vocabulary, less than 1% of tokens are context-dependent (requiring full pushdown automaton stack inspection to validate); the 99% of context-independent tokens can be precomputed into adaptive bitmasks per automaton state, shrinking per-state masks. The persistent execution stack supports constant-time rollback via pointer manipulation, enabling grammar work to overlap with GPU execution. Combining constrained decoding with speculative decoding / MTP is nontrivial: each speculated token must be validated against the grammar state produced by _all prior_ speculated tokens, creating a sequential dependency that prevents parallel grammar checking and can cause high rejection rates.

**Reference Suggestions:**

- Dong et al., [XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models](https://arxiv.org/abs/2411.15100) (MLSys 2025), [code](https://github.com/mlc-ai/xgrammar)
- Willard & Louf, [Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702) (2023), [Outlines implementation](https://github.com/dottxt-ai/outlines)
- [LMSYS Blog: Fast JSON Decoding with Compressed Finite State Machine](https://lmsys.org/blog/2024-02-05-compressed-fsm/), jump-forward decoding in SGLang
- [vLLM Blog: Structured Decoding Introduction](https://blog.vllm.ai/2025/01/14/struct-decode-intro.html)
- [SGLang Tool Parser](https://docs.sglang.io/advanced_features/tool_parser.html)
- Engineering discussion: [vLLM #12388](https://github.com/vllm-project/vllm/pull/12388)

---

### 11. System Architecture Design of a Modern LLM Serving Engine

The last topic is an architecture presentation that should synthesize the previous topics into a coherent picture: how vLLM, SGLang, or TensorRT-LLM combine all of the above techniques into a single system. This talk should trace a request through the full serving pipeline, from arrival through scheduling, prefill, caching, decode, and output and discuss how the components interact and where the remaining bottlenecks lie. As such, the talk should focus on the broader framework architecture, and the various model runners.

You can introduce vLLM's V1 architecture using persistent batches which improve upon the V0 architecture by caching the input tensors and isolating the EngineCore into a separate process connected via ZeroMQ. Then, you could introduce V2 (see the model runner v2 design document), and describe how the architecture was changed to fundamentally address issues arising from patches of V1.

You could also run the audience through Mini-SGLang, a learning implementation of only overlap scheduling, high-performance kernels and radix-tree prefix caching, which could be a good pick-up point for a broader discussion of architectural complexity, design choices, and lock-in due to user pressure.

**Reference Suggestions:**

- [vLLM V2 Design Document](https://docs.vllm.ai/en/latest/design/model_runner_v2/)
- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- Zheng et al., [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)
- [vLLM architecture overview](https://docs.vllm.ai/en/latest/design/arch_overview/) and [SGLang documentation](https://docs.sglang.io/)
- [Mini-SGLang: Efficient Inference Engine in a Nutshell](https://lmsys.org/blog/2025-12-17-minisgl/)
- Engineering discussion: [vLLM #12568](https://github.com/vllm-project/vllm/issues/12568) --- a collection of practical grievances with V1 :)

---

## Schedule

| Week | Date   | Topic                                                                           |
| ---- | ------ | ------------------------------------------------------------------------------- |
| 1    | Apr 13 | Introduction & Overview; Topic Assignment                                       |
| 2    | Apr 20 | _No Presentation_ - FAQ / Topic Office Hours                                    |
| 3    | Apr 27 | _No class_                                                                      |
| 4    | May 4  | Topic 1: KV Caching and Paged Attention                                         |
| 5    | May 11 | Topic 2: Request Scheduling and Overlap                                         |
| 6    | May 18 | Topic 3: Speculative Decoding and Modern Multi-Token-Prediction                 |
| 7    | May 25 | _No class (holiday)_                                                            |
| 8    | Jun 1  | Topic 4: The Properties of Modern GPU Hardware and how to write Software for it |
| 9    | Jun 8  | Topic 5: CUDA Graphs and torch.compile for LLM Serving                          |
| 10   | Jun 15 | Topic 6: Practical Quantization for LLM Inference                               |
| 11   | Jun 22 | Topic 7: Prefill-Decode Disaggregation and Chunked Prefill                      |
| 12   | Jun 29 | Topic 8: Serving Parallelism: TP, PP, EP, and Other -P Letter Combinations      |
| 13   | Jul 6  | Topic 9: Long-Context Inference                                                 |
| 14   | Jul 13 | Topic 10: Structured Output, Constrained Decoding and Tool Use                  |
| 15   | Jul 20 | Topic 11: System Architecture Design of a Modern LLM Serving Engine             |
