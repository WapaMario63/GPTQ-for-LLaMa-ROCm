# This repository is now obsolete (for inference at least)
Use [ExLlama](https://github.com/turboderp/exllama) instead, it performs far better than GPTQ-For-LLaMa and works perfectly in ROCm (21-27 tokens/s on an RX 6800 running LLaMa 2!).
Maybe give the very new [ExLlamaV2](https://github.com/turboderp/exllamav2) a try too if you want to risk with something more bleeding edge.

I only made this as a rather quick port as it only changes few things to make the HIP kernel compile, just so I can mess around with LLMs on my AMD GPU. I do not have experience in programming anything AI/Machine Learning related or even much in GPGPU in general, the most experience I have is with Vulkan and even I wouldn't say I am great at it. If I had the skills, interest and time, I would have tried making a sort of Vulkan kernel backend of some sort.

# GPTQ-for-LLaMA
4 bits quantization of [LLaMA](https://arxiv.org/abs/2302.13971) using [GPTQ](https://arxiv.org/abs/2210.17323)

GPTQ is SOTA one-shot weight quantization method

**This code is based on [GPTQ](https://github.com/IST-DASLab/gptq)**

This is a fork that adds support for ROCm's HIP to use in AMD GPUs, only supported on linux. This has been tested only inside oobabooga's [text generation](https://github.com/oobabooga/text-generation-webui) on an RX 6800 on Manjaro (Arch based distro). Make sure to use pytorch 1.13.1 as 2.0.0 has a bug that breaks this.

Need a guide for installing this with oobabooga's text generation? [Check this wiki page here!](https://github.com/WapaMario63/GPTQ-for-LLaMa-ROCm/wiki/Install-with-Text-Generation)

**NOTICE:** If you are using models that were quantized with the newer cuda kernels from upstream use the ```rocm-latest``` branch! This default branch, for now, is kept for stable usage in [text generation](https://github.com/oobabooga/text-generation-webui)!

Running test_kernel.py on an RX 6800 yields these results:
```
Benchmarking LLaMa-7B FC2 matvec ...
FP16: 0.0010962586402893067
2bit: 0.0004005000591278076
2bit: 0.0004368207454681397 (faster)
3bit: 0.0003001902103424072
3bit: 0.00033979535102844237 (faster)
4bit: 0.00027743816375732424
4bit: 0.00030428624153137207 (faster)
8bit: 0.0002059342861175537
```
Average inference performance using textgen on an RX 6800 (5 tests):
| Model (4bit) (128g) | Tokens/s | 
| --------------------| -------- | 
| gpt4-x-alpaca-13b   |  14.97   | 
| alpaca-13b-lora     |  16.39   | 
| llama-13b           |  16.32   |

Expect an average of around 14-15 tokens/s during normal usage if you use the same GPU.

## New Features
Changed to support new features proposed by [GPTQ](https://github.com/IST-DASLab/gptq#new-features).

* Slightly adjusted preprocessing of C4 and PTB for more realistic evaluations (used in our updated results); can be activated via the flag --new-eval.
* Optimized cuda kernels, which are considerably faster especially on the A100, e.g. 1.9x -> 3.25x generation speedup for OPT-175B; can be activated via --faster-kernel.
* two new tricks:--act-order (quantizing columns in order of decreasing activation size) and --true-sequential (performing sequential quantization even within a single Transformer block). Those fix GPTQ's strangely bad performance on the 7B model (from 7.15 to 6.09 Wiki2 PPL) and lead to slight improvements on most models/settings in general. 

**Currently, `groupsize` and `act-order` do not work together and you must choose one of them.**

## Result

<details>
<summary>LLaMA-7B(click me)</summary>

| [LLaMA-7B](https://arxiv.org/abs/2302.13971)       | Bits | group-size | memory(MiB) | Wikitext2 | checkpoint size(GB) |
| -------------------------------------------------- | ---- | ---------- | ----------- | --------- | ------------------- |
| FP16                                               |  16  |     -      |    13940    |    5.68   |         12.5        |
| RTN                                                |  4   |     -      |      -      |    6.29   |          -          |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  4   |     -      |     4740    |    6.09   |          3.5        |
| RTN                                                |  3   |     -      |      -      |   25.54   |          -          |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  3   |     -      |     3852    |    8.07   |          2.7        |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  3   |    128     |     4116    |    6.61   |          3.0        |

</details>

<details>
<summary>LLaMA-13B</summary>

| [LLaMA-13B](https://arxiv.org/abs/2302.13971)      | Bits | group-size | memory(MiB) | Wikitext2 | checkpoint size(GB) |
| -------------------------------------------------- | ---- | ---------- | ----------- | --------- | ------------------- |
| FP16                                               |  16  |     -      |     OOM     |    5.09   |         24.2        |
| RTN                                                |  4   |     -      |      -      |    5.53   |          -          |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  4   |     -      |     8410    |    5.36   |          6.5        |
| RTN                                                |  3   |     -      |      -      |   11.40   |          -          |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  3   |     -      |     6870    |    6.63   |          5.1        |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  3   |    128     |     7277    |    5.62   |          5.4        |

</details>

<details>
<summary>LLaMA-33B</summary>

| [LLaMa-33B](https://arxiv.org/abs/2302.13971)      | Bits | group-size | memory(MiB) | Wikitext2 | checkpoint size(GB) |
| -------------------------------------------------- | ---- | ---------- | ----------- | --------- | ------------------- |
| FP16                                               |  16  |     -      |     OOM     |    4.10   |         60.5        |
| RTN                                                |  4   |     -      |      -      |    4.54   |          -          |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  4   |     -      |    19493    |    4.45   |         15.7        |
| RTN                                                |  3   |     -      |      -      |   14.89   |          -          |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  3   |     -      |    15493    |    5.69   |         12.0        |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  3   |    128     |    16566    |    4.80   |         13.0        |

</details>

<details>
<summary>LLaMA-65B</summary>

| [LLaMA-65B](https://arxiv.org/abs/2302.13971)      | Bits | group-size | memory(MiB) | Wikitext2 | checkpoint size(GB) |
| -------------------------------------------------- | ---- | ---------- | ----------- | --------- | ------------------- |
| FP16                                               |  16  |     -      |     OOM     |    3.53   |         121.0       |
| RTN                                                |  4   |     -      |      -      |    3.92   |          -          |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  4   |     -      |     OOM     |    3.84   |         31.1        |
| RTN                                                |  3   |     -      |      -      |   10.59   |          -          |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  3   |     -      |     OOM     |    5.04   |         23.6        |
| [GPTQ](https://arxiv.org/abs/2210.17323)           |  3   |    128     |     OOM     |    4.17   |         25.6        |
</details>

Quantization requires a large amount of CPU memory. However, the memory required can be reduced by using swap memory.

Depending on the GPUs/drivers, there may be a difference in performance, which decreases as the model size increases.(https://github.com/IST-DASLab/gptq/issues/1)

According to [GPTQ paper](https://arxiv.org/abs/2210.17323), As the size of the model increases, the difference in performance between FP16 and GPTQ decreases.

## Installation
If you don't have [conda](https://docs.conda.io/en/latest/miniconda.html), install it first.
```
conda create --name gptq python=3.9 -y
conda activate gptq

# For CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
# Or, if you're having trouble with conda, use pip with python3.9:
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# For ROCm. Use pytorch 1.13.1, 2.0.0 currently has a bug that makes this break.
pip install torch==1.13.1+rocm5.2 torchvision==0.14.1+rocm5.2 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/rocm5.2

git clone https://github.com/WapaMario63/GPTQ-for-LLaMa-ROCm
cd GPTQ-for-LLaMa-ROCm
pip install -r requirements.txt

# For CUDA
python setup_cuda.py install
# For ROCm
python setup_rocm.py install

# Benchmark performance for FC2 layer of LLaMa-7B
CUDA_VISIBLE_DEVICES=0 python test_kernel.py
```
## Dependencies

* `torch`: tested on v2.0.0+cu117 or v2.0.0+rocm5.4.2
* `transformers`: tested on v4.28.0.dev0
* `datasets`: tested on v2.10.1
* `safetensors`: tested on v0.3.0
* (to run 4-bit kernels: setup for compiling PyTorch CUDA extensions, see also https://pytorch.org/tutorials/advanced/cpp_extension.html, tested on CUDA 11.7)

All experiments were run on a single NVIDIA RTX3090.

# Language Generation
## LLaMA

```
#convert LLaMA to hf
python convert_llama_weights_to_hf.py --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir ./llama-hf

# Benchmark language generation with 4-bit LLaMA-7B:

# Save compressed model
CUDA_VISIBLE_DEVICES=0 python llama.py ./llama-hf/llama-7b c4 --wbits 4 --true-sequential --act-order --save llama7b-4bit.pt
# Or save compressed `.safetensors` model
CUDA_VISIBLE_DEVICES=0 python llama.py ./llama-hf/llama-7b c4 --wbits 4 --true-sequential --act-order --save_safetensors llama7b-4bit.safetensors
# Benchmark generating a 2048 token sequence with the saved model
CUDA_VISIBLE_DEVICES=0 python llama.py ./llama-hf/llama-7b c4 --wbits 4 --load llama7b-4bit.pt --benchmark 2048 --check
# Benchmark FP16 baseline, note that the model will be split across all listed GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python llama.py ./llama-hf/llama-7b c4 --benchmark 2048 --check

# model inference with the saved model
CUDA_VISIBLE_DEVICES=0 python llama_inference.py ./llama-hf/llama-7b --wbits 4 --load llama7b-4bit.pt --text "this is llama"
# model inference with the saved model with offload(This is very slow. This is a simple implementation and could be improved with technologies like flexgen(https://github.com/FMInference/FlexGen).
CUDA_VISIBLE_DEVICES=0 python llama_inference_offload.py ./llama-hf/llama-7b --wbits 4 --load llama7b-4bit.pt --text "this is llama" --pre_layer 16
It takes about 180 seconds to generate 45 tokens(5->50 tokens) on single RTX3090 based on LLaMa-65B. pre_layer is set to 50.
```
CUDA Kernels support 2,3,4,8 bits and Faster CUDA Kernels support 2,3,4 bits.

Basically, 4-bit quantization and 128 groupsize are recommended.

# Acknowledgements
This code is based on [GPTQ](https://github.com/IST-DASLab/gptq)

Thanks to Meta AI for releasing [LLaMA](https://arxiv.org/abs/2302.13971), a powerful LLM.
