# nanorlhf

## Motivation
A few years ago, it still made sense for an individual to think “I’ll just train my own model.”
I had no GPUs of my own, but thanks to some open-source libraries I’d built, I was lucky enough to work with excellent teams like Hugging Face, the DeepSpeed team, and EleutherAI.
Through that, I got access to 512 GPUs and used them to train [Polyglot-Ko](https://github.com/EleutherAI/polyglot), the first commercially usable Korean LLM.
That was meaningful back then.
In 2025, though, companies train models like Qwen3 on tens of trillions of tokens with what is likely tens of thousands of GPUs, then release them for free.
In that world, a small, weaker model trained by one person on a few GPUs does not move the needle very much.

The same thing is happening with libraries.
Modern LLM frameworks like Megatron-LM or verl are maintained by full-time teams.
An individual can still hack together a good framework after work, but it is very hard to keep up with the maintenance and feature velocity of a dedicated corporate team.
Projects like OpenRLHF showed what a small group can do, but once a similar library with stronger backing appears, it naturally becomes the default choice.

So I changed the question.
Instead of “How do I compete with that?”, I started thinking “What can a single person, with no GPUs and a few hours a day, still do that is genuinely useful?”
That brought me back to what Andrej Karpathy did with his “nano” series: small, from-scratch implementations that teach, rather than compete.
nanorlhf, and the other “nano” libraries I’m building, follow that idea.
They are not meant to be the fastest or most complete frameworks.
They are meant to be small, readable, PyTorch-only references that show how RLHF and LLM infrastructure actually work, and later come with notebooks and free lectures.
These projects will never have the scale or efficiency of company-level systems, but I still believe that even without huge capital or full-time hours, an individual can create something meaningful, and that work can positively influence others.


## Contents 
| Packages                                                                          | Description                                                 | Reference                                                                                                                                   | Examples                                                                            |
|-----------------------------------------------------------------------------------|-------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| [`nanosets`](https://github.com/hyunwoongko/nanorlhf/tree/main/nanorlhf/nanosets) | Scratch implementation of zero-copy dataset library         | [arrow](https://github.com/apache/arrow), [datasets](https://github.com/huggingface/datasets)                                               | [available](https://github.com/hyunwoongko/nanorlhf/tree/main/examples/nanosets.py) |
| [`nanotron`](https://github.com/hyunwoongko/nanorlhf/tree/main/nanorlhf/nanotron) | Scratch implementation of various parallelisms              | [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [oslo](https://github.com/EleutherAI/oslo)                                            | [available](https://github.com/hyunwoongko/nanorlhf/tree/main/examples/nanotron.py) |
| `nanovllm`                                                                        | Scratch implementation of high performance inference engine | [vllm](https://github.com/vllm-project/vllm), [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)                                      | not available                                                                       |
| [`nanoray`](https://github.com/hyunwoongko/nanorlhf/tree/main/nanorlhf/nanoray)   | Scratch implementation of distributed computing engine      | [ray](https://github.com/ray-project/ray)                                                                                                   | [available](https://github.com/hyunwoongko/nanorlhf/tree/main/examples/nanoray.py)  |
| `nanorlhf`                                                                        | Scratch implementation of asynchronous RL framework         | [verl](https://github.com/volcengine/verl), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [AReaL](https://github.com/inclusionAI/AReaL) | not available                                                                       |
| [`kernels`](https://github.com/hyunwoongko/nanorlhf/tree/main/nanorlhf/kernels)   | Scratch implementation of various triton kernels            | [trident](https://github.com/kakaobrain/trident)                                                                                            | not available                                                                       |
