# phi-3-playground
Working on using [Phi-3](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) with PyTorch in multiple environments

**Local:** MacBook Pro 16-inch 2023 w/ M2 Pro & 32 GB RAM *(Typical Footprint: ~16 GB in RAM)*
**Remote:** [Adroit](https://researchcomputing.princeton.edu/systems/adroit) *(Typical Footprint: ~9 GB in GPU)*

## Model Location

You'll have to update *_PHI_PATH to match wherever you've downloaded Phi-3

## Environment

I'm using an Anaconda environment with:
- python==3.11
- torch==2.2.1 (-c pytorch -c nvidia)
- transformers==4.40 (pip)

When using GPU:
- accelerate==0.29.3 (pip)
- flash-attn==2.5.8 (pip)
