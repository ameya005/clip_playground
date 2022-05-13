# clip_playground
## Code for VisML project. 

### Description

Zero-shot learning models like CLIP leverage natural language text as supervisory signals unlike traditional supervised learning approaches. They show great performance gains over other approaches. However, given that the labels consist of natural text, the choice of words and phrases used becomes important. We present a novel topological based approach that allows us to select text prompts that are most representative of a classes. We rely on \textsc{MAPPER}, a topology preserving projection algorithm to construct a Reeb graph of the embedding space, and further propose a margin-based approach to select subgraphs that provide the greatest predictive value. We show empirical evidence of the efficacy of our algorithm on zero-shot classification for Imagenette, a subset of the large Imagenet dataset. Consequently, we also discuss on the use of topological methods to analyse the effect of synonyms and other hypernyms on the performance of CLIP. 


Implementation for prompt engineering can be found [here](https://github.com/ameya005/clip_playground/blob/main/CLIP/prompt_analysis-imagenette.ipynb).

MAPPER graphs for the 10 Imagenette class labels can be found [here](https://github.com/ameya005/clip_playground/tree/main/CLIP/result_html_files).
