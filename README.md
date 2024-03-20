# m&ms
m&ms is a benchmark for evaluating large language model (LLM) agents' tool-use abilities on multi-step multi-modal tasks. You can find the evaluation dataset on [HuggingFace](https://huggingface.co/datasets/zixianma/mms).

## Dataset examples
<img src="dataset_examples.png">

## Dataset details
This dataset contains 4K+ multi-step multi-modal tasks involving 33 tools that include 13 multi-modal models, 9 (free) public APIs, and 11 image processing modules. For each of these task queries, we provide automatically generated plans using this realistic toolset. We further provide a high-quality subset of 1,565 human-verified task plans and 882 human-verified, filtered, and correctly executable plans.

## Dataset generation
<img src="dataset_gen.png">

## Evaluation
Code coming soon!

## Citation
```
@misc{ma2024mms,
      title={m&m's: A Benchmark to Evaluate Tool-Use for multi-step multi-modal Tasks}, 
      author={Zixian Ma and Weikai Huang and Jieyu Zhang and Tanmay Gupta and Ranjay Krishna},
      year={2024},
      eprint={2403.11085},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
