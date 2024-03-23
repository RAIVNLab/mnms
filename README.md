# m&ms
m&ms is a benchmark for evaluating large language model (LLM) agents' tool-use abilities on multi-step multi-modal tasks. You can find the evaluation dataset on [HuggingFace](https://huggingface.co/datasets/zixianma/mms).

## Dataset examples
<img src="dataset_examples.png">

## Dataset details
This dataset contains 4K+ multi-step multi-modal tasks involving 33 tools that include 13 multi-modal models, 9 (free) public APIs, and 11 image processing modules. For each of these task queries, we provide automatically generated plans using this realistic toolset. We further provide a high-quality subset of 1,565 human-verified task plans and 882 human-verified, filtered, and correctly executable plans. Below is a table summarizing the statistics of m&ms:
<p align="center">
  <img src="dataset_stats.png" width="500px">
</p>



## Dataset generation
<p align="center">
  <img src="dataset_gen.png" width="800px">
</p>

## Installation
Please make sure you install all the required packages in ```requirements.txt``` by running:
```
pip install -r requirements.txt
```

## Evaluation
To evaluate the predicted plans against the groundtruth plans in m&ms, simply run this line:
```
python -m evaluation.run --output-csv <output.csv> --plan-format json --preds-file <predictions.json>
```
Note that our code works with a ```predictions.json``` file where each line is a dictionary containing "id" and "prediction" like the example below:
```
{"id": 10, "prediction": [{"id": 0, "name": "text classification", "args": {"text": "forced, familiar and thoroughly condescending."}}]}
{"id": 24, "prediction": [{"id": 0, "name": "text generation", "args": {"text": "Who exactly became the King of the Canary Islands?"}}]}
...
```

But you are welcome to modify it to work with other file formats. 

## Execution

To execute the groundtruth plans, you can run this line:
```
HUGGINGFACE_HUB_CACHE=<cache_dir> RAPID_API_KEY=<rapid_api_key> OMDB_API_KEY=<omdb_api_key> OPENAI_API_KEY=<openai_api_key> python -m execution.run --plan-format code
```
You can generate your own API keys on [Rapid](https://rapidapi.com/), [OMDb](https://www.omdbapi.com/), and [OpenAI](https://openai.com/).

To execute predicted plans, you will need to additionally provide a ```predictions.json``` file like this:
```
python -m execution.run --input-file <predictions.json> --plan-format json
```

## Citation
Please cite us if you find our work helpful!
```
@article{ma2024mms,
  title={m&m's: A Benchmark to Evaluate Tool-Use for multi-step multi-modal Tasks}, 
  author={Zixian Ma and Weikai Huang and Jieyu Zhang and Tanmay Gupta and Ranjay Krishna},
  year={2024},
  journal={arXiv preprint arXiv:2403.11085},
}
```
