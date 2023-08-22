# Software Entity Recognition with Self-regularization

Official implementation for our paper ["Software Entity Recognition with Noise-robust Learning"](https://arxiv.org/abs/2308.10564), ASE 2023.


## WikiSER
WikiSER corpus includes 1.7M sentences with named entity labels extracted from Wikipedia.
The corpus contains a high number of software-related entities labeled under 12 fine-grained categories:

| Type             | Examples                                              |
|------------------|-------------------------------------------------------|
| Algorithm        | Auction algorithm, Collaborative filtering            |
| Application      | Adobe Acrobat, Microsoft Excel                       |
| Architecture     | Graphics processing unit, Wishbone                   |
| Data_Structure   | Array, Hash table, mXOR linked list                  |
| Device           | Samsung Gear S2, iPad, Intel T5300                    |
| Error Name       | Buffer overflow, Memory leak                         |
| General_Concept  | Memory management, Nouvelle AI                       |
| Language         | C++, Java, Python, Rust                               |
| Library          | Beautiful Soup, FastAPI                               |
| License          | Cryptix General License, MIT License                  |
| Operating_System | Linux, Ubuntu, Red Hat OS, MorphOS                   |
| Protocol         | TLS, FTPS, HTTP 404                                   |


To download the full dataset, please navigate to this [folder](src/data/wikiser).

## Models
Our finetuned checkpoints are available through HuggingFace: [wikiser-bert-case](https://huggingface.co/taidng/wikiser-bert-base) and [wikiser-bert-large](https://huggingface.co/taidng/wikiser-bert-large).

You can load in the model by the standard API:
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("taidng/wikiser-bert-base")
model = AutoModelForTokenClassification.from_pretrained("taidng/wikiser-bert-base")
```

## Train with Self-regularization
We suggest using [conda](https://docs.conda.io/en/main/miniconda.html#installing) to set up your environment. To begin, we create a new environment using `environment.yml`, naming it "ser" by default.
```
conda env create -f environment.yml
```

To start train script:
```
python3 train_nll.py --model_name_or_path=bert-base-cased  --alpha=10 --n_model=2 --data_dir=data/wikiser-small --epochs=25
```
* Datasets: WikiSER_small, [S-NER](https://ieeexplore.ieee.org/document/7476633), and relabeled [SoftNER](https://arxiv.org/abs/2005.01634).

By default, training loss and evaluation statistics are stored in [wandb](https://wandb.ai/site).

## Citation
If you find our work helpful, please cite:
```bibtex
@inproceedings{nguyen2023software,
  title={Software Entity Recognition with Noise-Robust Learning},
  author={Nguyen, Tai and Di, Yifeng and Lee, Joohan and Chen, Muhao and Zhang, Tianyi},
  booktitle={Proceedings of the 38th IEEE/ACM International Conference on Automated Software Engineering (ASE'23)},
  year={2023},
  organization={IEEE/ACM}
}
```


