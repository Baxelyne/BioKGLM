# BioKGLM, PLM with Biomedical Knowledge Graph for Biomedical Inforamtion Extraction

**Source codes for the `Briefings in Bioinformatics` paper "[Enriching contextualized language model from knowledge graph for biomedical information extraction](https://academic.oup.com/bib/article/22/3/bbaa110/5854405)"**

--------------

# Installing Reqirements:

* Pytorch>=0.4.1
* Python
* tqdm
* boto3
* requests
* transformers==3.0.0


# Prepare the Pre-trained Bakcbone (Bio)BERT Model

Download the pre-trained BERT and load the parameters as the backbone LM of our BioLM.

- [BERT-Base, Uncased](https://github.com/google-research/bert)
- [BERT-Large, Uncased](https://github.com/google-research/bert)


Also, it is preferable to use the well-trained BioBERT model, which is learned with biomedical corpus, e.g., PubMed abstracts and PMC articles.

- [BioBERT-Base](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2)
- [BioBERT-Large](http://nlp.dmis.korea.edu/projects/biobert-2020-checkpoints/biobert_large_v1.1_pubmed.tar.gz)

# Post-training on the biomedical knowledge graph for knowledge injection

## Prepare databases of biomedical knowledge graph:

- [SemMedDB](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3509487/)
- [BioGrakn](https://github.com/vaticle/biograkn)
- [Reactome](https://reactome.org/what-is-reactome)
- [HGNC](https://www.genenames.org/)
- [UniprotKB](https://www.uniprot.org/help/uniprotkb)

## Represent the bio KG with knowledge graph embedding models:

* Pre-training the KG representations using the [THU-OpenKE](https://github.com/thunlp/OpenKE) open tools.
* Download pre-trained bioKG embedding from [Google Drive](https://drive.google.com/open?id=1IbJ20YEW81Vcm2GdJCiS21MSIdAAp1Ex), and unzip it into `checkpoints` fold.

## Post-training the Bakcbone PLM over the bioKG:

```bash
python run_post_training.py
```
Properly configurate all the runtime parameters.


## Fine-tuning on the downstream biomedical-domain training sets:

- named entity recognition task:

```bash
python run_bioNER.py 
```

    
- relation extraction task:

```bash
python run_bioRelClz.py 
```

- event extraction task:

```bash
python run_bioEventExt.py 
```

Properly setup the datasets.

# Testing on the downstream information extraction tasks:

- named entity recognition task:

```bash
python eval_bioNER.py 
```
    
- relation extraction task:

```bash
python eval_bioRelClz.py 
```

- event extraction task:

```bash
python eval_bioEventExt.py 
```

Properly setup the datasets.


# Citation

If you use the code, please cite this paper:

```
@article{fei2021enriching,
  title={Enriching contextualized language model from knowledge graph for biomedical information extraction},
  author={Fei, Hao and Ren, Yafeng and Zhang, Yue and Ji, Donghong and Liang, Xiaohui},
  journal={Briefings in bioinformatics},
  volume={22},
  number={3},
  pages={bbaa110},
  year={2021},
  publisher={Oxford University Press}
}
```




# License

The code is released under Apache License 2.0 for Noncommercial use only. 

