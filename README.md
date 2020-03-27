# Dialogue Knowledge Transfer Networks (DiKTNet)

This codebase is shared by the following papers:
"Few-Shot Dialogue Generation Without Annotated Data: A Transfer Learning Approach" by [Igor Shalyminov](https://ishalyminov.github.io/), [Sungjin Lee](https://www.linkedin.com/in/sungjinlee/), [Arash Eshghi](https://sites.google.com/site/araesh81/), and [Oliver Lemon](https://sites.google.com/site/olemon/). [[SigDial 2019 publication]](https://www.aclweb.org/anthology/W19-5904.pdf) [[bibtex]](https://www.aclweb.org/anthology/W19-5904.bib) [[Poster]](https://drive.google.com/file/d/1_0jPct70HyChxCTQtxa-EuUv2QBaogDe/view?usp=sharing)

"Data-Efficient Goal-Oriented Conversation with Dialogue Knowledge Transfer Networks" by [Igor Shalyminov](https://ishalyminov.github.io/), [Sungjin Lee](https://www.linkedin.com/in/sungjinlee/), [Arash Eshghi](https://sites.google.com/site/araesh81/), and [Oliver Lemon](https://sites.google.com/site/olemon/). [[EMNLP 2019 publication]](https://www.aclweb.org/anthology/D19-1183.pdf) [[bibtex]](https://www.aclweb.org/anthology/D19-1183.bib) [[Poster]](https://drive.google.com/file/d/1jH4H3pQC5HjYPD7n_wsmURL8I_ohHBt_/view?usp=sharing)

If you find it useful for your work, please cite the papers above.

The versions of the code corresponding to each publication can be found using the git release tags.



Pre-requisites
==
1. Download the [Maluuba MetaLWOz](https://www.microsoft.com/en-us/research/project/metalwoz/) (previously called BLIS) dataset (assuming you extract it into `metalwoz-v1` folder)
2. Prepare environment and dependencies. Below are the steps for Conda:

```
  conda create -n diktnet python=3.6
  conda activate diktnet
  git submodule update --init
  pip install -r requirements.txt
```

3. Create a vocabulary for LAED training:
`python make_vocabulary.py <data-dir> <corpus client type> <vocab_file.json>`

Given the data sources we have, it may be `NeuralDialog-ZSDG/data/stanford`/`ZslStanfordCorpus`, `NeuralDialog-ZSDG/data/simdial`/`SimDialCorpus` or `metalwoz-v1`/`LAEDBlisCorpus`. Vocabularies of a higher coverage can be produced by merging the primary ones.

How To
=

1. Training a LAED model - StED (skip-thought dialog-level)
```
python st_ed.py \
    metalwoz-v1 \
    LAEDBlisCorpus \
    vocabs/maluuba.json \
   --exlcude_domains ... \
   --y_size 10 
```

  1.1 Training a LAED model - vanilla VAE (dialog-level)
  ```
  python vae.py \
    metalwoz-v1 \
    LAEDBlisCorpus \
    vocabs/maluuba.json \
   --exlcude_domains ... \
   --y_size 1
   --k 100
  ```

2. Generating LAED features - StED (skip-thought dialog-level)
```
python generate_laed_features.py \
    laed_models/st_ed_maluuba_${TARGET_DOMAIN} \
    laed_features/st_ed_maluuba__smd_${TARGET_DOMAIN} \
    --model_name StED \
    --model_type dialog \
    --data_dir NeuralDialog-ZSDG/data/stanford \
    --corpus_client ZslStanfordCorpus \
    --data_loader SMDDialogSkipLoader \
    --vocab maluuba.json
```

3. Training a Few-Shot Dialog Generation model
```
python train_fsdg.py \
    LAZslStanfordCorpus \
    --data_dir NeuralDialog-ZSDG/data/stanford \
    --laed_z_folders laed_features/st_ed_3x5_maluuba__smd_${domain} \
    --black_domains $domain \
    --black_ratio 0.9 \
    --action_match False \
    --target_example_cnt 0 \
    --random_seed $rnd 
```
