# Dialog Knowledge Transfer

See the paper "Few-Shot Dialogue Generation Without Annotated Data: A Transfer Learning Approach" by [Igor Shalyminov](https://ishalyminov.github.io/), [Sungjin Lee](https://www.linkedin.com/in/sungjinlee/), [Arash Eshghi](https://sites.google.com/site/araesh81/), and [Oliver Lemon](https://sites.google.com/site/olemon/). [[SigDial 2019 publication]](https://arxiv.org/pdf/1908.05854.pdf) [[Poster]](https://drive.google.com/file/d/1_0jPct70HyChxCTQtxa-EuUv2QBaogDe/view?usp=sharing)

Pre-requisites
==
1. Download the [Maluuba MetaLWOz](https://www.microsoft.com/en-us/research/project/metalwoz/) (previously called BLIS) dataset (assuming you extract it into `metalwoz-v1` folder)
2. Create a vocabulary for LAED training:
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
