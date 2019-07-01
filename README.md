# Dialog Knowledge Transfer

See the paper "Few-Shot Dialogue Generation Without Annotated Data: A Transfer Learning Approach" by [Igor Shalyminov](https://ishalyminov.github.io/), [Sungjin Lee](https://www.microsoft.com/en-us/research/people/sule/), [Arash Eshghi](https://sites.google.com/site/araesh81/), and [Oliver Lemon](https://sites.google.com/site/olemon/). [INTERSPEECH 2019 publication]

How To
=

1. Training a LAED model - StED (skip-thought dialog-level)
```
python st_ed.py \
    data/crowdsourced_task-oriented_dialogues/blis_collected_dialogues.json \
    LAEDBlisCorpus \
    vocabs/maluuba.json \
   --exlcude_domains ... \
   --y_size 10 
```

  1.1 Training a LAED model - vanilla VAE (dialog-level)
  ```
  python vae.py \
    data/crowdsourced_task-oriented_dialogues/blis_collected_dialogues.json \
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
    --vocab vocabs/stanford_maluuba.json
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
