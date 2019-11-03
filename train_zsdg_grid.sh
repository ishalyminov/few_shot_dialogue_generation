domain=$1
black_ratio=$2
log_dir=$3

for rnd in 271 272 273 274 275 276 277 278 279 280; do
  python train.py ZslStanfordCorpus \
    --data_dir NeuralDialog-ZSDG/data/stanford \
    --log_dir $log_dir \
    --black_domains $domain \
    --black_ratio $black_ratio \
    --action_match True \
    --target_example_cnt 200 \
    --source_example_cnt 1000 \
    --random_seed $rnd \
    --domain_description annotation 
done

# --laed_z_folders laed_features/st_ed_10x5_maluuba__smd_$domain laed_features/di_vae_10x5_maluuba__smd_$domain \
