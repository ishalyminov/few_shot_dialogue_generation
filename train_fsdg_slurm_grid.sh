domain=$1
black_ratio=$2

for rnd in 271 272 273 274 275 276 277 278 279 280; do
  sbatch train_fsdg_slurm.sh LAZslStanfordCorpus \
    --data_dir NeuralDialog-ZSDG/data/stanford \
    --log_dir logs_fsdg \
    --laed_z_folders laed_features/st_ed_10x5_maluuba__smd_$domain laed_features/di_vae_10x5_maluuba__smd_$domain \
    --black_domains $domain \
    --black_ratio $black_ratio \
    --action_match False \
    --target_example_cnt 0 \
    --source_example_cnt 0 \
    --random_seed $rnd \
    --domain_description annotation
done
