domain=$1

for rnd in 271 272 273 274 275 276 277 278 279 280; do
  sbatch train_zsdg_slurm.sh ZslStanfordCorpus \
    --data_dir NeuralDialog-ZSDG/data/stanford_annotated \
    --log_dir logs_zsdg \
    --black_domains $domain \
    --black_ratio 1.0 \
    --action_match True \
    --target_example_cnt 200 \
    --source_example_cnt 1000 \
    --random_seed $rnd \
    --domain_description nlu
done

# --laed_z_folders laed_features/st_ed_10x5_maluuba__smd_$domain laed_features/di_vae_10x5_maluuba__smd_$domain \
