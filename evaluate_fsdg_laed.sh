sess=$1
rnd=$2
domain=$3
black_ratio=$4
log_dir=logs_${domain}_${black_ratio}

python train.py \
  --data_dir NeuralDialog-ZSDG/data/stanford \
  --laed_z_folders laed_features/di_vae_10x5_maluuba__smd_$domain laed_features/st_ed_10x5_maluuba__smd_$domain \
  --log_dir $log_dir \
  --black_domains $domain \
  --black_ratio $black_ratio \
  --action_match False \
  --target_example_cnt 0 \
  --source_example_cnt 0 \
  --random_seed $rnd \
  --domain_description annotation \
  --load_sess $sess \
  --forward_only \
  --lowercase
