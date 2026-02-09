# first, pre-process the raw data with sample_factor
python -m utils.0_preprocess_data \
  --raw_dir ./example_raw_data \
  --cvt_dir ./train_data \
  --sample_factor 20

# then, generate training data with bin-sampling strategy
python -m utils.1_generate_data \
  --base-dir ./train_data \
  --score-bins 25 \
  --gap-bins 4 \
  --oversample-factor 100 \
  --zero-ratio 0.05 \
  --max_sample_num 1000

# finally, post-process the sampled data for fine-tuning
python -m utils.2_posprocess_data \
  --root-dir ./train_data \
  --merged-json ./train_data/train_jsons/finetune_data_wo_replace.json \
  --final-json ./train_data/train_jsons/finetune_data_final.json \
  --replace-prob 0.75
