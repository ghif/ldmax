nohup python -m src.scripts.train \
  --config configs/celeba_tpu_b128.yaml \
  --output_dir ./outputs/celeba_tpu_b128 > train_celeba_tpu_b128.log 2>&1 &