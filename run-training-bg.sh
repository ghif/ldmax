# nohup python -m src.scripts.train --config configs/celeba_tpu_b256.yaml --output_dir ./outputs/celeba_tpu_b256_opt > train_celeba_tpu_b256_opt.log 2>&1 &

# nohup python -m src.scripts.train --config configs/celeba_tpu_b128.yaml --output_dir ./outputs/celeba_tpu_b128_opt > train_celeba_tpu_128_opt.log 2>&1 &

nohup python -m src.scripts.train --config configs/cifar10_tpu_b256.yaml --output_dir ./outputs/cifar10_tpu_b256_opt > train_cifar10_tpu_b256_opt.log 2>&1 &