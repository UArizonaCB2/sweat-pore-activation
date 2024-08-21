chmod +x ./scripts/train.sh # Excute permission

# The user can add --tag 'info' for naming the ML model (defualt is None)
python train.py --batchSize 8 --CNN 'SimpleCNN_p32' --epochs 100 --device 'mps' --tag 'stratified_5Folds'