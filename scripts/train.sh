chmod +x ./scripts/train.sh # Excute permission

# The user can add --tag 'info' for naming the ML model (defualt is None)
python train.py --batchSize 8 --CNN 'SimpleCNN_p32' --epochs 150 --TrainingPercentage 0.8 --TestingPercentage 0.2 --device 'mps' --tag 'stratified_5Folds'