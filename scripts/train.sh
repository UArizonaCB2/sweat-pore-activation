chmod +x ./scripts/train.sh # Excute permission

python train.py --patchSize 17 --CNN 'SimpleCNN_p17' --batchSize 8 --TrainingPercentage 0.8 --TestingPercentage 0.2 --epochs 40 --device 'mps'