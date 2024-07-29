chmod +x ./scripts/train.sh # Excute permission

python train.py --batchSize 8 --CNN 'SimpleCNN_p32' --epochs 200 --TrainingPercentage 0.8 --TestingPercentage 0.2 --device 'mps'