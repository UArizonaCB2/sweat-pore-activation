chmod +x ./scripts/runModel.sh # Excute permission

python train.py --patchSize 32 --TrainingPercentage 0.8 --TestingPercentage 0.2 --CNN 'SimpleCNN' --epochs 200 --device 'mps'