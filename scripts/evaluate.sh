chmod +x ./scripts/evaluate.sh # Excute permission

python evaluate.py --batchSize 8 --CNNmodel 'SimpleCNN_p32_e100_stratified_5Folds.model' --device 'mps'