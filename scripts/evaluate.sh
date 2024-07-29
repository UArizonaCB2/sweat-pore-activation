chmod +x ./scripts/evaluate.sh # Excute permission

python evaluate.py --patchSize 17 --CNNmodel 'SimpleCNN_p17_e40.model' --device 'mps' --batchSize 8