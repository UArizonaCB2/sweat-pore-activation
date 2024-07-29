chmod +x ./scripts/evaluate.sh # Excute permission

python evaluate.py --batchSize 8 --CNNmodel 'SimpleCNN_p32_e200.model' --device 'mps'