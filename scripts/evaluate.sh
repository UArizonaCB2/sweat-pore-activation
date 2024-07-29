chmod +x ./scripts/evaluate.sh # Excute permission

python evaluate.py --CNNmodel 'SimpleCNN_p32_e200.model' --device 'mps' --patchSize 32