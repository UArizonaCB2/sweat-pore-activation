chmod +x ./scripts/train.sh # Excute permission

# The user can add --tag 'info' for naming the ML model (defualt is None)
python train.py --batchSize 8 --CNN 'CNN4Layers_p32' --epochs 30 --device 'mps'