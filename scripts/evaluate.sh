chmod +x ./scripts/evaluate.sh # Excute permission

python evaluate.py --batchSize 8 --CNNmodel 'CNN4Layers_p32_e20_135bmps.model' --prediction '2bmp' --device 'mps'