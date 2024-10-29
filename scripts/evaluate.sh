chmod +x ./scripts/evaluate.sh # Excute permission

# Golbal variables
imgName="6" #ex. 2.bmp ---> 2
patchesDir="Preprocessing/centralizedPatches/32X32/"

# 1. Run the preprocessing step
echo "Running preprocessing to create prediction on the unseen patches..."
rm -rf Preprocessing/centralizedPatches/32X32/* #Delete the existing patches 
python Preprocessing/centralize_sweatpores.py --imgName $imgName
# 2. Run the evaluation
echo "Starting evaluation..."
python evaluate.py --batchSize 8 --CNNmodel 'CNN4Layers_p32_e20_135bmps.model' --prediction $imgName'bmp' --device 'mps' --patchesDir $patchesDir