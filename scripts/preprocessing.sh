chmod +x ./scripts/preprocessing.sh # Excute permission

# Custom patch size 
patch_size=32
processedImg="2bmp"

# Input directory
inputDirRaw="Preprocessing/input_images/testingModel/$processedImg/raw"
inputDirAnnotated="Preprocessing/input_images/testingModel/$processedImg/annotated"
# Output Directory
coordinatesDir="Preprocessing/testingModel_output_patches/$processedImg/centroid_coordinates"
centroidsDir="Preprocessing/testingModel_output_patches/$processedImg/contour_images"
patchesDir="Preprocessing/testingModel_output_patches/$processedImg/patch_size"

python Preprocessing/img_segmentation/getPatches.py --patchSize $patch_size --rawDir "$inputDirRaw" --annotatedDir "$inputDirAnnotated" --coordinatesDir "$coordinatesDir" --centroidsDir "$centroidsDir" --patchesDir "$patchesDir"
#python Preprocessing/img_segmentation/prepareDataset.py --patchSize $patch_size  --TrainingPercentage 0.8 --TestingPercentage 0.2 --processedImg "$processedImg"