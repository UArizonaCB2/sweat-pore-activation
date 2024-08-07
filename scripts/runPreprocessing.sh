chmod +x ./scripts/runPreprocessing.sh # Excute permission
# Input directory
inputDirRaw='Preprocessing/input_images/smallBrushesNoBackground/raw'
inputDirAnnotated='Preprocessing/input_images/smallBrushesNoBackground/annotated'
# Output Directory
coordinatesDir='Preprocessing/output_patches/centroid_coordinates'
centroidsDir='Preprocessing/output_patches/contour_images'
patchesDir='Preprocessing/output_patches/patch_size'


python Preprocessing/img_segmentation/getPatches.py --patchSize 32 --rawDir "$inputDirRaw" --annotatedDir "$inputDirAnnotated" --coordinatesDir "$coordinatesDir" --centroidsDir "$centroidsDir" --patchesDir "$patchesDir"