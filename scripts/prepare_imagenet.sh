echo "...Preparing ImageNet dataset..."

cd data

num_files=`unzip -l imagenet-object-localization-challenge.zip | wc -l`

echo "-> Unzipping imagenet-object-localization-challenge.zip file: ${num_files} files"
unzip -o imagenet-object-localization-challenge.zip | tqdm --desc "Extracting files:" --unit files --unit_scale --total $num_files > /dev/null
echo "-> Unzipped imagenet-object-localization-challenge.zip file"

rm imagenet-object-localization-challenge.zip
echo "-> Removed imagenet-object-localization-challenge.zip file"

echo "-> Moving train, val and test images files to data/ImageNet"
mkdir ImageNet

mv ILSVRC/Data/CLS-LOC/train ImageNet/
echo "-> Moved train files"

mv ILSVRC/Data/CLS-LOC/val ImageNet/
echo "-> Moved val files"

mv ILSVRC/Data/CLS-LOC/test ImageNet/
echo "-> Moved test files"

rm -rf ILSVRC
rm LOC_sample_submission.csv
rm LOC_synset_mapping.txt
rm LOC_train_solution.csv
rm LOC_val_solution.csv

echo "-> Removed redundant files and directories"

echo "-> Parsing val files to match train files structure"
cd ImageNet/val
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
cd ..

echo "-> Downloading wordnet labels mapping"
wget https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json
mv imagenet_class_index.json wordnet_labels.yaml


echo "Data statistics:"
echo "number of train samples: `find train/ -name "*.JPEG" | wc -l`"
echo "number of val samples: `find val/ -name "*.JPEG" | wc -l`"
echo "number of test samples: `find test/ -name "*.JPEG" | wc -l`"
echo ""

cd ../..
echo "Directory structure:"
tree data/ImageNet -L 1
