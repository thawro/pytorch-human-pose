cd data
mkdir ImageNet
mv ILSVRC/Data/CLS-LOC/train ImageNet/
mv ILSVRC/Data/CLS-LOC/val ImageNet/
mv ILSVRC/Data/CLS-LOC/test ImageNet/

rm -rf ILSVRC

cd ImageNet/val
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
cd ..

wget https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json
mv imagenet_class_index.json wordnet_labels.yaml

echo "Directory structure: \n`tree -L 1`"
echo "number of train samples: `find train/ -name "*.JPEG" | wc -l`"
echo "number of val samples: `find val/ -name "*.JPEG" | wc -l`"

cd ../..
