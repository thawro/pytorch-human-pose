echo "...Preparing COCO dataset..."

cd data
mkdir COCO
cd COCO

mkdir images annotations

echo "-> Processing zip files (download zip -> unzip -> remove zip)"

echo "-> Processing trainval2017 annotations"
wget images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
rm annotations_trainval2017.zip

echo "-> Processing train2017 images files"
wget images.cocodataset.org/zips/train2017.zip
num_train_files=`unzip -l train2017.zip | wc -l`
unzip -o train2017.zip -d images | tqdm --desc "Extracting train files:" --unit files --unit_scale --total $num_train_files > /dev/null
rm train2017.zip

echo "-> Processing val2017 images files"
wget images.cocodataset.org/zips/val2017.zip
num_val_files=`unzip -l val2017.zip | wc -l`
unzip -o val2017.zip -d images | tqdm --desc "Extracting val files:" --unit files --unit_scale --total $num_val_files > /dev/null
rm val2017.zip

echo "-> Processing test2017 images files"
wget images.cocodataset.org/zips/test2017.zip
num_test_files=`unzip -l test2017.zip | wc -l`
unzip -o test2017.zip -d images | tqdm --desc "Extracting test files:" --unit files --unit_scale --total $num_test_files > /dev/null
rm test2017.zip



echo "Data statistics:"
echo "number of train samples: `find images/train2017 -name "*.jpg" | wc -l`"
echo "number of val samples: `find images/val2017 -name "*.jpg" | wc -l`"
echo "number of test samples: `find images/test2017 -name "*.jpg" | wc -l`"
echo ""

cd ../..
echo "Directory structure:"
tree data/COCO -L 2