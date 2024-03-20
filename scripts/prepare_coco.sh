echo "...Preparing COCO dataset..."

cd data
mkdir COCO
cd COCO

mkdir images annotations

# train2017 images
wget images.cocodataset.org/zips/train2017.zip
unzip train2017.zip -d images
rm train2017.zip

# val2017 images
wget images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d images
rm val2017.zip

# test2017 images
wget images.cocodataset.org/zips/test2017.zip
unzip test2017.zip -d images
rm test2017.zip

# trainval2017 annotations
wget images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
rm annotations_trainval2017.zip


echo "number of train samples: `find images/train2017 -name "*.jpg" | wc -l`"
echo "number of val samples: `find images/val2017 -name "*.jpg" | wc -l`"
echo "number of test samples: `find images/test2017 -name "*.jpg" | wc -l`"

cd ../..
echo "Directory structure:"
tree data/COCO -L 2