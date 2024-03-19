cd data
mkdir COCO
cd COCO

wget images.cocodataset.org/zips/train2017.zip
wget images.cocodataset.org/zips/val2017.zip
wget images.cocodataset.org/zips/test2017.zip
wget images.cocodataset.org/annotations/annotations_trainval2017.zip

mkdir images annotations

unzip train2017.zip -d images
unzip val2017.zip -d images
unzip test2017.zip -d images
unzip annotations_trainval2017.zip

rm train2017.zip val2017.zip test2017.zip annotations_trainval2017.zip

echo "Directory structure: \n`tree -L 2`"
echo "number of train samples: `find images/train2017 -name "*.jpg" | wc -l`"
echo "number of val samples: `find images/val2017 -name "*.jpg" | wc -l`"
echo "number of test samples: `find images/test2017 -name "*.jpg" | wc -l`"

cd ../..