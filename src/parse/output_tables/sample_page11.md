Table 11. Detection results on the PASCAL VOC 2012 test set ( http://host.robots.ox.ac.uk:8080/leaderboard/ displaylb.php?challengeid=11&amp;compid=4 ). The baseline is the Faster R-CNN system. The system 'baseline+++' include box refinement, context, and multi-scale testing in Table 9.

| system      | net        | data        | mAP areo bike bird boat bottle bus car cat chair cow table dog horse mbike person plant sheep sofa train tv   |
|-------------|------------|-------------|---------------------------------------------------------------------------------------------------------------|
| baseline    | VGG-16     | 07++12      | 70.4 84.9 79.8 74.3 53.9 49.8 77.5 75.9 88.5 45.6 77.1 55.3 86.9 81.7 80.9 79.6 40.1 72.6 60.9 81.2 61.5      |
| baseline    | ResNet-101 | 07++12      | 73.8 86.5 81.6 77.2 58.0 51.0 78.6 76.6 93.2 48.6 80.4 59.0 92.1 85.3 84.8 80.7 48.1 77.3 66.5 84.7 65.6      |
| baseline+++ | ResNet-101 | COCO+07++12 | 83.8 92.1 88.4 84.8 75.9 71.4 86.3 87.8 94.2 66.8 89.4 69.2 93.9 91.9 90.9 89.6 67.9 88.2 76.8 90.3 80.0      |