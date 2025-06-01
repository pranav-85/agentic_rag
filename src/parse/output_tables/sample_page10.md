Table 10. Detection results on the PASCAL VOC 2007 test set. The baseline is the Faster R-CNN system. The system 'baseline+++' include box refinement, context, and multi-scale testing in Table 9.

| system      | net        | data       |   mAP | areo bike bird boat bottle bus car cat chair cow table dog horse mbike person plant sheep sofa train tv   |
|-------------|------------|------------|-------|-----------------------------------------------------------------------------------------------------------|
| baseline    | VGG-16     | 07+12      |  73.2 | 76.5 79.0 70.9 65.5 52.1 83.1 84.7 86.4 52.0 81.9 65.7 84.8 84.6 77.5 76.7 38.8 73.6 73.9 83.0 72.6       |
| baseline    | ResNet-101 | 07+12      |  76.4 | 79.8 80.7 76.2 68.3 55.9 85.1 85.3 89.8 56.7 87.8 69.4 88.3 88.9 80.9 78.4 41.7 78.6 79.8 85.3 72.0       |
| baseline+++ | ResNet-101 | COCO+07+12 |  85.6 | 90.0 89.6 87.8 80.8 76.1 89.9 89.9 89.6 75.5 90.0 80.7 89.6 90.3 89.1 88.7 65.4 88.1 85.6 89.0 86.8       |