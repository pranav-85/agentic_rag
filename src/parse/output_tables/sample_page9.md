| training data                      | COCO train   | COCO train   | COCO trainval   | COCO trainval   |
|------------------------------------|--------------|--------------|-----------------|-----------------|
| test data                          | COCO val     | COCO val     | COCO test-dev   | COCO test-dev   |
| mAP                                | @.5          | @[.5, .95]   | @.5             | @[.5, .95]      |
| baseline Faster R-CNN (VGG-16)     | 41.5         | 21.2         |                 |                 |
| baseline Faster R-CNN (ResNet-101) | 48.4         | 27.2         |                 |                 |
| +box refinement                    | 49.9         | 29.9         |                 |                 |
| +context                           | 51.1         | 30.0         | 53.3            | 32.2            |
| +multi-scale testing               | 53.8         | 32.5         | 55.7            | 34.9            |
| ensemble                           |              |              | 59.0            | 37.4            |