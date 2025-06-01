Table 13. Localization error (%) on the ImageNet validation. In the column of 'LOC error on GT class' ([41]), the ground truth class is used. In the 'testing' column, '1-crop' denotes testing on a center crop of 224 × 224 pixels, 'dense' denotes dense (fully convolutional) and multi-scale testing.

| LOC method         | LOC network   | testing             | LOC error on GT CLS   | classification network   | top-5 LOC error on predicted CLS   |
|--------------------|---------------|---------------------|-----------------------|--------------------------|------------------------------------|
| VGG's [41] RPN RPN | VGG-16        | 1-crop 1-crop dense | 33.1 [41] 13.3        |                          |                                    |
|                    | ResNet-101    |                     |                       |                          |                                    |
|                    | ResNet-101    |                     | 11.7                  |                          |                                    |
| RPN                | ResNet-101    | dense               |                       | ResNet-101               | 14.4                               |
| RPN+RCNN           | ResNet-101    | dense               |                       | ResNet-101               | 10.6                               |
| RPN+RCNN           | ensemble      | dense               |                       | ensemble                 | 8.9                                |