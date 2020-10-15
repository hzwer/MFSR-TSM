# MFSR-TSM
Multi-Frame Super-Resolution based on Temporal Shift Module

We took part in the [Megvii 1st Open-Source Super Resolution Competition](https://studio.brainpp.com/competition) and got PSNR31.07 in Round 1. We need 10 hours for training our model in a V100 GPU card.

We use TSM modules to fuse information from multiple frames, and the backbone of the network is a simplified DenseNet.

# Dependencies
[megengine](https://megengine.org.cn/install/)
