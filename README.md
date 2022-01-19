# StyleGAN_jittor
An implementation of StyleGAN in jittor

# 项目内容
使用jittor实现StyleGAN模型

## 训练
python train.py

## 测试
python generate.py

## 结果展示
可以大体感觉到，SourceB（列1）影响了生成图片的颜色，SourceA（行1）影响了生成图片的形状。

此外，重复生成的符号较多，可能也是训练不够充分导致的。

![Image text](https://github.com/Sunmeng1997/StyleGAN_jittor/blob/main/sample_mixing_9.png)

## 参考文献
https://github.com/xUhEngwAng/StyleGAN-jittor

https://github.com/tomguluson92/StyleGAN_PyTorch
