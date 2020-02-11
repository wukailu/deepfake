# deepfake
 Kaggle-DFDC

# Progress

* 训练自己的模型
  * 与时序无关，单帧识别
    * Resnet backbone + paired trainning + lots of argumentations(刚开始超参数搜索)
    * 基于人脸识别模型来做识别（未开始）
  * 与时序相关(未开始)
    * 先单帧再LSTM
    * 人脸识别+LSTM
    * end to end


* 改进0.46的版本
  * TTA
    * Flip（已测试，对于本版本没啥效果）
  * Take more frames
    * 13, 43, 53 frames per video（大于20帧后，效果不显著）
  * 根据人脸个数，分类讨论（未开始）


* 预处理
  * split dataset
  * extract all faces(partial)
  * 