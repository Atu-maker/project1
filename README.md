Please download the dataset, related models, and files from this link: 
Link: https://pan.baidu.com/s/1QtaB3PhqJfW1lhz3HSlbAQ  
Extract code: tttt 

You need to download the following folder and model files from this link:
--eval
--train
--test_b
--result
--vgg_model
--Swingv2
-model2.pth

The final structure of the project is as follows::

--eval
  -hzImage
  -lbImage
--train
  -hzImage
  -lbImage
--test_b
--result
--vgg_model
--Swingv2
--pytorch_msssim
--model2.pth
--Block.py
--model.py
--mymodel.py
--test.py
--train.py


#Running instructions: 
#If you want to retrain and reproduce, run the following command: 
Training Run: python train.py 
#If you want to directly use our trained model (model2. pth) to infer and reproduce the results: 
Inference run: python test.py
