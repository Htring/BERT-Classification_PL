上篇文章:[【Pytorch Lightning】基于Pytorch Lighting和TextCNN的中文文本情感分析模型实现](https://blog.csdn.net/meiqi0538/article/details/123466819?spm=1001.2014.3001.5501) 介绍了基于textcnn模型效果。而基于Bert的效果有将如何呢？本文就介绍如何使用Bert构建一个中文文本情感分类模型。

## 技术选型
### 编程包
python 3.7
pytorch 1.10
pytorch_lightning 1.5
transformers 4.7.0

本文选取的预训练模型是：roberta-wwm-ext


### 模型选择
Bert 微调。
## 数据获取
测试的数据来自于开源项目：[bigboNed3/chinese_text_cnn](https://github.com/bigboNed3/chinese_text_cnn)

## 程序书写

Bert微调有两种方式，一种使用`BertForSequenceClassification`去实现，一种是在`BertModel`的基础上进行调整。本文选择前者。
代码具体介绍可参见我的博文：[【Pytorch Lightning】基于Pytorch Lighting和Bert的中文文本情感分析模型实现](https://blog.csdn.net/meiqi0538/article/details/123720263?spm=1001.2014.3001.5501)

### 测试结果

```text
             precision    recall  f1-score   support

           0       0.95      0.98      0.97      3144
           1       0.98      0.95      0.97      3156

    accuracy                           0.97      6300
   macro avg       0.97      0.97      0.97      6300
weighted avg       0.97      0.97      0.97      6300

--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'f1_score': 0.967301607131958, 'val_loss': 0.10145504027605057}
--------------------------------------------------------------------------------
```

测试结果，相比于textcnn模型效果高出了2个多百分点，这效果真是谁用谁知道。

## 联系我

1. 我的github：[https://github.com/Htring](https://github.com/Htring)
2. 我的csdn：[科皮子菊](https://piqiandong.blog.csdn.net/)
3. 我订阅号：AIAS编程有道
  ![AIAS编程有道](https://s2.loli.net/2022/05/05/DS37LjhBQz2xyUJ.png)
4. 知乎：[皮乾东](https://www.zhihu.com/people/piqiandong)