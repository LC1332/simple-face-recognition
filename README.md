# Simple Face Recognition Baseline Build by Cheng Li
# 李鲁鲁的Face Recognition Baseline

这个项目是 春日社区的SD学习 在参加 FaceChain比赛中的副产物。

目标非常简单，我们想看看在现在hugging face的管线下如何简易搭建一个性能不错的人脸识别

# How to use

目前CLIP + LDA的代码已经上线，之后有时间将再做一个end2end的版本

直接查看这个colab <a href="https://colab.research.google.com/github/LC1332/simple-face-recognition/blob/main/notebook/minimal_pipeline.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## 人脸正规化

```python
from MPCropAndNorm import MPCropAndNorm
import cv2
detector = MPCropAndNorm()
image = cv2.imread('example.png')
faces = detector.crop_and_norm(image)

cropped_face = faces[0]
cv2.imwrite(save_name, cropped_face)
```

crop后的face是这样的

<p align="center">
        <img src= "https://github.com/LC1332/simple-face-recognition/blob/main/figures/cropped_face.png" height="400">
</p>

对于人脸识别来说，crop更紧一些有可能是更好的。但是因为我们这个是和CeleHQ对齐的，还有一些人脸生成的任务相关，所以我们不做更紧致的crop了。


## 抽取CLIP特征

这里使用openai/clip-vit-base-patch16模型，我们后面会基于这个tuning一个版本

```python
from CLIPExtractor import CLIPExtractor

extractor = CLIPExtractor()
features = extractor.extract([save_name])
raw_clip_feature = features[0]
```

## project到人脸识别特征

```python
from huggingface_hub import hf_hub_download
import pickle

lda_model_path = hf_hub_download(repo_id="silk-road/simple-face-recognition", filename="lda_openai_clip_model.pkl")

with open(lda_model_path, 'rb') as f:
    lda_model = pickle.load(f)

def project_to_lda(feature):
    return lda_model.transform([feature])

face_recognition_feature = project_to_lda(raw_clip_feature)
```

这里face_recognition_feature直接就可以使用了，可以用consine similarity来计算两个人脸的相似度

## Training Set

这里使用了imdb face中的37283个identity大约70万张照片进行训练。

由于年代久远，我只找到了crop后的数据集

完整的imdb face有些对不齐了。

# TODO List

- [x] 在imdb-face上训练 CLIP-LDA
- [x] 完成发布LDA的测试代码
- [ ] 做个明星识别的gradio demo
- [ ] 重新训练一个专属的vit
- [ ] 发布专属vit的测试代码

## Motivation

一方面主要是想看看现在hugging face的管线下如何简易搭建一个性能不错的人脸识别

另一方面我们在做比赛的时候，想去测试生成后的人脸identity保有情况，却发现没有什么好用的开源工具

## Citation

如果要cite这个工作就cite我imdb-face那个论文吧

```
@article{wang2018devil,
	title={The Devil of Face Recognition is in the Noise},
	author={Wang, Fei and Chen, Liren and Li, Cheng and Huang, Shiyao and Chen, Yanjie and Qian, Chen and Loy, Chen Change},
	journal={arXiv preprint arXiv:1807.11649},
	year={2018}
}
```