# Overview

_TransT is a very simple and efficient tracker, without online update module, using the same model and hyparameter for all test sets._


<p float="left">
  <img src="https://github.com/supervisely-ecosystem/trans-t/blob/main/supervisely/demo/demo1.gif?raw=true" style="width:49%;"/>
  <img src="https://github.com/supervisely-ecosystem/trans-t/blob/main/supervisely/demo/demo2.gif?raw=true" style="width:49%;"/>
</p>



# Original work

[**[Paper]**](https://arxiv.org/abs/2103.15436)
[**[Models(google)]**](https://drive.google.com/drive/folders/1GVQV1GoW-ttDJRRqaVAtLUtubtgLhWCE?usp=sharing)
[**[Models(baidu:iiau)]**](https://pan.baidu.com/s/1geI1cIv_AdLUd7qYKWIqzw)
[**[Raw Results]**](https://drive.google.com/file/d/1FSUh6NSzu8H2HzectIwCbDEKZo8ZKUro/view?usp=sharing)

We used as a basis the implementation of the TransT tracker from https://github.com/chenxin-dlut/TransT  


## TransT architecture

<img src="https://imgur.com/M5djthP.png" style="width:100%;"/>

TransT uses the input image to extract information about the appearance of the target and the surrounding area.  
It extracts feature maps from two patch images:

1. manually marked region of the image
2. the nearest area of the marked region

Next, feature maps are converted for use in the Feature Fusion Network, as shown in the dotted box in the Figure.  
* **ECAs — two ego-context extensions**, focus on **useful semantic context adaptively through the self-attention of multiple heads** to improve feature presentation.  
* **CFA — two cross function expansion modules**, obtain the performance maps of both their own and the other branch at the same time, and combine the two function maps through a multi-head cross-focus.  

Thus, two ECAs and two CFAs form a merge layer.  

After the Feature Fusion Network, the feature maps are fed into the predicted head, which calculates the coordinates of the object and classifies the pixels as background and foreground.



## TransT results


<img src="https://imgur.com/VhJ0lk8.png" style="width:100%;"/>
<img src="https://imgur.com/0x7VUpc.png" style="width:100%;"/>
