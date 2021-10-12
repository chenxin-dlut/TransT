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

<p>State-of-the-art comparison on TrackingNet, LaSOT, and GOT-10k. The best two results are shown in <b>red</b> and <b>blue</b> fonts:</p>
<img src="https://imgur.com/VhJ0lk8.png" style="width:100%;"/>

<p><b>AUC</b> scores of different attributes on the <b>LaSOT</b> dataset:</p>
<img src="https://imgur.com/0x7VUpc.png" style="width:100%;"/>



# Supervisely integration

### `placeholder`


# Demo

See how TransT works with Supervisely tools

## People


<table>
  <tr style="width: 100%">
    <th>
      <a data-key="sly-embeded-video-link" href="https://youtu.be/fNqMP-C7MA0" data-video-code="fNqMP-C7MA0">     <img src="https://imgur.com/EjbHbX0.jpg" alt="SLY_EMBEDED_VIDEO_LINK"  style="width:100%;"> </a>
    </th>
    <th>
      <a data-key="sly-embeded-video-link" href="https://youtu.be/Nv-45hoh4GQ" data-video-code="Nv-45hoh4GQ">     <img src="https://imgur.com/iPajKlb.jpg" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;"> </a> 
    </th>
  </tr>
  <tr>
    <td>
      <a data-key="sly-embeded-video-link" href="https://youtu.be/HLlgsf1ClXI" data-video-code="HLlgsf1ClXI">     <img src="https://imgur.com/6B3l6Dp.jpg" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;"> </a> 
    </td>
    <td>
      <a data-key="sly-embeded-video-link" href="https://youtu.be/ZkroK9_OH6Y" data-video-code="ZkroK9_OH6Y">     <img src="https://imgur.com/7i290SG.jpg" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;"> </a> 
    </td>
  </tr>
  <tr>
    <td>
      <a data-key="sly-embeded-video-link" href="https://youtu.be/zKUCtnQAhKU" data-video-code="zKUCtnQAhKU">     <img src="https://imgur.com/0lE1Kmx.jpg" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;"> </a>
    </td>
    <td>
      <a data-key="sly-embeded-video-link" href="https://youtu.be/jQW8jipqle8" data-video-code="jQW8jipqle8">     <img src="https://imgur.com/wbX5cl2.jpg" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;"> </a>
    </td>
  </tr>
</table>





