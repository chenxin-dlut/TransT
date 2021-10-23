<div align="center" markdown>
  
# TransT object tracking (CVPR2021)

state-of-the art interactive object tracking (CVPR2021) integrated into Supervisely Videos Labeling tool

---

<p float="left">
  <img src="https://github.com/supervisely-ecosystem/trans-t/releases/download/v0.0.1/demo1.gif?raw=true" style="width:49%;"/>
  <img src="https://github.com/supervisely-ecosystem/trans-t/releases/download/v0.0.1/demo2.gif?raw=true" style="width:49%;"/>
</p>

<p align="center">
  <a href="#Original-work">Original work</a> •
    <a href="#How-to-run">How to run</a> •
  <a href="#How-to-use">How to use</a> •
    <a href="#Controls">Controls</a> •
  <a href="#Demo">Demo</a> 
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/trans-t/supervisely/serve)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/trans-t)
[![views](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/trans-t/supervisely/serve&counter=views&label=views)](https://supervise.ly)
[![used by teams](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/trans-t/supervisely/serve&counter=downloads&label=used%20by%20teams)](https://supervise.ly)
[![runs](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/trans-t/supervisely/serve&counter=runs&label=runs&123)](https://supervise.ly)

</div>




# Original work

Original work available by hyperlinks: [**paper (CVPR2021)**](https://arxiv.org/abs/2103.15436) and [**code**](https://github.com/chenxin-dlut/TransT).


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



# How to run


### ⚠️ Notice  
 * The application may already be launched by the instance administrator (**Enterprise**) or the Supervisely team (**Community**). If the app is not available in dropdown menu in Videos Labeling tool, please contact us. If the TransT responds slowly, please run additional application sessions in your team.
 * **Enterprise only**: You can share started application with all users on your instance using **share** button in front of running session. We recommend to run multiple sessions if large number of users are using TransT simultaneously.

---

1. Add [TransT object tracking (CVPR2021)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/trans-t/supervisely/serve) from Ecosystem

<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/trans-t/supervisely/serve" src="https://imgur.com/huFAU48.png" width="600px" style='padding-bottom: 20px'/>  


2. Run app from **Plugins & Apps** page:

<img src="https://imgur.com/LHbMGZ4.png" width="100%"/>


3. Run app on agent with `GPU`

<img src="https://imgur.com/VC6Kf5y.png" width="100%"/>  

4. The model has been successfully deployed

<img src="https://imgur.com/iXiK6xb.png" width="100%"/>  

5. Use in `Videos Annotator` 

<img src="https://imgur.com/VzWfVRK.png" width="100%"/>  



# How to use


<table>
  <tr style="width: 100%">
    <td>
      <a data-key="sly-embeded-video-link" href="https://youtu.be/EMvqTFu1ILE" data-video-code="EMvqTFu1ILE">     <img src="https://imgur.com/a19csV9.jpg" alt="SLY_EMBEDED_VIDEO_LINK"  style="width:100%;"> </a>
    </td>
    <td>
      <a data-key="sly-embeded-video-link" href="https://youtu.be/Xa6WeIgw_mI" data-video-code="Xa6WeIgw_mI">     <img src="https://imgur.com/n2P5qSL.jpg" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;"> </a> 
    </td>
  </tr>
</table>


# Controls

| Key                                                           | Description                               |
| ------------------------------------------------------------- | ------------------------------------------|
| <kbd>5</kbd>                                       | Rectangle Tool                |
| <kbd>Ctrl + Space</kbd>                                       | Complete Annotating Object                |
| <kbd>Space</kbd>                                              | Complete Annotating Figure                |
| <kbd>Shift + T</kbd>                                          | Track Selected     |
| <kbd>Shift + Enter</kbd>                                      | Play Segment     |




# Demo

We have prepared a videos and demonstrated how TransT works on the following domains:

* <a href="#People">People</a>  
* <a href="#Automobiles">Automobiles</a>  
* <a href="#Animals">Animals</a> 
* <a href="#Things">Things</a> 
* <a href="#Conveyor">Conveyor</a> 


## People


<table>
  <tr style="width: 100%">
    <td>
      <a data-key="sly-embeded-video-link" href="https://youtu.be/fNqMP-C7MA0" data-video-code="fNqMP-C7MA0">     <img src="https://imgur.com/EjbHbX0.jpg" alt="SLY_EMBEDED_VIDEO_LINK"  style="width:100%;"> </a>
    </td>
    <td>
      <a data-key="sly-embeded-video-link" href="https://youtu.be/Nv-45hoh4GQ" data-video-code="Nv-45hoh4GQ">     <img src="https://imgur.com/iPajKlb.jpg" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;"> </a> 
    </td>
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


## Automobiles


<table>
  <tr style="width: 100%">
    <td>
      <a data-key="sly-embeded-video-link" href="https://youtu.be/Om3EjoV_B9Q" data-video-code="Om3EjoV_B9Q">     <img src="https://imgur.com/vSEQTD8.jpg" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;"> </a> 
    </td>
    <td>
       <a data-key="sly-embeded-video-link" href="https://youtu.be/Ebth1bWiDrU" data-video-code="Ebth1bWiDrU">     <img src="https://imgur.com/QUlhUXN.jpg" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;"> </a> 
    </td>
  </tr>
  <tr style="width: 100%">
    <td>
      <a data-key="sly-embeded-video-link" href="https://youtu.be/rMBusxCQPAY" data-video-code="rMBusxCQPAY">     <img src="https://imgur.com/W0G6ZSE.jpg" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;"> </a> 
    </td>
    <td>
       <a data-key="sly-embeded-video-link" href="https://youtu.be/lk2BSSwv_G8" data-video-code="lk2BSSwv_G8">     <img src="https://imgur.com/ZCjA02a.jpg" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;"> </a> 
    </td>
  </tr>
  </tr>
  <tr style="width: 100%">
    <td>
      <a data-key="sly-embeded-video-link" href="https://youtu.be/_769hB_Nm9s" data-video-code="_769hB_Nm9s">     <img src="https://imgur.com/2FlTHlp.jpg" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;"> </a> 
    </td>
  </tr>
</table>


## Animals


<table>
  <tr style="width: 100%">
    <td>
      <a data-key="sly-embeded-video-link" href="https://youtu.be/Zmc-G3Wiy0k" data-video-code="Zmc-G3Wiy0k"> <img src="https://imgur.com/KRCk1PZ.jpg" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;"> </a> 
    </td>
    <td>
      <a data-key="sly-embeded-video-link" href="https://youtu.be/rD78jMwyCpo" data-video-code="rD78jMwyCpo"> <img src="https://imgur.com/rkoOwJs.jpg" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;"> </a> 
    </td>
  </tr>
  <tr style="width: 100%">
    <td>
      <a data-key="sly-embeded-video-link" href="https://youtu.be/X1I3E6O6BOk" data-video-code="X1I3E6O6BOk"> <img src="https://imgur.com/QLOeU72.jpg" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;"> </a> 
    </td>
  </tr>
</table>



## Things


<table>
  <tr style="width: 100%">
    <td>
      <a data-key="sly-embeded-video-link" href="https://youtu.be/3Dks8Vp9kSA" data-video-code="3Dks8Vp9kSA"> <img src="https://imgur.com/6rksiaj.jpg" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;"> </a> 
    </td>
    <td>
      <a data-key="sly-embeded-video-link" href="https://youtu.be/qFN5fNyq7EQ" data-video-code="qFN5fNyq7EQ"> <img src="https://imgur.com/AmFEs4f.jpg" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;"> </a> 
    </td>
  </tr>
  <tr style="width: 100%">
    <td>
      <a data-key="sly-embeded-video-link" href="https://youtu.be/ns5_tNuSmkw" data-video-code="ns5_tNuSmkw"> <img src="https://imgur.com/EU6dAek.jpg" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;"> </a> 
    </td>
    <td>
      <a data-key="sly-embeded-video-link" href="https://youtu.be/rRsex6BqRGs" data-video-code="rRsex6BqRGs"> <img src="https://imgur.com/GeDONRT.jpg" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;"> </a> 
    </td>
  </tr>
  <tr style="width: 100%">
    <td>
      <a data-key="sly-embeded-video-link" href="https://youtu.be/qhrsWzxA7Js" data-video-code="qhrsWzxA7Js"> <img src="https://imgur.com/aRnNDdZ.jpg" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;"> </a> 
    </td>
    <td>
      <a data-key="sly-embeded-video-link" href="https://youtu.be/u_b_AP99jLI" data-video-code="u_b_AP99jLI"> <img src="https://imgur.com/CSnYQPI.jpg" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;"> </a> 
    </td>
  </tr>
  <tr style="width: 100%">
    <td>
      <a data-key="sly-embeded-video-link" href="https://youtu.be/ocDRgXHcDuw" data-video-code="ocDRgXHcDuw"> <img src="https://imgur.com/D1PDsK8.jpg" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;"> </a> 
    </td>
  </tr>
</table>


## Conveyor


<table>
  <tr style="width: 100%">
    <td>
      <a data-key="sly-embeded-video-link" href="https://youtu.be/MQ6ZZ8_F870" data-video-code="MQ6ZZ8_F870"> <img src="https://imgur.com/2FdDTZ2.jpg" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;"> </a> 
    </td>
    <td>
      <a data-key="sly-embeded-video-link" href="https://youtu.be/FW-AL5Pc1Vc" data-video-code="FW-AL5Pc1Vc"> <img src="https://imgur.com/WgsINiu.jpg" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;"> </a> 
    </td>
  </tr>
  <tr style="width: 100%">
    <td>
      <a data-key="sly-embeded-video-link" href="https://youtu.be/sJGjmvCkJBU" data-video-code="sJGjmvCkJBU"> <img src="https://imgur.com/voOkvmv.jpg" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;"> </a> 
    </td>
    <td>
      <a data-key="sly-embeded-video-link" href="https://youtu.be/YJ8Xone3y7U" data-video-code="YJ8Xone3y7U"> <img src="https://imgur.com/gWxkRmw.jpg" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;"> </a> 
    </td>
  </tr>
</table>










