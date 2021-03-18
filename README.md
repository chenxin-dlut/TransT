# TransT - Transformer Tracking [CVPR2021]
Official implementation of the TransT (CVPR2021) , including training code and trained models.

## Results

<table>
  <tr>
    <th>Model</th>
    <th>LaSOT<br>AUC (%)</th>
    <th>TrackingNet<br>AUC (%)</th>
    <th>GOT-10k<br>AO (%)</th>
    <th>Speed<br></th>
    <th>Params<br></th>
  </tr>
  <tr>
    <td>TransT-N2</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>TransT-N4</td>
    <td></td>
    <td></td>
    <td>72.3</td>
    <td>47fps</td>
    <td>23M</td>
  </tr>
  <tr>
    <td>TransT-N6</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>

## Installation
This document contains detailed instructions for installing the necessary dependencied for **TransT**. The instructions 
have been tested on Ubuntu 18.04 system.

#### Install dependencies
* Create and activate a conda environment 
    ```bash
    conda create -n transt python=3.7
    conda activate transt
    ```  
* Install PyTorch
    ```bash
    conda install -c pytorch pytorch=1.5 torchvision=0.6.1 cudatoolkit=10.2
    ```  

* Install other packages
    ```bash
    conda install matplotlib pandas tqdm
    pip install opencv-python tb-nightly visdom scikit-image tikzplotlib gdown
    conda install cython scipy
    pip install pycocotools jpeg4py
    pip install wget yacs
    pip install shapely==1.6.4.post2
    ```  
* Setup the environment                                                                                                 
Create the default environment setting files.

    ```bash
    # Change directory to <PATH_of_TransT>
    cd TransT
    
    # Environment settings for pytracking. Saved at pytracking/evaluation/local.py
    python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"
    
    # Environment settings for ltr. Saved at ltr/admin/local.py
    python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
    ```
You can modify these files to set the paths to datasets, results paths etc.
* Add the project path to environment variables  
Open ~/.bashrc, and add the following line to the end. Note to change <path_of_TransT> to your real path.
    ```
    export PYTHONPATH=<path_of_TransT>:$PYTHONPATH
    ```
* Download the pre-trained networks   
Download the network for [TransT](https://drive.google.com/file/d/1Pq0sK-9jmbLAVtgB9-dPDc2pipCxYdM5/view?usp=sharing)
and put it in the directory set by "network_path" in "pytracking/evaluation/local.py". By default, it is set to 
pytracking/networks.

## Quick Start
#### Traning
* Modify [local.py](ltr/admin/local.py) to set the paths to datasets, results paths etc.
* Runing the following commands to train the TransT. You can customize some parameters by modifying [transt.py](ltr/train_settings/transt/transt.py)
    ```bash
    conda activate transt
    cd TransT/ltr
    python run_training.py transt transt
    ```  

#### Evaluation
* We integrated [GOT-10k Python Toolkit](https://github.com/got-10k/toolkit) to eval on [GOT-10k](http://got-10k.aitestunion.com/), [OTB (2013/2015)](http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html), [VOT (2013~2018)](http://votchallenge.net), [DTB70](https://github.com/flyers/drone-tracking), [TColor128](http://www.dabi.temple.edu/~hbling/data/TColor-128/TColor-128.html), [NfS (30/240 fps)](http://ci2cv.net/nfs/index.html), [UAV (123/20L)](https://ivul.kaust.edu.sa/Pages/pub-benchmark-simulator-uav.aspx), [LaSOT](https://cis.temple.edu/lasot/) and [TrackingNet](https://tracking-net.org/) benchmarks. 
Please refer to [got10k_toolkit](/got10k_toolkit) for details.
For convenience, We provide some python files to test and eval on the corresponding benchmarks. For example [test_got.py](got10k_toolkit/toolkit/test_got.py) and [evaluate_got.py](got10k_toolkit/toolkit/evaluate_got.py). 

    You need to specify the path of the model and dataset in the these files.
    ```python
    net_path = '/path_to_model' #Absolute path of the model
    dataset_root= '/path_to_datasets' #Absolute path of the datasets
    ```  

    Then run the following commands.

    ```bash
    conda activate TransT
    cd TransT
    python got10k_toolkit/toolkit/test_got.py #test tracker
    python got10k_toolkit/toolkit/evaluate_got.py #eval tracker
    ```  

* We also integrated [PySOT](https://github.com/STVIR/pysot), You can use it to eval on [VOT2019](http://votchallenge.net). 
    
    You need to specify the path of the model and dataset in the [test.py](pysot_toolkit/test.py).
    ```python
    net_path = '/path_to_model' #Absolute path of the model
    dataset_root= '/path_to_datasets' #Absolute path of the datasets
    ```  
    Then run the following commands.
    ```bash
    conda activate TransT
    cd TransT
    python -u pysot_toolkit/test.py --dataset VOT2019 #test tracker #test tracker
    python pysot_toolkit/eval.py --tracker_path pysot_toolkit/results/ --dataset VOT2019 --num 1 #eval tracker
    ```  
* You can also use [pytracking](pytracking) to test and evaluate tracker. 
But we have not carefully tested it, the results might be slightly different with the two methods above due to the slight difference in implementation (pytracking saves results as integers, got-10k toolkit saves the results as decimals).

## Acknowledgement
This is a modified version of the python framework [PyTracking](https://github.com/visionml/pytracking) based on **Pytorch**, 
also borrowing from [PySOT](https://github.com/STVIR/pysot) and [GOT-10k Python Toolkit](https://github.com/got-10k/toolkit). 
We would like to thank their authors for providing great frameworks and toolkits.

## Contact
* Xin Chen (email:chenxin3131@mail.dlut.edu.cn)
