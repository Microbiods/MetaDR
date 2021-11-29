# MetaDR: Metagenome-based Human Disease Prediction by Multi-information Fusion and Machine Learning Algorithms



### For maintenance or other codes/datasets requests, please contact:

##### Xingjian Chen (xingjchen3-c@my.cityu.edu.hk)



MetaDR is a pipeline that can integrate various information to predict human diseases. MetaDR consists of a predictor and an interpreter. The predictor can embed the taxonomic relationship into microbial features and ensembles the prediction results from multiple perspectives. The interpreter can extract and elucidate biological insights from different microbial contexts. 

MetaDR can provide reference biomarkers from the combination of both known and unknown microbial organisms for the metagenomic dataset as well as achieving competitive prediction performance for human diseases.



### Update (25, Nov, 2021):

The framework can also be utilized for OTU-based prediciton, please see here () for more details.



# Description

MetaDR consists of the following 2 modules:

- Ensemble Phylogenetic Convolutional Neural Network (EPCNN) for disease prediction
- Weighted Random Forest (WRF) for feature selection to sort out the important reference biomarkers 



# Dependencies

### Softwares:

MicroPro
https://github.com/zifanzhu/MicroPro

Mash v.2.0
https://github.com/marbl/Mash/releases

PhyLoT v2
https://phylot.biobyte.de/

### MetaDR requires Python 3 (>= 3.7.6) with the following packages:



Tensorflow  2.4.1 

By default, the codes are running on GPU, therefore, please set up your GPU environment according to the official guideline provided by: 

https://www.tensorflow.org/install/gpu.

Basically, you need to install the corresponding GPU driver, CUDA 11.0 and cuDNN 8.0 according to the tested build configuration:

https://tensorflow.google.cn/install/source_windows?hl=en#gpu

Please uninstall the previous Keras package before installing Tensorflow!
(Simply execute "pip uninstall keras")

scikit-learn 0.24.1

xlrd 1.2.0

ete3 3.1.2

pandas-1.3.2

numpy-1.21.2

#### The necessary python packages can be installed with the following command:

```sh
$ pip install -r requirements.txt
```



##### Note: Please remember to uninstall Keras before installing Tensorflow  2.4.1, since there may be package conflicts.



# Data preparation

#### Note:

1. The known and unknown features are obtained by using MicroPro with default parameters. 

2. For details of MicroPro, please refer to the document on https://github.com/zifanzhu/MicroPro.

3. Our pipeline utilizes the MicrobialPip part of MicroPro and in terms of the output of MicroPro, MicrobialPip will provide an abundance table for all the microbes.

4. The pipeline takes about 15 min and all the abundance files can be found in the 'res/' folder if everything is installed properly.

### Outputs of MicroPro

The known and unknown abundance tables are stored in folder `res/`. Each of them has a 'csv' version which can be easily opened and edited. 

Every table contains a sample-by-organism matrix with each entry representing a known/unknown organism's relative abundance in a sample. Note that a microbe is output in the abundance table only if it appears in at least one sample. 

### Inputs of our MetaDR

The output of MicroPro can be direct as the input of our pipeline. Optionally, the users can also prepare thier data in the following formats based on other analysis pipelines .

There are 4 files that need to be prepared as the input for our pipeline, assume the name of the example set is 'Karlsson_T2D', then the file names should be  'Karlsson_T2D_known'.csv,
'Karlsson_T2D_unknown'.csv, 'Karlsson_T2D_y'.csv, and Unknown_name.xlsx'. Where the first two files are the abundance tables of known and unknown features. 'Karlsson_T2D_y'.csv is the label file for each patient. Unknown_name.xlsx' includes the genus-level assignments for each MAG.

##### 1. Known abundance table (with the file name of 'Karlsson_T2D_known'.csv) : 

In MicroPro, the taxonomy annotations and abundance table for known features can be directly obtained by using Centrifuge. So the abundance table for known features should be a m*n matrix, where m is the number of samples and n is the number of the known features. The matrix should be like:

```markdown
|           |   195   |   197   |   ...   |   287   |
| ERR260139 |  0.830  |  0.034  |   ...   |  0.008  |
| ERR260140 |  0.126  |  0.144  |   ...   |  0.021  |
|   ...     |  ...    |  ...    |   ...   |  ...    |
| ERR260144 |  0.010  |  0.125  |   ...   |  0.072  |
```

Here the first column represents the sample id while the first row represents the taxon id in the NCBI database (i.e., the annotions of known features). The values in matrix represent the relative abundances (Ab) of the known organisms.

##### 2. Unknown abundance table (with the file name of 'Karlsson_T2D_unknown'.csv) : 

In MicroPro, the abundance table for known features can be directly obtained by using Megahit and MetaBAT2. For unknown features, we follow MicroPro to classify the microbes into the genus level, and the obtained taxonomy annotations can be used to generate the phylogenetic tree for the unknown features. The abundance table for unknown features should be a m*p matrix, where m is the number of samples and p is the number of the unknown features. The matrix should be like:

```markdown
|           |   Bin1  |   Bin2  |   ...   |   Bin3  |
| ERR260139 |  0.430  |  0.134  |   ...   |  0.237  |
| ERR260140 |  0.071  |  0.157  |   ...   |  0.211  |
|   ...     |  ...    |  ...    |   ...   |  ...    |
| ERR260144 |  0.023  |  0.363  |   ...   |  0.001  |
```

Here the first column represents the same sample id while the first row represents the bin id of assembled contigs. The value in unknown features represents the relative abundance (Ab) of the unknown organisms.

##### 3. Label file (with the file name of 'Karlsson_T2D_y'.csv) : 

The label file, i.e., the states or existence of a certain disease for each sample.The label file should be a m*1 matrix, where m is the number of samples. The matrix should be like:

```markdown
|           | study_condition  | 
| ERR260139 |      T2D         |
| ERR260140 |      T2D         |
|   ...     |      ...         | 
| ERR260144 |     control      |
```

Here the first column represents the same sample id while the second column represents the  the states or existence of a certain disease for each sample. 

##### 4. The taxa annotions for unknown features (with the file name of 'Unknown_name'.xlsx) : 

In MicroPro, the file can be directly obtained by using Mash and Centrifuge. The matrix should be like:

```markdown
| Bin | Assignment (genus) | Taxa_id  | 
|  10 |      Roseburia     |   841    | 
| 137 |   Parabacteroides  |  375288  | 
| ... |        ...         |   ...    | 
| 158 |     Scardovia      |  196081  | 
```

Here the first column represents the bin id of assembled contigs,  the second column represents the distributed gene for each bin, and the third column represents the taxon id for each assigned gene in the NCBI database.



# Running WRF

Since WRF is based on RF which only accepts the dimensional vector, therefore we just need the features and the labels here without the phylogenetic trees. We offer two features for ERF which are select the important biomarkers as well as evaluation.


- Evaluation


```sh
python3 WRF_EV.py --fn Karlsson_T2D --rs 2 --ts 0.3
```

Where 'WRF_EV' is using WRF for evaluation.  '--rs' means the repeat times using different seeds. And '--ts' means the ratio of test data. The output will be saved in 'Karlsson_T2DWRF_ev.txt'.

- Feature selection (Biomarkers)

```sh
python3 WRF_FS.py --fn Karlsson_T2D --rs 2 --tp 30
```

Where 'WRF_FS' is using WRF for feature selection.  '--rs' means the repeat times using different seeds. And '--tp' means the numbers of selected features. The output will be saved in 'Karlsson_T2DWRF_fs.txt'.



# Running EPCNN


- Phylogenetic-sorting

```sh
python3 EPCNN_PS.py --fn Karlsson_T2D
```

Where 'EPCNN_PS' is using EPCNN for phylogenetic-sorting.  '--fn' means datasets.The outputs are four files which are: postorder phylogenetic-sorting based on known features, level phylogenetic-sorting based on known features, postorder phylogenetic-sortingbased on unknown features, and level phylogenetic-sorting based on unknown features. 

- Prediction

```sh
python3 EPCNN_EV.py --fn Karlsson_T2D --et t --ts 0.3 --rs 2
```

Where 'EPCNN_EV' is using EPCNN for prediction.  '--fn' means datasets. '--et' means types of earlystopping where 't' is to train on training data and other situation is to train on separate validaiton set. '--rs' means the repeat times using different seeds. And '--ts' means the ratio of test data.



### Options

    -fn, --fn		Dataset                  
    -et, --et		Earlystopping                   
    -ts, --ts		The ratio of test data                      
    -rs, --rs		Repeat times
    -tp, --tp		Number of select features
