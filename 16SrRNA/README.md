# MetaDR for OTU-based Prediction



### For maintenance or other codes/datasets requests, please contact:

##### Xingjian Chen (xingjchen3-c@my.cityu.edu.hk)



The framework can also be utilized for OTU-based prediciton, see the tutorial () for more details.



# Data preparation

### Inputs of our MetaDR

There are only 2 files that need to be prepared as the input for our MetaDR. And the files can be directly obtained by 16S amplicon analysis.

##### 1. Abundance table ("abundance.csv") : 

The abundance table should be a m*n matrix, where m is the number of OTUs and n is the number of the samples. The matrix should be like:

```markdown
|                   |   1     |     2   |   ...   |   200   |
| Methanobrevibacter|  0.830  |  0.034  |   ...   |  0.008  |
| Methanosphaera    |  0.126  |  0.144  |   ...   |  0.021  |
|   ...             |  ...    |  ...    |   ...   |  ...    |
| Acidobacteriaceae |  0.010  |  0.125  |   ...   |  0.072  |
```

Here the first row represents the sample id while the first column represents the taxon name for each OTU. The values in matrix represent the relative abundances (Ab).

##### 2. Label file (labels.txt) : 

The label file, i.e., the states or existence of a certain disease for each sample. The data structure should be like:

```markdown
|      T2D         | 
|      T2D         |
|      T2D         |
|      ...         | 
|     control      |
```

##### Note:

Since the OTU features are obtained by the 16S amplicon analysis pipeline, which is different from the shotgun metagenomic analysis pipeline utilized in our study, we cannot obtain the abundance profiles (or features) of unknown microbial organisms from 16S amplicon analysis. Therefore, we can only use EPCNN tool of MetaDR for disease prediction.

### The tutorial can be seen from .

In this tutorial, we evaluated the prediction module of MetaDR on a publicly available dataset to predict type 2 diabetes (T2D). The T2D dataset was a combination of two studies [1, 2] yielding a total of 223 patients with T2D and 223 healthy subjects. We also compared MetaDR with PopPhy-CNN [3], which is a state-of-the-art tool for OTU-based prediction. 

Surprisingly, our MetaDR achieved an average AUC of **0.72210** while PopPhy-CNN is simply **0.6810**, which means our pipeline also has promising generalization and better performance on OTU-level data. 

[1] J. Qin et al., “A metagenome-wide association study of gut microbiota in type 2 diabetes,” Nature, vol. 490, no. 7418, pp. 55–60, 2012.

[2] F. Karlsson et al., “Gut metagenome in european women with normal, impaired and diabetic glucose control,” Nature, vol. 498, no. 7452, pp. 99103, 2013.

[3] Reiman D, Metwally A A, Sun J, et al. PopPhy-CNN: a phylogenetic tree embedded architecture for convolutional neural networks to predict host phenotype from metagenomic data[J]. IEEE journal of biomedical and health informatics, 2020, 24(10): 2993-3001.

### Parameters

    -et, --et		Earlystopping                   
    -ts, --ts		The ratio of test data                      
    -rs, --rs		Repeat times
    -tp, --tp		Number of select features

