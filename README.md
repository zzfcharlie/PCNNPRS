# PCNNPRS
PCNNPRS is a novel method to perform PRS calculations with a partially connected neural network, where the input of PCNN is a multi-PRS calculated on each chromosome with different clumping and thresholding (C+T) parameters.
Compared to traditional methods, PCNN shows that neural networks could be flexible enough to predict the nonlinear relationships between C+T scores and continuous phenotypes, emphasizing the practicality of our method for continuous trait predictions.

## Installation:
### From source:
Download a local copy of PCNNPRS:
```
git clone https://github.com/zzfcharlie/PCNNPRS.git
```
### Dependencies:
Python: Pytorch-cpu, Scikit-learn, Numpy, Pandas. 

R: bigsnpr, dplyr, data.table, Matrix, doParallel, reticulate, and all of their dependencies.

## An example of training PCNN model with example data:

### STEP 1: Set up the Python environment and install R packages.
We recommend you use conda to manage dependencies in different environments. If Conda hasn't been installed on your system yet, please visit https://www.anaconda.com/download for detailed installation information. Our analyses are currently conducted on CPU, and we are considering using GPU devices to train the PyTorch model in the future. Refer to https://pytorch.org/get-started/locally/ for instructions on installing PyTorch, with steps varying based on your operating system.

First, create and activate a conda environment and install the Python packages. 
```
# The example is performed under Windows 10.
conda create -n yourenv python=3.9
conda activate yourenv
# We found that 'Pytorch' installed from the conda source would have a conflict with R packages 'reticulate'. Thus, we use 'pip install' instead of 'conda install'.
pip3 install torch torchvision torchaudio
pip install scikit-learn pandas numpy
```
Please record your conda environment name. We use ```yourenv``` in our example.

Then, install R packages as follows:
```R
install.packages('bigsnpr')
install.packages('dplyr')
install.packages('doParallel')
install.packages('reticulate')
install.packages('data.table')
```

### STEP 2: Set R workspace into 'PCNNPRS' and source 'train_with_pc.r'.
```R
setwd('PCNNPRS')
source('Train/train_with_pc.r')
source('Predict/predict_with_pc.r')
```
### STEP 3: Load training dataset and summary statistics.
We simulated continuous outcomes on 22 chromosomes(842619 SNPs) for 500 subjects. Then, we divided the whole dataset into two parts: 400 for training and 100 for testing. These example data were stored in directory '/data/', where 'train.bed', 'train.bim', 'train.fam' were the training bfiles, 'test.bed', 'test.bim', 'test.fam' were the testing bfiles, 'sumstats.txt' was the summary statistics, 'y_train.txt' and 'y_test.txt' were the corresponding simulated training and testing phenotypes, 'ID_list.txt' was the individual ID information of testing samples.
```R
obj.bigSNP <- read_file(
  input_file_dir = 'data/train',
  #plink_dir = 'plink',
  #bfile_out_dir = 'material_out/'
)
G <- obj.bigSNP$genotypes
map <- obj.bigSNP$map[-3]
names(map) <- c('chr', 'rsid', 'pos', 'a1', 'a0')
#sumstats should contain 'beta' and 'pval' columns.
sumstats <- fread('data/sumstats.txt', header = TRUE)
names(sumstats) <- c('chr', 'rsid', 'pos', 'a1', 'a0', 'beta', 'p')
y_train <- fread('data/y_train.txt', header=FALSE)
```
* ```read_file()``` accepts two types of input: VCF file(.vcf) and PLINK files (.bed, .bim, .fam) without extension.
* ```bfile_out_dir``` refers to the path where to store PLINK files and is only enabled when the input is in VCF format. Moreover, if you don't specify ```bfile_out_dir```, PLINK files will be generated under the same directory as your input file. 
* ```G``` refers to genotyped variants coded in '0/1/2'.
* ```map``` refers to variants information.
  
| chr | rsid        | pos   | a1 | a0 |
| --- | ----------- | ----- | -- | -- |
| 1   | rs13303291  | 862093| T  | C  |
| 1   | rs4040604   | 863124| G  | T  |
| ... | ...         | ...   | ...| ...|
| 22  | rs5771014   | 51216731 | C  | T  |
| 22  | rs28729663  | 51219006 | A  | G  |
| 22  | rs9616978   | 51220319 | G  | C  |

where chr, pos, a0, and a1 represent chromosome, position, reference allele, and alternative allele respectively.
* *sumstats* refer to summary statistics.

| chr | rsid        | pos       | a1 | a0 | beta              | p                |
| --- | ----------- | --------- | -- | -- | ----------------- | ---------------- |
| 1   | rs13303037  | 868981    | C  | T  | 2.469677e-02      | 1.1751364e-02    |
| 1   | rs76723341  | 872952    | T  | C  | 2.563849e-02      | 1.8113471e-02    |
| ... | ...         | ...       | ...| ...| ...               | ...              |
| 22  | rs5770815    | 51128693  | T  | C  | 1.1097896e-03     | 9.147904e-01     |
| 22  | rs9616818    | 51135545  | T  | C  | -4.814950e-04     | 9.620711e-01     |
| 22  | rs9616941    | 51136646  | T  | C  | -4.2036828e-03    | 7.383258e-01     |

* Our model doesn't handle missing values. You can use ```snp_fastImpute()``` or ```snp_fastImputeSimple()``` in ```bigsnpr``` to impute missing values of genotyped variants before training.

### STEP 4: Compute multi-PRS and train with PCNN.
```R
train_with_pcnn(
    G,
    y_train,
    map,
    sumstats,
    env_name = "yourenv",
    material_out_dir = "material_out/",
    max_evals = 20,
    Ncores = 2,
    seed = 32)
```
* ```G``` refers to genotyped variants coded in '0/1/2'.
* ```y_train``` refers to the continuous phenotype of the training dataset.
* ```map``` refers to variants information.
* ```env_name``` refers to the name of the conda environment we've created before.
* ```material_out_dir``` is a directory used to store temporary files during training. If you don't specify this directory, temporary files will be generated under the default path(the directory of the current R workspace). ``` This directory contains the temporary files generated during the training process that will be used 
during the prediction process.```
* ```max_evals``` is the number of rounds in a random search for hyperparameters tuning.
* ```Ncores``` represents the number of CPU cores used for parallel processing.
* ```seed``` refers to a random seed used to initialize the random number generator.



### STEP 5: Calculate reverse weight for variants and obtain ```deep_map.txt```.
The 'deep_map.txt' is generated during training process(STEP 4) and stored in 'material_out_dir', which looks like

| chr | rsid        | pos   | a1 | a0 | beta            |
| --- | ----------- | ----- | -- | -- | --------------- |
| 1   | rs13303037  | 868981| C  | T  | -4.450902e-04   |
| 1   | rs76723341  | 872952| T  | C  | -9.758492e-04   |
| ... | ...         | ...   | ...| ...| ...             |
| 22  | rs5770815    | 51128693 | T  | C  | -1.792037e-04   |
| 22  | rs9616818    | 51135545 | T  | C  | -6.468541e-05   |
| 22  | rs9616941    | 51136646 | T  | C  | -9.271383e-04   |


## An example to make a prediction based on the trained-PCNN model:
### STEP 1: Source 'predict_with_pc.r' and load testing dataset.
The example testing bfiles(test.bim, test.bed, test.fam) are stored in '/data/'.
```R
source('Predict/Predict_with_pc.r')
obj.bigSNP.test <- read_file(
  input_file_dir = 'data/test',
  #plink_dir = 'plink',
  #bfile_out_dir = 'path/to/store_test_bfile'
)
G.test <- obj.bigSNP.test$genotypes
map.test <- obj.bigSNP.test$map[-3]
names(map.test) <- c('chr', 'rsid', 'pos', 'a1', 'a0')
```

### STEP 2: Make prediction.
```R
predict_result <- predict_function(
                           G.test,
                           map.test,
                           env_name = "yourenv",
                           material_save_dir = 'material_out/',
                           ID_list_dir = 'data/ID_list.txt',
                           result_dir = 'result/',
                           Ncores = 2)

```
* ```material_save_dir``` : the same directory as ```material_out_dir``` that we used in training process.
* ```ID_list_dir``` : the path of a text file, including 'sample.ID', which should contain a header with 'ID'. If NULL, we use ```obj.bigSNP.test$fam$family.ID``` as sample ID.
* ```result_dir``` : the directory to save the prediction result. If NULL, the output file will be saved in the default directory(the directory of the current R workspace).

### The prediction result will be stored in 'result/pred.txt'.
| ID                                | pred                  |
|-----------------------------------|-----------------------|
| 0002fe444c0b23f3adec06e6b00bc20c | -0.790                |
| 000de66839a885808b03f5f8f426211b |  0.419                |
| 001f47b7adc95858bc8caba825f370d4 | -0.100                |
| 001f676054f6155a47820c012630d891 |  0.669                |
| ...                               | ...                   |
| 003abe243f194dc67d0173a920819a7e | -0.322                |
| 008f2ab0ab0c7ca06f0e821b616ed170 | -0.403                |
| 00c9a5fadab0b5ac73791decb7b6c9e0 | -0.551                |
| 00f0e515f89ff6249b822ccf28375a80 |  0.004                |




