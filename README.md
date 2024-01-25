# PCNNPRS
PCNNPRS is a novel method to perform PRS calculations with a partially connected neural network, where the input of PCNN is a multi-PRS calculated on each chromosome with different clumping and thresholding (C+T) parameters.
Compared to traditional methods, PCNN shows that neural network could be flexible enough to predict the nonlinear relationships between PRS scores and continuous phenotype, emphasizing the practicality of our method for continuous trait predictions.

## Installation:
Download a local copy of PCNNPRS and unzip it to a target directory.
## Dependencies:
Python: Pytorch-cpu, Scikit-learn, Numpy, Pandas. 

R: bigsnpr, dplyr, data.table, Matrix, doParallel, recticulate, and all of their dependencies.

## An example to train PCNN model and make prediction on target data.

### Set up Python and R environment
We recommend you use conda to manage dependencies in different environments. If Conda hasn't been installed on your system yet, please visit https://www.anaconda.com/download for detailed installation information. Our analyses are currently conducted on CPU, and we are considering using GPU devices to train PyTorch model in the future. Refer to https://pytorch.org/get-started/locally/ for instructions on installing PyTorch, with steps varying based on your operating system.

First, create a conda environment and install the Python packages. (Please skip this step if you've already done this before.)
```
# The example is performed under Windows 10.
conda create -n yourenv python=3.9
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install scikit-learn pandas numpy
```
Then activate your conda environment and record the Python interpreter path under **yourenv**.

```
conda activate yourenv
# Use "which python" on Mac or Linux.
where python
# Record the path below.
D:/path/to/anaconda3/envs/yourenv/python.exe
# or /Users/anaconda3/envs/yourenv/bin/python
```

Finally, create a new Rscript on your local device and set the working directory to **PCNNPRS**. Don't forget to source the Rscript ```train_with_pc.r``` after installing the R packages.
```R
setwd('path/to/PCNNPRS')
install.packages(c('bigsnpr', 'dplyr', 'doParallel', 'reticulate', 'data.table'))
library(c('bigsnpr', 'dplyr', 'doParallel', 'reticulate', 'data.table'))
source('Train/train_with_pc.r')
```

### Load training dataset and summary statistics.
```R
obj.bigSNP <- read_file(
  input_file_dir = 'data/train',
  plink_dir = 'plink',
  bfile_out_dir = 'path/to/store_bfile'
)
G <- obj.bigSNP$genotypes
map <- obj.bigSNP$map[-3]
names(map) <- c('chr', 'rsid', 'pos', 'a1','a0')
#sumstats should contain 'beta' and 'pval' columns.
sumstats <- fread('path/to/sumstats.txt',header = TRUE)
names(sumstats) <- c('chr', 'rsid', 'pos', 'a1', 'a0', 'beta', 'p')
y_train <- fread('path/to/y_train.txt', header=FALSE)
```
* ```read_file()``` accepts two types of input: VCF file(.vcf) and PLINK files (.bed, .bim, .fam) without extension.
* ```bfile_out_dir ``` refers to the path where to store PLINK files and is only enabled when the input is in VCF format. Moreover, if you don't specify ```bfile_out_dir```, PLINK files will be generated under the same directory as your input file. 
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
* ```sumstats``` refer to summary statistcs.

| chr | rsid        | pos       | a1 | a0 | beta              | p                |
| --- | ----------- | --------- | -- | -- | ----------------- | ---------------- |
| 1   | rs13303037  | 868981    | C  | T  | 2.469677e-02      | 1.1751364e-02    |
| 1   | rs76723341  | 872952    | T  | C  | 2.563849e-02      | 1.8113471e-02    |
| ... | ...         | ...       | ...| ...| ...               | ...              |
| 22  | rs5770815    | 51128693  | T  | C  | 1.1097896e-03     | 9.147904e-01     |
| 22  | rs9616818    | 51135545  | T  | C  | -4.814950e-04     | 9.620711e-01     |
| 22  | rs9616941    | 51136646  | T  | C  | -4.2036828e-03    | 7.383258e-01     |



* Our model doesn't handle missing values. You can use ```snp_fastImpute()``` or ```snp_fastImputeSimple()``` in ```bigsnpr``` to impute missing values of genotyped variants before training.

### Compute multi-PRS and train with PCNN.
```R
train_with_pcnn(
  G,
  y_train,
  map,
  sumstats,
  material_out_dir = "path/to/store/material",
  python_dir = "path/to/anaconda3/envs/yourenv/python",
  max_evals = 5,
  Ncores = ncores
)
```
* ```material_out_dir``` is a directory used to store temporary files during training.
* ```python_dir``` refers to the python interpreter under **yourenv**.
* ```max_evals``` is the total number of rounds in a random search for hyperparameters tuning.
* ```Ncores``` represents the number of CPU cores used for parallel processing during training.


### Calculate reverse weight for variants and obtain ```deep_map.txt```.

| chr | rsid        | pos   | a1 | a0 | beta            |
| --- | ----------- | ----- | -- | -- | --------------- |
| 1   | rs13303037  | 868981| C  | T  | -4.450902e-04   |
| 1   | rs76723341  | 872952| T  | C  | -9.758492e-04   |
| ... | ...         | ...   | ...| ...| ...             |
| 22  | rs5770815    | 51128693 | T  | C  | -1.792037e-04   |
| 22  | rs9616818    | 51135545 | T  | C  | -6.468541e-05   |
| 22  | rs9616941    | 51136646 | T  | C  | -9.271383e-04   |

### Prediction.

```R
source('Predict/Predict_with_pc.r')
obj.bigSNP.test <- read_file(
  input_file_dir = 'data/test',
  plink_dir = 'plink',
  bfile_out_dir = 'path/to/store_test_bfile'
)
G.test <- obj.bigSNP.test$genotypes
map.test <- obj.bigSNP.test$map[-3]
names(map.test) <- c('chr', 'rsid', 'pos', 'a1', 'a0')
```

### Run ```predict_function()```.
```R
predict_result <- predict_function(
  G.test,
  map.test,
  material_out_dir = "path/to/material_out_dir/",
  ID_list_dir = "path/to/ID_list.txt",
  python_dir = "path/to/anaconda3/envs/yourenv/python",
  result_dir = "path/to/result/",
  Ncores = ncores
)

```
* ```ID_list_dir``` refers to the path of a text file, including 'sample.ID', which should contain a header with 'ID'.

| ID                                |
|-----------------------------------|
| 0002fe444c0b23f3adec06e6b00bc20c |
| 000de66839a885808b03f5f8f426211b |
| 001f47b7adc95858bc8caba825f370d4 |
| 001f676054f6155a47820c012630d891 |
| ...                              |
| 003abe243f194dc67d0173a920819a7e |
| 008f2ab0ab0c7ca06f0e821b616ed170 |
| 00c9a5fadab0b5ac73791decb7b6c9e0 |
| 00f0e515f89ff6249b822ccf28375a80 |

### Final result.
The final result is stored in ```result_dir/```.
| ID                                | pred                  |
|-----------------------------------|-----------------------|
| 0002fe444c0b23f3adec06e6b00bc20c | -7.90064036846161e-01 |
| 000de66839a885808b03f5f8f426211b |  4.19244110584259e-01 |
| 001f47b7adc95858bc8caba825f370d4 | -1.00259378552437e-01 |
| 001f676054f6155a47820c012630d891 |  6.68807446956635e-01 |
| ...                               | ...                   |
| 003abe243f194dc67d0173a920819a7e | -3.22202265262604e-01 |
| 008f2ab0ab0c7ca06f0e821b616ed170 | -4.02684330940247e-01 |
| 00c9a5fadab0b5ac73791decb7b6c9e0 | -5.51103949546814e-01 |
| 00f0e515f89ff6249b822ccf28375a80 |  3.70040535926819e-03 |


## Application on human height prediction:

### Source ```height_pred.R``` and load target dataset.
```R
source('Height Prediction/height_pred.r')
obj.bigSNP <- read_file(
  input_file_dir = 'data/height',
  plink_dir = 'plink',
  bfile_out_dir = 'path/to/store_height_bfile'
)

G <- obj.bigSNP$genotypes
map <- obj.bigSNP$map[-3]
names(map) <- c('chr', 'rsid', 'pos', 'a1', 'a0')
```
We set a coverage rate (matching ratio between variants and sumstats) threshold to ensure reliable prediction. When the coverage rate is below 0.4, the prediction process will stop. When the coverage rate is between 0.4 and 0.8, the predictions may not be particularly accurate and should be interpreted with caution. When coverage rate is above 0.8, more reliable results will be obtained.

Then run ```height_predict()``` to generate predictions.
```R
height_predict(
  G,
  map,
  gender_ID_list_dir = 'path/to/gender_ID_list.txt',
  match_by_pos = TRUE,
  material_dir = 'Height Prediction/height_material/',
  result_dir = 'path/to/height_result/',
  Ncores = ncores
)
```
* ```gender_ID_list_dir``` refers to the path of a text file that includes 'sample.ID' and 'gender'. The file should have a header with 'ID' and 'gender'.

| ID                                | gender |
| --------------------------------- | ------ |
| 000861b9d9b502b5939136c8a8e66d0b | 2      |
| 001ca04596d464db83a8d3747a3167be | 2      |
| 002d43d7e8a42d32b8bd668e3dbc40b2 | 2      |
| 003e3c8006e11f9bb2faf166a336f898 | 1      |
| 004e0559b7d68acf4fdd7fa44aa31778 | 1      |
| 007da44b9de4ad3931efdf647b938e02 | 2      |
| 009fdd5194b699da3be445d4714f5860 | 2      |
| 009fe3c2d4b3ca29698ed595690c6ccc | 1      |
| 0012c8fab85c015c975bed170505eb3a | 2      |
| 0014c58204d4c306a8d3019eafdf4a57 | 1      |

Where '1' represents male and '2' represents female.
* ```match_by_pos```: ```TRUE``` means matched by 'chromosome:position' and ```FALSE``` means matched by 'rsid'.
 
The outputs will be stored in ```result/dir``` as a text file, which looks like:
|                                 ID | gender | height |
| ---------------------------------- | ------ | ------ |
| 000861b9d9b502b5939136c8a8e66d0b |      2 |  159.1 |
| 001ca04596d464db83a8d3747a3167be |      2 |  167.8 |
| 002d43d7e8a42d32b8bd668e3dbc40b2 |      2 |  163.5 |
| 003e3c8006e11f9bb2faf166a336f898 |      1 |  180.2 |
| 004e0559b7d68acf4fdd7fa44aa31778 |      1 |  177.5 |
| 007da44b9de4ad3931efdf647b938e02 |      2 |  160.3 |
| 009fdd5194b699da3be445d4714f5860 |      2 |  161.2 |
| 009fe3c2d4b3ca29698ed595690c6ccc |      1 |  175.8 |
| 0012c8fab85c015c975bed170505eb3a |      2 |  163.6 |
| 0014c58204d4c306a8d3019eafdf4a57 |      1 |  177.5 |



