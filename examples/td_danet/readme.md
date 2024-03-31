### To use

```
python train.py -c .\config\xxxx.yaml -g number_x
```
`xxxx` is the name of the dataset being used, and `number_x` is the id of GPU.

like movie_cla, movie_reg, MSLR

### Data preprocessing

using `svm2pkl()` to transform train.txt, vali.txt, test.txt in .\data\dataset\MSRL-WEB10K\Fold1 into .csv files, using `csv2pkl()` to transform train.csv, valid.csv, test.csv in .\dataset\movie_cla\movies into .pkl files


### structure in ./data

link：https://pan.baidu.com/s/1JkcYgdjWJecT4zKq94Y9xA?pwd=h2um 
code：h2um 

Unzip and put the unzipped repo in ./data, you'll have structure stated below:

-data
--movie_cla
--movie_reg
--MSLR-WEB10K
--data_util.py
--dataset.py