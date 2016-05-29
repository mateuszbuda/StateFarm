### Prerequisites ###

- [Get the Data](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data) and place it in `src/imgs`.
- Download a list of training images to `src/driver_imgs_list.csv`.
- Get the [VGG_16.caffemodel](https://gist.github.com/ksimonyan/211839e770f7b538e2d8) and save it as `src/VGG_16.caffemodel`.


### Dependencies ###

- python2
- python3
- [caffe](http://caffe.berkeleyvision.org)
- numpy
- [scikit-neuralnetwork](http://scikit-neuralnetwork.readthedocs.io/en/latest/guide_installation.html)
- scipy
- pandas


### Training and validation sets ###

Assuming you have downloaded and unzipped image to `src/imgs`, to create validation set run:

```
python2 split_train_val.py
```

It should create a new folder `scr/imgs/validation/c{0-9}` containing images and two files `src/trainset.csv` and `src/validationset.csv`.

Training and validation sets are split so that each driver appears only in one of them.


### Feature extraction ###

To extract training set features use `extract_features.py` script. If you have correctly setup `imgs/` folder structure and have saved model according to the instructions, default parameter values will do. Otherwise run `python extract_features.py --help` to show usage.

Notice: GPU is disabled by default. To use it run the script with 
`--gpu` flag.

For test and validation set you need to specify the folder containing images:

```
python2 extract_features.py --filesdir imgs/validation/ --out validationFeatures
python2 extract_features.py --filesdir imgs/test/ --out testFeatures
```


### Classification ###

Assumes you have extracted features in  `src/{train|test|validation}Features/`.

```
python3 classify.py
```


### Finetuning ###

Append path to `scr/imgs/validation/` and `scr/imgs/test/`.

```
python add_path.py validationset.csv <path>
python add_path.py trainset.csv <path>
```

And now you can run the finetuning. In case you don't have cuda capable GPU device, just skip `-gpu all` parameter.

```
caffe train -solver solver.prototxt -weights VGG_16.caffemodel -gpu all
```

### Testing ###

```
python caffe_test.py -m VGG_16_finetune_iter_1000.caffemodel -fd imgs/test -gpu all
```

Will output submission file for kaggle. 