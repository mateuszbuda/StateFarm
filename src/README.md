### Prerequisites ###

- [Get the Data](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data) and place it in `src/imgs`.
- Get the [VGG_16.caffemodel](https://gist.github.com/ksimonyan/211839e770f7b538e2d8) and save it as `src/VGG_16.caffemodel`.


### Dependencies ###

- [caffe](http://caffe.berkeleyvision.org)
- [scikit-learn develop](http://scikit-learn.org/stable/developers/contributing.html#retrieving-the-latest-code) (with MLPClassifier)
- scipy


### Feature extraction ###

To extract training set features use `extract_features.py` script.
If you have not changed imgs folder sctructure and saved model according to the instructions above, 
default parameter values will do.
Otherwise run `python extract_features.py --help` to show usage.

Notice: GPU is disabled by default. To use it run the script with `--gpu` flag.

For test set you need to specify the folder containing test images:
```
python extract_features.py --filesdir imgs/test/
```


### Classification ###

```
python classify.py
```

Assumes you have `features_c{0-10}.p` pickle files stored in `features/` and containing features generated with the procedure above.


