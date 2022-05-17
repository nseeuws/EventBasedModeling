# EventNet

## Example scripts

Artefact data (TUAR dataset):
```commandline
python tuar_training.py --data_path <hdf5 data set> --batch_size 16 --lr 1e-3 --n_epochs 100 --lambda_r 5 --duration_threshold 10
```

Seizure data (TUSZ dataset):
```commandline
python tusz_training.py
```
