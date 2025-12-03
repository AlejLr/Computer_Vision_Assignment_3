"""
Script to test training on a small subset of the 20bn-jester dataset.
--> Run with (3 epochs):
python scripts\train_baseline.py --data_root data\raw\small-20bn-jester-v1 --train_csv data\raw\jester-v1-small-train.csv --val_csv data\raw\jester-v1-validation.csv --labels_csv data\raw\jester-v1-labels.csv --batch_size 32 --epochs 3

See scripts/train_baseline.py for more details.
--> Run with (3 epochs):
python scripts\train_temporal.py --data_root data\raw\small-20bn-jester-v1 --train_csv data\raw\jester-v1-small-train.csv --val_csv data\raw\jester-v1-validation.csv --labels_csv data\raw\jester-v1-labels.csv --batch_size 4 --epochs 3


"""