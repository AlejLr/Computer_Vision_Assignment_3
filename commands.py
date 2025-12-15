"""
Script to test training on a small subset of the 20bn-jester dataset.
--> Run with (3 epochs):
python scripts\train_baseline.py --data_root data\raw\small-20bn-jester-v1 --train_csv data\raw\jester-v1-small-train.csv --val_csv data\raw\jester-v1-validation.csv --labels_csv data\raw\jester-v1-labels.csv --batch_size 32 --epochs 3

See scripts/train_baseline.py for more details.
--> Run with (3 epochs):
python scripts\train_temporal.py --data_root data\raw\small-20bn-jester-v1 --train_csv data\raw\jester-v1-small-train.csv --val_csv data\raw\jester-v1-validation.csv --labels_csv data\raw\jester-v1-labels.csv --batch_size 4 --epochs 3


--> evaluate baseline model:
python scripts/eval_model.py --data_root data\raw\small-20bn-jester-v1 --val_csv data\raw\jester-v1-validation.csv --labels_csv data\raw\jester-v1-labels.csv --model_type baseline --ckpt_path experiments\baseline_model\model_best.pth --batch_size 32 --save_dir experiments\eval


--> evaluate temporal model:
python scripts/eval_model.py --data_root data\raw\small-20bn-jester-v1 --val_csv data\raw\jester-v1-validation.csv --labels_csv data\raw\jester-v1-labels.csv --model_type temporal --ckpt_path experiments\temporal_model\model_best.pth --batch_size 4 --num_frames 8 --save_dir experiments\eval

"""