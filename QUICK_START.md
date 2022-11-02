## Getting ready.
Copy ```resnets.py``` in the main directory to the following location in the virtual environment:
```
venv/lib/python3.9/site-packages/pl_bolts/models/self_supervised/resnets.py
```

This fixes a versioning issue in one of the installed dependecies.

Dependencies can be installed as
```python -m pip install -r requirements.txt```

Make sure to put the preprocessed data in a directory called `./ecg_data_processed` in the base directory location.

## Example Usage
```
python3 custom_simclr_bolts.py --batch_size 4096 --epochs 2000 --precision 16 --trafos RandomResizedCrop TimeOut --datasets ./ecg_data_processed/zheng_fs100 ./ecg_data_processed/ribeiro_fs100 --log_dir=experiment_logs --gpus 0 --num_nodes 1 --resume
```
