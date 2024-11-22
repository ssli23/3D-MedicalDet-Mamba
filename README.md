## 3D MedicalDet-Mamba: A hybrid Mamba-CNN Network for Medical Object Detection and Localization.



## Training
Configs can be trained with:
```bash
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr='127.0.0.1' --master_port=12345  scripts/train2.py
```
By default, we use 2 GPUs with total batch size as 4 for training.

## Evaluation
Model evaluation can be done as follows:
```bash
 python scripts/test.py --run --save_preds
```
