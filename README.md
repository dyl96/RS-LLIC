## RS-LLIC: A Lightweight Learned Image Compression Model with Knowledge Distillation for Onboard Remote Sensing
(Currently under review at IEEE Transactions on Geoscience and Remote Sensing)

#### Required environments:
* Linux
* Python 3.8+
* PyTorch 1.10+
* CUDA 11.x
* CompressAI [https://github.com/InterDigitalInc/CompressAI]


#### 🚀 Training
Use the following command to train RS-LLIC:

```bash
python train_distillation.py \
    -d data_path \
    --cuda \
    --N 128 \
    --M 320 \
    -lr 1e-4 \
    --lambda 0.05 \
    --epochs 600 \
    --lr_epoch 400 550 \
    --save_path save path \
    --save \
    --checkpoint teacher checkpoint
```


#### 🧪 Evaluation

Use the following command to evaluate a trained model:

```bash
python eval.py \
    --N 128 \
    --M 320 \
    --checkpoint checkpoint_path \
    --data test_data_path \
    --cuda
```





