# Multi-Label Classification using SGM and Bert

## Dataset
Download [data](https://nam11.safelinks.protection.outlook.com/?url=http%3A%2F%2Fkdd.ics.uci.edu%2Fdatabases%2Freuters21578%2Freuters21578.html&data=04%7C01%7Ckzhang8%40wpi.edu%7C89b9f1ac965a4578d84408da0434765f%7C589c76f5ca1541f9884b55ec15a0672a%7C0%7C0%7C637826920638497504%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C3000&sdata=frJjN8WQsaVxUrhalXoponDY8YuvSxSqeiqPSMbvg7E%3D&reserved=0) here.

## SGM Usage
### Pre-requirements:
* Pythorch version >= 1.0.0
* Python version >=3.0

### Loading with checkpoints
Using the .pt file we provide to ./checkpoints folder to indicate our trained parameters to model.

### Train
Please use the following command to train the model.
'''python3 train.py -gpus gpu_id -config model_config -log save_path -restore "" '''
If you don't have GPUs in your device, please add 'torch.device('cuda' if torch.cuda.is_available() else 'cpu')' in train.py.

### Test
Please use the following command to test.
'''python3 predict.py -gpus gpu_id -data save_data_path -batch_size batch_size -log log_path'''

The code follows the work of [SGM](https://github.com/lancopku/SGM)

## Bert Usage
#### Pre-requirements:
* Pytorch version >= 1.0.0
* Transformers
* tqdm

Details can be seen in the jupyter notebook, we provide a trainable version without using GPU, if you would like use GPUs, indicate it the code will be ok.

## Results
![results](https://github.com/MatthewKKai/matthewkkai.github.io/blob/main/results.JPG)
