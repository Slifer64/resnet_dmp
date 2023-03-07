### Source code for the paper: 
## [<ins>**From RGB images to Dynamic Movement Primitives for planar tasks**</ins>](https://doi.org/10.48550/arxiv.2303.03204)

Tested with Ubuntu 20.04

# Virtual Env Setup


Install dependencies for virtual-env:
```bash
sudo apt-get install python3-tk python3-pip
sudo pip3 install virtualenv
```

Create virtual environment:
```bash
virtualenv ~/my_env --python=python3 --prompt='[my env] '
```

---


# Download the package and install dependencies

```bash
git clone https://github.com/Slifer64/resnet_dmp.git
cd resnet_dmp/src/stem_unveiling
source ~/my_env/bin/activate
pip install -e .
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

---

# Data Augmentation - Training - Evaluation 
```bash
cd resnet_dmp/src/stem_unveiling/scripts
source ~/my_env/bin/activate
./run.sh
```
This will augment the demo dataset, train the models, calculate the RMSE and finally vizualize the results on the test set.
This is done separately for simulation and real data.
You can enable/disable some of the above steps by setting accordingly the variables `SKIP_AUG, SKIP_TRAIN, SKIP_CALC_RMSE, SKIP_COMPARE` in `run.sh`.
To see details about the arguments of each python script run:
```bash
python3 <script_name>.py -h
```


---

# Reproducability of results

Notice that the results reported in the paper differ slightly from the results you will get by running the above code.
Nevertheless, the discrepancies are very small, leading pretty much to the **same results qualitatively and almost the same quantitatively**.

This was due to our having ommited to set `torch.backends.cudnn.deterministic = True`, which otherwise resulted in small differences despite using the same random seeds.
This issue is now fixed, and using the same random seed should produce the same results.



