# Update pip
python3 -m pip install --upgrade pip
# Fix setup tools version (https://github.com/pytorch/pytorch/issues/69894)
python3 -m pip --no-cache-dir install --upgrade setuptools==59.5.0

# Install dependencies
python3 -m pip install -r requirements.txt

python3 -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
python3 -m pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
python3 -m pip install mmdet==2.28.2