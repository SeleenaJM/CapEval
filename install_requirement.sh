#create a conda environment for tiger
conda create --name tiger  python=2.7
source activate tiger
conda install pytorch=0.4.1 cuda92 -c pytorch
conda install -c pytorch torchvision
conda install -c anaconda nltk
conda install -c anaconda scipy

# create an empty __init__.py file on the SCAN folder
echo "" >> SCAN/__init__.py