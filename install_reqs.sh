git config --global user.name "AWS VM"
git config --global user.email "william.a@alum.mit.edu"
conda install --yes --file requirements.txt
conda install matplotlib
conda install pmdarima
pip install git+https://github.com/amazon-science/chronos-forecasting.git
