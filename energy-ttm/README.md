Code for pre-training and inferencing of Energy-TTM model


> git clone 

conda create -n energypt python=3.10.13
conda activate energypt

cd energy-ttm
pip install -r requirements.txt

cd granite-tsfm
pip install -e .
cd ..


python test_out_cpu.py --config-file ./config/dataset-out.json 
python test_in_cpu.py --config-file ./config/dataset-in.json 

