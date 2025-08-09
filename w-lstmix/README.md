Code for pre-training and inferencing of w-lstmix model


## for testing the mode on in and out distribution datasets.

conda create -n energypt python=3.10.13
conda activate energypt

cd w-lstmix
pip install -r requirements.txt

python test_in_cpu.py
python test_out_cpu.py



