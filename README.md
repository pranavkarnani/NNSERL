# Steps to run

1. Clone the repository
2. Run this bash command to download the data: bash NNSERL/get_transfer_data.sh
3. Install the requirements: pip3 install -r NNSERL/requirements.txt
4. Move into the directory: cd NNSERL
5. Clone the transformers repository: https://github.com/huggingface/transformers.git
6. Replace the modeling_bert.py file: cp ./modeling_bert.py ./transformers/src/transformers/models/bert/modeling_bert.py
7. Install the transformers package pip3 install -e .
7. Run the file: python3 sts_nnclr.py


To change any hyperparameters, please make changes to the config dictionary in the sts_nnclr.py
