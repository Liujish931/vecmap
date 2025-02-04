#!/bin/bash


### aaai2018
python3 map_embeddings.py --aaai2018 data/dictionaries/en-it.train.txt data/embeddings/original/en.emb.txt data/embeddings/original/it.emb.txt data/embeddings/en.txt data/embeddings/it.txt

python3 eval_translation.py data/embeddings/en.txt data/embeddings/it.txt -d data/dictionaries/en-it.test.txt

### acl2018
python3 map_embeddings.py --acl2018 --verbose data/embeddings/original/ELMo/en.txt data/embeddings/original/ELMo/fr.txt data/embeddings/mapped/ELMo/en-fr/en.txt data/embeddings/mapped/ELMo/en-fr/fr.txt
### acl2018 NGC
python3 map_embeddings.py --acl2018 --verbose --cuda /home/ubuntu/data/vecamp_data/data/embeddings/original/ELMo/en.txt /home/ubuntu/data/vecamp_data/data/embeddings/original/ELMo/fr.txt /home/ubuntu/data/vecamp_data/data/embeddings/mapped/ELMo/en-fr/en.txt /home/ubuntu/data/vecamp_data/data/embeddings/mapped/ELMo/en-fr/fr.txt
### acl 2018 no stochastic dictionary induction
python3 map_embeddings.py --acl2018 --verbose --cuda --stochastic_initial 1.0 --stochastic_interval 1 /home/ubuntu/data/vecamp_data/data/embeddings/original/ELMo/en.txt /home/ubuntu/data/vecamp_data/data/embeddings/original/ELMo/fr.txt /home/ubuntu/data/vecamp_data/data/embeddings/mapped/ELMo/en-fr/en.txt /home/ubuntu/data/vecamp_data/data/embeddings/mapped/ELMo/en-fr/fr.txt
