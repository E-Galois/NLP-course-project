NLP course project
- 
download models before running:
1. download `BERT-wwm-ext, Chinese` (or other `pytorch` bert models) from https://github.com/ymcui/Chinese-BERT-wwm.
2. unzip and move files to directory `model`.

sentiment train & test:
- `python sentiment.py`

NER train & test:
- `python named_entity.py`

combine results and generate submit file:
1. please manually modify path in combine_results.py
2. run `python combine_results.py`
