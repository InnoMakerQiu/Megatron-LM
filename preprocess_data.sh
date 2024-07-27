python tools/preprocess_data.py \
       --input codeparrot_data.json \
	--output-prefix codeparrot \
       --vocab-file vocab.json \
       --merge-file merges.txt \
       --tokenizer-type GPT2BPETokenizer \
	--workers 2 \
       --append-eod
