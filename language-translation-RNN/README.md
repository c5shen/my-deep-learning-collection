# Sequence-to-sequence language translation model
A sequence-to-sequence language translation model to translate French to 
English, with gradient recurrent units (GRUs), Attention module,
and Beam Search algorithm.

To see options for the seq2seq model with Attention module:
```
python seq2seq_attention.py -h
```

#### Example 1
To train on existing data (provided in `data/translation_data.bin`),
with beam search of width 4:
```
python seq2seq_attention.py train --beam_width 4 --save_model_prefix [name] 
```

#### Example 2
To get the average likelihood of a beam search of width 8:
```
python seq2seq_attention.py beam_avg_likelihood --load_model [path/to/model.pth] --beam_width 8
```

#### Unit Testing
To run tests on the current implementation:
```
python tests.py -m unittest tests
```

