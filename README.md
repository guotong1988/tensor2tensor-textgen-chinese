# USAGE

Step 1, put file `ptb2.py` to directory `data_generators` of tensor2tensor.

Step 2, data files named `train.txt` and `valid.txt` are put to the `tmp_dir` parameter of `t2t_trainer.py` of tensor2tensor.

Step 3, edit tensor2tensor's `all_problems.py` to registry `ptb2.py`.

Step 4, edit tensor2tensor's `flags.py`'s parameter `problem` to `languagemodel_chinese`.

Step 5, run `t2t_trainer.py` with other default parameters.
