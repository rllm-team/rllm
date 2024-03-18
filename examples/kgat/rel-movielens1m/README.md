To run KGAT on rel-movielens, you need to build knowledge graph using the code in `build_kg`. For regression task, run `python build_reg.py`. For classification task, run `python build_reg.py` and `python build_clf.py`.
`build_reg.py` This file creates knowledge graph needed to run regression task, the output files are:
`train.txt, test.txt, val.txt`: User/Movie interactions of train/test/val set
`test_rating.txt, val_rating.txt`: User-rating-movie triplets of test/val set
`build_clf.py` creates extra knowledge graph needed to run classification task, the output files are:
`train_category.txt, test_category.txt, val_category.txt`: movie-category of train/test/val set

This is an adaption from KGAT. Instead of training on CF and KG loss as described in the paper, we train the model on MSE loss between predicted ratings and true ratings for both regression and classification task. By doing this, we can reduce the training time drastically.