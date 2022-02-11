# Characterizing and overcoming the greedy nature of learning in multi-modal deep neural networks 

We provide the source code for the **balanced multi-modal learning algorithm** proposed in the above paper, along with implementations for the derived metrics, **conditional utilization rate** and **conditional learning speed**. 

[[Paper]](https://arxiv.org/pdf/****.pdf)

## Dependencies: 
* Python 3.8 / gin-config / numpy / pandas / pytorch / scikit-learn / scipy / torchvision / skimage / PIL

## Workflow

We take the 3D object classification task using the [ModelNet40 dataset](http://maxwell.cs.umass.edu/mvcnn-data/) as an example. One can train the multi-modal DNN via the *balanced multi-modal learning algorithm* :

* `python3 train.py $RESULTS_DIR/random configs/training_guided.gin`

or its *random* version:

* `python3 train.py $RESULTS_DIR/random configs/training_random.gin`

To analysis multi-modal DNNs' *conditional utilization rate*, run the following two scripts consecutively:

* `python3 eval.py $RESULTS_DIR/random configs/recording.gin`
* `python3 eval.py $RESULTS_DIR/random configs/eval.gin`

## Citation
Please cite this work if you find the analysis or the proposed method useful for your research.

```
@inproceedings{wu2022greedymultimodal,
  title={Characterizing and overcoming the greedy nature of learning in multi-modal deep neural networks},
  author={Nan Wu, Stanis{\l}aw Jastrz\k{e}bski, Kyunghyun Cho, Krzysztof J. Geras},
  booktitle={ArXiv},
  year={2022}
}
```

