# Characterizing and overcoming the greedy nature of learning in multi-modal deep neural networks 

We provide the source code for the **balanced multi-modal learning algorithm** proposed in the above paper, along with implementations for the derived metrics, **conditional utilization rate** and **conditional learning speed**. 

**Accepted by ICML 2022 ** [[Paper]](https://arxiv.org/abs/2202.05306.pdf)

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
@misc{wu2022characterizing,
      title={Characterizing and overcoming the greedy nature of learning in multi-modal deep neural networks}, 
      author={Nan Wu and Stanisław Jastrzębski and Kyunghyun Cho and Krzysztof J. Geras},
      year={2022},
      eprint={2202.05306},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

