# greedy_multimodal_learning

We provide a training script for the proposed balanced multi-modal algorithm:

* `python3 train.py $RESULTS_DIR/random configs/training_guided.gin`

and the training script for the random version alternative:

* `python3 train.py $RESULTS_DIR/random configs/training_random.gin`

To analysis the model's performance, especially its conditional utilization, we provide the two scripts:

* `python3 eval.py $RESULTS_DIR/random configs/recording.gin`
* `python3 eval.py $RESULTS_DIR/random configs/eval.gin`

