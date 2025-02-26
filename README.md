In this tutorial, we explore some of the infrastructure and platform requirements for large model training, and to support the training of many models by many teams. We focus specifically on 

* experiment tracking (using [MLFlow](https://mlflow.org/))
* and scheduling training jobs on a GPU cluster (using [Ray](https://www.ray.io/))

Follow along at [Model training infrastructure and platforms](https://teaching-on-testbeds.github.io/mltrain-chi/).

Note: this tutorial requires advance reservation of specific hardware! You will need a node with 2 GPUs suitable for model training. You should reserve a 3-hour block for the MLFlow section and a 3-hour block for the Ray section. (They are designed to run independently.)

You can use either:

* a `gpu_mi100` at CHI@TACC (but, make sure the one you select has 2 GPUs), or
* a `compute_liqid` at CHI@TACC (again, make sure the one you select has 2 GPUs)

---

This material is based upon work supported by the National Science Foundation under Grant No. 2230079.
