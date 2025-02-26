
::: {.cell .markdown}

## Track a Lightning experiment

In the previous experiment, we manually added a lot of MLFlow logging code to our Pytorch training script. However, [for many ML frameworks](https://mlflow.org/docs/latest/tracking/autolog.html#supported-libraries), MLFLow can automatically log relevant details. In this section, we will convert our Pytorch training script to a Pytorch Lightning script, for which automatic logging is supported in MLFlow, to see how this capability works.

After completing this section, you should be able to:

* understand what Pytorch Lightning is, and some benefits of using Lightning over "vanilla" Pytorch
* understand how to use autologging in MLFlow

:::

::: {.cell .markdown}

Switch to the `lightning` branch of the `gourmetgram-train` repository:

```bash
# run in a terminal inside jupyter container, from the "work/gourmetgram-train" directory
git switch lightning
```

The `train.py` script in this branch has already been modified for Pytorch Lightning. Open it, and note that we have:

* added imports
* left the data section as is. We could have wrapped it in a [`LightningDataModule`](https://lightning.ai/docs/pytorch/stable/data/datamodule.html), but for now, we don't need to.
* moved the training and validation functions, and the model definition, inside a class `LightningFood11Model` which inherits from Lightning's [`LightningModule`](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html).
* used build-in Lightning callbacks instead of hand-coding `ModelCheckpoint`, `EarlyStopping`, and `BackboneFinetuning`
* replaced the training loops with a Lightning `Trainer`. This also includes baked-in support for distributed training across GPUs - we set `devices="auto"` and let it figure out by itself how many GPUs are available, and how to use them.


To test this code, run

```bash
# run in a terminal inside jupyter container, from the "work/gourmetgram-train" directory
python3 train.py
```

Note that we are *not* tracking this experiment with MLFlow.

:::

::: {.cell .markdown}

### Autolog with MLFlow

Let's try adding MLFlow tracking code, using their `autolog` feature. Open `train.py`, and:

1. In the imports section, add

```python
import mlflow
import mlflow.pytorch
```

2. Just before `trainer.fit`, add:


```python
mlflow.set_experiment("food11-classifier")
mlflow.pytorch.autolog()
mlflow.start_run(log_system_metrics=True)
```

(note that we do not need to set tracking URI, because it is set in an environment variable).

3. At the end, add

```python
mlflow.end_run()
```

and then save the code.

Commit your changes to `git` (you can use Ctrl+C to stop the existing run):

```bash
# run in a terminal inside jupyter container, from the "work/gourmetgram-train" directory
git add train.py
git commit -m "Add MLFlow logging to Lightning version"
```

and note the commit hash. Then, test it with

```bash
# run in a terminal inside jupyter container, from the "work/gourmetgram-train" directory
python3 train.py
```

You will see this logged in MLFlow. But because the training script runs on each GPU with distributed training, it will be represented as two separate runs in MLFlow. The runs from "secondary" GPUs will have just system metrics, and the runs from the "primary" GPU will have model metrics as well.

Let's make sure that only the "primary" process logs to MLFlow. Open `train.py`, and

1. Change

```python
mlflow.set_experiment("food11-classifier")
mlflow.pytorch.autolog()
mlflow.start_run(log_system_metrics=True)
```

to

```python
if trainer.global_rank==0:
    mlflow.set_experiment("food11-classifier")
    mlflow.pytorch.autolog()
    mlflow.start_run(log_system_metrics=True)
```

2. Change

```python
mlflow.end_run()
```

to

```python
if trainer.global_rank==0:
    mlflow.end_run()
```

Commit your changes to `git` (you can stop the running job with Ctrl+C):

```bash
# run in a terminal inside jupyter container, from the "work/gourmetgram-train" directory
git add train.py
git commit -m "Make sure only rank 0 process logs to MLFlow"
```

and note the commit hash. Then, test it with

```bash
# run in a terminal inside jupyter container, from the "work/gourmetgram-train" directory
python3 train.py
```

You will see this logged in MLFlow as a single run. Note from the system metrics that both GPUs are utilized.

Look in the "Parameters" table and observe that all of these parameters are automatically logged - we did not make any call to `mlflow.log_params`. (We could have added `mlflow.log_params(config)` after `mlflow.start_run()` if we want, to log additional parameters that are not automatically logged - such as those related to data augmentation.)

Look in the "Metrics" table, and note that anything that appears in the Lightning progress bar during training, is also logged to MLFlow automatically. 

Let this training job run to completion. (On a node with two GPUs, it should take less than 10 minutes.)

:::

::: {.cell .markdown}

### Compare experiments

MLFlow also makes it easy to directly compare training runs.

Open `train.py`, change any logged parameter in the `config` dictionary - e.g. `lr` or `total_epochs` - then save the file, and re-run:

```bash
# run in a terminal inside jupyter container, from the "work/gourmetgram-train" directory
python3 train.py
```

(On a node with two GPUs, it should take less than 10 minutes.)

Then, in the "Runs" section of MLFlow, select this experiment run and your last one. Click "Compare".

Scroll to the "Parameters" section, which shows a table with the parameters of the two runs side-by-side. The parameter that you changed should be highlighted.

Then, scroll to the "Metrics" section, which shows a similar table with the metrics logged by both runs. Scroll within the table to see e.g. the test accuracy of each run.

In the "Artifacts" section, you can also see a side-by-side view of the model summary - in case these runs involved different models with different layers, you could see them here.

:::

