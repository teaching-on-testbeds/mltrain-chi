
::: {.cell .markdown}

## Track a Pytorch experiment

Now, we will use our MLFlow tracking server to track a Pytorch training job. After completing this section, you should be able to:

* understand what type of artifacts, parameters, and metrics may be logged to an experiment tracking service (MLFlow or otherwise)
* configure a Python script to connect to an MLFlow tracking server and associate with a particular experiment
* configure system metrics logging in MLFlow
* log hyperparametrics and metrics of a Pytorch training job to MLFlow
* log a trained Pytorch model as an artifact to MLFlow
* use MLFlow to compare experiments visually

:::


::: {.cell .markdown}

The premise of this example is as follows: You are working at a machine learning engineer at a small startup company called GourmetGram. They are developing an online photo sharing community focused on food. You have developed a convolutional neural network in Pytorch that automatically classifies photos of food into one of a set of categories: Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup, and Vegetable/Fruit. 

An original Pytorch training script is available at: [gourmetgram-train/train.py](https://github.com/teaching-on-testbeds/gourmetgram-train/blob/main/train.py). The model uses a MobileNetV2 base layer, adds a classification head on top, trains the classification head, and then fine-tunes the entire model, using the [Food-11 dataset](https://www.epfl.ch/labs/mmspg/downloads/food-image-datasets/).

:::


::: {.cell .markdown}

### Run a non-MLFlow training job


Open a terminal inside this environment ("File > New > New Terminal") and `cd` to the `work` directory. Then, clone the  [gourmetgram-train](https://github.com/teaching-on-testbeds/gourmetgram-train/) repository:

```bash
# run in a terminal inside jupyter container
cd ~/work
git clone https://github.com/teaching-on-testbeds/gourmetgram-train
```

In the `gourmetgram-train` directory, open `train.py`, and view it directly there.


Then, run `train.py`: 

```bash
# run in a terminal inside jupyter container
cd ~/work/gourmetgram-train
python3 train.py
```

(note that the location of the Food-11 dataset has been specified in an environment variable passed to the container.)

Don't let it finish (it would take a long time) - this is just to see how it works, and make sure it doesn't crash. Use Ctrl+C to stop it running after a few minutes.

:::

::: {.cell .markdown}

### Add MLFlow logging to Pytorch code


After working on this model for a while, though, you realize that you are not being very effective because it's difficult to track, compare, version, and reproduce all of the experiments that you run with small changes.  To address this, at the organization level, the ML Platform team at GourmetGram has set up a tracking server that all GourmetGram ML teams can use to track their experiments. Moving forward, your training scripts should log all the relevant details of each training run to MLFlow.

Switch to the `mlflow` branch of the `gourmetgram-train` repository:

```bash
# run in a terminal inside jupyter container, from the "work/gourmetgram-train" directory
git fetch -a
git switch mlflow
```

The `train.py` script in this branch has already been augmented with MLFlow tracking code. Run the following to see a comparison betweeen the original and the modified training script. 

```bash
# run in a terminal inside jupyter container, from the "work/gourmetgram-train" directory
git diff main..mlflow
```

(press `q` after you have finished reviewing this diff.)

The changes include:

**Add imports for MLFlow**:

```python
import mlflow
import mlflow.pytorch
```

MLFlow includes framework-specific modules for many machine learning frameworks, including [Pytorch](https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html), [scikit-learn](https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html), [Tensorflow](https://mlflow.org/docs/latest/python_api/mlflow.tensorflow.html), [HuggingFace/transformers](https://mlflow.org/docs/latest/python_api/mlflow.transformers.html), and many more. In this example, most of the functions we will use come from base `mlflow`, but we will use an `mlflow.pytorch`-specific function to save the Pytorch model.

**Configure MLFlow**:

The main configuration that is required for MLFlow tracking is to tell the MLFlow client where to send everything we are logging! By default, MLFlow assumes that you want to log to a local directory named `mlruns`. Since we want to log to a remote tracking server, you'll have to override this default.

One way to specify the location of the tracking server would be with a call to `set_tracking_uri`, e.g.

```python
mlflow.set_tracking_uri("http://A.B.C.D:8000/") 
```

where `A.B.C.D` is the IP address of your tracking server. However, we may prefer not to hard-code the address of the tracking server in our code (for example, because we may occasionally want the same code to log to different tracking servers). 

In these experiments, we will instead specify the location of the tracking server with the `MLFLOW_TRACKING_URI` environment variable, which we have already passed to the container. 

(A list of other environment variables that MLFLow uses is available in [its documentation](https://mlflow.org/docs/latest/python_api/mlflow.environment_variables.html). )

We also set the "experiment". In MLFlow, an "experiment" is a group of related "runs", e.g. different attempts to train the same type of model. If we don't specify any experiment, then MLFlow logs to a "default" experiment; but we will specify that runs of this code should be organized inside the "food11-classifier" experiment.

```python
mlflow.set_experiment("food11-classifier")
```

**Start a run**: 

In MLFlow, each time we train a model, we start a new run. Before we start training, we call

```python
mlflow.start_run()
```

or, we can put all the training inside a 

```python
with mlflow.start_run():
    # ... do stuff
```

block.  In this example, we actually start a run inside a 


```python
try: 
    mlflow.end_run() # end pre-existing run, if there was one
except:
    pass
finally:
    mlflow.start_run()
```

block, since we are going to interrupt training runs with Ctrl+C, and without "gracefully" ending the run, we may not be able to start a new run.

**Track system metrics**: 

Also, when we called `start_run`, we passed a `log_system_metrics=True` argument. This directs MLFlow to automatically start tracking and logging details of the host on which the experiment is running: CPU utilization and memory, GPU utilization and memory, etc.

Note that to automatically log GPU metrics, we must have installed `pyrsmi` (for AMD GPUs) or `pynvml` (for NVIDIA GPUs) - we installed these libraries inside the container image already. (But if we would build a new container image, we'd want to remember that.)

Besides for the details that are tracked automatically, we also decided to get the output of `rocm-smi` (for AMD GPUs) or `nvidia-smi` (for NVIDIA GPUs), and save the output as a text file in the tracking server. This type of logged item is called an artifact - unlike some of the other data that we track, which is more structured, an artifact can be any kind of file.

We used

```python
mlflow.log_text(gpu_info, "gpu-info.txt")
```

to save the contents of the `gpu_info` variable as a text file artifact named `gpu-info.txt`.

**Log hyperparameters**:

Of course, we will want to save all of the hyperparameters associated with our training run, so that we can go back later and identify optimal values. Since we have already saved all of our hyperparameters as a dictionary at the beginning, we can just call

```python
mlflow.log_params(config)
```

passing that entire dictionary. This practice of defining hyperparameters in one place (a dictionary, an external configuration file) rather than hard-coding them throughout the code, is less error-prone but also easier for tracking.

**Log metrics during training**: 

Finally, the thing we most want to track: the metrics of our model during training! We use `mlflow.log_metrics` inside each training run:

```python
mlflow.log_metrics(
{"epoch_time": epoch_time,
    "train_loss": train_loss,
    "train_accuracy": train_acc,
    "val_loss": val_loss,
    "val_accuracy": val_acc,
    "trainable_params": trainable_params,
    }, step=epoch)
```

to log the training and validation metrics per epoch. We also track the time per epoch (because we may want to compare runs on different hardware or different distributed training strategies) and the number of trainable parameters (so that we can sanity-check our fine tuning strategy).

**Log model checkpoints**:

During the second part of our fine-tuning, when we un-freeze the backbone/base layer, we log the same metrics. In this training loop, though, we additionally log a model checkpoint at the end of each epoch if the validation loss has improved:

```python
mlflow.pytorch.log_model(food11_model, "food11")
```

The model *and* many details about it will be saved as an artifact in MLFlow.

**Log test metrics**:

At the end of the training run, we also log the evaluation on the test set:

```python
mlflow.log_metrics(
    {"test_loss": test_loss,
    "test_accuracy": test_acc
    })
```

and finally, we finish our run with

```python
mlflow.end_run()
```

:::

::: {.cell .markdown}

### Run Pytorch code with MLFlow logging

To test this code, run

```bash
# run in a terminal inside jupyter container, from the "work/gourmetgram-train" directory
python3 train.py
```

(Note that we already passed the `MLFLOW_TRACKING_URI` and `FOOD11_DATA_DIR` to the container, so we do not need to specify this environment variable again when launching the training script.)

While this is running, in another tab in your browser, open the URL

```
http://A.B.C.D:8000/
```

where in place of `A.B.C.D`, substitute the floating IP address assigned to *your* instance. You will see the MLFlow browser-based interface. Now, in the list of experiments on the left side, you should see the "food11-classifier" experiment. Click on it, and make sure you see your run listed. (It will be assigned a random name, since we did not specify the run name.)

Click on your run to see an overview. Note that in the "Details" field of the "Source" table, the exact Git commit hash of the code we are running is logged, so we know exactly what version of our training script generated this run.

As the training script runs, you will see a "Parameters" table and a "Metrics" table on this page, populated with values logged from the experiment.

* Look at the "Parameters" table, and note that the hyperparameters in the `config` dictionary, which we logged with `log_params`, are all there.
* Look at the "Metrics" section, and note that (at least) the most recent value of each of the system metrics appear there. Once an epoch has passed, model metrics will also appear there.

Click on the "System metrics" tab for a visual display of the system metrics over time. In particular, look at the time series chart for the `gpu_0_utilization_percentage` metric, which logs the utilization of the first GPU over time. Wait until a few minutes of system metrics data has been logged. (You can use the "Refresh" button in the top right to update the display.)


Notice that the GPU utilization is low - the training script is not keeping the GPU busy! This is not good, but in a way it is good - because it suggest some potential for speeding up our training.

Let's see if there is something we can do. Open `train.py`, and change

```python
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
```

to 


```python
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=16)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=16)
```

and save this code. 

Now, our training script will use multiple subprocesses to prepare the data, hopefully feeding it to the GPU more efficiently and reducing GPU idle time.

Let's see if this helps! Make sure that at least one epoch has passed in the running training job. Then, use Ctrl+C to stop it. Commit your changes to `git`:

```bash
# run in a terminal inside jupyter container, from the "work/gourmetgram-train" directory
git config --global user.email "netID@nyu.edu" # substitue your own email
git config --global user.name "Your Name"  # substitute your own name
git add train.py
git commit -m "Increase number of data loader workers, to improve low GPU utilization"
```

(substituting your own email address and name.) Next, run 

```bash
# run in a terminal inside jupyter container, from the "work/gourmetgram-train" directory
git log -n 2
```

to see a log of recent changes tracked in version control, and their associated commit hash. You should see the last commit before your changes, and the commit corresponding to your changes.

Now, run the training script again with

```bash
# run in a terminal inside the jupyter container, from inside the "work/gourmetgram-train" directory
python3 train.py
```

In the MLFlow interface, find this new run, and open its overview. Note that the commit hash associated with this updated code is logged. You can also write a note to yourself, to remind yourself later what the objective behind this experiment was; click on the pencil icon next to "Description" and then put text in the input field, e.g.

> Checking if increasing num_workers helps bring up GPU utilization.

then, click "Save". Back on the "Experiments > food11-classifier" page in the MLFlow UI, click on the "Columns" drop-down menu, and check the "Description" field, so that it is included in this overview table of all runs.

Once a few epochs have passed, we can compare these training runs in the MLFlow interface. From the main "Experiments > food11-classifier" page in MLFlow, click on the "📈" icon near the top to see a chart comparing all the training runs in this experiment, across all metrics. 

(If you have any false starts/training runs to exclude, you can use the 👁️ icon next to each run to determine whether it is hidden or visible in the chart. This way, you can make the chart include only the runs you are interested in.)

Note the difference between these training runs in:

* the utilization of GPU 0 (logged as `gpu_0_utilization_percentage`, under system metrics)
* and the time per epoch (logged as `epoch_time`, under model metrics)

we should see that your system metrics logging has allowed us to **substantially** speed up training by realizing that the GPU utilization aws low, and taking steps to address it. At the end of the training run, you will save these two plot panels for your reference.

Once the training script enters the second fine-tuning phase, it will start to log models. From the "run" page, click on the "Artifacts" tab, and find the model, as well as additional details about the model which are logged automatically (e.g. Python library dependencies, size, creation time).

Let this training script run to completion (it may take up to 15 minutes), and note that the test scores are also logged in MLFlow. 

<!-- 

Full training run should take: 6 minutes on Liqid, 13 minutes on mi100

-->


:::


::: {.cell .markdown}

### Register a model

MLFlow also includes a model registry, with which we can manage versions of our models. 

From the "run" page, click on the "Artifacts" tab, and find the model. Then, click "Register model" and in the "Model" menu, "Create new model". Name the model `food11` and save.

Now, in the "Models" tab in MLFlow, you can see the latest version of your model, with its lineage (i.e. the specific run that generated it) associated with it. This allows us to version models, identify exactly the code, system settings, and hyperparameters that created the model, and manage different model versions in different parts of the lifecycle (e.g. staging, canary, production deployment).

:::