
::: {.cell .markdown}

## Submit jobs to the Ray cluster

Now that we have a Ray cluster running, we can learn how to use it! After you finish this section, 

* you should understand how to specify the resource requirements and runtime environment for a job, and submit it to Ray
* and you should be able to modify a Pytorch Lightning script to use Ray Train, including its checkpointing, fault tolerance, and distributed training capabilities, and to use Ray Tune for hyperparameter optimization.


:::


::: {.cell .markdown}

### Submit a job with no modifications

To start, let's see how we can submit a training job to our Ray cluster, without modifying the code of our training job at all.

The premise of this example is as follows: You are working at a machine learning engineer at a small startup company called GourmetGram. They are developing an online photo sharing community focused on food. You have developed a convolutional neural network in Pytorch that automatically classifies photos of food into one of a set of categories: Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup, and Vegetable/Fruit. 

An original Pytorch Lightning training script is available at: [gourmetgram-train/train.py](https://github.com/teaching-on-testbeds/gourmetgram-train/blob/lightning/train.py). The model uses a MobileNetV2 base layer, adds a classification head on top, trains the classification head, and then fine-tunes the entire model, using the [Food-11 dataset](https://www.epfl.ch/labs/mmspg/downloads/food-image-datasets/).


Open a terminal inside this Jupyter environment ("File > New > New Terminal") and `cd` to the `work` directory. Then, clone the `lightning` branch of the [gourmetgram-train](https://github.com/teaching-on-testbeds/gourmetgram-train/) repository:

```bash
# run in a terminal inside jupyter container
cd ~/work
git clone https://github.com/teaching-on-testbeds/gourmetgram-train -b lightning
```

In the `gourmetgram-train` directory, open `train.py`, and view it directly there.


To run it on a worker node, though, we must give Ray some instructions about how to set up the runtime environment on the worker nodes. Two files necessary for this, `requirements.txt` and `runtime.json`, are inside the "work" directory:

* We assume that the worker nodes already have the Food-11 dataset at `/mnt/Food-11`, since we attached our data volume to those containers. So we don't have to worry about getting the data to the worker node in this case. We will have to make sure that the environment variable `FOOD11_DATA_DIR` is set, so that the training script can find the data. (In general, we will need to make sure that either worker nodes have access to the data, or they download it at the beginning of the training job.)
* We need to make sure that the worker nodes have the Python packages necessary to run our script. We'll put the list of packages in `requirements.txt`.
* And, we need to direct Ray to run this on a GPU node.

In `runtime.json`:

```json
{
    "pip": "requirements.txt",
    "env_vars": {
        "FOOD11_DATA_DIR": "/mnt/Food-11"
    }
}
```

we specify that when setting up a worker node to run our job, Ray should:

* install the Python packages listed in `requirements.txt`
* and set the `FOOD11_DATA_DIR` directory.

With this in hand, we can submit our job! In a terminal inside the Jupyter environment, run

```bash
# runs on jupyter container inside node-mltrain, from inside the "work" directory
ray job submit --runtime-env runtime.json --entrypoint-num-gpus 1 --entrypoint-num-cpus 8 --verbose  --working-dir .  -- python gourmetgram-train/train.py 
```

where we pass 

* the runtime environment specification, 
* the number of GPUs and CPUs our job requires, 
* we specify that we would like to see verbose output, 
* that the current working directory should be packaged up and shipped to the worker nodes,
* and that the command to run is: `python gourmetgram-train/train.py`.

While it is running, click on the "Overview", "Cluster", and "Jobs" tabs in the Ray dashboard.

* Initially, the job will be a in PENDING state, as the runtime environment is set up. This is slow the first time (because of downloading the Python packages), but faster in subsequent runs because the packages are cached on the workers.
* Then, the job will be in RUNNING state. Eventually, it should go to SUCCEEDED.
* You will see the job's requested GPU and CPU resource in the "Resource Status" section of the "Overview" page, which shows the cumulative resource requests of all jobs running on the cluster.
* As the job runs, if you are using NVIDIA GPUs you'll see one of the worker nodes has higher GPU utilization, in the "Cluster" tab. (Ray does not support this visualization for AMD GPUs, so if a node has an AMD GPU it will show "NA" in this field, even though the GPU is used by the worker to execute jobs.)
* You can click on the job and, in the "Logs", see the output of the job.

Let the training job finish, and get to SUCCEEDED state. (This may take up to 10-15 minutes.)

:::


::: {.cell .markdown}

### Submit an infeasible job

Next, let's see what happens if we submit a job for which there is no node that satisfies the resource requirements. Run

```bash
# runs on jupyter container inside node-mltrain, from inside the "work" directory
ray job submit --runtime-env runtime.json --entrypoint-num-gpus 2 --entrypoint-num-cpus 8 --verbose  --working-dir .  -- python gourmetgram-train/train.py 
```

noting that we have no node with 2 GPUs - only two nodes, each with 1 GPU. 

In the Ray dashboard "Overview" page, observe that this request is listed in "Demands" in the "Resource Status" section.

The job will be stuck in PENDING state until we add a node with 2 GPUs to the cluster, at which time it can be scheduled.

In a commercial cloud, when deployed with Kubernetes, a Ray cluster could [autoscale](https://docs.ray.io/en/latest/cluster/vms/user-guides/configuring-autoscaling.html) in this situation to accommodate the demand that could not be satisfied. Our cluster is not auto-scaling and we are not going to add a node with 2 GPUs, so this job will wait forever.

Use Ctrl+C to stop the process in the Jupyter terminal. (The job is still submitted and PENDING, but not consuming worker resources, since it cannot be scheduled.)

:::

::: {.cell .markdown}

### Use Ray Train

While this is enough to run training jobs on the cluster, we're not making full use of Ray by running jobs this way. We can use Ray Train's [`TorchTrainer`](https://docs.ray.io/en/latest/train/getting-started-pytorch-lightning.html) to get additional features, including:

* fault tolerance - we'll save checkpoints in a MinIO object store and, if the worker node assigned to a job dies, it will resume training from the most recent checkpoint on a different worker.
* distributed training across different workers in the cluster (they can even be on different hosts! Although in this experiment, we're not using different hosts because it makes reservations complicated.)
* and, together with Ray Tune, intelligent hyperparameter optimization.

It is simple to [wrap a Pytorch Lightning script with Ray Train](https://docs.ray.io/en/latest/train/getting-started-pytorch-lightning.html). Run

```bash
# run in a terminal inside jupyter container
cd ~/work/gourmetgram-train
git stash # stash any changes you made to the current branch
git fetch -a
git switch ray
cd ~/work
```

The `train.py` script in this branch has already been modified for Pytorch Lightning. Open it, and note that we have:

* added imports
* wrapped *all** the Lightning code in a `def train_func(config)` block
* changed our `Lightning` trainer from 

```python
trainer = Trainer(
    max_epochs=config["total_epochs"],
    accelerator="gpu",
    devices="auto",
    callbacks=[checkpoint_callback, early_stopping_callback, backbone_finetuning_callback]
)
```

to use Ray-specific functions for checkpointing and distributed training,

```python
trainer = Trainer(
    max_epochs=config["total_epochs"],
    devices="auto",
    accelerator="auto",
    strategy=ray.train.lightning.RayDDPStrategy(),
    plugins=[ray.train.lightning.RayLightningEnvironment()],
    callbacks=[early_stopping_callback, backbone_finetuning_callback, ray.train.lightning.RayTrainReportCallback()]
)
```

* at the end, we run the training function with

```python
run_config = RunConfig(storage_path="s3://ray", failure_config=FailureConfig(max_failures=2))
scaling_config = ScalingConfig(num_workers=1, use_gpu=True, resources_per_worker={"GPU": 1, "CPU": 8})
trainer = TorchTrainer(
    train_func, scaling_config=scaling_config, run_config=run_config, train_loop_config=config
)
result = trainer.fit()
```

Here, 

* the `RunConfig` specifies where to put checkpoint files - we have created a MinIO bucket named `ray` for this. The MinIO credentials are passed to the Ray containers as environment variables in the Docker Compose. We also specify what to do if a job fails (we'll get to that soon!)
* the `ScalingConfig` defines the cluster resources we are going to request.
* and, we pass it all to a `Ray` `TorchTrainer`.


Let's try our Ray Train script. Since we define the resource requirements in the script itself, we **won't** use `--entrypoint-num-gpus` or `--entrypoint-num-cpus` this time - 


```bash
# runs on jupyter container inside node-mltrain, from inside the "work" directory
ray job submit --runtime-env runtime.json  --working-dir .  -- python gourmetgram-train/train.py 
```

Submit the job, and note that it runs mostly as before. Let it run until it is finished.

:::

::: {.cell .markdown}

### Use Ray Train with multiple workers

This version of our Ray Train script doesn't improve much on the previous way of directly submitting a Pytorch or Pytorch Lightning job, but it can!

We can easily use Ray Train to scale to multiple nodes. In `train.py`, change

```python
scaling_config = ScalingConfig(num_workers=1, use_gpu=True, resources_per_worker={"GPU": 1, "CPU": 8})
```

to 

```python
scaling_config = ScalingConfig(num_workers=2, use_gpu=True, resources_per_worker={"GPU": 1, "CPU": 8})
``` 

to scale to two worker nodes, each with 1 GPU and 8 GPUs assigned to the job. Save it, and run with

```bash
# runs on jupyter container inside node-mltrain, from inside the "work" directory
ray job submit --runtime-env runtime.json  --working-dir .  -- python gourmetgram-train/train.py 
```

On the Ray dashboard, in the "Resource Status" section of the "Overview" tab, you should see the increased resource requirements reflected in the "Usage" section.

In a terminal on the "node-mltrain" host (*not* inside the Jupyter container), run

```bash
# runs on node-mltrain
nvtop
```

and confirm that both GPUs are busy.


:::



::: {.cell .markdown}

### Use Ray Train with fractional GPUs

Just as we scaled up, we can scale down - we can ask for fractional GPUs. Ray won't enforce that a process only uses a fraction of the GPU, but if we know that a process does not fully utilize the GPU, we can specify fractional resources and then we can pack more processes on the same workers.

In `train.py`, change

```python
scaling_config = ScalingConfig(num_workers=2, use_gpu=True, resources_per_worker={"GPU": 1, "CPU": 8})
```

to 

```python
scaling_config = ScalingConfig(num_workers=1, use_gpu=True, resources_per_worker={"GPU": 0.5, "CPU": 4})
``` 

i.e. set the number of workers back to 1, and reduce the resources per worker. 


Then, open *three* terminals inside the Jupyter container. You are going to start three training jobs at the same time.

In each of the terminals, run


```bash
# runs on jupyter container inside node-mltrain, from inside the "work" directory
cd ~/work
ray job submit --runtime-env runtime.json  --working-dir .  -- python gourmetgram-train/train.py 
```

to submit the job three times.

On the Ray dashboard, in the "Resource Status" section of the "Overview" tab, you should see the total resource requirements reflected in the "Usage" section.


In a terminal on the "node-mltrain" host (not inside the container), run

```bash
# runs on node-mltrain
nvtop
```

to observe the effect on GPU utilization. You should be able to visually identify the GPU that has two jobs running on it, vs. the GPU that has only one. 

Wait for the training jobs to finish, and note the total time required to run 3 training jobs. (This is the time from "start of first job" until "end of last job to finish".) Since these jobs originally did not utilize a full GPU, they aren't slowed down much by sharing a GPU. 

Fractional GPU use allows us to increase the throughput of the cluster - it won't reduce the time to complete any one job, but if GPUs are underutilized, it can increase the number of jobs completed per unit time.

:::



::: {.cell .markdown}

### Use Ray Train for fault tolerance

Next, let's try out fault tolerance! If the worker node that runs our Ray Train job dies, Ray can resume from the most recent checkpoint on another worker node.

We already configured fault tolerance with

```python
run_config = RunConfig( ... failure_config=FailureConfig(max_failures=2))
```


So, let's create a failure to see how it works! Run

```bash
# runs on jupyter container inside node-mltrain, from inside the "work" directory
ray job submit --runtime-env runtime.json  --working-dir .  -- python gourmetgram-train/train.py 
```

to submit the job.

Wait until about half of the epochs have passed (e.g. about 10 epochs), so that there is a checkpoint to resume from.

In a terminal on the "node-mltrain" host (not inside the container), run

```bash
# runs on node-mltrain
nvtop
```

and identify whether the job is assigned to GPU 0 or GPU 1. Keep monitoring the `nvtop` output as you bring up a second terminal on "node-mltrain".

In that second terminal bring down the Docker container in which you Ray Train job is running (`ray-worker-0` has GPU 0 and `ray-worker-1` has GPU 1) - run one of these two commands -

```bash
# runs on node-mltrain
# docker stop ray-worker-0
# docker stop ray-worker-1
```

Observe in the `nvtop` output that the job is transferred to the other GPU. (Take a screenshot for your reference, during the interval when GPU usage is visible on both GPUs, showing the job transfer.) 

And, in the `ray job submit` output, you will see something like

```
(TorchTrainer pid=516, ip=172.19.0.4) Worker 0 has failed.
...
(TorchTrainer pid=512, ip=172.19.0.5) Restored on 172.19.0.5 from checkpoint: Checkpoint(filesystem=py::fsspec+('s3', 's3a'), path=ray/TorchTrainer_2025-02-25_17-47-34/TorchTrainer_a64e2_00000_0_2025-02-25_17-47-34/checkpoint_000004)
````

as the job switches to the other worker and resumes from checkpoint.

On the "Overview" page in the Ray dashboard, check the "Cluster status and autoscaler" visualization, and node that the node count has gone down. 

Wait for the training job to finish. 

Re-start the worker you stopped with one of - 

```bash
# runs on node-mltrain
# docker start ray-worker-0
# docker start ray-worker-1
```

:::


::: {.cell .markdown}

### Optional: Use Ray Tune for hyperparameter optimization

Finally, let's try using Ray Tune! Ray Tune makes it easy to run a distributed hyperparamter optimization, with intelligent scheduling e.g. aborting runs that are not looking promising. 

Switch to the `tune` branch to see this version of the code - 

```bash
# run in a terminal inside jupyter container
cd ~/work/gourmetgram-train
git stash # stash any changes you made to the current branch
git fetch -a
git switch tune
cd ~/work
```

In this version of the code,

* we have added some new imports
* we made some changes to our `config` to specify the hyperparameters we want to tune. We will consider two batch sizes, and we will randomly sample different dropout probabilities:

```
config = {
    "batch_size": tune.choice([32, 64]),
    "dropout_probability": tune.uniform(0.1, 0.8),
    ...
```

* the `config` is not passed directly to the `trainer` anymore, 
* and, instead of calling `fit` on `trainer`, we wrap our trainer in a Tune function, and `fit` that. We are using `ASHAScheduler`, a kind of [hyperband](https://arxiv.org/abs/1603.06560) scheduler which early-stops less promising runs:

```python

### New for Ray Tune
def tune_asha(num_samples:
    scheduler = ASHAScheduler(max_t=config["total_epochs"], grace_period=1, reduction_factor=2)
    tuner = tune.Tuner(
        trainer,   # this is the original trainer!
        param_space={"train_loop_config": config},  # pass our config here instead
        tune_config=tune.TuneConfig(
            metric="val_accuracy",
            mode="max",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    return tuner.fit()


results = tune_asha(num_samples=16)
```


Run

```bash
# runs on jupyter container inside node-mltrain, from inside the "work" directory
ray job submit --runtime-env runtime.json  --working-dir .  -- python gourmetgram-train/train.py 
```

to submit the job.


The initial output (which is also available from the job logs in the Ray dashboard, if you miss it in the terminal!) will show you which configurations were randomly samples, and it will start 8 training samples to evaluate those configurations.

However, as the training job progresses, you will see that some jobs are "TERMINATED" in fewer than 20 epochs, because other hyperparameter configurations were judged more effective. This saves resources on the cluster, compared to a grid search or a random search.


:::


::: {.cell .markdown}

### Stop Ray system

When you are finished with this section, stop the Ray cluster and its associated pieces (Grafana, object store).

For AMD GPUs:

```bash
# run on node-mltrain
docker compose -f mltrain-chi/docker/docker-compose-ray-rocm.yaml down
```

For NVIDIA GPUs:

```bash
# run on node-mltrain
docker compose -f mltrain-chi/docker/docker-compose-ray-cuda.yaml down
```

and then stop the Jupyter server with

```bash
# run on node-mltrain
docker stop jupyter
```


:::



