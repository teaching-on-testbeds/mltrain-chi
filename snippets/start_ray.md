
::: {.cell .markdown}

## Start the Ray cluster

The other major piece of our ML infrastructure and platform is the training job scheduler! When many teams are running training jobs on shared infrastructure, we need to be able to allocate resources among them.

After you finish this section, 

* you should be able to identify the parts of a Ray cluster
* and you should understand how to bring up these parts as Docker containers

:::

::: {.cell .markdown}

### Understand the Ray cluster

Our overall system in this experiment is illustrated in the following image:

![Ray cluster system.](images/5-ray-system.svg)

- We will operate a Ray cluster with a head node (responsible for scheduling and managing jobs and data, and serving a dashboard), and two worker nodes.
- For observability, the Ray head node uses [Prometheus](https://prometheus.io/) to collect metrics, and [Grafana](https://grafana.com/) to visualize them in a dashboard.
- The Ray worker nodes will use the MinIO object store for persistent storage from jobs. We will save model checkpoints in this MinIO storage, so that if a job is interrupted, a new Ray worker can resume from the last checkpoint.
- In addition to the elements that make up the Ray cluster, we will separately bring up a Jupyter notebook server container, in which we'll submit jobs to the cluster.


Ray is a general framework for programming distributed applications in Python. It is also a platform around this framework that includes many components - 

- Ray Cluster, for managing the underlying hardware resources and scheduling jobs on them
- Ray Train, for training ML models
- Ray Tune, for hyperparameter optimization
- Ray Data, for distributed data processing
- Ray Serve, for serving already-trained models

but we will focus specifically on the first three - Ray Cluster, Ray Train, and Ray Tune.

To bring up the cluster, follow the instructions for the GPU type that you are using - AMD or NVIDIA.

:::

::: {.cell .markdown}

### Start the Ray cluster - AMD GPUs

> **Note**: Follow these instructions only if you are running this experiment on a node with AMD GPUs.

First, we're going to build a container image for the Ray worker nodes, with Ray and ROCm installed. Run

```bash
# run on node-mltrain
docker build -t ray-rocm:2.42.1 -f mltrain-chi/docker/Dockerfile.ray-rocm .
```

It will take 5-10 minutes to build the container image.

We'll bring up our Ray cluster with Docker Compose. Run:

```bash
# run on node-mltrain
export HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4 )
docker compose -f mltrain-chi/docker/docker-compose-ray-rocm.yaml up -d
```

When it is finished, the output of 

```bash
# run on node-mltrain
docker ps
```

should show that the `ray-head`, `ray-worker-0`, and `ray-worker-1` containers are running.

Verify that a GPU is visible to each of the worker nodes.

```bash
# run on node-mltrain
docker exec ray-worker-0 "rocm-smi"
```

and

```bash
# run on node-mltrain
docker exec ray-worker-1 "rocm-smi"
```

:::


::: {.cell .markdown}

### Start the Ray cluster - NVIDIA GPUs

> **Note**: Follow these instructions only if you are running this experiment on a node with NVIDIA GPUs.

We'll bring up our Ray cluster with Docker Compose. Run:

```bash
# run on node-mltrain
export HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4 )
docker compose -f mltrain-chi/docker/docker-compose-ray-cuda.yaml up -d
```

When it is finished, the output of 

```bash
# run on node-mltrain
docker ps
```

should show that the `ray-head`, `ray-worker-0`, and `ray-worker-1` containers are running.

Although the host has 2 GPUs, we only passed one to each worker. Run

```bash
# run on node-mltrain
docker exec -it ray-worker-0 nvidia-smi --list-gpus
```

and 

```bash
# run on node-mltrain
docker exec -it ray-worker-1 nvidia-smi --list-gpus
```

and confirm that only one GPU appears in the output, and it is a different GPU (different UUID) in each.


:::


::: {.cell .markdown}

Next, let's start a Jupyter notebook container that does *not* have any GPUs attached. We'll use this container to submit jobs to the Ray cluster.


```bash
# run on node-mltrain
docker build -t jupyter-ray -f mltrain-chi/docker/Dockerfile.jupyter-ray .
```

Run

```bash
# run on node-mltrain
HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4 )
docker run  -d --rm  -p 8888:8888 \
    -v ~/mltrain-chi/workspace_ray:/home/jovyan/work/ \
    -e RAY_ADDRESS=http://${HOST_IP}:8265/ \
    --name jupyter \
    jupyter-ray
```


Then, run 

```bash
# run on node-mltrain
docker logs jupyter
```

and look for a line like

```
http://127.0.0.1:8888/lab?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Paste this into a browser tab, but in place of `127.0.0.1`, substitute the floating IP assigned to your instance, to open the Jupyter notebook interface.

In the file browser on the left side, open the `work` directory.

Open a terminal ("File > New > Terminal") inside the Jupyter server environment, and in this terminal, run

```bash
# runs on jupyter container inside node-mltrain
env
```

to see environment variables. Confirm that the `RAY_ADDRESS` is set, with the correct floating IP address.

:::


::: {.cell .markdown}

### Access Ray cluster dashboard

The Ray head node serves a dashboard on port 8265. In a browser, open

```
http://A.B.C.D:8265
```

where in place of `A.B.C.D`, substitute the floating IP associated with your server.

Click on the "Cluster" tab and verify that you see your head node and two worker nodes.

:::
