

# Train ML models with MLFlow and Ray

In this tutorial, we explore some of the infrastructure and platform requirements for large model training, and to support the training of many models by many teams. We focus specifically on 

* experiment tracking (using [MLFlow](https://mlflow.org/))
* and scheduling training jobs on a GPU cluster (using [Ray](https://www.ray.io/))

To run this experiment, you should have already created an account on Chameleon, and become part of a project. You must also have added your SSH key to the CHI@TACC site.



## Experiment resources 

For this experiment, we will provision one bare-metal node with GPUs. 

The MLFlow section is more interesting if we run it on a node with two GPUs, because then we can better understand how to configure logging in a distributed training run. But, if need be, we can run it on a node with one GPU.

The Ray section requires a node with two GPUs.

We can browse Chameleon hardware configurations for suitable node types using the [Hardware Browser](https://chameleoncloud.org/hardware/). For example, to find nodes with 2x GPUs: if we expand "Advanced Filters", check the "2" box under "GPU count", and then click "View", we can identify some suitable node types. 

We'll proceed with the `gpu_mi100` and `compute_liqid` node types at CHI@TACC.

* Most of the `gpu_mi100` nodes have two AMD MI100 GPUs. (One of the `gpu_mi100` nodes, `c03-04` has only one GPU; we'll avoid this one for the "Ray" section, which requires two GPUs.)
* The `compute_liqid` nodes at CHI@TACC have one or two NVIDIA A100 40GB GPUs. As of this writing, `liqid01` and `liqid02` have two GPUs, so we may use these two for the "Ray" section, which requires two GPUs. 

You can decide which type to use based on availability; but once you decide, make sure to follow the instructions specific to that GPU type. In some parts, there will be different instructions for setting up an AMD GPU node vs. and NVIDIA GPU node.




## Create a lease



To use bare metal resources on Chameleon, we must reserve them in advance. We can reserve two separate 3-hour blocks for this experiment: one for the MLFlow section and one for the Ray section. They are designed to run independently.

We can use the OpenStack graphical user interface, Horizon, to submit a lease for an MI100 or Liquid node at CHI@TACC. To access this interface,

* from the [Chameleon website](https://chameleoncloud.org/hardware/)
* click "Experiment" > "CHI@TACC"
* log in if prompted to do so
* check the project drop-down menu near the top left (which shows e.g. “CHI-XXXXXX”), and make sure the correct project is selected.



Then, 

* On the left side, click on "Reservations" > "Leases", and then click on "Host Calendar". In the "Node type" drop down menu, change the type to `gpu_mi100` or `compute_liqid` to see the schedule of availability. You may change the date range setting to "30 days" to see a longer time scale. Note that the dates and times in this display are in UTC. You can use [WolframAlpha](https://www.wolframalpha.com/) or equivalent to convert to your local time zone. 
* Once you have identified an available three-hour block in UTC time that works for you in your local time zone, make a note of:
  * the start and end time of the time you will try to reserve. (Note that if you mouse over an existing reservation, a pop up will show you the exact start and end time of that reservation.)
  * and the name of the node you want to reserve. (We will reserve nodes by name, not by type, to avoid getting a 1-GPU node when we wanted a 2-GPU node.)
* Then, on the left side, click on "Reservations" > "Leases", and then click on "Create Lease":
  * set the "Name" to <code>mltrain_<b>netID</b>_<b>X</b></code> where in place of <code><b>netID</b></code> you substitute your actual net ID, and in place of <code><b>X</b></code>, substitute an incrementing number (to differentiate multiple leases for different sections).
  * set the start date and time in UTC. To make scheduling smoother, please start your lease on an hour boundary, e.g. `XX:00`.
  * modify the lease length (in days) until the end date is correct. Then, set the end time. To be mindful of other users, you should limit your lease time to three hours as directed. Also, to avoid a potential race condition that occurs when one lease starts immediately after another lease ends, you should end your lease five minutes before the end of an hour, e.g. at `YY:55`.
  * Click "Next".
* On the "Hosts" tab, 
  * check the "Reserve hosts" box
  * leave the "Minimum number of hosts" and "Maximum number of hosts" at 1
  * in "Resource properties", specify the node name that you identified earlier.
* Click "Next". Then, click "Create". (We won't include any network resources in this lease.)
  
Your lease status should show as "Pending". Click on the lease to see an overview. It will show the start time and end time, and it will show the name of the physical host that is reserved for you as part of your lease. Make sure that the lease details are correct.



Since you will need the full lease time to actually execute your experiment, you should read *all* of the experiment material ahead of time in preparation, so that you make the best possible use of your time.



At the beginning of your lease time, you will continue with the next step, in which you bring up and configure a bare metal instance! Two alternate sets of instructions are provided for this part:

* a notebook that runs in the Chameleon Jupyter interface. This automates the setup process, so that you can "Run > Run Selected Cell and All Below" to let the setup mostly run without human intervention.
* or, instructions for using the Horizon GUI and an SSH session, in case you cannot or prefer not to use the Chameleon Jupyter interface




Before you begin, open this experiment on Trovi:

* Use this link: [Train ML models with MLFlow and Ray](https://chameleoncloud.org/experiment/share/d48d7684-cf6d-4c33-bcd6-5504266bc3d4) on Trovi
* Then, click “Launch on Chameleon”. This will start a new Jupyter server for you, with the experiment materials already in it.



## Launch and set up server - options

The next step will be to provision the bare metal server you have reserved, and configure it to run the containers associated with our experiment. The specific details will depend on what type of GPU server you have reserved - 

* If you have reserved an AMD GPU server, work through the notebook `1_create_server_amd.ipynb` inside the Chameleon Jupyter environment. Or if you don't have access to the Chameleon Jupyter environment or prefer to work through it by hand, you can follow [these instructions to provision and configure the AMD server](snippets/create_server_by_hand_amd.html).
* If you have reserved an NVIDIA GPU server, work through the notebook `1_create_server_nvidia.ipynb`  inside the Chameleon Jupyter environment. Or if you don't have access to the Chameleon Jupyter environment or prefer to work through it by hand, you can follow [these instructions to provision and configure the NVIDIA server](snippets/create_server_by_hand_nvidia.html).

After you have finished setting up the server, you will return to this page for the next section.




## Prepare data

For the rest of this tutorial, we'll be training models on the [Food-11 dataset](https://www.epfl.ch/labs/mmspg/downloads/food-image-datasets/). We're going to prepare a Docker volume with this dataset already prepared on it, so that the containers we create later can attach to this volume and access the data. 




First, create the volume:

```bash
# runs on node-mltrain
docker volume create food11
```

Then, to populate it with data, run

```bash
# runs on node-mltrain
docker compose -f mltrain-chi/docker/docker-compose-data.yaml up -d
```

This will run a temporary container that downloads the Food-11 dataset, organizes it in the volume, and then stops. It may take a minute or two. You can verify with 

```bash
# runs on node-mltrain
docker ps
```

that it is done - when there are no running containers.

Finally, verify that the data looks as it should. Start a shell in a temporary container with this volume attached, and `ls` the contents of the volume:

```bash
# runs on node-mltrain
docker run --rm -it -v food11:/mnt alpine ls -l /mnt/Food-11/
```

it should show "evaluation", "validation", and "training" subfolders.


## Start the tracking server

Now, we are ready to get our MLFlow tracking server running! After you finish this section, 

* you should be able to identify the parts of the remote MLFlow tracking server system, and what each part is for
* and you should understand how to bring up these parts as Docker containers



### Understand the MLFlow tracking server system


The MLFLow experiment tracking system [can scale](https://mlflow.org/docs/latest/tracking.html#common-setups) from a "personal" deployment on your own machine, to a larger scale deployment suitable for use by a team. Since we are interested in building and managing ML platforms and systems, not only in using them, we are of course going to bring up a larger scale instance.

The "remote tracking server" system includes:

* a database in which to store structured data for each "run", like the start and end time, hyperparameter values, and the values of metrics that we log to the server. In our deploymenet, this will be realized by a PostgreSQL server.
* an object store, in which MLFlow will log artifacts - model weights, images (e.g. PNGs), and so on. In our deployment, this will be realized by MinIO, an open source object storage system that is compatible with AWS S3 APIs (so it may be used as a drop-in self-managed replacement for AWS S3).
* and of course, the MLFlow tracking server itself. Users can interact with the MLFlow tracking server directly through a browser-based interface; user code will interact with the MLFlow tracking server through its APIs, implemented in the `mlflow` Python library.



We'll bring up each of these pieces in Docker containers. To make it easier to define and run this system of several containers, we'll use [Docker Compose](https://docs.docker.com/reference/compose-file/), a tool that lets us define the configuration of a set of containers in a YAML file, then bring them all up in one command. 

(However, unlike a container orchestration framework such as Kubernetes, it does not help us launch containers across multiple hosts, or have scaling capabilities.)

You can see our YAML configuration at: [docker-compose-mlflow.yaml](https://github.com/teaching-on-testbeds/mltrain-chi/tree/main/docker/docker-compose-mlflow.yaml)



Here, we explain the contents of the Docker compose file and describe an equivalent `docker run` (or `docker volume`) command for each part, but you won't actually run these commands - we'll bring up the system with a `docker compose` command at the end.

First, note that our Docker compose defines two volumes:

```
volumes:
  minio_data:
  postgres_data:
```

which will provide persistent storage (beyond the lifetime of the containers) for both the object store and the database backend. This part of the Docker compose is equivalent to

```
docker volume create minio_data
docker volume create postgres_data
```

Next, let's look at the part that specifies the MinIO container:

```
  minio:
    image: minio/minio
    restart: always
    expose:
      - "9000"
    ports:  
      - "9000:9000"  # The API for object storage is hosted on port 9000
      - "9001:9001"  # The web-based UI is on port 9001
    environment:
      MINIO_ROOT_USER: "your-access-key"
      MINIO_ROOT_PASSWORD: "your-secret-key"
    healthcheck:
      test: timeout 5s bash -c ':> /dev/tcp/127.0.0.1/9000' || exit 1
      interval: 1s
      timeout: 10s
      retries: 5
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data  # Use a volume so minio storage persists beyond container lifetime
```

This specification is similar to running 

```
docker run -d --name minio \
  --restart always \
  -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER="your-access-key" \
  -e MINIO_ROOT_PASSWORD="your-secret-key" \
  -v minio_data:/data \
  minio/minio server /data --console-address ":9001"
````

where we start a container named `minio`, publish ports 9000 and 9001, pass two environment variables into the container (`MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD`), and attach a volume `minio_data` that is mounted at `/data` inside the container. The container image is [`minio/minio`](https://hub.docker.com/r/minio/minio/tags), and we specify that the command 

```
server /data --console-address ":9001"
```

should run inside the container as soon as it is launched.

However, we also define a health check: we test that the `minio` container accepts connections on port 9000, which is where the S3-compatible API is hosted. This will allow us to make sure that other parts of our system are brought up only once MinIO is ready for them.

Next, we have 

```
  minio-create-bucket:
    image: minio/mc
    depends_on:
      minio:
        condition: service_healthy
    entrypoint: >
      /bin/sh -c "
      mc alias set minio http://minio:9000 your-access-key your-secret-key &&
      if ! mc ls minio/mlflow-artifacts; then
        mc mb minio/mlflow-artifacts &&
        echo 'Bucket mlflow-artifacts created'
      else
        echo 'Bucket mlflow-artifacts already exists';
      fi"
```

which creates a container that starts only once the `minio` container has passed a health check; this container uses an image with the MinIO client `mc`, `minio/mc`, and it just authenticates to the `minio` server that is running on the same Docker network, then creates a storage "bucket" named `mlflow-artifacts`, and exits:

```
mc alias set minio http://minio:9000 your-access-key your-secret-key
mc mb minio/mlflow-artifacts
```

The PostgreSQL database backend is defined in 

```
  postgres:
    image: postgres:latest
    container_name: postgres
    restart: always
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: mlflowdb
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data  # use a volume so storage persists beyond container lifetime
```

which is equivalent to 

```
docker run -d --name postgres \
  --restart always \
  -p 5432:5432 \
  -e POSTGRES_USER=user \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=mlflowdb \
  -v postgres_data:/var/lib/postgresql/data \
  postgres:latest
```

where like the MinIO container, we specify the container name and image, the port to publish (`5432`), some environment variables, and we attach a volume. 

Finally, the MLFlow tracking server is specified:

```
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.20.2
    container_name: mlflow
    restart: always
    depends_on:
      - minio
      - postgres
      - minio-create-bucket  # make sure minio and postgres services are alive, and bucket is created, before mlflow starts
    environment:
      MLFLOW_TRACKING_URI: http://0.0.0.0:8000
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000  # how mlflow will access object store
      AWS_ACCESS_KEY_ID: "your-access-key"
      AWS_SECRET_ACCESS_KEY: "your-secret-key"
    ports:
      - "8000:8000"
    command: >
      /bin/sh -c "pip install psycopg2-binary boto3 &&
      mlflow server --backend-store-uri postgresql://user:password@postgres/mlflowdb 
      --artifacts-destination s3://mlflow-artifacts/ --serve-artifacts --host 0.0.0.0 --port 8000"
```

which is similar to running

```bash
docker run -d --name mlflow \
  --restart always \
  -p 8000:8000 \
  -e MLFLOW_TRACKING_URI="http://0.0.0.0:8000" \
  -e MLFLOW_S3_ENDPOINT_URL="http://minio:9000" \
  -e AWS_ACCESS_KEY_ID="your-access-key" \
  -e AWS_SECRET_ACCESS_KEY="your-secret-key" \
  --network host \
  ghcr.io/mlflow/mlflow:v2.20.2 \
  /bin/sh -c "pip install psycopg2-binary boto3 &&
  mlflow server --backend-store-uri postgresql://user:password@postgres/mlflowdb 
  --artifacts-destination s3://mlflow/ --serve-artifacts --host 0.0.0.0 --port 8000"
```

and starts an MLFlow container that runs the command:

```bash
pip install psycopg2-binary boto3
mlflow server --backend-store-uri postgresql://user:password@postgres/mlflowdb --artifacts-destination s3://mlflow/ --serve-artifacts --host 0.0.0.0 --port 8000
```

Additionally, in the Docker Compose file, we specify that this container should be started only after the `minio`, `postgres` and `minio-create-bucket` containers come up, since otherwise the `mlflow server` command will fail.




In addition to the three elements that make up the MLFlow tracking server system, we will separately bring up a Jupyter notebook server container, in which we'll run ML training experiments that will be tracked in MLFlow. So, our overall system will look like this:

![MLFlow experiment tracking server system.](images/5-mlflow-system.svg)





### Start MLFlow tracking server system

Now we are ready to get it started! Bring up our MLFlow system with:


```bash
# run on node-mltrain
docker compose -f mltrain-chi/docker/docker-compose-mlflow.yaml up -d
```

which will pull each container image, then start them.

When it is finished, the output of 

```bash
# run on node-mltrain
docker ps
```

should show that the `minio`, `postgres`, and `mlflow` containers are running.




### Access dashboards for the MLFlow tracking server system


Both MLFlow and MinIO include a browser-based dashboard. Let's open these to make sure that we can find our way around them.

The MinIO dashboard runs on port 9001. In a browser, open

```
http://A.B.C.D:9001
```

where in place of `A.B.C.D`, substitute the floating IP associated with your server.

Log in with the credentials we specified in the Docker Compose YAML:

* Username: `your-access-key`
* Password: `your-secret-key`

Then,

* Click on the "Buckets" section and note the `mlflow-artifacts` storage bucket that we created as part of the Docker Compose. 
* Click on "Monitoring > Metrics" and note the dashboard that shows the storage system health. MinIO works as a distributed object store with many advanced capabilities, although we are not using them; this dashboard lets operators keep an eye on system status.
* Click on "Object Browser". In this section, you can look at the files that have been uploaded to the object store - but, we haven't used MLFlow yet, so for now there is nothing interesting here. However, as you start to log artifacts to the MLFlow server, you will see them appear here.

Next, let's look at the MLFlow UI. This runs on port 8000. In a browser, open

```
http://A.B.C.D:8000
```

where in place of `A.B.C.D`, substitute the floating IP associated with your server.

The UI shows a list of tracked "experiments", and experiment "runs". (A "run" corresponds to one instance of training a model; an "experiment" groups together related runs.) Since we have not yet used MLFlow, for now we will only see a "Default" experiment and no runs. But, that will change very soon!



### Start a Jupyter server

Finally, we'll start the Jupyter server container, inside which we will run experiments that are tracked in MLFlow. Make sure your container image build, from the previous section, is now finished - you should see a "jupyter-mlflow" image in the output of:


```bash
# run on node-mltrain
docker image list
```


The command to run will depend on what type of GPU node you are using - 

If you are using an AMD GPU (node type `gpu_mi100`), run

```bash
# run on node-mltrain IF it is a gpu_mi100
HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4 )
docker run  -d --rm  -p 8888:8888 \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add $(getent group | grep render | cut -d':' -f 3) \
    --shm-size 16G \
    -v ~/mltrain-chi/workspace_mlflow:/home/jovyan/work/ \
    -v food11:/mnt/ \
    -e MLFLOW_TRACKING_URI=http://${HOST_IP}:8000/ \
    -e FOOD11_DATA_DIR=/mnt/Food-11 \
    --name jupyter \
    jupyter-mlflow
```

Note that we intially get `HOST_IP`, the floating IP assigned to your instance, as a variable; then we use it to specify the `MLFLOW_TRACKING_URI` inside the container. Training jobs inside the container will access the MLFlow tracking server using its public IP address.

Here,

* `-d` says to start the container and detach, leaving it running in the background
* `-rm` says that after we stop the container, it should be removed immediately, instead of leaving it around for potential debugging
* `-p 8888:8888` says to publish the container's port `8888` (the second `8888` in the argument) to the host port `8888` (the first `8888` in the argument)
* `--device=/dev/kfd --device=/dev/dri` pass the AMD GPUs to the container
* `--group-add video --group-add $(getent group | grep render | cut -d':' -f 3)` makes sure that the user inside the container is a member of a group that has permission to use the GPU(s) - the `video` group and the `render` group. (The `video` group always has the same group ID, by convention, but [the `render` group does not](https://github.com/ROCm/ROCm-docker/issues/90), so we need to find out its group ID on the host and pass that to the container.)
* `--shm-size 16G` increases the memory available for interprocess communication
* the host directory `~/mltrain-chi/workspace_mlflow` is mounted inside the workspace as `/home/jovyan/work/`
* the volume `food11` is mounted inside the workspace as `/mnt/`
* and we pass `MLFLOW_TRACKING_URI` and `FOOD11_DATA_DIR` as environment variables.

If you are using an NVIDIA GPU (node type `compute_liqid`), run

```bash
# run on node-mltrain IF it is a compute_liqid
HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4 )
docker run  -d --rm  -p 8888:8888 \
    --gpus all \
    --shm-size 16G \
    -v ~/mltrain-chi/workspace_mlflow:/home/jovyan/work/ \
    -v food11:/mnt/ \
    -e MLFLOW_TRACKING_URI=http://${HOST_IP}:8000/ \
    -e FOOD11_DATA_DIR=/mnt/Food-11 \
    --name jupyter \
    jupyter-mlflow
```

Note that we intially get `HOST_IP`, the floating IP assigned to your instance, as a variable; then we use it to specify the `MLFLOW_TRACKING_URI` inside the container. Training jobs inside the container will access the MLFlow tracking server using its public IP address.

* `-d` says to start the container and detach, leaving it running in the background
* `-rm` says that after we stop the container, it should be removed immediately, instead of leaving it around for potential debugging
* `-p 8888:8888` says to publish the container's port `8888` (the second `8888` in the argument) to the host port `8888` (the first `8888` in the argument)
* `--gus all` pass the NVIDIA GPUs to the container
* `--shm-size 16G` increases the memory available for interprocess communication
* the host directory `~/mltrain-chi/workspace_mlflow` is mounted inside the workspace as `/home/jovyan/work/`
* the volume `food11` is mounted inside the workspace as `/mnt/`
* and we pass `MLFLOW_TRACKING_URI` and `FOOD11_DATA_DIR` as environment variables.

Then, run 

```
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

to see environment variables. Confirm that the `MLFLOW_TRACKING_URI` is set, with the correct floating IP address.



## Track a Pytorch experiment

Now, we will use our MLFlow tracking server to track a Pytorch training job. After completing this section, you should be able to:

* understand what type of artifacts, parameters, and metrics may be logged to an experiment tracking service (MLFlow or otherwise)
* configure a Python script to connect to an MLFlow tracking server and associate with a particular experiment
* configure system metrics logging in MLFlow
* log hyperparametrics and metrics of a Pytorch training job to MLFlow
* log a trained Pytorch model as an artifact to MLFlow
* use MLFlow to compare experiments visually




The premise of this example is as follows: You are working at a machine learning engineer at a small startup company called GourmetGram. They are developing an online photo sharing community focused on food. You have developed a convolutional neural network in Pytorch that automatically classifies photos of food into one of a set of categories: Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup, and Vegetable/Fruit. 

An original Pytorch training script is available at: [gourmetgram-train/train.py](https://github.com/teaching-on-testbeds/gourmetgram-train/blob/main/train.py). The model uses a MobileNetV2 base layer, adds a classification head on top, trains the classification head, and then fine-tunes the entire model, using the [Food-11 dataset](https://www.epfl.ch/labs/mmspg/downloads/food-image-datasets/).




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





### Register a model

MLFlow also includes a model registry, with which we can manage versions of our models. 

From the "run" page, click on the "Artifacts" tab, and find the model. Then, click "Register model" and in the "Model" menu, "Create new model". Name the model `food11` and save.

Now, in the "Models" tab in MLFlow, you can see the latest version of your model, with its lineage (i.e. the specific run that generated it) associated with it. This allows us to version models, identify exactly the code, system settings, and hyperparameters that created the model, and manage different model versions in different parts of the lifecycle (e.g. staging, canary, production deployment).


## Track a Lightning experiment

In the previous experiment, we manually added a lot of MLFlow logging code to our Pytorch training script. However, [for many ML frameworks](https://mlflow.org/docs/latest/tracking/autolog.html#supported-libraries), MLFLow can automatically log relevant details. In this section, we will convert our Pytorch training script to a Pytorch Lightning script, for which automatic logging is supported in MLFlow, to see how this capability works.

After completing this section, you should be able to:

* understand what Pytorch Lightning is, and some benefits of using Lightning over "vanilla" Pytorch
* understand how to use autologging in MLFlow



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




## Use MLFlow outside of training runs

We can interact with the MLFLow tracking service through the web-based UI, but we can also use its Python API. For example, we can systematically "promote" the model from the highest-scoring run as the registered model, and then trigger a CI/CD pipeline using the new model.

After completing this section, you should be able to:

* use the MLFlow Python API to search runs 
* and use the MLFlow Python API to interact with the model registry

The code in this notebook will run in the "jupyter" container on "node-mltrain". Inside the "work" directory in your Jupyter container on "node-mltrain", open the `mlflow_api.ipynb` notebook, and follow along there to execute this notebook.




First, let's create an MLFlow client and connect to our tracking server:


```python
import mlflow
from mlflow.tracking import MlflowClient

# We don't have to set MLflow tracking URI because we set it in an environment variable
# mlflow.set_tracking_uri("http://A.B.C.D:8000/") 

client = MlflowClient()
```



Now, let's specify get the ID of the experiment we are interesting in searching:


```python
experiment = client.get_experiment_by_name("food11-classifier")
experiment
```


We'll use this experiment ID to query our experiment runs. Let's ask MLFlow to return the two runs with the largest value of the `test_accuracy` metric:


```python
runs = client.search_runs(experiment_ids=[experiment.experiment_id], 
    order_by=["metrics.test_accuracy DESC"], 
    max_results=2)
```


Since these are sorted, the first element in `runs` should be the run with the highest accuracy:


```python
best_run = runs[0]  # The first run is the best due to sorting
best_run_id = best_run.info.run_id
best_test_accuracy = best_run.data.metrics["test_accuracy"]
model_uri = f"runs:/{best_run_id}/model"

print(f"Best Run ID: {best_run_id}")
print(f"Test Accuracy: {best_test_accuracy}")
print(f"Model URI: {model_uri}")
```



Let's register this model in the MLFlow model registry. We'll call it "food11-staging":


```python
model_name = "food11-staging"
registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
print(f"Model registered as '{model_name}', version {registered_model.version}")
```


and, we should see it if we click on the "Models" section of the MLFlow UI. 




Now, let's imagine that a separate process - for example, part of a CI/CD pipeline - wants to download the latest version of the "food11-staging" model, in order to build a container including this model and deploy it to a staging environment.



```python
import mlflow
from mlflow.tracking import MlflowClient

# We don't have to set MLflow tracking URI because we set it in an environment variable
# mlflow.set_tracking_uri("http://A.B.C.D:8000/") 

client = MlflowClient()
model_name = "food11-staging"

```


We can get all versions of the "food11-staging" model:



```python
model_versions = client.search_model_versions(f"name='{model_name}'")
```


We can find the version with the highest version number (latest version):



```python
latest_version = max(model_versions, key=lambda v: int(v.version))

print(f"Latest registered version: {latest_version.version}")
print(f"Model Source: {latest_version.source}")
print(f"Status: {latest_version.status}")
```




and now, we can download the model artifact (e.g. in order to build it into a Docker container):



```python
local_download = mlflow.artifacts.download_artifacts(latest_version.source, dst_path="./downloaded_model")
```


In the file browser on the left side, note that the "downloaded_model" directory has appeared, and the model has been downloaded from the registry into this directory. 
 



### Stop MLFlow system

When you are finished with this section, stop the MLFlow tracking server and its associated pieces (database, object store) with

```bash
# run on node-mltrain
docker compose -f mltrain-chi/docker/docker-compose-mlflow.yaml down
```

and then stop the Jupyter server with

```bash
# run on node-mltrain
docker stop jupyter
```







## Start the Ray cluster

The other major piece of our ML infrastructure and platform is the training job scheduler! When many teams are running training jobs on shared infrastructure, we need to be able to allocate resources among them.

After you finish this section, 

* you should be able to identify the parts of a Ray cluster
* and you should understand how to bring up these parts as Docker containers



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



### Start the Ray cluster - AMD GPUs

> **Note**: Follow these instructions only if you are running this experiment on a node with AMD GPUs.

First, we're going to build a container image for the Ray worker nodes, with Ray and ROCm installed. Run

```bash
# run on node-mltrain
docker build -t ray-rocm:2.42.1 -f mltrain-chi/docker/Dockerfile.ray-rocm .
```

It will take 5-10 minutes to build the container image.

You can see this Dockerfile here: [Dockerfile.ray-rocm](https://github.com/teaching-on-testbeds/mltrain-chi/blob/main/docker/Dockerfile.ray-rocm).


We'll bring up our Ray cluster with Docker Compose. Run:

```bash
# run on node-mltrain
export HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4 )
docker compose -f mltrain-chi/docker/docker-compose-ray-rocm.yaml up -d
```

You can see this Docker Compose YAML here: [docker-compose-ray-rocm.yaml](https://github.com/teaching-on-testbeds/mltrain-chi/blob/main/docker/docker-compose-ray-rocm.yaml).


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




### Start the Ray cluster - NVIDIA GPUs

> **Note**: Follow these instructions only if you are running this experiment on a node with NVIDIA GPUs.

We'll bring up our Ray cluster with Docker Compose. Run:

```bash
# run on node-mltrain
export HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4 )
docker compose -f mltrain-chi/docker/docker-compose-ray-cuda.yaml up -d
```

You can see this Docker Compose YAML here: [docker-compose-ray-cuda.yaml](https://github.com/teaching-on-testbeds/mltrain-chi/blob/main/docker/docker-compose-ray-cuda.yaml).

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




### Access Ray cluster dashboard

The Ray head node serves a dashboard on port 8265. In a browser, open

```
http://A.B.C.D:8265
```

where in place of `A.B.C.D`, substitute the floating IP associated with your server.

Click on the "Cluster" tab and verify that you see your head node and two worker nodes.



## Submit jobs to the Ray cluster

Now that we have a Ray cluster running, we can learn how to use it! After you finish this section, 

* you should understand how to specify the resource requirements and runtime environment for a job, and submit it to Ray
* and you should be able to modify a Pytorch Lightning script to use Ray Train, including its checkpointing, fault tolerance, and distributed training capabilities, and to use Ray Tune for hyperparameter optimization.





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







<hr>

<small>Questions about this material? Contact Fraida Fund</small>

<hr>

<small>This material is based upon work supported by the National Science Foundation under Grant No. 2230079.</small>

<small>Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.</small>