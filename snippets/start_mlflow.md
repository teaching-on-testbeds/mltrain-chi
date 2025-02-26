
::: {.cell .markdown}

## Start the tracking server

Now, we are ready to get our MLFlow tracking server running! After you finish this section, 

* you should be able to identify the parts of the remote MLFlow tracking server system, and what each part is for
* and you should understand how to bring up these parts as Docker containers

:::

::: {.cell .markdown}

### Understand the MLFlow tracking server system


The MLFLow experiment tracking system [can scale](https://mlflow.org/docs/latest/tracking.html#common-setups) from a "personal" deployment on your own machine, to a larger scale deployment suitable for use by a team. Since we are interested in building and managing ML platforms and systems, not only in using them, we are of course going to bring up a larger scale instance.

The "remote tracking server" system includes:

* a database in which to store structured data for each "run", like the start and end time, hyperparameter values, and the values of metrics that we log to the server. In our deploymenet, this will be realized by a PostgreSQL server.
* an object store, in which MLFlow will log artifacts - model weights, images (e.g. PNGs), and so on. In our deployment, this will be realized by MinIO, an open source object storage system that is compatible with AWS S3 APIs (so it may be used as a drop-in self-managed replacement for AWS S3).
* and of course, the MLFlow tracking server itself. Users can interact with the MLFlow tracking server directly through a browser-based interface; user code will interact with the MLFlow tracking server through its APIs, implemented in the `mlflow` Python library.

:::

::: {.cell .markdown}

We'll bring up each of these pieces in Docker containers. To make it easier to define and run this system of several containers, we'll use [Docker Compose](https://docs.docker.com/reference/compose-file/), a tool that lets us define the configuration of a set of containers in a YAML file, then bring them all up in one command. 

(However, unlike a container orchestration framework such as Kubernetes, it does not help us launch containers across multiple hosts, or have scaling capabilities.)

You can see our YAML configuration at: [docker-compose-mlflow.yaml](https://github.com/teaching-on-testbeds/mltrain-chi/tree/main/docker/docker-compose-mlflow.yaml)

:::

::: {.cell .markdown}

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


:::

::: {.cell .markdown}

In addition to the three elements that make up the MLFlow tracking server system, we will separately bring up a Jupyter notebook server container, in which we'll run ML training experiments that will be tracked in MLFlow. So, our overall system will look like this:

![MLFlow experiment tracking server system.](images/5-mlflow-system.svg)


:::


::: {.cell .markdown}

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


:::

::: {.cell .markdown}

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

:::

::: {.cell .markdown}

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

:::
