
::: {.cell .markdown}

## Use MLFlow outside of training runs

We can interact with the MLFLow tracking service through the web-based UI, but we can also use its Python API. For example, we can systematically "promote" the model from the highest-scoring run as the registered model, and then trigger a CI/CD pipeline using the new model.

After completing this section, you should be able to:

* use the MLFlow Python API to search runs 
* and use the MLFlow Python API to interact with the model registry

The code in this notebook will run in the "jupyter" container on "node-mltrain". Inside the "work" directory in your Jupyter container on "node-mltrain", open the `mlflow_api.ipynb` notebook, and follow along there to execute this notebook.

:::


::: {.cell .markdown}

First, let's create an MLFlow client and connect to our tracking server:

:::

::: {.cell .code}
```python
import mlflow
from mlflow.tracking import MlflowClient

# We don't have to set MLflow tracking URI because we set it in an environment variable
# mlflow.set_tracking_uri("http://A.B.C.D:8000/") 

client = MlflowClient()
```
:::


::: {.cell .markdown}

Now, let's specify get the ID of the experiment we are interesting in searching:

:::

::: {.cell .code}
```python
experiment = client.get_experiment_by_name("food11-classifier")
experiment
```
:::

::: {.cell .markdown}

We'll use this experiment ID to query our experiment runs. Let's ask MLFlow to return the two runs with the largest value of the `test_accuracy` metric:

:::

::: {.cell .code}
```python
runs = client.search_runs(experiment_ids=[experiment.experiment_id], 
    order_by=["metrics.test_accuracy DESC"], 
    max_results=2)
```
:::

::: {.cell .markdown}

Since these are sorted, the first element in `runs` should be the run with the highest accuracy:

:::

::: {.cell .code}
```python
best_run = runs[0]  # The first run is the best due to sorting
best_run_id = best_run.info.run_id
best_test_accuracy = best_run.data.metrics["test_accuracy"]
model_uri = f"runs:/{best_run_id}/model"

print(f"Best Run ID: {best_run_id}")
print(f"Test Accuracy: {best_test_accuracy}")
print(f"Model URI: {model_uri}")
```
:::


::: {.cell .markdown}

Let's register this model in the MLFlow model registry. We'll call it "food11-staging":

:::

::: {.cell .code}
```python
model_name = "food11-staging"
registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
print(f"Model registered as '{model_name}', version {registered_model.version}")
```
:::

::: {.cell .markdown}

and, we should see it if we click on the "Models" section of the MLFlow UI. 

:::


::: {.cell .markdown}

Now, let's imagine that a separate process - for example, part of a CI/CD pipeline - wants to download the latest version of the "food11-staging" model, in order to build a container including this model and deploy it to a staging environment.

:::


::: {.cell .code}
```python
import mlflow
from mlflow.tracking import MlflowClient

# We don't have to set MLflow tracking URI because we set it in an environment variable
# mlflow.set_tracking_uri("http://A.B.C.D:8000/") 

client = MlflowClient()
model_name = "food11-staging"

```
:::

::: {.cell .markdown}

We can get all versions of the "food11-staging" model:

:::


::: {.cell .code}
```python
model_versions = client.search_model_versions(f"name='{model_name}'")
```
:::

::: {.cell .markdown}

We can find the version with the highest version number (latest version):

:::


::: {.cell .code}
```python
latest_version = max(model_versions, key=lambda v: int(v.version))

print(f"Latest registered version: {latest_version.version}")
print(f"Model Source: {latest_version.source}")
print(f"Status: {latest_version.status}")
```
:::



::: {.cell .markdown}

and now, we can download the model artifact (e.g. in order to build it into a Docker container):

:::


::: {.cell .code}
```python
local_download = mlflow.artifacts.download_artifacts(latest_version.source, dst_path="./downloaded_model")
```
:::

::: {.cell .markdown}

In the file browser on the left side, note that the "downloaded_model" directory has appeared, and the model has been downloaded from the registry into this directory. 
 
:::


::: {.cell .markdown}

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


:::



