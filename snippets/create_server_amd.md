

::: {.cell .markdown}

## Launch and set up AMD MI100 server - with python-chi

At the beginning of the lease time, we will bring up our GPU server. We will use the `python-chi` Python API to Chameleon to provision our server. 

> **Note**: if you don't have access to the Chameleon Jupyter environment, or if you prefer to set up your AMD MI100 server by hand, the next section provides alternative instructions! If you want to set up your server "by hand", skip to the next section.


We will execute the cells in this notebook inside the Chameleon Jupyter environment.

Run the following cell, and make sure the correct project is selected:

:::

::: {.cell .code}
```python
from chi import server, context, lease
import os, time

context.version = "1.0" 
context.choose_project()
context.choose_site(default="CHI@TACC")
```
:::

::: {.cell .markdown}

Change the string in the following cell to reflect the name of *your* lease (**with your own net ID**), then run it to get your lease:

:::

::: {.cell .code}
```python
l = lease.get_lease(f"mltrain_netID") 
l.show()
```
:::

::: {.cell .markdown}

The status should show as "ACTIVE" now that we are past the lease start time.

The rest of this notebook can be executed without any interactions from you, so at this point, you can save time by clicking on this cell, then selecting "Run" > "Run Selected Cell and All Below" from the Jupyter menu.  

As the notebook executes, monitor its progress to make sure it does not get stuck on any execution error, and also to see what it is doing!

:::

::: {.cell .markdown}

We will use the lease to bring up a server with the `CC-Ubuntu24.04-hwe` disk image. (The default Ubuntu 24.04 kernel is not compatible with the AMD GPU on these nodes.)

> **Note**: the following cell brings up a server only if you don't already have one with the same name! (Regardless of its error state.) If you have a server in ERROR state already, delete it first in the Horizon GUI before you run this cell.


:::


::: {.cell .code}
```python
username = os.getenv('USER') # all exp resources will have this prefix
s = server.Server(
    f"node-mltrain-{username}", 
    reservation_id=l.node_reservations[0]["id"],
    image_name="CC-Ubuntu24.04-hwe"
)
s.submit(idempotent=True)
```
:::

::: {.cell .markdown}

Note: security groups are not used at Chameleon bare metal sites, so we do not have to configure any security groups on this instance.

:::

::: {.cell .markdown}

Then, we'll associate a floating IP with the instance, so that we can access it over SSH.

:::

::: {.cell .code}
```python
s.associate_floating_ip()
```
:::

::: {.cell .code}
```python
s.refresh()
s.check_connectivity()
```
:::

::: {.cell .markdown}

In the output below, make a note of the floating IP that has been assigned to your instance (in the "Addresses" row).

:::

::: {.cell .code}
```python
s.refresh()
s.show(type="widget")
```
:::




::: {.cell .markdown}

## Retrieve code and notebooks on the instance

Now, we can use `python-chi` to execute commands on the instance, to set it up. We'll start by retrieving the code and other materials on the instance.

:::

::: {.cell .code}
```python
s.execute("git clone --recurse-submodules https://github.com/teaching-on-testbeds/mltrain-chi")
```
:::


::: {.cell .markdown}

## Set up Docker

To use common deep learning frameworks like Tensorflow or PyTorch, and ML training platforms like MLFlow and Ray, we can run containers that have all the prerequisite libraries necessary for these frameworks. Here, we will set up the container framework.

:::

::: {.cell .code}
```python
s.execute("curl -sSL https://get.docker.com/ | sudo sh")
s.execute("sudo groupadd -f docker; sudo usermod -aG docker $USER")
```
:::

::: {.cell .markdown}

## Set up the AMD GPU


Before we can use the AMD GPUs, we need to set up the driver using the `amdgpu-install` utility. 

Let's follow [AMD's instructions for setting up `amdgpu-install`](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/install-methods/amdgpu-installer/amdgpu-installer-ubuntu.html#installation):

:::

::: {.cell .code}
```python
s.execute("sudo apt update; wget https://repo.radeon.com/amdgpu-install/6.3.3/ubuntu/noble/amdgpu-install_6.3.60303-1_all.deb")
s.execute("sudo apt -y install ./amdgpu-install_6.3.60303-1_all.deb; sudo apt update")
```
:::


::: {.cell .markdown}

To [run containers using ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html) (Radeon Open Compute Platform), an open-source software stack from AMD that allows users to program AMD GPUs (similar to NVIDIA's CUDA), we need to install the `amdgpu-dkms` driver:

:::


::: {.cell .code}
```python
s.execute("amdgpu-install -y --usecase=dkms")
```
:::

::: {.cell .markdown}

And, we'll also install the `rocm-smi` utility, so that we can monitor the GPU from the host:

:::


::: {.cell .code}
```python
s.execute("sudo apt -y install rocm-smi")
```
:::

::: {.cell .markdown}

Finally, we will add the `cc` user to the `video` and `render` groups, which are needed for access to the GPU:

:::


::: {.cell .code}
```python
s.execute("sudo usermod -aG video,render $USER")
```
:::

::: {.cell .markdown}


That's all we will need on the host - the rest of ROCm will be installed inside the containers. 

To apply the changes to the kernel, we need to reboot, and wait for the server to come back up.

:::


::: {.cell .code}
```python
s.execute("sudo reboot")
time.sleep(30)
```
:::


::: {.cell .code}
```python
s.refresh()
s.check_connectivity()
```
:::

::: {.cell .markdown}

Run

:::

::: {.cell .code}
```python
s.execute("rocm-smi")
```
:::

::: {.cell .markdown}

and verify that you can see the GPU(s).

:::


::: {.cell .markdown}

We can also install `nvtop` to monitor GPU usage - we'll install from source, because the older version in the Ubuntu package repositories does not support AMD GPUs:

:::


::: {.cell .code}
```python
s.execute("sudo apt -y install cmake libncurses-dev libsystemd-dev libudev-dev libdrm-dev libgtest-dev")
s.execute("git clone https://github.com/Syllo/nvtop")
s.execute("mkdir -p nvtop/build && cd nvtop/build && cmake .. -DAMDGPU_SUPPORT=ON && sudo make install")
```
:::


::: {.cell .markdown}

###  Build a container image - for MLFlow section


Finally, we will build a container image in which to work in the MLFlow section, that has:

* a Jupyter notebook server
* Pytorch and Pytorch Lightning
* ROCm, which allows deep learning frameworks like Pytorch to use the AMD GPU accelerator
* and MLFlow

You can see our Dockerfile for this image at: [Dockerfile.jupyter-torch-mlflow-rocm](https://github.com/teaching-on-testbeds/mltrain-chi/tree/main/docker/Dockerfile.jupyter-torch-mlflow-rocm)


Building this container will take a **very long** time (ROCm is huge). But that's OK: we can get it started and then continue to the next section while it builds in the background, since we don't need this container immediately. We just need it to finish by the "Start a Jupyter server" subsection of the "Start the tracking server" section.


:::


::: {.cell .code}
```python
s.execute("docker build -t jupyter-mlflow -f mltrain-chi/docker/Dockerfile.jupyter-torch-mlflow-rocm .")
```
:::


::: {.cell .markdown}

Leave that cell running, and in the meantime, open an SSH sesson on your server. From your local terminal, run

```
ssh -i ~/.ssh/id_rsa_chameleon cc@A.B.C.D
```

where

* in place of `~/.ssh/id_rsa_chameleon`, substitute the path to your own key that you had uploaded to CHI@TACC
* in place of `A.B.C.D`, use the floating IP address you just associated to your instance.


:::