
::: {.cell .markdown}

## Launch and set up AMD MI100 server - with python-chi

At the beginning of the lease time, we will bring up our GPU server. We will use the `python-chi` Python API to Chameleon to provision our server.

We will execute the cells in this notebook inside the Chameleon Jupyter environment.

Run the following cell, and make sure the correct project is selected:

:::

::: {.cell .code}
```python
# run in Chameleon Jupyter environment
from chi import server, context, lease
import os

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
# run in Chameleon Jupyter environment
l = lease.get_lease(f"mltrain_netID")
l.show()
```
:::

::: {.cell .markdown}

The status should show as "ACTIVE" now that we are past the lease start time.

The rest of this notebook can be executed without any interactions from you, so at this point, you can save time by clicking on this cell, then selecting "Run" > "Run Selected Cell and All Below" from the Jupyter menu.

As the notebook executes, monitor its progress to make sure it does not get stuck on any execution error, and also to see what it is doing.

:::

::: {.cell .markdown}

We will use the lease to bring up a server with the `CC-Ubuntu24.04-ROCm` disk image. (The default Ubuntu 24.04 kernel is not compatible with the AMD GPU on these nodes.)

> **Note**: the following cell brings up a server only if you do not already have one with the same name (regardless of its error state). If you have a server in ERROR state already, delete it first in the Horizon GUI before you run this cell.

:::

::: {.cell .code}
```python
# run in Chameleon Jupyter environment
username = os.getenv('USER')
s = server.Server(
    f"node-mltrain-{username}",
    reservation_id=l.node_reservations[0]["id"],
    image_name="CC-Ubuntu24.04-ROCm"
)
s.submit(idempotent=True)
```
:::

::: {.cell .markdown}

Note: security groups are not used at Chameleon bare metal sites, so we do not have to configure any security groups on this instance.

:::

::: {.cell .markdown}

Then, we will associate a floating IP with the instance, so that we can access it over SSH.

:::

::: {.cell .code}
```python
# run in Chameleon Jupyter environment
s.associate_floating_ip()
```
:::

::: {.cell .code}
```python
# run in Chameleon Jupyter environment
s.refresh()
s.check_connectivity()
```
:::

::: {.cell .markdown}

In the output below, make a note of the floating IP that has been assigned to your instance (in the "Addresses" row).

:::

::: {.cell .code}
```python
# run in Chameleon Jupyter environment
s.refresh()
s.show(type="widget")
```
:::

::: {.cell .markdown}

## Retrieve code and notebooks on the instance

Now, we can use `python-chi` to execute commands on the instance to set it up. We will start by retrieving the code and other materials on the instance.

:::

::: {.cell .code}
```python
# run in Chameleon Jupyter environment
s.execute("git clone --branch amd --single-branch https://github.com/teaching-on-testbeds/mltrain-chi")
```
:::

::: {.cell .markdown}

## Set up Docker

To use common deep learning frameworks like Tensorflow or PyTorch, and distributed training platforms like Ray, we can run containers that have all the prerequisite libraries necessary for these frameworks. Here, we will set up the container framework.

:::

::: {.cell .code}
```python
# run in Chameleon Jupyter environment
s.execute("curl -sSL https://get.docker.com/ | sudo sh")
s.execute("sudo groupadd -f docker; sudo usermod -aG docker $USER")
```
:::

::: {.cell .markdown}

## Verify AMD GPU access

Run

:::

::: {.cell .code}
```python
# run in Chameleon Jupyter environment
s.execute("rocm-smi")
```
:::

::: {.cell .markdown}

and verify that you can see the GPU(s).

:::

::: {.cell .markdown}

We can also install `nvtop` to monitor GPU usage. We install from source because older versions in Ubuntu package repositories do not support AMD GPUs.

:::

::: {.cell .code}
```python
# run in Chameleon Jupyter environment
s.execute("sudo apt -y install cmake libncurses-dev libsystemd-dev libudev-dev libdrm-dev libgtest-dev")
s.execute("git clone https://github.com/Syllo/nvtop")
s.execute("mkdir -p nvtop/build && cd nvtop/build && cmake .. -DAMDGPU_SUPPORT=ON && sudo make install")
```
:::

::: {.cell .markdown}

Leave that cell running, and in the meantime, open an SSH session on your server. From your local terminal, run

```
ssh -i ~/.ssh/id_rsa_chameleon cc@A.B.C.D
```

where

* in place of `~/.ssh/id_rsa_chameleon`, substitute the path to your own key that you uploaded to CHI@TACC
* in place of `A.B.C.D`, use the floating IP address you just associated with your instance.

:::
