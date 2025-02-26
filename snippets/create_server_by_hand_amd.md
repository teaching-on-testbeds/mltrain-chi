
::: {.cell .markdown}

## Launch and set up AMD MI100 server - by hand

> **Note**: if you set up your AMD MI100 server using the `python-chi` notebook, you will skip this section! This section describes how to set up the server "by hand", in case you do not have access to the Chameleon Jupyter environment in which to run the `python-chi` notebook, or in case you prefer to do it "by hand".

At the beginning of the lease time, we will bring up our GPU server. We will use Horizon GUI at CHI@TACC to provision our server. 


:::


::: {.cell .markdown}

To access this interface,

* from the [Chameleon website](https://chameleoncloud.org/hardware/)
* click "Experiment" > "CHI@TACC"
* log in if prompted to do so
* check the project drop-down menu near the top left (which shows e.g. "CHI-XXXXXX"), and make sure the correct project is selected.

* On the left side of the interface, expand the "Compute" menu
* Choose the "Instances" option
* Click the "Launch Instance" button 

You will be prompted to set up your instance step by step using a graphical "wizard".

* On the first ("Details") tab, set the instance name to  <code>node-mltrain-<b>netID</b></code> where in place of <code><b>netID</b></code> you substitute your own net ID (e.g. `ff524` in my case). 
* From the "Reservation" drop-down menu, select your reservation.
* Leave other settings at their default values, and click "Next".
* In the second ("Source") tab, we specify the source disk from which the instance should boot. In the "Select Boot Source" menu, choose "Image". Then, in the "Available" list at the bottom, search for `CC-Ubuntu24.04-hwe` (the default Ubuntu 24.04 disk image is incompatible with the MI100 GPU - this image has a Hardware Enablement kernel with which the GPU *is* compatible). Click the arrow next to this entry. You will see the `CC-Ubuntu24.04-hwe` image appear in the "Allocated" list. Click "Next".
* In the third ("Flavor") page, click "Next". (There are no "flavors" when provisioning a bare metal instance.)
* In the fourth ("Networks") tab, we will attach the instance to a network provided by the infrastructure provider which is connected to the Internet.
  * From the "Available" list, click on the arrow next to `sharednet1`. It will appear as item 1 in the "Allocated" list. 
  * Click "Next".
* In the fifth ("Ports") tab, click "Next".
* In the seventh ("Key Pair") tab, find the SSH key associated with your laptop on the "Available" list. Click on the arrow next to it to move it to the "Allocated" section. 
* Finally, click "Launch Instance" (the remaining tabs are not required).

Note: security groups are not used at Chameleon bare metal sites, so we do not have to configure any security groups on this instance.

You will see your instance appear in the list of compute instances. Within 10-20 minutes, it should go to the "Running" state. 

:::

::: {.cell .markdown}

Then, we'll associate a floating IP with the instance, so that we can access it over SSH.

* On the left side of the interface, expand the "Network" menu
* Choose the "Floating IPs" option
* Click "Allocate IP to project"
* In the "Pool" menu, choose "public"
* In the "Description" field, write: <code>MLTrain IP for <b>netID</b></code>, where in place of <code><b>netID</b></code> you use your own net ID.
* Click "Allocate IP"
* Then, choose "Associate" next to "your" IP in the list.
* In the "Port" menu, choose the port associated with your <code>node-mltrain-<b>netID</b></code> instance on the `shared1` network.
* Click "Associate".

Now, you should be able to access your instance over SSH! Test it now. From your local terminal, run

```
ssh -i ~/.ssh/id_rsa_chameleon cc@A.B.C.D
```

where

* in place of `~/.ssh/id_rsa_chameleon`, substitute the path to your own key that you had uploaded to CHI@TACC
* in place of `A.B.C.D`, use the floating IP address you just associated to your instance.

You will run the rest of the commands in this section inside your SSH session on "node-mltrain".

:::



::: {.cell .markdown}

## Retrieve code and notebooks on the instance

We'll start by retrieving the code and other materials on the instance.

```bash
# run on node-mltrain
git clone --recurse-submodules https://github.com/teaching-on-testbeds/mltrain-chi
```

:::


::: {.cell .markdown}


## Set up Docker

To use common deep learning frameworks like Tensorflow or PyTorch, and ML training platforms like MLFlow and Ray, we can run containers that have all the prerequisite libraries necessary for these frameworks. Here, we will set up the container framework.

```bash
# run on node-mltrain
curl -sSL https://get.docker.com/ | sudo sh
```

Then, give the `cc` user permission to run `docker` commands:

```bash
# run on node-mltrain
sudo groupadd -f docker; sudo usermod -aG docker $USER
```

After running this command, for the change in permissions to be effective, you must open a new SSH session - use `exit` and then reconnect. When you do, the output of `id` should show that you are a member of the `docker` group..

:::


::: {.cell .markdown}

### Set up the AMD GPU


Before we can use the AMD GPUs, we need to set up the driver using the `amdgpu-install` utility. 

Let's follow [AMD's instructions for setting up `amdgpu-install`](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/install-methods/amdgpu-installer/amdgpu-installer-ubuntu.html#installation):

```bash
# run on node-mltrain
sudo apt update
wget https://repo.radeon.com/amdgpu-install/6.3.3/ubuntu/noble/amdgpu-install_6.3.60303-1_all.deb
sudo apt -y install ./amdgpu-install_6.3.60303-1_all.deb
sudo apt update
```

To [run containers using ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html) (Radeon Open Compute Platform), an open-source software stack from AMD that allows users to program AMD GPUs (similar to NVIDIA's CUDA), we need to install the `amdgpu-dkms` driver:

```bash
# run on node-mltrain
amdgpu-install --usecase=dkms
```

And, we'll also install the `rocm-smi` utility, so that we can monitor the GPU from the host:

```bash
# run on node-mltrain
sudo apt -y install rocm-smi
```

Finally, we will add the `cc` user to the `video` and `render` groups, which are needed for access to the GPU:

```bash
# run on node-mltrain
sudo usermod -aG video,render $USER
```

That's all we will need on the host - the rest of ROCm will be installed inside the containers. 

To apply the changes to the kernel, we need to reboot:

```bash
# run on node-mltrain
sudo reboot
```

wait until the host comes back up, then reconnect using SSH. Run


```bash
# run on node-mltrain
rocm-smi
```

and verify that you can see the GPU(s).

We can also install `nvtop` to monitor GPU usage - we'll install from source, because the older version in the Ubuntu package repositories does not support AMD GPUs:

```bash
# run on node-mltrain
sudo apt -y install cmake libncurses-dev libsystemd-dev libudev-dev libdrm-dev libgtest-dev
git clone https://github.com/Syllo/nvtop
mkdir -p nvtop/build && cd nvtop/build && cmake .. -DAMDGPU_SUPPORT=ON && sudo make install
cd ~  # return to home directory
```

:::

::: {.cell .markdown}

###  Build a container image


Finally, we will build a container image in which to work, that has:

* a Jupyter notebook server
* Pytorch and Pytorch Lightning
* ROCm, which allows deep learning frameworks like Pytorch to use the AMD GPU accelerator
* and MLFlow

You can see our Dockerfile for this image at: [Dockerfile.jupyter-torch-mlflow-rocm](https://github.com/teaching-on-testbeds/mltrain-chi/tree/main/docker/Dockerfile.jupyter-torch-mlflow-rocm)


Building this container will take a **very long** time (ROCm is huge). But that's OK: we can get it started and then continue to the next section while it builds in the background, since we don't need this container immediately. We just need it to finish by the "Start a Jupyter server" subsection of the "Start the tracking server" section.

```bash
# run on node-mltrain
docker build -t jupyter-mlflow -f mltrain-chi/docker/Dockerfile.jupyter-torch-mlflow-rocm .
```

In the meantime, open another SSH session on "node-mltrain", so that you can continue with the next section.

:::
