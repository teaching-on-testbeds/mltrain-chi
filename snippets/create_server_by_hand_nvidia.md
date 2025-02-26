

## Launch and set up NVIDIA A100 40GB server - by hand

> **Note**: if you set up your NVIDIA A100 server using the `python-chi` notebook, you will skip this section! This section describes how to set up the server "by hand", in case you do not have access to the Chameleon Jupyter environment in which to run the `python-chi` notebook, or in case you prefer to do it "by hand".

At the beginning of the lease time, we will bring up our GPU server. We will use Horizon GUI at CHI@TACC to provision our server. 


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
* In the second ("Source") tab, we specify the source disk from which the instance should boot. In the "Select Boot Source" menu, choose "Image". Then, in the "Available" list at the bottom, search for `CC-Ubuntu24.04-CUDA`. Click the arrow next to this entry. You will see the `CC-Ubuntu24.04-CUDA` image appear in the "Allocated" list. Click "Next".
* In the third ("Flavor") page, click "Next". (There are no "flavors" when provisioning a bare metal instance.)
* In the fourth ("Networks") tab, we will attach the instance to a network provided by the infrastructure provider which is connected to the Internet.
  * From the "Available" list, click on the arrow next to `sharednet1`. It will appear as item 1 in the "Allocated" list. 
  * Click "Next".
* In the fifth ("Ports") tab, click "Next".
* In the seventh ("Key Pair") tab, find the SSH key associated with your laptop on the "Available" list. Click on the arrow next to it to move it to the "Allocated" section. 
* Finally, click "Launch Instance" (the remaining tabs are not required).

Note: security groups are not used at Chameleon bare metal sites, so we do not have to configure any security groups on this instance.

You will see your instance appear in the list of compute instances. Within 10-20 minutes, it should go to the "Running" state. 


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


## Retrieve code and notebooks on the instance

We'll start by retrieving the code and other materials on the instance.

```bash
# run on node-mltrain
git clone --recurse-submodules https://github.com/teaching-on-testbeds/mltrain-chi
```

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

## Set up the NVIDIA container toolkit


We will also install the NVIDIA container toolkit, with which we can access GPUs from inside our containers.

```bash
# run on node-mltrain
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  
sudo apt update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
# for https://github.com/NVIDIA/nvidia-container-toolkit/issues/48
sudo jq 'if has("exec-opts") then . else . + {"exec-opts": ["native.cgroupdriver=cgroupfs"]} end' /etc/docker/daemon.json | sudo tee /etc/docker/daemon.json.tmp > /dev/null && sudo mv /etc/docker/daemon.json.tmp /etc/docker/daemon.json
sudo systemctl restart docker
```

and we can install `nvtop` to monitor GPU usage:

```bash
# run on node-mltrain
sudo apt update
sudo apt -y install nvtop
```

###  Build a container image - for MLFlow section


Finally, we will build a container image in which to work in the MLFlow section, that has:

* a Jupyter notebook server
* Pytorch and Pytorch Lightning
* CUDA, which allows deep learning frameworks like Pytorch to use the NVIDIA GPU accelerator
* and MLFlow

You can see our Dockerfile for this image at: [Dockerfile.jupyter-torch-mlflow-cuda](https://github.com/teaching-on-testbeds/mltrain-chi/tree/main/docker/Dockerfile.jupyter-torch-mlflow-cuda)


Building this container may take a bit of time, but that's OK: we can get it started and then continue to the next section while it builds in the background, since we don't need this container immediately.

```bash
# run on node-mltrain
docker build -t jupyter-mlflow -f mltrain-chi/docker/Dockerfile.jupyter-torch-mlflow-cuda .
```

In the meantime, open another SSH session on "node-mltrain", so that you can continue with the next section.


