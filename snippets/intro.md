
::: {.cell .markdown}

# Model training infrastructure and platforms

In this tutorial, we explore some of the infrastructure and platform requirements for large model training, and to support the training of many models by many teams. We focus specifically on 

* experiment tracking (using [MLFlow](https://mlflow.org/))
* and scheduling training jobs on a GPU cluster (using [Ray](https://www.ray.io/))

To run this experiment, you should have already created an account on Chameleon, and become part of a project. You must also have added your SSH key to the CHI@TACC site.

:::

::: {.cell .markdown}

## Experiment resources 

For this experiment, we will provision one bare-metal node with GPUs. 

The MLFlow section is more interesting if we run it on a node with two GPUs, because then we can better understand how to configure logging in a distributed training run. But, if need be, we can run it on a node with one GPU.

The Ray section requires a node with two GPUs.

We can browse Chameleon hardware configurations for suitable node types using the [Hardware Browser](https://chameleoncloud.org/hardware/). For example, to find nodes with 2x GPUs: if we expand "Advanced Filters", check the "2" box under "GPU count", and then click "View", we can identify some suitable node types. 

We'll proceed with the `gpu_mi100` and `compute_liqid` node types at CHI@TACC.

* Most of the `gpu_mi100` nodes have two AMD MI100 GPUs. (One of the `gpu_mi100` nodes, `c03-04` has only one GPU; we'll avoid this one for the "Ray" section, which requires two GPUs.)
* The `compute_liqid` nodes at CHI@TACC have one or two NVIDIA A100 40GB GPUs. As of this writing, `liqid01` and `liqid02` have two GPUs, so we may use these two for the "Ray" section, which requires two GPUs. 

You can decide which type to use based on availability; but once you decide, make sure to follow the instructions specific to that GPU type. In some parts, there will be different instructions for setting up an AMD GPU node vs. and NVIDIA GPU node.


:::

::: {.cell .markdown}

## Create a lease

:::

::: {.cell .markdown}

To use bare metal resources on Chameleon, we must reserve them in advance. We can reserve two separate 3-hour blocks for this experiment: one for the MLFlow section and one for the Ray section. They are designed to run independently.

We can use the OpenStack graphical user interface, Horizon, to submit a lease for an MI100 or Liquid node at CHI@TACC. To access this interface,

* from the [Chameleon website](https://chameleoncloud.org/hardware/)
* click "Experiment" > "CHI@TACC"
* log in if prompted to do so
* check the project drop-down menu near the top left (which shows e.g. “CHI-XXXXXX”), and make sure the correct project is selected.

:::

::: {.cell .markdown}

Then, 

* On the left side, click on "Reservations" > "Leases", and then click on "Host Calendar". In the "Node type" drop down menu, change the type to `gpu_mi100` or `compute_liqid` to see the schedule of availability. You may change the date range setting to "30 days" to see a longer time scale. Note that the dates and times in this display are in UTC. You can use [WolframAlpha](https://www.wolframalpha.com/) or equivalent to convert to your local time zone. 
* Once you have identified an available three-hour block in UTC time that works for you in your local time zone, make a note of:
  * the start and end time of the time you will try to reserve. (Note that if you mouse over an existing reservation, a pop up will show you the exact start and end time of that reservation.)
  * and the name of the node you want to reserve. (We will reserve nodes by name, not by type, to avoid getting a 1-GPU node when we wanted a 2-GPU node.)
* Then, on the left side, click on "Reservations" > "Leases", and then click on "Create Lease":
  * set the "Name" to <code>mltrain_<b>netID</b></code> where in place of <code><b>netID</b></code> you substitute your actual net ID.
  * set the start date and time in UTC. To make scheduling smoother, please start your lease on an even hour boundary, e.g. `XX:00` where `XX` is an even number.
  * modify the lease length (in days) until the end date is correct. Then, set the end time. To be mindful of other users, you should limit your lease time to three hours as directed. Also, to avoid a potential race condition that occurs when one lease starts immediately after another lease ends, you should end your lease five minutes before the end of an hour, e.g. at `YY:55`.
  * Click "Next".
* On the "Hosts" tab, 
  * check the "Reserve hosts" box
  * leave the "Minimum number of hosts" and "Maximum number of hosts" at 1
  * in "Resource properties", specify the node name that you identified earlier.
* Click "Next". Then, click "Create". (We won't include any network resources in this lease.)
  
Your lease status should show as "Pending". Click on the lease to see an overview. It will show the start time and end time, and it will show the name of the physical host that is reserved for you as part of your lease. Make sure that the lease details are correct.

:::

::: {.cell .markdown}

Since you will need the full lease time to actually execute your experiment, you should read *all* of the experiment material ahead of time in preparation, so that you make the best possible use of your time.

:::

::: {.cell .markdown}

At the beginning of your lease time, you will continue with the next step, in which you bring up and configure a bare metal instance! Two alternate sets of instructions are provided for this part:

* a notebook that runs in the Chameleon Jupyter interface. This automates the setup process, so that you can "Run > Run Selected Cell and All Below" to let the setup mostly run without human intervention.
* or, instructions for using the Horizon GUI and an SSH session, in case you cannot or prefer not to use the Chameleon Jupyter interface

:::
