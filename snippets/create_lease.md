
::: {.cell .markdown}

## Create a lease

To use bare metal resources on Chameleon, we must reserve them in advance. We can reserve a 3-hour block for this experiment.

We can use the OpenStack graphical user interface, Horizon, to submit a lease at CHI@TACC. To access this interface,

* from the [Chameleon website](https://chameleoncloud.org/hardware/)
* click "Experiment" > "CHI@TACC"
* log in if prompted to do so
* check the project drop-down menu near the top left (which shows e.g. "CHI-XXXXXX"), and make sure the correct project is selected.

:::

::: {.cell .markdown .gpu-amd}

Then,

* On the left side, click on "Reservations" > "Leases", and then click on "Host Calendar". In the "Node type" drop down menu, change the type to `gpu_mi100` to see the schedule of availability. You may change the date range setting to "30 days" to see a longer time scale. Note that the dates and times in this display are in UTC. You can use [WolframAlpha](https://www.wolframalpha.com/) or equivalent to convert to your local time zone.
* Once you have identified an available three-hour block in UTC time that works for you in your local time zone, make a note of:
  * the start and end time of the time you will try to reserve. (Note that if you mouse over an existing reservation, a pop up will show you the exact start and end time of that reservation.)
  * and the name of the node you want to reserve. (We will reserve nodes by name, not by type, to avoid getting a 1-GPU node when we wanted a 2-GPU node.)
* Then, on the left side, click on "Reservations" > "Leases", and then click on "Create Lease":
  * set the "Name" to `mltrain_netID`, where `netID` is your actual net ID.
  * set the start date and time in UTC. To make scheduling smoother, please start your lease on an hour boundary, e.g. `XX:00`.
  * modify the lease length (in days) until the end date is correct. Then, set the end time. To be mindful of other users, you should limit your lease time to three hours as directed. Also, to avoid a potential race condition that occurs when one lease starts immediately after another lease ends, you should end your lease ten minutes before the end of an hour, e.g. at `YY:50`.
  * Click "Next".
* On the "Hosts" tab,
  * check the "Reserve hosts" box
  * leave the "Minimum number of hosts" and "Maximum number of hosts" at 1
  * in "Resource properties", specify the node name that you identified earlier.
* Click "Next". Then, click "Create". (We will not include any network resources in this lease.)

Your lease status should show as "Pending". Click on the lease to see an overview. It will show the start time and end time, and it will show the name of the physical host that is reserved for you as part of your lease. Make sure that the lease details are correct.

:::

::: {.cell .markdown .gpu-nvidia}

Then,

* On the left side, click on "Reservations" > "Leases", and then click on "Host Calendar". In the "Node type" drop down menu, change the type to `compute_liqid` to see the schedule of availability. You may change the date range setting to "30 days" to see a longer time scale. Note that the dates and times in this display are in UTC. You can use [WolframAlpha](https://www.wolframalpha.com/) or equivalent to convert to your local time zone.
* Once you have identified an available three-hour block in UTC time that works for you in your local time zone, make a note of:
  * the start and end time of the time you will try to reserve. (Note that if you mouse over an existing reservation, a pop up will show you the exact start and end time of that reservation.)
  * and the name of the node you want to reserve. (We will reserve nodes by name, not by type, to avoid getting a 1-GPU node when we wanted a 2-GPU node.)
* Then, on the left side, click on "Reservations" > "Leases", and then click on "Create Lease":
  * set the "Name" to `mltrain_netID`, where `netID` is your actual net ID.
  * set the start date and time in UTC. To make scheduling smoother, please start your lease on an hour boundary, e.g. `XX:00`.
  * modify the lease length (in days) until the end date is correct. Then, set the end time. To be mindful of other users, you should limit your lease time to three hours as directed. Also, to avoid a potential race condition that occurs when one lease starts immediately after another lease ends, you should end your lease ten minutes before the end of an hour, e.g. at `YY:50`.
  * Click "Next".
* On the "Hosts" tab,
  * check the "Reserve hosts" box
  * leave the "Minimum number of hosts" and "Maximum number of hosts" at 1
  * in "Resource properties", specify the node name that you identified earlier.
* Click "Next". Then, click "Create". (We will not include any network resources in this lease.)

Your lease status should show as "Pending". Click on the lease to see an overview. It will show the start time and end time, and it will show the name of the physical host that is reserved for you as part of your lease. Make sure that the lease details are correct.

:::

::: {.cell .markdown}

## Open this experiment on Trovi

Since you will need the full lease time to execute your experiment, you should read *all* of the experiment material ahead of time in preparation, so that you make the best possible use of your time.

When you are ready to begin, open this experiment on Trovi:

:::

::: {.cell .markdown .gpu-nvidia}

* Use this link: [Train ML models with Ray (NVIDIA)](https://trovi.chameleoncloud.org/dashboard/artifacts/b4578c82-e84a-4353-83d2-fbecf153eefd) on Trovi

:::

::: {.cell .markdown .gpu-amd}

* Use this link: [Train ML models with Ray (AMD)](https://trovi.chameleoncloud.org/dashboard/artifacts/d48d7684-cf6d-4c33-bcd6-5504266bc3d4) on Trovi

:::

::: {.cell .markdown}

* Then, click "Launch on Chameleon". This will start a new Jupyter server for you, with the experiment materials already in it.

At the beginning of your lease time, inside the `mltrain-chi` directory, continue with `2_create_server.ipynb`.

:::
