
::: {.cell .markdown}

# Train ML models with Ray

In this tutorial, we explore some of the infrastructure and platform requirements for large model training, and to support the training of many models by many teams. We focus specifically on scheduling training jobs on a GPU cluster (using [Ray](https://www.ray.io/)).

To run this experiment, you should have already created an account on Chameleon, and become part of a project. You must also have added your SSH key to the CHI@TACC site.

:::

::: {.cell .markdown}

## Experiment resources

For this experiment, we will provision one bare-metal node with GPUs.

This lab requires a node with two GPUs.

We can browse Chameleon hardware configurations for suitable node types using the [Hardware Browser](https://chameleoncloud.org/hardware/). For example, to find nodes with 2x GPUs: if we expand "Advanced Filters", check the "2" box under "GPU count", and then click "View", we can identify some suitable node types.

:::

::: {.cell .markdown .gpu-amd}

For AMD GPUs, we will use the `gpu_mi100` node type at CHI@TACC.

Most of the `gpu_mi100` nodes have two AMD MI100 GPUs. (One of the `gpu_mi100` nodes, `c03-04`, has only one GPU; avoid this one.)

:::

::: {.cell .markdown .gpu-nvidia}

For NVIDIA GPUs, we will use the `compute_liqid` node type at CHI@TACC.

The `compute_liqid` nodes at CHI@TACC have one or two NVIDIA A100 40GB GPUs. As of this writing, `liqid01` and `liqid02` have two GPUs and are suitable for this lab.

:::

::: {.cell .markdown}

Once you decide which GPU type to use, continue with `1_create_lease.ipynb`.

:::
