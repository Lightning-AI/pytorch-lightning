###############
Troubleshooting
###############

Learn how to troubleshoot possible causes for common issues related to CUDA, NCCL, and distributed training.


----


*********
Multi-GPU
*********

If your program is stuck at

.. code-block::

    Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/4

it indicates that PyTorch can't set up the communication between GPUs, and that your system is not configured correctly.
Run the `diagnose` command from the Fabric CLI to investigate:

.. code-block:: bash

    fabric diagnose

This tool will run basic multi-GPU tests using only PyTorch.
Any issues raised here will confirm that the problem is with your system and not with Lightning.
Common solutions:

- **Wrong driver version:** The NVIDIA driver for your GPU is too old or too new.
  You can check the version of the driver by running

  .. code-block:: bash

      nvidia-smi --id=0 --query-gpu=driver_version --format=csv,noheader

  *Solution*: Install a recent driver.
  Search online for instructions how to update the driver on your platform.

- **Peer-to-peer connection is broken:** The GPUs can't communicate with each other.
  *Solution*: Try to set the environment variable ``NCCL_P2P_DISABLE=1``.
  If you rerun your scipt and it fixes the problem, this means that peer-to-peer transport is not working properly (your training will run but it will be slow).
  This is likely because of driver compatibility issues (see above) or because your GPU does not support peer-to-peer (e.g., certain RTX cards).


----


**********
Multi-node
**********

Before troubleshooting multi-node connectivity issues, first ensure that multi-GPU within a single machine is working correctly by following the steps above.
If single-node execution works, but multi-node hangs at

.. code-block::

    Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/4

it indicates that there is a connection issue between the nodes.
Common solutions:

- **Wrong network interface:** Some servers have multiple network interfaces.
  There is usually only one that can send and receive traffic from the network of the other nodes, but sometimes it is not set as the default.
  In this case, you need to set it manually:

  .. code-block:: bash

    export GLOO_SOCKET_IFNAME=eno1
    export NCCL_SOCKET_IFNAME=eno1
    fabric run ...

  You can find the interface name by parsing the output of the ``ifconfig`` command.
  The name of this interface **may differ on each node**.

- **NCCL can't communicate between the nodes:**

  Follow the steps in the `NCCL troubleshooting guide <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html>`_.
  In particular, take note of the network section that describes restricting the port range and firewall rules.

  .. code-block:: bash

      echo "net.ipv4.ip_local_port_range = 50000 51000" >> /etc/sysctl.conf
      sysctl --system
      ufw allow 50000:51000/tcp
