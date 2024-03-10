###############
Troubleshooting
###############


----


*********
Multi-GPU
*********


**My program is stuck initializing at startup. What is causing this?**

You are seeing a message like this in the logs, but nothing happens:

.. code-block::

    Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/4

The most likely reasons and how to fix it:


.. code-block:: bash

    fabric diagnose


----


**********
Multi-node
**********


**My program is stuck initializing at startup. What is causing this?**

You are seeing a message like this in the logs, but nothing happens:

.. code-block::

    Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/4

The most likely reasons and how to fix it:

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


**My program crashes with an NCCL error, but it is not helpful**

Launch your command by prepending ``NCCL_DEBUG=INFO`` to get more info.

.. code-block:: bash

    NCCL_DEBUG=INFO fabric run ...
