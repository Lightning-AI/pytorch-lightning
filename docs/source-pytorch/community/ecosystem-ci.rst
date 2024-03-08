Ecosystem CI
############

`Ecosystem CI <https://github.com/Lightning-AI/ecosystem-ci>`_ automates issue discovery for your projects against Lightning nightly and releases.
It is a lightweight repository that provides easy configuration of Continues Integration running on CPUs and GPUs.
Any user who wants to keep their project aligned with current and future Lightning releases can use the EcoSystem CI to configure their integrations.
Read more: `Stay Ahead of Breaking Changes with the New Lightning Ecosystem CI <https://devblog.pytorchlightning.ai/stay-ahead-of-breaking-changes-with-the-new-lightning-ecosystem-ci-b7e1cf78a6c7>`_


----


***********************
Integrate a New Project
***********************

Follow the instructions below to add a new project to the PyTorch Lightning ecosystem.

1. Fork the ecosystem CI repository to be able to create a `new Pull Request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork>`_ and work within a specific branch.
2. Create a new config file in ``configs/<Organization-name>`` folder and call it ``<project-name>.yaml``.
3. Define runtime for CPU and link the config for GPU:
   For CPU integrations, list OS and Python version combination to be running with GitHub actions.
   For GPU integrations, you only add the path to the config (OS/Linux and Python version is fixed) to be running with Azure pipelines.
4. Add a Contact to the ``.github/CODEOWNERS`` list for your organization folder or just a single project.
5. Create a Draft PR with all mentioned requirements.
6. Join our `Discord <https://discord.gg/VptPCZkGNa>`_ (Optional) channel ``#alerts-ecosystem-ci`` to be notified if your project is breaking.

To learn more about Ecosystem CI, please refer to the `Ecosystem CI repo <https://github.com/Lightning-AI/ecosystem-ci>`_.
Also, note that some particular implementation details described above may evolve over time.
