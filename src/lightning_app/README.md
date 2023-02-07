<div align="center">

<img src="https://pl-flash-data.s3.amazonaws.com/brandmark.png" width="400px">

**With Lightning Apps, you build exactly what you need: from production-ready, multi-cloud ML systems to simple research demos.**

______________________________________________________________________

<p align="center">
  <a href="https://lightning.ai/">Website</a> •
  <a href="https://lightning.ai/lightning-docs">Docs</a> •
  <a href="#getting-started">Getting started</a> •
  <a href="#asking-for-help">Help</a> •
  <a href="https://www.pytorchlightning.ai/community">Slack</a>
</p>

![readme-gif](https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/lightning-gif-888777nslpiijdbcvctyvwhe.gif)

</div>

## From production-ready, multi-cloud ML systems to simple research demos.

Lightning Apps enable researchers, data scientists, and software engineers to build, share and iterate on highly scalable, complex AI workflows using the tools and technologies of their choice without any of the cloud boilerplate.

With Lightning Apps, your favorite components can work together on any machine at any scale.

# Getting started

## Install Lightning

<details>

<summary>Prerequisites</summary>

> TIP: We strongly recommend creating a virtual environment first.
> Don’t know what this is? Follow our [beginner guide here](https://lightning.ai/docs/stable/install/installation.html).

- Python 3.8.x or later (3.8.x, 3.9.x, 3.10.x, ...)
- Git
- Set up an alias for python=python3
- Add the root folder of Lightning to the Environment Variables to PATH
- (quick-start app requirement) Install Z shell (zsh)

</details>

```bash
pip install -U lightning
```

## Run your first Lightning App

1. Install a simple training and deployment app by typing:

```bash
lightning install app lightning/quick-start
```

2. If everything was successful, move into the new directory:

```bash
cd lightning-quick-start
```

3. Run the app locally

```bash
lightning run app app.py
```

4. Alternatively, run it on the public Lightning Cloud to share your app!

```bash
lightning run app app.py --cloud
```

[Read this guide](https://lightning.ai/docs/stable/levels/basic/) to learn the basics of Lightning Apps in 15 minutes.

# Features

Lightning Apps consist of a root [LightningFlow](https://lightning.ai/docs/stable/glossary/app_tree.html) component, that optionally contains a tree of 2 types of components: [LightningFlow](https://lightning.ai/lightning-docs/core_api/lightning_flow.html) 🌊 and [LightningWork](https://lightning.ai/lightning-docs/core_api/lightning_work/) ⚒️. Key functionality includes:

- A shared state between components.
- A constantly running event loop for reactivity.
- Dynamic attachment of components at runtime.
- Start and stop functionality of your works.

Lightning Apps can run [locally](https://lightning.ai/lightning-docs/workflows/run_on_private_cloud.html) 💻 or [on the cloud](https://lightning.ai/lightning-docs/core_api/lightning_work/compute.html) 🌩️.

Easy communication 🛰️ between components is supported with:

- [Directional state updates](https://lightning.ai/lightning-docs/core_api/lightning_app/communication.html?highlight=directional%20state) from the Works to the Flow creating an event: When creating interactive apps, you will likely want your components to share information with each other. You might to rely on that information to control their execution, share progress in the UI, trigger a sequence of operations, or more.
- [Storage](https://lightning.ai/lightning-docs/api_reference/storage.html): The Lightning Storage system makes it easy to share files between LightningWork so you can run your app both locally and in the cloud without changing the code.
  - [Path](https://lightning.ai/lightning-docs/api_reference/generated/lightning_app.storage.path.Path.html): The Path object is a reference to a specific file or directory from a LightningWork and can be used to transfer those files to another LightningWork (one way, from source to destination).
  - [Payload](https://lightning.ai/lightning-docs/api_reference/generated/lightning_app.storage.payload.Payload.html): The Payload object enables transferring of Python objects from one work to another in a similar fashion as Path.
  - [Drive](https://lightning.ai/lightning-docs/api_reference/generated/lightning_app.storage.drive.Drive.html): The Drive object provides a central place for your components to share data. The drive acts as an isolated folder and any component can access it by knowing its name.

Lightning Apps have built-in support for [adding UIs](https://lightning.ai/lightning-docs/workflows/add_web_ui/) 🎨:

- [StaticWebFrontEnd](https://lightning.ai/lightning-docs/api_reference/generated/lightning_app.frontend.web.StaticWebFrontend.html): A frontend that serves static files from a directory using FastAPI.
- [StreamlitFrontend](https://lightning.ai/lightning-docs/api_reference/generated/lightning_app.frontend.stream_lit.StreamlitFrontend.html): A frontend for wrapping Streamlit code in your LightingFlow.
- [ServeGradio](https://lightning.ai/docs/stable/api_reference/generated/lightning_app.components.serve.gradio_server.ServeGradio.html): This class enables you to quickly create a `gradio` based UI for your Lightning App.

[Scheduling](https://lightning.ai/lightning-docs/glossary/scheduling.html) ⏲️: The Lightning Scheduling system makes it easy to schedule your components execution with any arbitrary conditions.

Advanced users who need full control over the environment a LightningWork runs in can [specify a custom Docker image](https://lightning.ai/lightning-docs/glossary/build_config/build_config_advanced.html?highlight=docker) 🐋 that will be deployed in the cloud.

[Environment variables](https://lightning.ai/lightning-docs/glossary/environment_variables.html?highlight=environment%20variables) 💬: If your app is using secrets or values, such as API keys or access tokens, use environment variables to avoid sticking them in the source code.

Ready to use [built-in components](https://lightning.ai/lightning-docs/api_reference/components.html?highlight=built%20components) 🧱:

- [PopenPythonScript](https://lightning.ai/lightning-docs/api_reference/generated/lightning_app.components.python.popen.PopenPythonScript.html#lightning_app.components.python.popen.PopenPythonScript): This class enables you to easily run a Python Script.
- [ModelInferenceAPI](https://lightning.ai/lightning-docs/api_reference/generated/lightning_app.components.serve.serve.ModelInferenceAPI.html#lightning_app.components.serve.serve.ModelInferenceAPI): This class enables you to easily get your model served.

# App gallery

The [Lightning AI website](https://lightning.ai/) features a curated gallery of Lightning Apps and components that makes it easy to get started. A few highlights:

| App                            | Description                                                                                                                                                                                                  |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Train & Demo PyTorch Lightning | Train a model using PyTorch Lightning and deploy it to an interactive demo. Use this Lightning App as a starting point for building more complex apps around your models.                                    |
| Lightning Sweeper              | Run a hyperparameter sweep over any model script across hundreds of cloud machines at once. This Lightning App uses Optuna to provide advanced tuning algorithms (from grid and random search to Hyperband). |
| Flashy                         | Flashy, the auto-AI Lightning App, selects the best deep learning model for your image or text datasets. It automatically uses state-of-the-art models from Torchision, TIMM and Hugging Face.               |

## Current limitations

- Lightning requires Python 3.8.x or later (3.8.x, 3.9.x, 3.10.x).
- For now, you can only run a single app locally at a time.
- You are required to install the Lightning App requirements locally, even when starting the app on the cloud.
- Multiple works cannot share the same machine.
- To run on the cloud, you will need access to a browser.
- Frontends only support the HTTP protocol. TCP support is coming in the future.
- App Flow Frontends cannot be changed after startup, but you the layout can be updated reactively.
- Authentication is not supported.

## Asking for help

If you have any questions please:

1. [Read the docs](https://lightning.ai/lightning-docs/).
1. [Search through existing Discussions](https://github.com/Lightning-ai/lightning/discussions), or [add a new question](https://github.com/Lightning-ai/lightning/discussions/new)
1. [Join our Slack community](https://www.pytorchlightning.ai/community).
