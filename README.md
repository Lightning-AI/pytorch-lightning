<div align="center">

# Lightning AI

______________________________________________________________________

<p align="center">
  <a href="https://lightning.ai/">Website</a> •
  <a href="https://lightning.ai/lightning-docs">Docs</a> •
  <a href="https://www.pytorchlightning.ai/community">Support</a>
</p>

</div>

## What is Lightning AI?

Lightning AI allows researchers, data scientists, and software engineers to build, share and iterate on highly scalable, production-ready AI apps using the tools and technologies of their choice, regardless of their expertise. To solve any kind of AI problem from research to deployment and production-ready pipelines, users can simply group components of their choice into a Lightning App and customize the underlying code as needed. Lightning Apps can then be republished back into the community for future use, or kept private in users’ personal libraries.

Lightning AI combines a wide variety of extant tools into a modular, intuitive platform for building AI applications in research, enterprise and personal contexts. It is the foundation of the growing Lightning ecosystem, which provides developers with a suite of ready-to-use tools and required infrastructure and compute resources, as well as community support for building AI applications.

The Lightning AI platform includes:

- The new Lightning framework, which extends PyTorch Lightning’s simple, modular, and flexible design principles to the entire app development process.
- A collection of tools and functionalities relevant to machine learning, including workflow scheduling for distributed computing, infrastructure-as-code, and connecting web UIs.
- A gallery of AI apps, curated by the Lightning team, which can be used instantly or further built upon.
- A library of components that add functionalities to users’ apps, such as extracting data from streaming video.
- A hosting platform for running and maintaining private and public AI apps on the cloud.
- The ability to build and run Lightning Apps on private cloud infrastructure or in an on-prem enterprise environment.

# Models

# Apps

## Turn any PyTorch Lightning model ito a Lightning App

Turn your PyTorch Lightning Script into a Lightning App with a cool UI for tracking your run, no code changes needed! Run on the cloud (including multi-GPU), add deployment components, build a research demo, plug-in feature stores, add notifications via Slack or text — and a ton more!

1. Create an app from your PyTorch Lightning script

```
cd path/to/your/source/code
lightning init pl-app path/to/your/script.py
```

2. Run the app locally

```
lightning run app pl-app/app.py
```

3. Run the app on the Public Lightning cloud

```
lightning run app pl-app/app.py --cloud
```
