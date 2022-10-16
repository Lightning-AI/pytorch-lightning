# Creation GPU self-hosted agent pool

## Prepare the machine

This is a slightly modified version of the script from
https://docs.microsoft.com/en-us/azure/devops/pipelines/agents/docker

```bash
apt-get update
apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    jq \
    git \
    iputils-ping \
    libcurl4 \
    libunwind8 \
    netcat \
    libssl1.0

curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
mkdir /azp
```

## Stating the agents

```bash
export TARGETARCH=linux-x64
export AZP_URL="https://dev.azure.com/Lightning-AI"
export AZP_TOKEN="xxxxxxxxxxxxxxxxxxxxxxxxxx"
export AZP_POOL="lit-rtx-3090"

for i in {0..7..2}
do
     nohup bash .azure/start.sh \
        "AZP_AGENT_NAME=litGPU-YX_$i,$((i+1))" \
        "CUDA_VISIBLE_DEVICES=$i,$((i+1))" \
     > "agent-$i.log" &
done
```

## Check running agents

```bash
ps aux | grep start.sh
```
