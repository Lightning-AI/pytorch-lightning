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

# Machine maintenance

As most of our jobs/checks are running in a docker container the OS/machine get polluted soon d fails to run anymore with error such as

```
No space left on device : '/azp/agent-litGPU-21_0,1/_diag/pages/8bb191f4-a8c2-419a-8788-66e3f0522bea_1.log'
```

in such case you need to login to the machine and run `docker system prune`

## Automated ways

Let's explore adding cron for periodical nuke all docker caches:

1. Open your user's cron tab for editing: `crontab -e`
1. Schedule/add the command with `--force` flag to force the pruning without interactive confirmation:
   ```bash
   # every day at 2:00 AM clean docker caches
   0 2 * * * docker system prune --force
   ```
1. Verify the Entry: `crontab -l`

Note, you may need yourself to the docker group `sudo usermod -aG docker <your_username>` to have permission to execute this command without needing `sudo` and inserting password.
