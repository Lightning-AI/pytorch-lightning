local base_tpuvm = import './base_tpuvm.libsonnet';

{
  PyTorchTpuVmMixin:: base_tpuvm.BaseTpuVmMixin {
    local config = self,
    podTemplate+:: {
      spec+: {
        containerMap+:: {
          monitor: null,
          train+: {
            local scriptSettings = {
              testCommand:
                std.join(
                  ' ',
                  config.command,
                ),
            },
            args: null,
            // Tests are structured as bash scripts that run directly
            // on the Cloud TPU VM instead of using docker images.
            command: [
              'bash',
              '-c',
              |||
                set -x
                set -u
                ssh -i scripts/id_rsa -o StrictHostKeyChecking=no xl-ml-test@$(cat /scripts/tpu_ip) \
                  'sudo apt-get -y update && sudo apt-get -y install nfs-common git google-perftools'
                ssh -i scripts/id_rsa -o StrictHostKeyChecking=no xl-ml-test@$(cat /scripts/tpu_ip) \
                  'sudo mkdir /datasets && sudo mount $(PYTORCH_DATA_LOCATION) /datasets'
                ssh -i scripts/id_rsa -o StrictHostKeyChecking=no xl-ml-test@$(cat /scripts/tpu_ip) << 'TEST_SCRIPT_EOF'
                  export XRT_TPU_CONFIG='localservice;0;localhost:51011'
                  export LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4'
                  %(testCommand)s
                TEST_SCRIPT_EOF
                exit_code=$?
                bash /scripts/cleanup.sh
                exit $exit_code
              ||| % scriptSettings,
            ],
          },
        },
      },
    },
  },
}
