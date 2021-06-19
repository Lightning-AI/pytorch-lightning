local base = import 'templates/base.libsonnet';
local metrics = import 'templates/metrics.libsonnet';

{
  local CloudAcceleratorTest = base.BaseTest {
    local config = self,

    publisherImage: 'gcr.io/xl-ml-test/publisher:stable',

    tpuSettings+: {
      requireTpuAvailableLabel: true,
    },

    metricConfig:
      metrics.CompatMetrics(config.metricCollectionConfig, config.regressionTestConfig),

    // Add experimental TPU health monitor to Job.
    podTemplate+:: {
      spec+: {
        containerMap+: if config.accelerator.type == 'tpu' then
          {
            monitor: {
              name: 'monitor',
              image: 'gcr.io/xl-ml-test/health-monitor:stable',
              imagePullPolicy: 'Always',
              env: [
                {
                  name: 'POD_NAME',
                  valueFrom: {
                    fieldRef: {
                      fieldPath: 'metadata.name',
                    },
                  },
                },
                {
                  name: 'POD_NAMESPACE',
                  valueFrom: {
                    fieldRef: {
                      fieldPath: 'metadata.namespace',
                    },
                  },
                },
              ],
            },
          }
        else {},
      } + if config.accelerator.type == 'gpu' then {
        priorityClassName: 'gpu-%(version)s' % config.accelerator,
      } else if config.accelerator.type == 'tpu' && std.endsWith(config.testName, '-1vm') then {
        priorityClassName: if config.accelerator.replicas == 1 then
          'tpu-device' else 'tpu-pod',
      } else {},
    },

    cronJob+:: {
      metadata+: {
        namespace: 'automated',
      },
    },
  },
  local PyTorchBaseTest = CloudAcceleratorTest {
    configMaps+: ['pytorch-nfs-ip'],
    regressionTestConfig+: {
      metric_subset_to_alert: [
        'ExecuteTime__Percentile_99_sec_final',
        'total_wall_time',
        'Accuracy/test_final',
        'aten_ops_sum_final',
      ],
      metric_success_conditions+: {
        ExecuteTime__Percentile_99_sec_final: {
          success_threshold: {
            stddevs_from_mean: 5.0,
          },
          comparison: 'less',
          wait_for_n_points_of_history: 10,
        },
        aten_ops_sum_final: {
          success_threshold: {
            stddevs_from_mean: 0.0,
          },
          comparison: 'less_or_equal',
        },
      },
    },

    metricCollectionConfig+: {
      tags_to_ignore: ['LearningRate'],
    },
  },
  PyTorchTest:: PyTorchBaseTest {
    local config = self,

    image: 'gcr.io/xl-ml-test/pytorch-xla',
    volumeMap+: {
      dshm: volumes.MemoryVolumeSpec {
        name: 'dshm',
        mountPath: '/dev/shm',
      },
    },

    cpu: '4.5',
    memory: '8Gi',

    podTemplate+:: {
      spec+: {
        containerMap+: {
          train+: {
            envMap+: {
              XLA_USE_BF16: '0',
            },
          },
        },
      },
    },
  },
}
