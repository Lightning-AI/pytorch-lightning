import { toPath } from "lodash";

import { Stack, Typography } from "lightning-ui/src/design-system/components";

import ProgressBar from "./ProgressBar";

export default function ProgressBarGroup(props: any) {
  const trainer_progress = props.trainer_progress;
  const trainer_state = props.trainer_state;

  const primary_bar_title = "Training";
  var secondary_bar_title = "";
  var current = 0;
  var total = 0;

  switch (trainer_state?.stage) {
    case "validating":
      secondary_bar_title = "Validation";
      current = trainer_progress?.fit?.val_batch_idx;
      total = trainer_progress?.fit?.total_val_batches;
      break;
    case "testing":
      secondary_bar_title = "Test";
      current = trainer_progress?.test?.test_batch_idx;
      total = trainer_progress?.test?.total_test_batches;
      break;
    case "predicting":
      secondary_bar_title = "Prediction";
      current = trainer_progress?.predict?.predict_batch_idx;
      total = trainer_progress?.predict?.total_predict_batches;
      break;
    default:
      secondary_bar_title = "Hidden";
  }

  return (
    <Stack>
      <Typography>{primary_bar_title}</Typography>
      <ProgressBar
        current={trainer_progress?.fit?.train_batch_idx}
        total={trainer_progress?.fit?.total_train_batches}
      />
      <div style={secondary_bar_title == "Hidden" ? { visibility: "hidden" } : {}}>
        <Typography>{secondary_bar_title}</Typography>
        <ProgressBar current={current} total={total} />
      </div>
    </Stack>
  );
}
