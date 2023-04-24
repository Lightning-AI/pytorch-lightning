import Typography from "@mui/material/Typography";

import Timer from "components/Timer";
import { Grid, Stack } from "lightning-ui/src/design-system/components";

export default function ExecutionSummary(props: any) {
  const trainer_progress = props?.trainer_progress;
  const trainer_state = props?.trainer_state;
  const script_running = props?.script_running;

  const global_step = trainer_progress?.fit?.global_step || 0;
  const estimated_stepping_batches = trainer_progress?.fit?.estimated_stepping_batches || Infinity;
  const total_estimated_progress = Math.round((100 * global_step) / estimated_stepping_batches);

  return (
    <Stack direction="row" spacing={2} justifyContent={"space-between"} width={"100%"}>
      <Grid item>
        <Stack spacing={0} marginTop={0} marginBottom={0}>
          <Grid item margin="0">
            <Typography m={1}>Duration</Typography>
          </Grid>
          <Grid item>
            <Typography m={1} variant="h5">
              <Timer isActive={script_running} />
            </Typography>
          </Grid>
        </Stack>
      </Grid>
      <Grid item>
        <Stack spacing={0} marginTop={0} marginBottom={0}>
          <Grid item margin="0">
            <Typography m={1}>Stage</Typography>
          </Grid>
          <Grid item>
            <Typography m={1} variant="h5" sx={{ textTransform: "uppercase" }}>
              {trainer_state?.stage != undefined ? trainer_state?.stage : "-"}
            </Typography>
          </Grid>
        </Stack>
      </Grid>
      <Grid item>
        <Stack spacing={0} marginTop={0} marginBottom={0}>
          <Grid item margin="0">
            <Typography m={1}>Epoch</Typography>
          </Grid>
          <Grid item>
            <Typography m={1} variant="h5">
              {trainer_progress?.fit?.current_epoch != undefined ? trainer_progress?.fit?.current_epoch : "-"}
            </Typography>
          </Grid>
        </Stack>
      </Grid>
      <Grid item>
        <Stack spacing={0} marginTop={0} marginBottom={0}>
          <Grid item margin="0">
            <Typography m={1}>Batch</Typography>
          </Grid>
          <Grid item>
            <Typography m={1} variant="h5">
              {trainer_progress?.fit?.train_batch_idx != undefined ? trainer_progress?.fit?.train_batch_idx : "-"}
            </Typography>
          </Grid>
        </Stack>
      </Grid>
      <Grid item>
        <Stack spacing={0} marginTop={0} marginBottom={0}>
          <Grid item margin="0">
            <Typography m={1}>Total Progress</Typography>
          </Grid>
          <Grid item>
            <Typography m={1} variant="h5">
              {trainer_progress?.fit != undefined ? total_estimated_progress + "%" : "-"}
            </Typography>
          </Grid>
        </Stack>
      </Grid>
    </Stack>
  );
}
