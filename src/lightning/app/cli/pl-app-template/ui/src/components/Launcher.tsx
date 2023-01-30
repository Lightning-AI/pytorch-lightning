import { useState } from "react";

import PlayCircleFilledWhiteOutlinedIcon from "@mui/icons-material/PlayCircleFilledWhiteOutlined";
import cloneDeep from "lodash/cloneDeep";

import { useLightningState } from "hooks/useLightningState";
import {
  Banner,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Label,
  Select,
  Stack,
  TextField,
  Typography,
  useSnackbar,
} from "lightning-ui/src/design-system/components";

import EnvironmentConfigurator, { data2dict } from "./EnvironmentConfigurator";

export default function Launcher(props: any) {
  // We had to pass the updateLightningState as props because accessing them through useLightningState does not work
  const { updateLightningState } = useLightningState();
  const [hardwareType, setHardwareType] = useState("cpu-small");
  const [environmentVariables, setEnvironmentVariables] = useState([{ name: "", value: "" }]);
  const [showHardwareDialog, setShowHardwareDialog] = useState(false);
  const { enqueueSnackbar } = useSnackbar();
  const [scriptArgs, setScriptArgs] = useState<string>("");

  let status_text = "";
  let status_color: "default" | "primary" | "success" | "error" | "warning" | "purple" = "default";
  if (props.script_running) {
    status_text = "Running";
    status_color = "success";
  } else if (props.start_triggered && !props.script_running) {
    status_text = "Starting";
    status_color = "warning";
  } else if (props.script_succeeded) {
    status_text = "Finished";
    status_color = "success";
  } else if (props.script_failed) {
    status_text = "Failed";
    status_color = "error";
  }

  const onStartClick = async () => {
    setShowHardwareDialog(true);
  };

  const onHardwareDialogCancelClick = () => {
    setShowHardwareDialog(false);
  };

  const onHardwareDialogConfirmClick = () => {
    setShowHardwareDialog(false);
    enqueueSnackbar({
      title: "Hardware request sent",
      severity: "info",
      children: "Your script will start once the hardware is ready.",
    });
    if (props.lightningState) {
      const newLightningState = cloneDeep(props.lightningState);
      newLightningState.flows.script_orchestrator.vars.triggered =
        !newLightningState.flows.script_orchestrator.vars.triggered;

      newLightningState.flows.script_orchestrator.vars.cloud_compute_args = {
        name: hardwareType,
      };
      newLightningState.flows.script_orchestrator.vars.environment_variables = data2dict(environmentVariables);
      newLightningState.flows.script_orchestrator.vars.script_args =
        scriptArgs.length > 0 ? scriptArgs.trim().split(/[ ]+/) : [];
      updateLightningState(newLightningState);
    }
  };

  const handleHardwareTypeChange = (new_value: any) => {
    setHardwareType(new_value);
  };

  const handleScriptArgsChange = (new_value: string | null) => {
    if (new_value !== null) {
      setScriptArgs(new_value);
    }
  };

  return (
    <div>
      <Stack direction="row" spacing={2} marginY={3} alignItems="center">
        <Button
          onClick={onStartClick}
          size="medium"
          text="Start"
          icon={<PlayCircleFilledWhiteOutlinedIcon />}
          disabled={props.script_running || props.start_triggered}
        />

        <Typography variant="h5">{props.script_path}</Typography>

        {status_text ? <Label color={status_color} text={status_text} /> : <div></div>}
      </Stack>

      <Dialog open={showHardwareDialog}>
        <DialogTitle subtext="" text="Launch Configuration" onClick={onHardwareDialogCancelClick} />
        <DialogContent>
          <Typography marginY={1}>Hardware</Typography>
          <Stack direction="row" spacing={2} marginY={2}>
            <Select
              helperText=""
              label="Type"
              onChange={handleHardwareTypeChange}
              options={[
                {
                  label: "cpu-small",
                  value: "CPU, 2x, small",
                },
                {
                  label: "cpu-medium",
                  value: "CPU, 8x, medium",
                },
                {
                  label: "gpu",
                  value: "GPU, T4, 1x",
                },
                {
                  label: "gpu-fast",
                  value: "GPU, V100, 1x",
                },
                {
                  label: "gpu-fast-multi",
                  value: "GPU, V100, 4x",
                },
              ]}
              size="medium"
              statusText=""
              value={hardwareType}
              disabled={!props.running_in_cloud}
            />
          </Stack>
          <Banner show={!props.running_in_cloud} variant="info">
            Hardware selection is only available in the cloud.
            <br />
            Hint: Try running the app with --cloud.
          </Banner>

          <Typography marginY={1}>Script Arguments</Typography>
          <TextField
            helperText=""
            label=""
            onChange={handleScriptArgsChange}
            placeholder="--option value"
            size="medium"
            statusText=""
            type="text"
            value={scriptArgs}
            fullWidth={true}
          />

          <Typography marginY={1}>Environment Variables</Typography>
          <EnvironmentConfigurator data={environmentVariables} setData={setEnvironmentVariables} />
        </DialogContent>

        <DialogActions>
          <Button color="grey" onClick={onHardwareDialogCancelClick} text="Cancel" />
          <Button onClick={onHardwareDialogConfirmClick} text="Confirm" />
        </DialogActions>
      </Dialog>
    </div>
  );
}
