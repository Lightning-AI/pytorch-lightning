import { useEffect } from "react";

import { QueryClient, QueryClientProvider } from "react-query";
import { BrowserRouter } from "react-router-dom";

import ErrorPanel from "components/ErrorPanel";
import HyperparameterSummary from "components/HyperparameterSummary";
import Launcher from "components/Launcher";
import ProgressBarGroup from "components/ProgressBarGroup";
import {
  Breadcrumbs,
  Card,
  CardContent,
  CardHeader,
  Grid,
  SnackbarProvider,
  Stack,
  useSnackbar,
} from "lightning-ui/src/design-system/components";
import ThemeProvider from "lightning-ui/src/design-system/theme";

import ExecutionSummary from "./components/ExecutionSummary";
import { useLightningState } from "./hooks/useLightningState";

const queryClient = new QueryClient();

function AppContainer() {
  const { lightningState } = useLightningState();

  const trainer_progress = lightningState?.flows.script_orchestrator.works.script_runner?.vars.trainer_progress;
  const trainer_state = lightningState?.flows.script_orchestrator.works.script_runner?.vars.trainer_state;
  const trainer_hparams = lightningState?.flows.script_orchestrator.works.script_runner?.vars.trainer_hparams;
  const model_hparams = lightningState?.flows.script_orchestrator.works.script_runner?.vars.model_hparams;

  const script_running = lightningState?.flows.script_orchestrator.vars.running;
  const script_succeeded = lightningState?.flows.script_orchestrator.vars.succeeded;
  const script_failed = lightningState?.flows.script_orchestrator.vars.failed;
  const start_triggered = lightningState?.flows.script_orchestrator.vars.triggered;
  const script_path = lightningState?.flows.script_orchestrator.vars.script_path;
  const running_in_cloud = lightningState?.vars.running_in_cloud;

  const breadCrumbItems = [
    { title: "Users", href: "url/to/href/1" },
    { title: "adrian", href: "url/to/href/2" },
    { title: "projects", href: "url/to/href/3" },
    { title: "app_name", href: "url/to/href/4" },
    { title: "source", href: "url/to/href/5" },
    { title: "train.py", href: "url/to/href/6" },
  ];

  const { enqueueSnackbar } = useSnackbar();
  const exception_message = lightningState?.flows.script_orchestrator.works.script_runner?.vars?.exception_message;
  useEffect(() => {
    if (exception_message) {
      enqueueSnackbar({
        title: "The script failed to complete",
        severity: "error",
        children: "See the error message",
      });
    }
  }, [exception_message]);

  return (
    <Stack sx={{ height: "100vh", margin: "auto", maxWidth: "1280px" }}>
      <Launcher
        lightningState={lightningState}
        script_running={script_running}
        start_triggered={start_triggered}
        script_succeeded={script_succeeded}
        script_failed={script_failed}
        script_path={script_path}
        running_in_cloud={running_in_cloud}
      />

      <Grid container spacing={3}>
        <Grid item xs={12} lg={6}>
          <Card>
            <CardContent>
              <ExecutionSummary
                trainer_progress={trainer_progress}
                trainer_state={trainer_state}
                script_running={script_running}
              />
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <ProgressBarGroup trainer_progress={trainer_progress} trainer_state={trainer_state} />
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <Card variant="elevation">
            <CardHeader title="Hyperparameters" subheader="" />
            <CardContent>
              <HyperparameterSummary trainer_hparams={trainer_hparams} model_hparams={model_hparams} />
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <ErrorPanel
            error_message={lightningState?.flows.script_orchestrator.works.script_runner?.vars?.exception_message}
          />
        </Grid>
      </Grid>
    </Stack>
  );
}

function App() {
  return (
    <ThemeProvider>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <SnackbarProvider>
            <AppContainer />
          </SnackbarProvider>
        </BrowserRouter>
      </QueryClientProvider>
    </ThemeProvider>
  );
}

export default App;
