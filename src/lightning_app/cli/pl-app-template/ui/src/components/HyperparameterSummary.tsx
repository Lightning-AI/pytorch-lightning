import { useState } from "react";

import { TextField } from "@mui/material";

import { Checkbox, Stack, Table, Typography } from "lightning-ui/src/design-system/components";

function HyperparametersTable(props: any) {
  return <Table header={["Name", "Value"]} rows={props.rows} />;
}

export default function HyperparameterSummary(props: any) {
  const model_hparams = props?.model_hparams ? props.model_hparams : {};
  const trainer_hparams = props?.trainer_hparams ? props.trainer_hparams : {};
  const model_hparams_keys = Object.keys(model_hparams);
  const trainer_hparams_keys = Object.keys(trainer_hparams);

  const [searched, setSearched] = useState<string>("");
  const filteredModelHparams = model_hparams_keys
    .filter(key => key.toLowerCase().includes(searched.toLowerCase()))
    .map(key => [key, model_hparams[key]]);
  const filteredTrainerHparams = trainer_hparams_keys
    .filter(key => key.toLowerCase().includes(searched.toLowerCase()))
    .map(key => [key, trainer_hparams[key]]);

  const [modelHparamsVisible, setModelHparamsVisible] = useState<boolean>(true);
  const [trainerHparamsVisible, setTrainerHparamsVisible] = useState<boolean>(true);

  const requestSearch = (event: any) => {
    const searchedVal = event?.target.value;
    setSearched(searchedVal);
  };

  const toggleModelHparamsCheckbox = (value: boolean) => {
    setModelHparamsVisible(value);
  };

  const toggleTrainerHparamsCheckbox = (value: boolean) => {
    setTrainerHparamsVisible(value);
  };

  return (
    <Stack spacing={3}>
      <Stack direction="row" spacing={1} alignItems="center">
        <Checkbox
          helperText=""
          label=""
          onChange={toggleTrainerHparamsCheckbox}
          size="small"
          statusText=""
          description={<>Trainer</>}
          checked={trainerHparamsVisible}
          disabled={filteredTrainerHparams.length == 0}
        />
        <Checkbox
          helperText=""
          label=""
          onChange={toggleModelHparamsCheckbox}
          size="small"
          statusText=""
          description={<>Model</>}
          checked={modelHparamsVisible}
          disabled={filteredModelHparams.length == 0}
        />
        <TextField
          onChange={requestSearch}
          placeholder="Search"
          size="small"
          type="text"
          value={searched}
          disabled={!trainer_hparams_keys.length && !model_hparams_keys.length}
        />
      </Stack>

      {(!model_hparams || Object.keys(model_hparams).length == 0) && (
        <p>Hyperparameters will appear when script is running.</p>
      )}

      <Stack sx={{ maxHeight: "40em", overflow: "auto" }}>
        {modelHparamsVisible && model_hparams_keys && model_hparams_keys.length > 0 && (
          <div>
            <Typography>Model</Typography>
            <HyperparametersTable rows={filteredModelHparams} />
          </div>
        )}

        {trainerHparamsVisible && trainer_hparams && Object.keys(trainer_hparams).length > 0 && (
          <div>
            <Typography>Trainer</Typography>
            <HyperparametersTable rows={filteredTrainerHparams} />
          </div>
        )}
      </Stack>
    </Stack>
  );
}
