// App.tsx

import { Button } from "@mui/material";
import { TextField } from "@mui/material";
import Box from "@mui/material/Box";
import { ChangeEvent } from "react";
import cloneDeep from "lodash/cloneDeep";

import "./App.css";
import { useLightningState } from "./hooks/useLightningState";

function App() {
  const { lightningState, updateLightningState } = useLightningState();

  const counter = lightningState?.vars.counter;

  const handleClick = async () => {
    if (lightningState) {
      const newLightningState = cloneDeep(lightningState);
      newLightningState.flows.react_ui.vars.should_print =
        !newLightningState.flows.react_ui.vars.should_print;

      updateLightningState(newLightningState);
    }
  };

  const handleTextField = async (event: ChangeEvent<HTMLInputElement>) => {
    if (lightningState) {
      const newLightningState = cloneDeep(lightningState);
      newLightningState.flows.react_ui.vars.message_to_print =
        event.target.value;

      updateLightningState(newLightningState);
    }
  };

  return (
    <div className="App">
      <div className="wrapper">
        <div>
          <Button variant="text" onClick={() => handleClick()}>
            <h2>
              {lightningState?.["flows"]?.["react_ui"]?.["vars"]?.[
                "should_print"
              ]
                ? "Stop printing"
                : "Start Printing"}
            </h2>
          </Button>
        </div>
        <Box
          component="form"
          sx={{
            "& .MuiTextField-root": { m: 1, width: "25ch" },
          }}
          noValidate
          autoComplete="off"
        >
          <div>
            <TextField
              defaultValue="Hello World!"
              onChange={handleTextField}
              helperText="Message to be printed in your terminal"
            />
          </div>
          <div>
            <h2>Total number of prints in your terminal: {counter}</h2>
          </div>
        </Box>
      </div>
    </div>
  );
}

export default App;
