import * as React from "react";

import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import Accordion from "@mui/material/Accordion";
import AccordionDetails from "@mui/material/AccordionDetails";
import AccordionSummary from "@mui/material/AccordionSummary";
import Typography from "@mui/material/Typography";

export default function SimpleAccordion(props: any) {
  return (
    <Accordion disabled={!props?.error_message}>
      <AccordionSummary expandIcon={<ExpandMoreIcon />} aria-controls="panel1a-content" id="panel1a-header">
        <Typography>Errors</Typography>
      </AccordionSummary>
      <AccordionDetails>
        <Typography>
          <pre>
            <code>{props?.error_message}</code>
          </pre>
        </Typography>
      </AccordionDetails>
    </Accordion>
  );
}
