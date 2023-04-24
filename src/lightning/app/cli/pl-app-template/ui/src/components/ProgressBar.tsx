import * as React from "react";

import Box from "@mui/material/Box";
import CircularProgress, { CircularProgressProps, circularProgressClasses } from "@mui/material/CircularProgress";
import LinearProgress, { linearProgressClasses } from "@mui/material/LinearProgress";
import Typography from "@mui/material/Typography";
import { styled } from "@mui/material/styles";

import { LIGHTNING_PURPLE } from "lightning-colors";

const BorderLinearProgress = styled(LinearProgress)(({ theme }) => ({
  height: 10,
  borderRadius: 5,
  [`&.${linearProgressClasses.colorPrimary}`]: {
    backgroundColor: theme.palette.grey[theme.palette.mode === "light" ? 200 : 800],
  },
  [`& .${linearProgressClasses.bar}`]: {
    borderRadius: 5,
    backgroundColor: { LIGHTNING_PURPLE },
  },
}));

export default function ProgressBar(props: any) {
  const percentage = props.current ? (props.current * 100) / props.total : 0;
  return (
    <Box sx={{ display: "flex", alignItems: "center" }}>
      <Box sx={{ width: "100%", mr: 1 }}>
        <BorderLinearProgress variant="determinate" value={percentage} />
      </Box>
      <Box sx={{ minWidth: 35 }}>
        <Typography variant="body2" color="text.secondary">{`${Math.round(percentage)}%`}</Typography>
      </Box>
    </Box>
  );
}
