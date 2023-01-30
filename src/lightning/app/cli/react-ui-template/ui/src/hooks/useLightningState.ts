import { useState, useEffect } from "react";

import type { LightingState } from "../types/lightning";

interface LightningState {
  subscribe(handler: (state: any) => void): () => void;
  next(state: any): void;
}

declare global {
  interface Window {
    LightningState: LightningState;
  }
}

export const useLightningState = () => {
  const [lightningState, setLightningState] = useState<LightingState>();

  useEffect(() => {
    const unsubscribe = window.LightningState.subscribe(setLightningState);

    return unsubscribe;
  }, []);

  const updateLightningState = window.LightningState.next;

  return {
    lightningState,
    updateLightningState,
  };
};
