/**
 * Represents the internal state of a Lightning app as exposed by
 * the `/state` endpoint of the Lightning HTTP API.
 */
export type LightingState = {
  vars: {
    _layout: Layout | Layout[];
    [key: string]: any;
  };
  calls: {
    [key: string]: {
      name: string;
      call_hash: string;
      ret: boolean;
    };
  };
  flows: {
    [key: string]: ChildState;
  };
  works: {
    [key: string]: any;
  };
  changes: {
    [key: string]: any;
  };
  app_state: {
    stage: AppStage;
  };
};

export type ChildState = Omit<LightingState, "app_state">;

export type Layout = LayoutBranch | LayoutLeaf;

export type LayoutBranch = {
  name: string;
  content: string;
};

export type LayoutLeaf = {
  name: string;
  type: LayoutType;
  source?: string;
  target: string;
};

export enum LayoutType {
  web = "web",
  streamlit = "streamlit",
}

export enum AppStage {
  blocking = "blocking",
  restarting = "restarting",
  running = "running",
  stopping = "stopping",
}
