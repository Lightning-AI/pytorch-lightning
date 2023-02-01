(function () {
  const channel = new MessageChannel();

  // We use document.referrer to get parent container's url
  window.parent.postMessage("Establish communication", document.referrer, [channel.port2]);

  class LightningState {
    static subscribe(componentHandler) {
      channel.port1.onmessage = message => {
        componentHandler(message.data);
      };

      return () => {
        channel.port1.onmessage = null;
      };
    }

    static next(state) {
      channel.port1.postMessage(state);
    }
  }

  window.LightningState = LightningState;
})();
