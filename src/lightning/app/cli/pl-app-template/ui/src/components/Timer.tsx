import { useEffect, useState } from "react";

export default function Timer(props: any) {
  const isActive = props?.isActive;
  var [totalSeconds, setTotalSeconds] = useState(0);

  var hours = Math.floor(totalSeconds / 3600);
  var totalSeconds = totalSeconds % 3600;
  var minutes = Math.floor(totalSeconds / 60);
  var seconds = totalSeconds % 60;

  useEffect(() => {
    let interval: any = null;
    if (isActive) {
      interval = setInterval(() => {
        setTotalSeconds(totalSeconds => totalSeconds + 1);
      }, 1000);
    } else if (!isActive && totalSeconds !== 0) {
      clearInterval(interval);
    }
    return () => clearInterval(interval);
  }, [isActive, totalSeconds]);

  return (
    <div>
      {("0" + hours).slice(-2)}:{("0" + minutes).slice(-2)}:{("0" + seconds).slice(-2)}
    </div>
  );
}
