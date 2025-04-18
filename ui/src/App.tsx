import axios from "axios";
import MapView from "./components/MapView";
import { useEffect, useState } from "react";
import { Button } from "./components/ui/button";
import { Loader, Pause, Play, RotateCcw } from "lucide-react";

export type status = "run" | "pause";
type btnStatus = "loading" | "active" | "disabled";

const REFRESH_INTERVAL = 1000;

const getBtnCont = (copy: string, status: btnStatus) => {
  switch (status) {
    case "loading":

    case "active":
    case "disabled":
  }
};

function App() {
  const [runStatus, setRunStatus] = useState<btnStatus>("active");
  const [pauseStatus, setPauseStatus] = useState<btnStatus>("active");
  const [data, setData] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await axios.get("http://localhost:8000/api/data");
        console.log(res.data);
        setData(res.data);
        const status = res.data.status;
        if (status == "pause") {
          if (pauseStatus == "loading") {
            setRunStatus("active");
            setPauseStatus("disabled");
          }
        } else if (status == "run") {
          if (runStatus == "loading") {
            setPauseStatus("active");
            setRunStatus("disabled");
          }
        }
      } catch (err) {
        console.error("Data fetch failed", err);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, REFRESH_INTERVAL);

    return () => clearInterval(interval);
  }, [pauseStatus, runStatus]);

  return (
    <main className="dark w-screen h-screen flex flex-col p-2 gap-2 bg-slate-900">
      <div className="flex gap-2">
        <Button
          disabled={runStatus != "active"}
          onClick={() => {
            axios.post("http://localhost:8000/api/start");
            setRunStatus("loading");
            setPauseStatus("active");
          }}
        >
          {runStatus == "loading" ? (
            <Loader className="animate-spin" />
          ) : (
            <Play />
          )}
          Run
        </Button>

        <Button
          disabled={pauseStatus != "active"}
          onClick={() => {
            axios.post("http://localhost:8000/api/pause");
            setPauseStatus("loading");
            setRunStatus("active");
          }}
        >
          {pauseStatus == "loading" ? (
            <Loader className="animate-spin" />
          ) : (
            <Pause />
          )}
          Pause
        </Button>

        <Button
          onClick={() => {
            axios.post("http://localhost:8000/api/restart");
          }}
        >
          <RotateCcw />
          Restart
        </Button>
      </div>
      <div className="relative w-full h-full border rounded-md overflow-hidden">
        <MapView data={data} />
      </div>
    </main>
  );
}

export default App;
