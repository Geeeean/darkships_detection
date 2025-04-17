import axios from "axios";
import MapView from "./components/MapView";
import { useState } from "react";

export type status = "run" | "pause";

function App() {
  const [status, setStatus] = useState<status>("pause");

  return (
    <main className="w-screen h-screen flex">
      <div className="relative w-[1000px] h-[500px] border border-black">
        <MapView status={status} />
      </div>
      <div className="flex gap-2">
        <div
          className="px-3 py-1 rounded-md border h-fit bg-blue-500"
          onClick={() => {
            axios.post("http://localhost:8000/api/start");
            setStatus("run");
          }}
        >
          run
        </div>
        <div
          className="px-3 py-1 rounded-md border h-fit bg-blue-500"
          onClick={() => {
            axios.post("http://localhost:8000/api/pause");
            setStatus("pause");
          }}
        >
          pause
        </div>
        <div
          className="px-3 py-1 rounded-md border h-fit bg-blue-500"
          onClick={() => {
            axios.post("http://localhost:8000/api/restart");
          }}
        >
          restart
        </div>
      </div>
    </main>
  );
}

export default App;
