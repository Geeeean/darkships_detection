from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from multiprocessing import Queue
from threading import Thread
import time


class Server:
    def __init__(self, read_queue: Queue, write_queue: Queue):
        self.read = read_queue
        self.write = write_queue
        self.latest_data = None
        self._start_queue_reader()
        self.sim_status = False

        self.app = FastAPI()
        self.app.mount(
            "/", StaticFiles(directory="./ui/dist/", html=True), name="static"
        )

        self._add_routes()

    def _add_routes(self):
        self.app.get("/data")(self.get_latest_data)
        self.app.post("/toggle")(self.toggle)

    def _start_queue_reader(self):
        def read_loop():
            while True:
                try:
                    data = self.read.get()
                    self.latest_data = data
                except Exception as e:
                    print(f"Error reading from queue: {e}")
                time.sleep(0.01)

        thread = Thread(target=read_loop, daemon=True)
        thread.start()

    def get_latest_data(self):
        if self.latest_data is None:
            return JSONResponse(content={"message": "No data yet"}, status_code=404)
        return self.latest_data

    def toggle(self):
        msg = ""

        if self.sim_status:
            self.write.put("PAUSE")
            msg = "Simulation paused"
        else:
            self.write.put("RUN")
            msg = "Simulation started"

        self.sim_status = not self.sim_status

        return {"message": msg}

    def run(self, host="0.0.0.0", port=8000):
        import uvicorn

        uvicorn.run(self.app, host=host, port=port, reload=False)
