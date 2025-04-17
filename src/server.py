from fastapi import FastAPI, Request
from pathlib import Path
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from multiprocessing import Queue
from threading import Thread
from fastapi.middleware.cors import CORSMiddleware
import time


class Server:
    def __init__(self, read_queue: Queue, write_queue: Queue):
        self.read = read_queue
        self.write = write_queue
        self.latest_data = None
        self._start_queue_reader()
        self.sim_status = False

        self.app = FastAPI()

        self._add_routes()
        self._add_fallback()
        self._set_cors()

    def _set_cors(self):
        origins = [
            "http://localhost",
            "http://localhost:5173",
        ]

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _add_routes(self):
        self.app.get("/api/data")(self.get_latest_data)
        self.app.post("/api/start")(self.start)
        self.app.post("/api/pause")(self.pause)
        self.app.post("/api/restart")(self.restart)
        self.app.mount(
            "/assets", StaticFiles(directory="./ui/dist/assets"), name="assets"
        )

    def _add_fallback(self):
        # Serve index.html for all unmatched GET routes
        @self.app.get("/{full_path:path}")
        async def serve_spa(full_path: str, request: Request):
            index_path = Path("./ui/dist/index.html")
            if index_path.exists():
                return FileResponse(index_path)
            return JSONResponse(
                content={"error": "index.html not found"}, status_code=404
            )

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

    def start(self):
        msg = "Simulation started"
        self.write.put("START")
        return {"message": msg}

    def pause(self):
        msg = "Simulation paused"
        self.write.put("PAUSE")
        return {"message": msg}

    def restart(self):
        msg = "Simulation restarted"
        self.write.put("RESTART")
        return {"message": msg}

    def run(self, host="0.0.0.0", port=8000):
        import uvicorn

        uvicorn.run(self.app, host=host, port=port, reload=False)
