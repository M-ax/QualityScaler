"""Latest-status queue shared by the GUI and backend."""

from queue import Empty


def write_process_status(process_status_q, step):
    # empty() is only a hint on multiprocessing queues; never follow it with a
    # blocking get, which could hang the orchestrator during shutdown.
    while True:
        try:
            process_status_q.get_nowait()
        except Empty:
            break
    process_status_q.put(str(step))
