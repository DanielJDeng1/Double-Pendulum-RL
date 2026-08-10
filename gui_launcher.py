"""
gui_launcher.py

Tkinter dashboard for managing RL runs without external web dependencies.
Runs train.py, evaluate.py, and live_view.py as isolated background processes.
Reads progress directly from local run logs.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


class ManagedProcess:
    def __init__(self, script: str, args: list[str], on_exit=None):
        self.script = script
        self.args = args
        self.on_exit = on_exit
        self.line_queue: queue.Queue[str] = queue.Queue()
        self.proc: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None

    def start(self):
        cmd = [sys.executable, "-u", os.path.join(PROJECT_DIR, self.script)] + self.args
        self.proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def _read_loop(self):
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            self.line_queue.put(line.rstrip("\n"))
        self.proc.wait()
        self.line_queue.put(f"__EXIT__:{self.proc.returncode}")
        if self.on_exit:
            self.on_exit(self.proc.returncode)

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if not self.is_running():
            return
        try:
            if os.name == "nt":
                self.proc.kill()
            else:
                self.proc.send_signal(signal.SIGTERM)
        except Exception:
            pass


class LogPane(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.text = tk.Text(self, height=14, wrap="none", state="disabled",
                            bg="#111", fg="#ddd", insertbackground="#ddd")
        yscroll = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=yscroll.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def write(self, line: str):
        self.text.configure(state="normal")
        self.text.insert("end", line + "\n")
        self.text.see("end")
        self.text.configure(state="disabled")

    def clear(self):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")


class TrainTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=10)
        self.proc: ManagedProcess | None = None
        self.watch_proc: ManagedProcess | None = None

        form = ttk.Frame(self)
        form.pack(fill="x")

        self.run_name = tk.StringVar(value="ppo_double_pendulum")
        self.env_id = tk.StringVar(value="InvertedDoublePendulum-v5")
        self.total_steps = tk.StringVar(value="1000000")
        self.num_envs = tk.StringVar(value="8")

        self._labeled_entry(form, "Run name", self.run_name, 0)
        self._labeled_entry(form, "Env ID", self.env_id, 1)
        self._labeled_entry(form, "Total steps", self.total_steps, 2)
        self._labeled_entry(form, "Num envs", self.num_envs, 3)

        btns = ttk.Frame(self)
        btns.pack(fill="x", pady=(8, 4))
        self.start_btn = ttk.Button(btns, text="Train", command=self.start_training)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(btns, text="Stop", command=self.stop_training, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        self.watch_btn = ttk.Button(btns, text="Watch", command=self.toggle_watch)
        self.watch_btn.pack(side="left", padx=6)

        self.status_var = tk.StringVar(value="idle")
        ttk.Label(self, textvariable=self.status_var, font=("TkDefaultFont", 10, "bold")).pack(
            anchor="w", pady=(2, 8))

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)

        chart_frame = ttk.LabelFrame(body, text="Avg return (last 50 episodes)")
        chart_frame.pack(side="left", fill="both", expand=True, padx=(0, 6))
        if HAVE_MPL:
            self.fig = Figure(figsize=(4.5, 3.2), dpi=100)
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlabel("global step")
            self.ax.set_ylabel("avg return")
            self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
        else:
            ttk.Label(chart_frame, text="Install matplotlib for telemetry plot").pack(padx=10, pady=10)

        log_frame = ttk.LabelFrame(body, text="Training log")
        log_frame.pack(side="left", fill="both", expand=True)
        self.log = LogPane(log_frame)
        self.log.pack(fill="both", expand=True)

        self._poll_output()
        self._poll_stats()

    def _labeled_entry(self, parent, label, var, row):
        ttk.Label(parent, text=label, width=12).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=var, width=28).grid(row=row, column=1, sticky="w", pady=2)

    @property
    def run_dir(self) -> str:
        return os.path.join(PROJECT_DIR, "runs", self.run_name.get())

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(PROJECT_DIR, "models", f"{self.run_name.get()}.pth")

    def start_training(self):
        if self.proc and self.proc.is_running():
            messagebox.showinfo("Already running", "A training run is already in progress.")
            return
        try:
            total_steps = int(self.total_steps.get())
            num_envs = int(self.num_envs.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Total steps and num envs must be integers.")
            return

        self.log.clear()
        args = [
            "--run-name", self.run_name.get(),
            "--env-id", self.env_id.get(),
            "--total-steps", str(total_steps),
            "--num-envs", str(num_envs),
        ]
        self.proc = ManagedProcess("train.py", args, on_exit=self._on_train_exit)
        self.proc.start()
        self.status_var.set("running…")
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

    def stop_training(self):
        if self.proc:
            self.status_var.set("stopping (saving checkpoint)…")
            self.proc.stop()
        self.stop_btn.configure(state="disabled")

    def _on_train_exit(self, returncode: int):
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    def toggle_watch(self):
        if self.watch_proc and self.watch_proc.is_running():
            self.watch_proc.stop()
            self.watch_proc = None
            self.watch_btn.configure(text="Watch")
        else:
            self.watch_proc = ManagedProcess("live_view.py", ["--checkpoint", self.checkpoint_path])
            self.watch_proc.start()
            self.watch_btn.configure(text="Stop Watch")

    def _poll_output(self):
        if self.proc:
            try:
                while True:
                    line = self.proc.line_queue.get_nowait()
                    if line.startswith("__EXIT__:"):
                        code = line.split(":", 1)[1]
                        self.log.write(f"--- process exited (code {code}) ---")
                    else:
                        self.log.write(line)
            except queue.Empty:
                pass

        if self.watch_proc and not self.watch_proc.is_running():
            self.watch_proc = None
            self.watch_btn.configure(text="Watch")

        self.after(150, self._poll_output)

    def _poll_stats(self):
        status = self._read_json(os.path.join(self.run_dir, "status.json"))
        stats = self._read_json(os.path.join(self.run_dir, "stats.json"))

        if status:
            state = status.get("state", "?")
            gs = status.get("global_step")
            sps = status.get("sps")
            avg_ret = status.get("avg_return")
            elapsed = None
            if status.get("start_time") and status.get("timestamp"):
                elapsed = status["timestamp"] - status["start_time"]
            parts = [f"state={state}"]
            if gs is not None:
                parts.append(f"step={gs:,}")
            if sps is not None:
                parts.append(f"sps={sps}")
            if avg_ret is not None:
                parts.append(f"avg_return={avg_ret:.1f}")
            if elapsed is not None:
                parts.append(f"elapsed={elapsed / 60:.1f} min")
            self.status_var.set(" | ".join(parts))

        if HAVE_MPL and isinstance(stats, list) and stats:
            xs = [s["global_step"] for s in stats if s.get("avg_return") is not None]
            ys = [s["avg_return"] for s in stats if s.get("avg_return") is not None]
            if xs:
                self.ax.clear()
                self.ax.plot(xs, ys, color="#3399ff")
                self.ax.set_xlabel("global step")
                self.ax.set_ylabel("avg return")
                self.canvas.draw_idle()

        self.after(1500, self._poll_stats)

    @staticmethod
    def _read_json(path: str):
        try:
            with open(path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None


class WatchTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=10)
        self.watch_proc: ManagedProcess | None = None
        self.eval_proc: ManagedProcess | None = None

        picker = ttk.Frame(self)
        picker.pack(fill="x")
        ttk.Label(picker, text="Checkpoint (.pth)").pack(side="left")
        self.ckpt_path = tk.StringVar(value=os.path.join("models", "ppo_double_pendulum.pth"))
        ttk.Entry(picker, textvariable=self.ckpt_path, width=48).pack(side="left", padx=6)
        ttk.Button(picker, text="Browse", command=self.browse).pack(side="left")
        ttk.Button(picker, text="Use latest", command=self.use_latest).pack(side="left", padx=4)

        watch_row = ttk.Frame(self)
        watch_row.pack(fill="x", pady=(10, 4))
        self.stochastic_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(watch_row, text="Stochastic policy",
                        variable=self.stochastic_var).pack(side="left")
        self.watch_btn = ttk.Button(watch_row, text="Watch", command=self.toggle_watch)
        self.watch_btn.pack(side="right")

        eval_frame = ttk.LabelFrame(self, text="Batch evaluation")
        eval_frame.pack(fill="x", pady=(12, 4))
        row = ttk.Frame(eval_frame)
        row.pack(fill="x", padx=6, pady=6)
        ttk.Label(row, text="Episodes").pack(side="left")
        self.episodes_var = tk.StringVar(value="20")
        ttk.Entry(row, textvariable=self.episodes_var, width=6).pack(side="left", padx=(4, 16))
        self.render_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="Render evaluation", variable=self.render_var).pack(side="left")
        ttk.Button(row, text="Evaluate", command=self.run_eval).pack(side="right")

        log_frame = ttk.LabelFrame(self, text="Output")
        log_frame.pack(fill="both", expand=True, pady=(8, 0))
        self.log = LogPane(log_frame)
        self.log.pack(fill="both", expand=True)

        self._poll_output()

    def browse(self):
        path = filedialog.askopenfilename(
            title="Select a checkpoint",
            initialdir=os.path.join(PROJECT_DIR, "models"),
            filetypes=[("PyTorch checkpoint", "*.pth"), ("All files", "*.*")],
        )
        if path:
            self.ckpt_path.set(path)

    def use_latest(self):
        sys.path.insert(0, PROJECT_DIR)
        from env_utils import latest_checkpoint
        path = latest_checkpoint(os.path.join(PROJECT_DIR, "models"))
        if path:
            self.ckpt_path.set(path)
        else:
            messagebox.showinfo("No checkpoints", "No .pth files found in models/.")

    def _resolved_ckpt(self) -> str | None:
        path = self.ckpt_path.get()
        if not os.path.isabs(path):
            path = os.path.join(PROJECT_DIR, path)
        if not os.path.exists(path):
            messagebox.showerror("Not found", f"Checkpoint not found:\n{path}")
            return None
        return path

    def toggle_watch(self):
        if self.watch_proc and self.watch_proc.is_running():
            self.watch_proc.stop()
            self.watch_proc = None
            self.watch_btn.configure(text="Watch")
            self.log.write("[gui] stopped live viewer.")
            return

        path = self._resolved_ckpt()
        if not path:
            return

        args = ["--checkpoint", path]
        if self.stochastic_var.get():
            args.append("--stochastic")

        self.watch_proc = ManagedProcess("live_view.py", args)
        self.watch_proc.start()
        self.watch_btn.configure(text="Stop Watch")

    def run_eval(self):
        path = self._resolved_ckpt()
        if not path:
            return
        try:
            episodes = int(self.episodes_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Episodes must be an integer.")
            return
        if self.eval_proc and self.eval_proc.is_running():
            messagebox.showinfo("Already running", "An evaluation is already in progress.")
            return
        args = ["--checkpoint", path, "--episodes", str(episodes)]
        if self.render_var.get():
            args.append("--render")
        self.log.clear()
        self.eval_proc = ManagedProcess("evaluate.py", args)
        self.eval_proc.start()

    def _poll_output(self):
        for proc in (self.watch_proc, self.eval_proc):
            if proc:
                try:
                    while True:
                        line = proc.line_queue.get_nowait()
                        if line.startswith("__EXIT__:"):
                            code = line.split(":", 1)[1]
                            self.log.write(f"process exited (code {code})")
                        else:
                            self.log.write(line)
                except queue.Empty:
                    pass

        if self.watch_proc and not self.watch_proc.is_running():
            self.watch_proc = None
            self.watch_btn.configure(text="Watch")

        if self.eval_proc and not self.eval_proc.is_running():
            self.eval_proc = None

        self.after(150, self._poll_output)


def main():
    root = tk.Tk()
    root.title("Double Pendulum PPO Control Panel")
    root.geometry("1000x650")

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=8, pady=8)

    train_tab = TrainTab(nb)
    watch_tab = WatchTab(nb)
    nb.add(train_tab, text="Train")
    nb.add(watch_tab, text="Evaluate")

    def on_close():
        for tab in (train_tab, watch_tab):
            for attr in ("proc", "watch_proc", "eval_proc"):
                p = getattr(tab, attr, None)
                if p and p.is_running():
                    p.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()