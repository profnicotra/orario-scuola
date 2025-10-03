
# webapp.py — Interfaccia web minimale per lanciare hybrid_timetable_postgres.py
# Avvio:  python webapp.py
# Requisiti: pip install Flask

import os, shlex, subprocess, datetime, pathlib
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash

APP_DIR = pathlib.Path(__file__).resolve().parent
OUTPUTS_DIR = APP_DIR / "outputs"
SCRIPT_PATH = (APP_DIR.parent / "hybrid_timetable_postgres.py").resolve()  # atteso nella cartella superiore

app = Flask(__name__)
app.config["SECRET_KEY"] = "dev-secret"  # cambia in produzione

def safe_bool(val):
    return str(val).lower() in ("1","true","on","yes")

@app.route("/", methods=["GET"])
def index():
    default_dsn = os.environ.get("DATABASE_URL", "postgresql://USER:PASS@localhost:5432/scuola_orari")
    next_monday = datetime.date.today() + datetime.timedelta(days=((7 - datetime.date.today().weekday()) % 7))
    return render_template("index.html",
                           default_dsn=default_dsn,
                           default_start=next_monday.isoformat())

@app.route("/run", methods=["POST"])
def run_job():
    dsn = request.form.get("dsn","").strip()
    start = request.form.get("start","").strip()
    days = request.form.get("days","").strip()
    weeks = request.form.get("weeks","").strip()
    use_afternoon = safe_bool(request.form.get("use_afternoon"))
    write_db = safe_bool(request.form.get("write_db"))
    clear_range = safe_bool(request.form.get("clear_range"))

    if not dsn:
        flash("DSN Postgres mancante.", "error")
        return redirect(url_for("index"))

    # Output folder sotto outputs/<timestamp>
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = OUTPUTS_DIR / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # Costruisci i parametri per il programma Python
    args = ["python", str(SCRIPT_PATH), "--dsn", dsn, "--out", str(out_dir)]
    if start:
        args += ["--start", start]
    if weeks:
        args += ["--weeks", weeks]
    elif days:
        args += ["--days", days]
    if use_afternoon:
        args.append("--use-afternoon")
    if write_db:
        args.append("--write-db")
    if clear_range:
        args.append("--clear-range")

    # Esegui e cattura l'output
    try:
        proc = subprocess.run(args, capture_output=True, text=True, cwd=str(APP_DIR.parent))
        stdout = proc.stdout
        stderr = proc.stderr
        rc = proc.returncode
    except Exception as e:
        stdout = ""
        stderr = f"Errore di esecuzione: {e}"
        rc = -1

    # Elenco file generati (solo sotto outputs/, ricorsivo)
    files = []
    for p in out_dir.rglob("*"):
        if p.is_file():
            rel = p.relative_to(OUTPUTS_DIR)
            files.append(str(rel))

    return render_template("result.html",
                           args=" ".join(shlex.quote(a) for a in args),
                           rc=rc, stdout=stdout, stderr=stderr,
                           ts=ts, files=sorted(files))

@app.route("/files/<path:subpath>")
def files(subpath):
    # subpath deve essere relativo a OUTPUTS_DIR
    return send_from_directory(OUTPUTS_DIR, subpath, as_attachment=True)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
