@echo off
REM Launch nowcaster training detached from any supervising shell.
REM
REM Why this exists: run as a Claude Code background task, training was stopped
REM three times by the host's low-memory reaper. The job is a long-lived
REM ~3-11h run that has no reason to be a child of a chat session, and this
REM machine commits ~11 GB to ordinary desktop software before training starts,
REM so the pressure is not going away. Started this way the run is an
REM independent process: no supervisor can reap it, and it survives closing the
REM session.
REM
REM Usage (from the repo root):
REM     start "" /B scripts\train_detached.bat
REM or from PowerShell:
REM     Start-Process -FilePath scripts\train_detached.bat -WindowStyle Hidden
REM
REM Resumes from the newest checkpoint by mtime. Appends to logs\nowcaster_train.log.
REM Stop it with: taskkill /PID <pid> /F   (pid is in logs\nowcaster_train.pid)

setlocal
set ROOT=%~dp0..
set PY=C:\Users\<username>\AppData\Local\Programs\Python\Python312\python.exe
set LOG=%ROOT%\logs\nowcaster_train.log

cd /d "%ROOT%"
echo === detached run started %DATE% %TIME% === >> "%LOG%"
"%PY%" -W ignore -u train.py --max-steps 300000 --batch-size 4 --num-workers 0 --resume auto >> "%LOG%" 2>&1
echo === detached run exited %DATE% %TIME% (exit %ERRORLEVEL%) === >> "%LOG%"
endlocal
