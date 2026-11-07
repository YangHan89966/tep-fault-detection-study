@echo off
setlocal DisableDelayedExpansion
set "F9_LAUNCH_ACTION=run"
"%~dp0runtime\windows_driver.cmd" %*
