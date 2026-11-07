@echo off
setlocal DisableDelayedExpansion
set "F9_LAUNCH_ACTION=setup"
"%~dp0runtime\windows_driver.cmd" %*
