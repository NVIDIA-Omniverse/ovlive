@echo off
REM SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
REM SPDX-License-Identifier: Apache-2.0
setlocal
if not defined LIVE_TEST_PYTHON set "LIVE_TEST_PYTHON=python"
"%LIVE_TEST_PYTHON%" "%~dp0test.py" %*
exit /b %errorlevel%
