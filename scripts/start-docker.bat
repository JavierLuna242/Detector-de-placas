@echo off
echo ===================================================
echo   Iniciando Contenedores Docker (Backend + Frontend)
echo ===================================================
docker compose up -d --build
echo.
echo ===================================================
echo   Servicios Listos:
echo   - Backend API:  http://localhost:8080/docs
echo   - Frontend Web: http://localhost:80
echo ===================================================
pause
