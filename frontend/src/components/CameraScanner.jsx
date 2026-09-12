// frontend/src/components/CameraScanner.jsx
import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Camera, SwitchCamera, Zap, ZapOff, RefreshCw, AlertCircle } from 'lucide-react';
import { predictPlate } from '../services/api';

export default function CameraScanner({ onScanResult, isProcessing, setIsProcessing }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const [cameraFacing, setCameraFacing] = useState('environment'); // 'environment' (rear) or 'user' (front)
  const [hasPermission, setHasPermission] = useState(null);
  const [autoScan, setAutoScan] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  // Stop camera stream
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  }, []);

  // Start camera stream
  const startCamera = useCallback(async () => {
    stopCamera();
    setErrorMessage('');
    
    try {
      const constraints = {
        video: {
          facingMode: { ideal: cameraFacing },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setHasPermission(true);
    } catch (err) {
      console.error('Error accediendo a la cámara:', err);
      setHasPermission(false);
      setErrorMessage('No se pudo acceder a la cámara. Asegúrate de otorgar permisos en el navegador.');
    }
  }, [cameraFacing, stopCamera]);

  useEffect(() => {
    startCamera();
    return () => {
      stopCamera();
    };
  }, [startCamera, stopCamera]);

  const toggleCameraFacing = () => {
    setCameraFacing((prev) => (prev === 'environment' ? 'user' : 'environment'));
  };

  // Capture frame from video and send to API
  const captureAndPredict = useCallback(async () => {
    if (!videoRef.current || isProcessing) return;

    const video = videoRef.current;
    const canvas = canvasRef.current || document.createElement('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const base64Image = canvas.toDataURL('image/jpeg', 0.85);

    setIsProcessing(true);
    try {
      const result = await predictPlate(base64Image);
      onScanResult(result);
    } catch (error) {
      console.error('Error en captura:', error);
    } finally {
      setIsProcessing(false);
    }
  }, [isProcessing, onScanResult, setIsProcessing]);

  // Auto scan interval effect
  useEffect(() => {
    let intervalId;
    if (autoScan && hasPermission && !isProcessing) {
      intervalId = setInterval(() => {
        captureAndPredict();
      }, 3500);
    }
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [autoScan, hasPermission, isProcessing, captureAndPredict]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', width: '100%' }}>
      <div className="scanner-viewport">
        {hasPermission === false ? (
          <div style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)' }}>
            <AlertCircle size={48} color="#ef4444" style={{ marginBottom: '12px' }} />
            <p style={{ fontWeight: 600, color: '#f8fafc', marginBottom: '8px' }}>Acceso a Cámara Denegado</p>
            <p style={{ fontSize: '0.85rem', marginBottom: '16px' }}>{errorMessage}</p>
            <button className="btn-primary" onClick={startCamera} style={{ display: 'inline-flex', width: 'auto' }}>
              <RefreshCw size={16} /> Reintentar Permisos
            </button>
          </div>
        ) : (
          <>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="scanner-video"
            />
            <canvas ref={canvasRef} style={{ display: 'none' }} />

            {/* Guía visual de encuadre para la placa */}
            <div className="camera-overlay">
              <div className="plate-guide-box">
                <div className="scanner-line"></div>
              </div>
              <span className="guide-text">Centra la placa dentro del recuadro</span>
            </div>
          </>
        )}
      </div>

      {/* Barra de Controles de la Cámara */}
      <div className="camera-controls">
        <button
          className={`icon-btn ${autoScan ? 'active' : ''}`}
          onClick={() => setAutoScan(!autoScan)}
          title={autoScan ? 'Auto-Escaneo Activado' : 'Activar Auto-Escaneo'}
        >
          {autoScan ? <Zap size={22} /> : <ZapOff size={22} />}
        </button>

        <button
          className="capture-btn"
          onClick={captureAndPredict}
          disabled={isProcessing || hasPermission === false}
          title="Capturar y Procesar Placa"
        >
          {isProcessing ? <div className="spinner" style={{ width: '28px', height: '28px' }} /> : <Camera size={30} />}
        </button>

        <button
          className="icon-btn"
          onClick={toggleCameraFacing}
          title="Cambiar Cámara (Trasera / Frontal)"
        >
          <SwitchCamera size={22} />
        </button>
      </div>
    </div>
  );
}
