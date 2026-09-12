// frontend/src/components/ApiConfigModal.jsx
import React, { useState, useEffect } from 'react';
import { Settings, Server, CheckCircle2, AlertTriangle, X, RefreshCw } from 'lucide-react';
import { getApiUrl, setApiUrl, checkApiHealth } from '../services/api';

export default function ApiConfigModal({ isOpen, onClose, onServerStatusChange }) {
  const [url, setUrlInput] = useState('');
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState(null);

  useEffect(() => {
    if (isOpen) {
      setUrlInput(getApiUrl());
      setTestResult(null);
    }
  }, [isOpen]);

  const handleTestConnection = async () => {
    setTesting(true);
    setTestResult(null);
    const savedUrl = setApiUrl(url);
    setUrlInput(savedUrl);

    const health = await checkApiHealth();
    setTesting(false);
    setTestResult(health);
    if (onServerStatusChange) {
      onServerStatusChange(health.online);
    }
  };

  const handleSave = async () => {
    const savedUrl = setApiUrl(url);
    setUrlInput(savedUrl);
    const health = await checkApiHealth();
    if (onServerStatusChange) {
      onServerStatusChange(health.online);
    }
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content glass-card" onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Server size={20} color="#10b981" />
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Configuración de Backend</h3>
          </div>
          <button 
            className="icon-btn" 
            style={{ width: '32px', height: '32px' }} 
            onClick={onClose}
          >
            <X size={18} />
          </button>
        </div>

        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
          Ingresa la URL pública o local del servidor FastAPI con el modelo YOLOv8.
        </p>

        <div className="form-group">
          <label className="form-label">URL del Servidor API</label>
          <input
            type="text"
            className="form-input"
            value={url}
            onChange={(e) => setUrlInput(e.target.value)}
            placeholder="http://localhost:8080"
          />
        </div>

        {testResult && (
          <div 
            style={{ 
              padding: '10px 12px', 
              borderRadius: '8px', 
              fontSize: '0.85rem',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              background: testResult.online ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
              border: `1px solid ${testResult.online ? 'rgba(16, 185, 129, 0.4)' : 'rgba(239, 68, 68, 0.4)'}`,
              color: testResult.online ? '#34d399' : '#f87171'
            }}
          >
            {testResult.online ? <CheckCircle2 size={18} /> : <AlertTriangle size={18} />}
            <span>{testResult.message}</span>
          </div>
        )}

        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginTop: '6px' }}>
          <button className="btn-secondary" onClick={handleTestConnection} disabled={testing}>
            {testing ? <div className="spinner" /> : <RefreshCw size={16} />}
            Probar Conexión
          </button>
          <button className="btn-primary" onClick={handleSave}>
            Guardar Cambios
          </button>
        </div>
      </div>
    </div>
  );
}
