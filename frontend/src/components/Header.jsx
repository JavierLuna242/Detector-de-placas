// frontend/src/components/Header.jsx
import React from 'react';
import { Camera, Sliders } from 'lucide-react';

export default function Header({ isOnline, isChecking, onOpenSettings }) {
  return (
    <header className="app-header">
      <div className="brand-container">
        <div className="brand-logo">
          <Camera size={22} color="#FFFFFF" />
        </div>
        <div>
          <h1 className="brand-title">ScanPlaca iOS</h1>
          <span className="brand-subtitle">YOLOv8 • EasyOCR</span>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <div 
          className="status-badge" 
          onClick={onOpenSettings}
          title="Verificar o cambiar servidor API"
        >
          <span 
            className={`status-dot ${
              isChecking ? 'checking' : isOnline ? 'online' : 'offline'
            }`} 
          />
          <span>
            {isChecking ? 'Buscando...' : isOnline ? 'En línea' : 'Sin conexión'}
          </span>
        </div>

        <button 
          className="icon-btn" 
          style={{ width: '38px', height: '38px' }} 
          onClick={onOpenSettings}
          title="Configuración API"
        >
          <Sliders size={18} />
        </button>
      </div>
    </header>
  );
}
