// frontend/src/components/ImageUploader.jsx
import React, { useRef, useState } from 'react';
import { UploadCloud, Image as ImageIcon, CheckCircle2 } from 'lucide-react';
import { predictPlate } from '../services/api';

export default function ImageUploader({ onScanResult, isProcessing, setIsProcessing }) {
  const fileInputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);

  const handleFileSelect = async (file) => {
    if (!file || !file.type.startsWith('image/')) return;

    const reader = new FileReader();
    reader.onload = (e) => setPreviewUrl(e.target.result);
    reader.readAsDataURL(file);

    setIsProcessing(true);
    try {
      const result = await predictPlate(file);
      onScanResult(result);
    } catch (error) {
      console.error('Error procesando imagen subida:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="glass-card" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <div style={{ textAlign: 'center' }}>
        <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '4px' }}>Subir Imagen de Vehículo</h3>
        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Selecciona una fotografía de tu galería o arrastra una imagen aquí.</p>
      </div>

      <div
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        style={{
          border: `2px dashed ${dragActive ? 'var(--accent-primary)' : 'var(--bg-card-border)'}`,
          borderRadius: 'var(--radius-md)',
          padding: '36px 16px',
          textAlign: 'center',
          cursor: 'pointer',
          background: dragActive ? 'rgba(16, 185, 129, 0.05)' : 'rgba(0, 0, 0, 0.2)',
          transition: 'all 0.2s ease',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '12px'
        }}
      >
        <input
          type="file"
          ref={fileInputRef}
          accept="image/*"
          style={{ display: 'none' }}
          onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
        />

        {isProcessing ? (
          <>
            <div className="spinner" style={{ width: '36px', height: '36px' }} />
            <p style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--accent-secondary)' }}>Analizando con YOLOv8 + OCR...</p>
          </>
        ) : previewUrl ? (
          <div style={{ width: '100%', maxHeight: '200px', overflow: 'hidden', borderRadius: '8px' }}>
            <img src={previewUrl} alt="Vista previa" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
          </div>
        ) : (
          <>
            <div style={{
              width: '54px',
              height: '54px',
              borderRadius: '50%',
              background: 'rgba(16, 185, 129, 0.15)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--accent-primary)'
            }}>
              <UploadCloud size={28} />
            </div>
            <div>
              <p style={{ fontWeight: 600, fontSize: '0.9rem', color: '#f8fafc' }}>Seleccionar Foto</p>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '2px' }}>JPG, PNG, WEBP (Máx 10MB)</p>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
