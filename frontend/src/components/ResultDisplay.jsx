// frontend/src/components/ResultDisplay.jsx
import React, { useState } from 'react';
import { Copy, Volume2, Check, AlertCircle, CheckCircle2 } from 'lucide-react';

export default function ResultDisplay({ result }) {
  const [copied, setCopied] = useState(false);
  const [speaking, setSpeaking] = useState(false);

  if (!result) return null;

  const hasPlates = result.placas && result.placas.length > 0;
  const processedImgSrc = result.image
    ? (result.image.startsWith('data:') ? result.image : `data:image/jpeg;base64,${result.image}`)
    : null;

  const handleCopy = (text) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleSpeak = (text) => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const spacedText = text.split('').join(' ');
      const utterance = new SpeechSynthesisUtterance(`Placa ${spacedText}`);
      utterance.lang = 'es-ES';
      utterance.onstart = () => setSpeaking(true);
      utterance.onend = () => setSpeaking(false);
      utterance.onerror = () => setSpeaking(false);
      window.speechSynthesis.speak(utterance);
    }
  };

  return (
    <div className="glass-card result-card">
      <div className="result-header">
        <div className="result-title">
          {hasPlates ? (
            <CheckCircle2 size={20} color="var(--ios-green)" />
          ) : (
            <AlertCircle size={20} color="var(--ios-amber)" />
          )}
          <span style={{ fontSize: '0.92rem', fontWeight: 700 }}>
            {hasPlates ? 'Placa Identificada' : 'Resultado del Escaneo'}
          </span>
        </div>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
          {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
        </span>
      </div>

      {hasPlates ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {result.placas.map((placa, index) => (
            <div key={index} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
              <div className="plate-badge">
                <span>🚘 {placa}</span>
              </div>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button
                  className="icon-btn"
                  style={{ width: '42px', height: '42px' }}
                  onClick={() => handleSpeak(placa)}
                  title="Escuchar número de placa"
                >
                  <Volume2 size={19} color={speaking ? 'var(--ios-green)' : '#FFFFFF'} />
                </button>
                <button
                  className="icon-btn"
                  style={{ width: '42px', height: '42px' }}
                  onClick={() => handleCopy(placa)}
                  title="Copiar texto"
                >
                  {copied ? <Check size={19} color="var(--ios-green)" /> : <Copy size={19} color="#FFFFFF" />}
                </button>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div 
          style={{ 
            padding: '12px 14px', 
            background: 'rgba(255, 159, 10, 0.12)', 
            borderRadius: '12px', 
            border: '1px solid rgba(255, 159, 10, 0.25)', 
            display: 'flex', 
            alignItems: 'center', 
            gap: '10px' 
          }}
        >
          <AlertCircle size={18} color="var(--ios-amber)" />
          <span style={{ fontSize: '0.84rem', color: '#FFD60A', fontWeight: 500 }}>
            {result.message || 'No se detectaron caracteres en la imagen.'}
          </span>
        </div>
      )}

      {processedImgSrc && (
        <div style={{ marginTop: '4px' }}>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginBottom: '8px', fontWeight: 600 }}>
            Detección Visual YOLOv8:
          </p>
          <div className="processed-image-box">
            <img src={processedImgSrc} alt="Procesada YOLOv8" />
          </div>
        </div>
      )}
    </div>
  );
}
