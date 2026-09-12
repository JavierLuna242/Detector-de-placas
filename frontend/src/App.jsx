// frontend/src/App.jsx
import React, { useState, useEffect, useCallback } from 'react';
import { Camera, Upload, History } from 'lucide-react';
import Header from './components/Header';
import CameraScanner from './components/CameraScanner';
import ImageUploader from './components/ImageUploader';
import ResultDisplay from './components/ResultDisplay';
import HistoryLog from './components/HistoryLog';
import ApiConfigModal from './components/ApiConfigModal';
import { checkApiHealth } from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('camera'); // 'camera', 'upload', 'history'
  const [isProcessing, setIsProcessing] = useState(false);
  const [lastResult, setLastResult] = useState(null);
  const [history, setHistory] = useState([]);
  
  const [isOnline, setIsOnline] = useState(false);
  const [isChecking, setIsChecking] = useState(true);
  const [isConfigOpen, setIsConfigOpen] = useState(false);

  // Load history from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('scanned_plates_history');
    if (saved) {
      try {
        setHistory(JSON.parse(saved));
      } catch (e) {
        console.error('Error cargando historial:', e);
      }
    }
  }, []);

  // Check API connectivity
  const verifyServer = useCallback(async () => {
    setIsChecking(true);
    const health = await checkApiHealth();
    setIsOnline(health.online);
    setIsChecking(false);
  }, []);

  useEffect(() => {
    verifyServer();
    const interval = setInterval(verifyServer, 15000);
    return () => clearInterval(interval);
  }, [verifyServer]);

  // Handle new scan result
  const handleScanResult = (result) => {
    setLastResult(result);
    if (result && result.placas && result.placas.length > 0) {
      const newItem = {
        id: Date.now(),
        timestamp: new Date().toISOString(),
        placas: result.placas,
      };

      setHistory((prev) => {
        const updated = [newItem, ...prev];
        localStorage.setItem('scanned_plates_history', JSON.stringify(updated.slice(0, 50)));
        return updated;
      });
    }
  };

  const handleClearHistory = () => {
    setHistory([]);
    localStorage.removeItem('scanned_plates_history');
  };

  return (
    <>
      <Header
        isOnline={isOnline}
        isChecking={isChecking}
        onOpenSettings={() => setIsConfigOpen(true)}
      />

      <main className="main-content">
        {/* Navegación por Pestañas */}
        <div className="nav-tabs">
          <button
            className={`tab-btn ${activeTab === 'camera' ? 'active' : ''}`}
            onClick={() => setActiveTab('camera')}
          >
            <Camera size={18} />
            <span>Cámara</span>
          </button>

          <button
            className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => setActiveTab('upload')}
          >
            <Upload size={18} />
            <span>Galería</span>
          </button>

          <button
            className={`tab-btn ${activeTab === 'history' ? 'active' : ''}`}
            onClick={() => setActiveTab('history')}
          >
            <History size={18} />
            <span>Historial ({history.length})</span>
          </button>
        </div>

        {/* Pestaña Cámara */}
        {activeTab === 'camera' && (
          <>
            <CameraScanner
              onScanResult={handleScanResult}
              isProcessing={isProcessing}
              setIsProcessing={setIsProcessing}
            />
            {lastResult && <ResultDisplay result={lastResult} />}
          </>
        )}

        {/* Pestaña Subir Imagen */}
        {activeTab === 'upload' && (
          <>
            <ImageUploader
              onScanResult={handleScanResult}
              isProcessing={isProcessing}
              setIsProcessing={setIsProcessing}
            />
            {lastResult && <ResultDisplay result={lastResult} />}
          </>
        )}

        {/* Pestaña Historial */}
        {activeTab === 'history' && (
          <HistoryLog
            history={history}
            onClearHistory={handleClearHistory}
          />
        )}
      </main>

      <ApiConfigModal
        isOpen={isConfigOpen}
        onClose={() => setIsConfigOpen(false)}
        onServerStatusChange={(online) => setIsOnline(online)}
      />
    </>
  );
}
