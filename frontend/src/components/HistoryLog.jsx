// frontend/src/components/HistoryLog.jsx
import React, { useState } from 'react';
import { History, Download, Trash2, Search, Car } from 'lucide-react';

export default function HistoryLog({ history, onClearHistory }) {
  const [searchTerm, setSearchTerm] = useState('');

  const filteredHistory = history.filter((item) =>
    item.placas.some((p) => p.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  const exportToCSV = () => {
    if (history.length === 0) return;
    const headers = ['Fecha', 'Hora', 'Placa'];
    const rows = [];

    history.forEach((item) => {
      item.placas.forEach((placa) => {
        const dateObj = new Date(item.timestamp);
        rows.push([
          dateObj.toLocaleDateString(),
          dateObj.toLocaleTimeString(),
          `"${placa}"`,
        ]);
      });
    });

    const csvContent =
      'data:text/csv;charset=utf-8,' +
      [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `historial_placas_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="glass-card" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <History size={20} color="var(--accent-secondary)" />
          <h3 style={{ fontSize: '1rem', fontWeight: 700 }}>Historial de Escaneos</h3>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          {history.length > 0 && (
            <>
              <button
                className="icon-btn"
                style={{ width: '34px', height: '34px' }}
                onClick={exportToCSV}
                title="Exportar a CSV"
              >
                <Download size={16} />
              </button>
              <button
                className="icon-btn"
                style={{ width: '34px', height: '34px', color: '#ef4444' }}
                onClick={onClearHistory}
                title="Borrar historial"
              >
                <Trash2 size={16} />
              </button>
            </>
          )}
        </div>
      </div>

      {history.length > 0 && (
        <div className="form-group" style={{ position: 'relative' }}>
          <input
            type="text"
            className="form-input"
            style={{ paddingLeft: '34px', fontSize: '0.85rem' }}
            placeholder="Buscar por número de placa..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
          <Search
            size={16}
            color="var(--text-muted)"
            style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)' }}
          />
        </div>
      )}

      {filteredHistory.length === 0 ? (
        <div style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)' }}>
          <Car size={36} opacity={0.4} style={{ marginBottom: '8px' }} />
          <p style={{ fontSize: '0.85rem' }}>
            {history.length === 0 ? 'Aún no has registrado placas.' : 'No se encontraron resultados.'}
          </p>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', maxHeight: '350px', overflowY: 'auto' }}>
          {filteredHistory.map((item) => (
            <div key={item.id} className="history-item">
              <div>
                <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                  {item.placas.map((placa, pIdx) => (
                    <span key={pIdx} className="history-plate">
                      {placa}
                    </span>
                  ))}
                </div>
                <span className="history-time">
                  {new Date(item.timestamp).toLocaleString()}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
