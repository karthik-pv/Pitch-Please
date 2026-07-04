import { useMemo } from 'react';
import { usePitchData } from './hooks/usePitchData';
import { usePitchTrace } from './hooks/usePitchTrace';
import PitchGraph from './components/PitchGraph';
import './App.css';

function App() {
  const { files, selectedFile, setSelectedFile, data, loading, error } =
    usePitchData();
  const trace = usePitchTrace(data);

  const stats = useMemo(() => {
    if (!data || data.length === 0) return null;
    const voiced = data.filter((p) => p.frequency !== null);
    const avgConf = data.reduce((s, p) => s + p.confidence, 0) / data.length;
    return {
      frames: data.length,
      voiced: voiced.length,
      avgConf,
    };
  }, [data]);

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="title">Pitch Please</h1>
        <div className="file-selector">
          <label htmlFor="file-select">Reference File:</label>
          <select
            id="file-select"
            value={selectedFile ?? ''}
            onChange={(e) => setSelectedFile(e.target.value || null)}
            disabled={files.length === 0}
          >
            {files.length === 0 && <option value="">No files available</option>}
            {files.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
        </div>
      </header>

      <main className="graph-container">
        {loading && <div className="status">Loading pitch data…</div>}
        {error && <div className="status error">{error}</div>}
        {!loading && !error && !trace && (
          <div className="status">No data to display.</div>
        )}
        {!loading && !error && trace && <PitchGraph trace={trace} />}
      </main>

      <footer className="app-footer">
        <span>Time</span>
        <span>Frequency</span>
        <span>
          Confidence
          {stats && (
            <em className="stats">
              {' '}— {stats.frames} frames · {stats.voiced} voiced · avg {stats.avgConf.toFixed(2)}
            </em>
          )}
        </span>
      </footer>
    </div>
  );
}

export default App;
