import { useCallback, useEffect, useState } from 'react';
import type { PitchPoint } from '../types/PitchPoint';

const API_BASE = 'http://127.0.0.1:5000/api';

interface UsePitchDataResult {
  files: string[];
  selectedFile: string | null;
  setSelectedFile: (file: string | null) => void;
  data: PitchPoint[] | null;
  loading: boolean;
  error: string | null;
}

export function usePitchData(): UsePitchDataResult {
  const [files, setFiles] = useState<string[]>([]);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [data, setData] = useState<PitchPoint[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch the list of available JSON files on mount.
  useEffect(() => {
    let cancelled = false;

    async function fetchFiles() {
      try {
        const res = await fetch(`${API_BASE}/files`);
        if (!res.ok) throw new Error(`Failed to list files (${res.status})`);
        const names: string[] = await res.json();
        if (cancelled) return;
        setFiles(names);
        if (names.length > 0 && selectedFile === null) {
          setSelectedFile(names[0]);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load file list');
        }
      }
    }

    fetchFiles();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Fetch the selected file's pitch data whenever the selection changes.
  useEffect(() => {
    if (selectedFile === null) {
      setData(null);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);

    async function fetchPitch() {
      try {
        const res = await fetch(`${API_BASE}/pitch/${selectedFile}`);
        if (!res.ok) throw new Error(`Failed to load ${selectedFile} (${res.status})`);
        const json: PitchPoint[] = await res.json();
        if (cancelled) return;
        setData(json);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load pitch data');
          setData(null);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    fetchPitch();
    return () => {
      cancelled = true;
    };
  }, [selectedFile]);

  const selectFile = useCallback((file: string | null) => {
    setSelectedFile(file);
  }, []);

  return { files, selectedFile, setSelectedFile: selectFile, data, loading, error };
}
