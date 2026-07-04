import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import type { Data, Layout, Config } from 'plotly.js';
import type { PitchTrace } from '../hooks/usePitchTrace';

interface PitchGraphProps {
  trace: PitchTrace | null;
}

const BG = '#0B0F14';
const PLOT_BG = '#111827';
const GRID = '#1f2937';
const AXIS = '#9ca3af';
const CYAN = '#00E5FF';

/**
 * Interactive pitch graph built on Plotly.
 *
 * Accepts a single trace today but is structured so multiple traces
 * (reference + user) can be passed later without changes.
 */
export default function PitchGraph({ trace }: PitchGraphProps) {
  const data: Data[] = useMemo(() => {
    if (!trace) return [];
    return [
      {
        type: 'scatter',
        mode: 'lines',
        x: trace.x,
        y: trace.y,
        customdata: trace.customdata,
        line: {
          color: CYAN,
          width: 2.5,
          shape: 'linear',
        },
        connectgaps: false, // break the line on null (unvoiced) frames
        hovertemplate:
          '<b>Time:</b> %{x:.3f} s<br>' +
          '<b>Frequency:</b> %{y:.2f} Hz<br>' +
          '<b>Confidence:</b> %{customdata:.2f}<extra></extra>',
        name: 'Reference Pitch',
      },
    ];
  }, [trace]);

  const layout: Partial<Layout> = useMemo(
    () => ({
      paper_bgcolor: BG,
      plot_bgcolor: PLOT_BG,
      font: {
        color: '#ffffff',
        family: 'system-ui, sans-serif',
        size: 13,
      },
      margin: { l: 70, r: 30, t: 20, b: 80 },
      showlegend: false,
      xaxis: {
        title: { text: 'Time (s)', font: { color: AXIS } },
        color: AXIS,
        gridcolor: GRID,
        zerolinecolor: GRID,
        rangeslider: { visible: true, thickness: 0.08 },
        rangeselector: undefined,
      },
      yaxis: {
        title: { text: 'Frequency (Hz)', font: { color: AXIS } },
        color: AXIS,
        gridcolor: GRID,
        zerolinecolor: GRID,
      },
      hoverlabel: {
        bgcolor: '#1f2937',
        bordercolor: CYAN,
        font: { color: '#ffffff', size: 13 },
      },
    }),
    [],
  );

  const config: Partial<Config> = useMemo(
    () => ({
      displayModeBar: true,
      scrollZoom: true,
      responsive: true,
      modeBarButtonsToRemove: [
        'select2d',
        'lasso2d',
        'autoScale2d',
        'toggleSpikelines',
        'hoverClosestCartesian',
        'hoverCompareCartesian',
      ],
      displaylogo: false,
    }),
    [],
  );

  return (
    <Plot
      data={data}
      layout={layout}
      config={config}
      style={{ width: '100%', height: '100%' }}
      useResizeHandler
    />
  );
}
