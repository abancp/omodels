// ScatterPlot.js
import React from 'react';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend
} from 'recharts';

const data = [
  { x: 100, y: 200, label: 'A' },
  { x: 120, y: 100, label: 'B' },
  { x: 170, y: 300, label: 'A' },
  { x: 140, y: 250, label: 'C' },
  { x: 150, y: 400, label: 'B' },
];

// Group data by label and assign a color
const labelColors = {
  A: '#8884d8',
  B: '#82ca9d',
  C: '#ffc658',
};

// Separate data by label
const groupedData = Object.keys(labelColors).map(label => ({
  label,
  data: data.filter(point => point.label === label),
  color: labelColors[label],
}));

const ScatterPlot = () => (
  <ScatterChart width={500} height={400} margin={{ top: 20, right: 20, bottom: 10, left: 10 }}>
    <CartesianGrid />
    <XAxis type="number" dataKey="x" name="X" />
    <YAxis type="number" dataKey="y" name="Y" />
    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
    <Legend />
    {groupedData.map(group => (
      <Scatter
        key={group.label}
        name={`Label ${group.label}`}
        data={group.data}
        fill={group.color}
      />
    ))}
  </ScatterChart>
);

export default ScatterPlot;

