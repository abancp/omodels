import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

const LossChart = ({ data }) => {
  return (
    <div style={{ width: '100%', height: 300 }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="epoch" />
          <YAxis />
          <Tooltip />
<Line
  type="linear"               // <-- use linear for sharp corners
  dataKey="loss"
  stroke="#8884d8"
  strokeWidth={2}
/>
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default LossChart;

