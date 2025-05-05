// ScatterPlot.js
import React, { useEffect, useState } from 'react';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend
} from 'recharts';

const ScatterPlot = ({ dataset }) => {

  let [data, setData] = useState([])

  useEffect(() => {
    if (dataset.inputs?.length > 0) {
      setData([])
      for (let i = 0; i < dataset.inputs.length; i++) {
        setData((p) => [...p, { x: dataset.inputs[i][0], y: dataset.inputs[i][1], label: String(dataset.outputs[i]) }])
      }
    }
  }, [dataset])

  const labelColors = {
    0: '#8884d8',
    1: '#82ca9d',
    2: '#ffc658',
  };

 

  useEffect(() => { console.log(data) }, [data])

  // Separate data by label
  const groupedData = Object.keys(labelColors).map(label => ({
    label,
    data: data.filter(point => point.label === label),
    color: labelColors[label],
  }));

  return (
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
  )
};

export default ScatterPlot;

