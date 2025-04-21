// App.jsx
/*import React, { useEffect, useState } from 'react';
import LossChart from './LossChart';

const App = () => {
  const TOTAL_EPOCHS = 200;

  // Initialize with fixed epochs and undefined loss
  const [data, setData] = useState(
    Array.from({ length: TOTAL_EPOCHS }, (_, i) => ({ epoch: i + 1, loss: null }))
  );

  useEffect(() => {
    let epoch = 0;
    const interval = setInterval(() => {
      if (epoch >= TOTAL_EPOCHS) {
        clearInterval(interval);
        return;
      }

      const newLoss = +(Math.random() * 0.5).toFixed(3);

      setData((prevData) => {
        const updated = [...prevData];
        updated[epoch] = { ...updated[epoch], loss: newLoss };
        return updated;
      });

      epoch++;
    }, 1);
  }, []);

  return (
    <div className="max-w-xl mx-auto mt-10">
      <h2 className="text-2xl font-bold mb-4 text-center">Training Loss</h2>
      <losschart data={data} />
    </div>
  );
};

export default App;
*/

import { useEffect, useState } from 'react';
import LossChart from './LossChart';

function App() {
  const [result, setResult] = useState();
  const [module, setModule] = useState()
  const [epoches, setEpoches] = useState(10)
  const [data, setData] = useState(
    Array.from({ length: 10 }, (_, i) => ({ epoch: i + 1, loss: null }))
  )

  useEffect(() => {
    const loadWasm = async () => {
      // Load the JS glue code dynamically using a script tag
      const script = document.createElement('script');
      script.src = '/wasm/math.js';
      script.async = true;

      script.onload = async () => {
        const Module = await window.createModule();
        setModule(Module)
        window.report_train = (epoch, loss) => {
          // setData((pr)=>[...pr,{epoch,loss}]);
          setData((prevData) => {
            const updated = [...prevData];
            updated[epoch] = { ...updated[epoch], loss: loss };
            return updated;
          });

        }
      }

      document.body.appendChild(script);
    };

    loadWasm();
  }, []);

  useEffect(() => {
    setData(
      Array.from({ length: epoches }, (_, i) => ({ epoch: i + 1, loss: null }))
    )
  }, [epoches])


  return (
    <div className="bg-gray-200 h-screen w-full flex gap-4 p-4">
      <div className='w-[15rem] h-full'></div>
      <div className='w-full rounded-lg p-4 bg-white'>
        <form onSubmit={(e) => { e.preventDefault(); module.train(Number(e.target.epochs.value), Number(e.target.lr.value)) }}>
          <input type="submit" value="train" />
          <label>number of epoches</label>
          <input type="range" max={1000} onChange={(e) => setEpoches(e.target.value)} min={1} name="epochs" />
          <input type="text" name="lr" />
        </form>
        <LossChart data={data} />
      </div>
    </div>
  );
}

export default App;
