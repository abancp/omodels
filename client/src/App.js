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
import NN from "./NN"

function App() {
  const [result, setResult] = useState();
  const [module, setModule] = useState()
  const [epoches, setEpoches] = useState(10)
  const [data, setData] = useState(
    Array.from({ length: 10 }, (_, i) => ({ epoch: i + 1, loss: null }))
  )
  const [ips, setIps] = useState(2)
  const [ops, setOps] = useState(3)

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

  const train = () => {
    const x = [
      [-1.0, 0.5], [-0.8, 0.3], [-1.2, 0.1],
      [1.0, 0.5], [0.8, 0.3], [1.2, 0.1],
      [0.0, 0.0], [0.2, -0.2], [-0.2, -0.2]
    ];
    const y = [0, 0, 0, 1, 1, 1, 2, 2, 2];
    const flatX = x.flat()
    const xPtr = module._malloc(flatX.length*8)
    const yPtr = module._malloc(y.length * 8)

  }


  return (
    <div className="bg-gray-200 h-screen w-full flex gap-4 p-4">
      <div className='w-[15rem] h-full flex p-4 flex-col gap-3 justify-start items-center'>
        <h1 className='text-4xl font-playwrite mt-2 mb-10'>Omodels</h1>
        <p className='cursor-pointer hover:bg-white duration-300 bg-gray-100 w-full text-md rounded-lg text-center '>Neural Network</p>
      </div>
      <div className='w-full rounded-lg p-4 px-6 bg-white flex flex-col justify-start gap-4 items-center h-full overflow-y-scroll'>
        <h1 className='text-3xl font-black'>Neural Network</h1>
        <h5 className='text-xl font-semibold w-full mt-5'> Creating </h5>
        <form className='flex gap-16 items-center'>
          <div>
            <label>Input Layer nuerons : </label>
            <input onChange={(e) => setIps(Number(e.target.value))} value={ips} min={1} placeholder='input layer' type="number" className="ring-1 focus:ring rounded-lg ring-blue-600 px-2 focus:outline-none" />
          </div>
          <>|</>
          <div>
            <label>Output Layer nuerons : </label>
            <input onChange={(e) => setOps(Number(e.target.value))} value={ops} min={1} placeholder="output layer" type="number" className="ring-1 focus:ring rounded-lg ring-blue-600 px-2 focus:outline-none" />
          </div>
        </form>
        <NN
          ips={ips} ops={ops}
        />

        <h5 className='text-xl font-semibold w-full mt-5'> Training </h5>


        <form className='flex flex-col gap-4' onSubmit={(e) => {
          e.preventDefault(); module.train(ips, ops, [[-1.0, 0.5], [-0.8, 0.3], [-1.2, 0.1],
          [1.0, 0.5], [0.8, 0.3], [1.2, 0.1],
          [0.0, 0.0], [0.2, -0.2], [-0.2, -0.2]], [0, 0, 0, 1, 1, 1, 2, 2, 2], Number(e.target.epochs.value), Number(e.target.lr.value) || 0.01)
        }}>
          <div className='flex gap-3'>
            <label>number of epoches</label>
            <input type="range" value={epoches} max={1000} onChange={(e) => setEpoches(e.target.value)} min={1} name="epochs" />
          </div>
          <input type="text" defaultValue={0.1} max="10" name="lr" className="ring-1 focus:ring rounded-lg ring-blue-600 px-2 focus:outline-none" />
          <input className='bg-blue-600 text-white rounded-lg text-xl' type="submit" value="train" />
        </form>

        <LossChart data={data} />
      </div>
    </div>
  );
}

export default App;
