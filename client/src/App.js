import { useEffect, useState, useSyncExternalStore } from 'react';
import LossChart from './components/LossChart';
import NN from "./components/NN"
import Vis from "./components/Vis"

function App() {
  const [showSideBar, setShowSideBar] = useState(true);
  const [collapseSideBar, setCollapseSideBar] = useState(false)
  const [module, setModule] = useState()
  const [epoches, setEpoches] = useState(100)
  const [data, setData] = useState(
    Array.from({ length: 10 }, (_, i) => ({ epoch: i + 1, loss: null }))
  )
  const [ips, setIps] = useState(2)
  const [ops, setOps] = useState(3)
  const [dataset, setDataset] = useState({
    inputs: [[-1.0, 0.5, 0.2], [-0.8, 0.3, 0.2], [-1.2, 0.1, 0.3],
    [1.0, 0.5, 0.4], [0.8, 0.3, 0.4], [1.2, 0.1, 0.4],
    [0.0, 0.0, 0.1], [0.2, -0.2, 0.1], [-0.2, -0.2, 0.1]], outputs: [0, 0, 0, 1, 1, 1, 2, 2, 2]
  })
  const [currectEpoch, setCurentEpoch] = useState(0)
  const [training, setTraining] = useState(false)
  const [output, setOutput] = useState()
  const [boudries, setBoundries] = useState({ topright: [0, 0], bottumleft: [0, 0], bottumright: [0, 0], topleft: [0, 0] })

  useEffect(() => {
    const loadWasm = async () => {
      // Load the JS glue code dynamically using a script tag
      const script = document.createElement('script');
      script.src = '/wasm/math.js';
      script.async = true;

      script.onload = async () => {
        const Module = await window.createModule();
        setModule(Module)
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

  const train = (lr) => {
    window.report_train = (epoch, loss) => {
      // setData((pr)=>[...pr,{epoch,loss}]);
      setCurentEpoch(epoch)
      console.log((epoch + 1) / epoches)
      console.log(epoches)
      console.log(Math.round((100 * ((epoch + 1) / epoches)), 1))
      if (Math.round((100 * ((epoch + 1) / epoches)), 1) === 100) {
        setTraining(false)
      }
      setData((prevData) => {
        const updated = [...prevData];
        updated[epoch] = { ...updated[epoch], loss: loss };
        return updated;
      });

    }

    setTraining(true)
    const x = [
      [-1.0, 0.5, 0.2], [-0.8, 0.3, 0.2], [-1.2, 0.1, 0.3],
      [1.0, 0.5, 0.4], [0.8, 0.3, 0.4], [1.2, 0.1, 0.4],
      [0.0, 0.0, 0.1], [0.2, -0.2, 0.1], [-0.2, -0.2, 0.1]
    ];
    const y = [0, 0, 0, 1, 1, 1, 2, 2, 2];

    const vecVectorDouble = new module.VectorVectorDouble();
    dataset.inputs.forEach((r_x) => {
      const vec = new module.VectorDouble();
      r_x.forEach((val) => vec.push_back(val))
      vecVectorDouble.push_back(vec)
    })

    const vectorDouble = new module.VectorDouble();
    dataset.outputs.forEach((val) => { vectorDouble.push_back(val) })
    module.train(ips, ops, vecVectorDouble, vectorDouble, epoches, lr || 0.01)
  }

  const inference = (form) => {
    let inputs = []
    for (let i = 0; i < ips; i++) {

      inputs.push(Number(form['ip' + i].value))
    }
    console.log(inputs)
    const vectorDouble = new module.VectorDouble();
    inputs.forEach((val) => { vectorDouble.push_back(Number(val)) })
    let out = module.inference(vectorDouble)
    setOutput(out)
    console.log(out)
  }


  useEffect(() => {
    if (collapseSideBar) {
      const collapseTime = setTimeout(() => setShowSideBar(false), 100)
      return () => clearTimeout(collapseTime)
    } else {
      setShowSideBar(true)
    }
  }, [collapseSideBar])

  useEffect(() => {
    if (dataset.inputs.length !== 0) {
      for (let i = 0; i < dataset.inputs.length; i++) {
        for (let j = 0; j < ips; j++) {
          if (dataset.inputs[i][j] === undefined || dataset.inputs[i][j] === null) {
            dataset.inputs[i][j] = 0
          }
        }
        if (dataset.inputs[i].length > ips) {
          dataset.inputs[i] = dataset.inputs[i].slice(0, ips)
        }
      }
    }
  }, [ips])



  return (
    <div className="bg-gray-200 h-screen w-full flex gap-4 p-4">
      {

        showSideBar &&
        <div className={`duration-200  h-full flex p-4 flex-col gap-3 justify-start overflow-hidden items-center ${collapseSideBar ? 'w-0 h-0' : 'w-[15rem]'}`}>
          <h1 className='text-4xl font-playwrite mt-2 mb-10'>Omodels</h1>
          <p className='cursor-pointer hover:bg-white duration-300 bg-gray-100 w-full text-md rounded-lg text-center '>Neural Network</p>
        </div>
      }
      <div className='w-full rounded-lg p-4 px-6 bg-white flex flex-col justify-start gap-4 items-center h-full overflow-y-scroll'>
        <div className='w-full h-fit flex flex-col justify-start gap-4 items-center'>
          <div className='w-full flex justify-start '>
            <svg onClick={() => { setCollapseSideBar((p) => !p) }} xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" class="bi cursor-pointer bi-caret-left-square" viewBox="0 0 16 16">
              <path d="M14 1a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H2a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1zM2 0a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V2a2 2 0 0 0-2-2z" />
              <path d="M10.205 12.456A.5.5 0 0 0 10.5 12V4a.5.5 0 0 0-.832-.374l-4.5 4a.5.5 0 0 0 0 .748l4.5 4a.5.5 0 0 0 .537.082" />
            </svg>
          </div>
          <h1 className='text-3xl font-blac0.1k'>Neural Network</h1>
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


          <h5 className='text-xl font-semibold w-full mt-5'> Dataset </h5>
          <div className='flex gap-10'>
            <div className='p-4 text-center rounded-lg  bg-gray-100 '>
              <h3>Input</h3>
              <div className='flex max-w-[30rem] overflow-x-scroll p-2 flex-col gap-2'>
                {
                  dataset.inputs.map((data_raw, i) => (
                    <div className='gap-2  flex '>
                      {data_raw.map((data, j) => (
                        <input
                          className="ring-1 text-center focus:ring w-[4rem] rounded-lg ring-blue-600 px-2 focus:outline-none"
                          value={data}
                          onChange={(e) => setDataset((prev) => ({
                            ...prev, inputs: prev.inputs.map((data_raw0, i0) =>
                              i === i0 ? data_raw0.map((data0, j0) => j === j0 ? Number(e.target.value) : data0) : data_raw0)
                          }))}
                        />
                      ))
                      }
                    </div>
                  ))
                }
              </div>
            </div>
            <div className='flex items-center bg-gray-100 rounded-lg p-2 flex-col gap-2'>

              <h3>output</h3>
              {
                dataset.outputs.map((data, i) => (
                  <div className='gap-2 flex '>

                    <input
                      onChange={(e) =>
                        setDataset((prev) =>
                          ({ ...prev, outputs: prev.outputs.map((val, i0) => i === i0 ? Number(e.target.value) : val) })
                        )}
                      className="ring-1 focus:ring rounded-lg w-[4rem] ring-blue-600 px-2 focus:outline-none"
                      value={data} />
                  </div>
                ))
              }
            </div>
            <Vis dataset={dataset} />

          </div>
          <div onClick={() => setDataset((d) => ({ inputs: [...d.inputs, Array.from({ length: ips }, (_) => 0)], outputs: [...d.outputs, 0] }))} className='cursor-pointer rounded-md w-[30rem] text-center hover:bg-blue-600 duration-300 text-xs border-black border h-8 '>
            Add data
          </div>

          <h5 className='text-xl font-semibold w-full mt-5'> Training </h5>

          <form className='flex flex-col gap-4' onSubmit={(e) => {
            e.preventDefault(); train(Number(e.target.lr.value))
          }}>
            <div className='flex gap-3'>
              <label>number of epoches</label>
              <input type="range" value={epoches} max={1000} onChange={(e) => setEpoches(Number(e.target.value))} min={1} name="epochs" />
            </div>
            <input type="text" defaultValue={0.1} max="10" name="lr" className="ring-1 focus:ring rounded-lg ring-blue-600 px-2 focus:outline-none" />
            <button disabled={training} className='relative w-[10rem] rounded-lg h-6 bg-blue-600 text-white rounded-lg text-xl' type="submit"  >
              <div
                style={{ width: training ? (100 * ((currectEpoch + 1) / epoches)) + "%" : 0 }}
                className='absolute z-0 top-0 left-0 h-full bg-blue-800 rounded-lg transition-all duration-200 ease-in-out'
              >
              </div>
              <h3 className='z-[100] absolute top-0 left-0 w-full text-center'>{training ? Math.round((100 * ((currectEpoch + 1) / epoches)), 1) + "%" : "train"}</h3>
            </button>
          </form>
          <LossChart data={data} />
        </div>
        <h5 className='text-xl font-semibold w-full mt-5'> Testing / Inference</h5>
        <form onSubmit={(e) => { e.preventDefault(); inference(e.target) }} className='flex justify-center items-center  flex-col gap-3'>
          <div className='flex items-center gap-3'>
            <p>Input : </p>
            {
              Array.from({ length: ips }).map((_, i) => (
                <input name={"ip" + i} defaultValue={0.2} className="ring-1 w-[3rem] focus:ring rounded-lg ring-blue-600 px-2 focus:outline-none" />
              ))
            }
          </div>
          <div className='flex gap-3'>
            <h5>Output : </h5>
            <input name="ip" value={output} className="ring-1 w-[3rem] focus:ring rounded-lg ring-blue-600 px-2 focus:outline-none" />
          </div>
          <input type="submit" value="predict" className="ring-1 focus:ring cursor-pointer rounded-lg bg-blue-600 px-2 focus:outline-none" />
        </form>
      </div>
    </div >
  );
}

export default App;
