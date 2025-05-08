import { useEffect, useState } from "react";
import LossChart from "../components/LossChart";
import NN from "../components/NN";
import Vis from "../components/Vis";
import * as tf from "@tensorflow/tfjs";

const App = () => {
  const [showSideBar, setShowSideBar] = useState(true);
  const [collapseSideBar, setCollapseSideBar] = useState(false);
  const [epoches, setEpoches] = useState(100);
  const [data, setData] = useState(
    Array.from({ length: 10 }, (_, i) => ({ epoch: i + 1, loss: null }))
  );
  const [ips, setIps] = useState("720,20");
  const [ops, setOps] = useState(3);
  const [dataset, setDataset] = useState({
    inputs: [
      [-1.0, 0.5, 0.2],
      [-0.8, 0.3, 0.2],
      [-1.2, 0.1, 0.3],
      [1.0, 0.5, 0.4],
      [0.8, 0.3, 0.4],
      [1.2, 0.1, 0.4],
      [0.0, 0.0, 0.1],
      [0.2, -0.2, 0.1],
      [-0.2, -0.2, 0.1],
    ],
    outputs: [0, 0, 0, 1, 1, 1, 2, 2, 2],
  });
  const [currectEpoch, setCurentEpoch] = useState(0);
  const [training, setTraining] = useState(false);
  const [output, setOutput] = useState();
  let model;

  useEffect(() => {
    setData(
      Array.from({ length: epoches }, (_, i) => ({ epoch: i + 1, loss: null }))
    );
  }, [epoches]);

  const train = (lr) => { };

  const inference = (form) => {
    let inputs = [];
    for (let i = 0; i < ips; i++) {
      inputs.push(Number(form["ip" + i].value));
    }
    console.log(inputs);
  };

  useEffect(() => {
    if (collapseSideBar) {
      const collapseTime = setTimeout(() => setShowSideBar(false), 100);
      return () => clearTimeout(collapseTime);
    } else {
      setShowSideBar(true);
    }
  }, [collapseSideBar]);

  useEffect(() => {
    if (dataset.inputs.length !== 0) {
      for (let i = 0; i < dataset.inputs.length; i++) {
        for (let j = 0; j < ips[0]; j++) {
          if (
            dataset.inputs[i][j] === undefined ||
            dataset.inputs[i][j] === null
          ) {
            dataset.inputs[i][j] = 0;
          }
        }
        if (dataset.inputs[i].length > ips) {
          dataset.inputs[i] = dataset.inputs[i].slice(0, ips);
        }
      }
    }
  }, [ips]);

  const createModel = () => {
    let config = ips.split(",");
    console.log(ips);
    model = tf.sequential();
    for (let i = 0; i < config.length; i++) {
      if (i === 0) {
        //first layer
        model.add(
          tf.layers.dense({
            inputShape: [Number(config[0])],
            units: Number(config[1]),
            activation: "relu",
          })
        );
      } else if (i === config.length - 1 && i != 1) {
        model.add(
          tf.layers.dense({ units: Number(config[i]), activation: "softmax" })
        );
      } else {
        if (i === 1) {
          continue;
        }
        model.add(
          tf.layers.dense({ units: Number(config[i]), activation: "relu" })
        );
      }
    }
    model.summary();
  };

  return (
    <div className="dark:bg-gray-900 bg-gray-200 h-screen w-full flex gap-4 p-4">
      {showSideBar && (
        <div
          className={`duration-200  h-full flex p-4 flex-col gap-3 justify-start overflow-hidden items-center ${collapseSideBar ? "w-0 h-0" : "w-[15rem]"
            }`}
        >
          <h1 className="text-4xl font-playwrite mt-2 mb-10">Omodels</h1>
          <p className="cursor-pointer dark:bg-gray-800 dark:hover:bg-gray-950 hover:bg-white duration-300 bg-gray-100 w-full text-md rounded-lg text-center ">
            Neural Network
          </p>
        </div>
      )}
      <div className="w-full rounded-lg p-4 px-6 dark:bg-gray-950 bg-white flex flex-col justify-start gap-4 items-center h-full overflow-y-scroll">
        <div className="w-full h-fit flex flex-col justify-start gap-4 items-center">
          <div className="w-full flex justify-start ">
            <svg
              onClick={() => {
                setCollapseSideBar((p) => !p);
              }}
              xmlns="http://www.w3.org/2000/svg"
              width="20"
              height="20"
              fill="currentColor"
              class="bi cursor-pointer bi-caret-left-square"
              viewBox="0 0 16 16"
            >
              <path d="M14 1a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H2a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1zM2 0a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V2a2 2 0 0 0-2-2z" />
              <path d="M10.205 12.456A.5.5 0 0 0 10.5 12V4a.5.5 0 0 0-.832-.374l-4.5 4a.5.5 0 0 0 0 .748l4.5 4a.5.5 0 0 0 .537.082" />
            </svg>
          </div>
          <h1 className="text-3xl font-blac0.1k">Neural Network</h1>
          <h5 className="text-xl font-semibold w-full mt-5"> Creating </h5>
          <form className="flex gap-2 items-center">
            <label>Layer (Seperated by commas) : </label>
            <input
              onChange={(e) => setIps(e.target.value)}
              value={ips}
              placeholder="2,3,4"
              type="text"
              className="ring-1 focus:ring rounded-lg ring-blue-600 px-2 focus:outline-none"
            />
          </form>
          <NN layers_init={ips.split(",").map((d) => Number(d))} />
          <button
            onClick={createModel}
            className="ring-1 focus:ring cursor-pointer rounded-lg bg-blue-600 px-2 focus:outline-none"
          >
            create
          </button>
          <h5 className="text-xl font-semibold w-full mt-5"> Dataset </h5>
          <div className="flex gap-10">
            <div className="p-4 text-center rounded-lg dark:bg-gray-900 bg-gray-100 ">
              <h3>Input</h3>
              <div className="flex max-w-[30rem] overflow-x-scroll p-2 flex-col gap-2">
                {dataset.inputs.map((data_raw, i) => (
                  <div className="gap-2  flex ">
                    {data_raw.map((data, j) => (
                      <input
                        className="ring-1 text-center focus:ring w-[4rem] rounded-lg ring-blue-600 px-2 focus:outline-none"
                        value={data}
                        onChange={(e) =>
                          setDataset((prev) => ({
                            ...prev,
                            inputs: prev.inputs.map((data_raw0, i0) =>
                              i === i0
                                ? data_raw0.map((data0, j0) =>
                                  j === j0 ? Number(e.target.value) : data0
                                )
                                : data_raw0
                            ),
                          }))
                        }
                      />
                    ))}
                  </div>
                ))}
              </div>
            </div>
            <div className="flex items-center dark:bg-gray-900 bg-gray-100 rounded-lg p-2 flex-col gap-2">
              <h3>output</h3>
              {dataset.outputs.map((data, i) => (
                <div className="gap-2 flex ">
                  <input
                    onChange={(e) =>
                      setDataset((prev) => ({
                        ...prev,
                        outputs: prev.outputs.map((val, i0) =>
                          i === i0 ? Number(e.target.value) : val
                        ),
                      }))
                    }
                    className="ring-1 focus:ring rounded-lg w-[4rem] ring-blue-600 px-2 focus:outline-none"
                    value={data}
                  />
                </div>
              ))}
            </div>
            <Vis dataset={dataset} />
          </div>
          <div
            onClick={() =>
              setDataset((d) => ({
                inputs: [...d.inputs, Array.from({ length: ips }, (_) => 0)],
                outputs: [...d.outputs, 0],
              }))
            }
            className="cursor-pointer rounded-md w-[30rem] text-center hover:bg-blue-600 duration-300 text-xs border-black border h-8 "
          >
            Add data
          </div>

          <h5 className="text-xl font-semibold w-full mt-5"> Training </h5>

          <form
            className="flex flex-col gap-4"
            onSubmit={(e) => {
              e.preventDefault();
              train(Number(e.target.lr.value));
            }}
          >
            <div className="flex gap-3">
              <label>number of epoches</label>
              <input
                type="range"
                value={epoches}
                max={1000}
                onChange={(e) => setEpoches(Number(e.target.value))}
                min={1}
                name="epochs"
              />
            </div>
            <input
              type="text"
              defaultValue={0.1}
              max="10"
              name="lr"
              className="ring-1 focus:ring rounded-lg ring-blue-600 px-2 focus:outline-none"
            />
            <button
              disabled={training}
              className="relative w-[10rem] rounded-lg h-6 bg-blue-600 text-white rounded-lg text-xl"
              type="submit"
            >
              <div
                style={{
                  width: training
                    ? 100 * ((currectEpoch + 1) / epoches) + "%"
                    : 0,
                }}
                className="absolute z-0 top-0 left-0 h-full bg-blue-800 rounded-lg transition-all duration-200 ease-in-out"
              ></div>
              <h3 className="z-[100] absolute top-0 left-0 w-full text-center">
                {training
                  ? Math.round(100 * ((currectEpoch + 1) / epoches), 1) + "%"
                  : "train"}
              </h3>
            </button>
          </form>
          <LossChart data={data} />
        </div>
        <h5 className="text-xl font-semibold w-full mt-5">
          {" "}
          Testing / Inference
        </h5>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            inference(e.target);
          }}
          className="flex justify-center items-center  flex-col gap-3"
        >
          <div className="flex items-center gap-3">
            <p>Input : </p>
            {Array.from({ length: ips[0] }).map((_, i) => (
              <input
                name={"ip" + i}
                defaultValue={0.2}
                className="ring-1 w-[3rem] focus:ring rounded-lg ring-blue-600 px-2 focus:outline-none"
              />
            ))}
          </div>
          <div className="flex gap-3">
            <h5>Output : </h5>
            <input
              name="ip"
              value={output}
              className="ring-1 w-[3rem] focus:ring rounded-lg ring-blue-600 px-2 focus:outline-none"
            />
          </div>
          <input
            type="submit"
            value="predict"
            className="ring-1 focus:ring cursor-pointer rounded-lg bg-blue-600 px-2 focus:outline-none"
          />
        </form>
      </div>
    </div>
  );
};

export default App;
