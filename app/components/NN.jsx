import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";

// Define colors
const colors = {
    input: "#ff6b6b",
    hidden: "#4ecdc4",
    output: "#ffd166",
    connection: "#778da9",
    pulse: "#ffffff",
};

const NeuralNetworkVisualization = ({ layers_init }) => {
    // Network architecture
    const [layers, setLayers] = useState([]);
    useEffect(() => {
        setLayers([])

        console.log("layers:", layers_init)
        for (let i = 0; i < layers_init?.length; i++) {
            if (!layers_init[i]) break;
            setLayers(l => [...l, { type: i === 0 ? "input" : i === layers_init?.length - 1 ? "output" : "hidden", neurons: layers_init[i] > 100 ? 100 : layers_init[i] }])
        }
    }, [layers_init])

    // State for animation triggers
    const [activations, setActivations] = useState([]);

    // Generate random activations every few seconds

    // Calculate positions
    const width = 900;
    const height = 400;
    const layerSpacing = width / (layers.length + 1);

    return (
        <div className="flex justify-center items-center w-full h-[30rem]">
            <svg width={width} height={height} className="overflow-visible">
                {/* Draw connections */}
                {layers.map((layer, layerIndex) => {
                    if (layerIndex === layers.length - 1) return null;

                    const nextLayer = layers[layerIndex + 1];
                    const startX = (layerIndex + 1) * layerSpacing;
                    const endX = (layerIndex + 2) * layerSpacing;

                    const connections = [];

                    for (let i = 0; i < layer.neurons; i++) {
                        const startY = height / (layer.neurons + 1) * (i + 1);

                        for (let j = 0; j < nextLayer.neurons; j++) {
                            const endY = height / (nextLayer.neurons + 1) * (j + 1);

                            connections.push(
                                <line
                                    key={`${layerIndex}-${i}-${j}`}
                                    x1={startX}
                                    y1={startY}
                                    x2={endX}
                                    y2={endY}
                                    stroke={colors.connection}
                                    strokeWidth={1}
                                    strokeOpacity={0.5}
                                />
                            );
                        }
                    }

                    return connections;
                })}

                {/* Draw weight animations */}
                {activations.map(activation => {
                    const startX = (activation.sourceLayer + 1) * layerSpacing;
                    const startY = height / (layers[activation.sourceLayer].neurons + 1) * (activation.sourceNeuron + 1);
                    const endX = (activation.targetLayer + 1) * layerSpacing;
                    const endY = height / (layers[activation.targetLayer].neurons + 1) * (activation.targetNeuron + 1);

                    // Calculate the path length for proper timing
                    const dx = endX - startX;
                    const dy = endY - startY;
                    const pathLength = Math.sqrt(dx * dx + dy * dy);

                    return (
                        <motion.circle
                            key={activation.id}
                            cx={startX}
                            cy={startY}
                            r={5}
                            fill={activation.weight > 0 ? "#4CAF50" : "#F44336"}
                            initial={{ opacity: 0 }}
                            animate={{
                                cx: endX,
                                cy: endY,
                                opacity: [0, 1, 1, 0],
                            }}
                            transition={{
                                duration: pathLength / 100,
                                ease: "linear",
                                times: [0, 0.1, 0.9, 1],
                            }}
                        />
                    );
                })}

                {/* Draw neurons */}
                {layers.map((layer, layerIndex) => {
                    const layerX = (layerIndex + 1) * layerSpacing;
                    const neuronColor = colors[layer.type];

                    return Array.from({ length: layer.neurons }).map((_, neuronIndex) => {
                        const neuronY = height / (layer.neurons + 1) * (neuronIndex + 1);

                        // Check if this neuron is being activated
                        const isActive = activations.some(
                            a => (a.sourceLayer === layerIndex && a.sourceNeuron === neuronIndex) ||
                                (a.targetLayer === layerIndex && a.targetNeuron === neuronIndex)
                        );

                        return (
                            <g key={`${layerIndex}-${neuronIndex}`}>
                                {/* Outer rotating ring */}
                                <motion.circle
                                    cx={layerX}
                                    cy={neuronY}
                                    r={18}
                                    stroke={neuronColor}
                                    strokeWidth={2}
                                    strokeDasharray="4 6"
                                    fill="transparent"
                                    animate={{ rotate: 360 }}
                                    transition={{
                                        duration: 8,
                                        ease: "linear",
                                        repeat: Infinity,
                                    }}
                                    style={{ transformOrigin: `${layerX}px ${neuronY}px` }}
                                />

                                {/* Neuron core */}
                                <motion.circle
                                    cx={layerX}
                                    cy={neuronY}
                                    r={12}
                                    fill={neuronColor}
                                    animate={{
                                        scale: isActive ? [1, 1.2, 1] : 1,
                                        opacity: isActive ? [0.8, 1, 0.8] : 0.8,
                                    }}
                                    transition={{
                                        duration: isActive ? 0.5 : 0,
                                        ease: "easeInOut",
                                    }}
                                />
                            </g>
                        );
                    });
                })}
            </svg>
        </div>
    );
};

export default NeuralNetworkVisualization;