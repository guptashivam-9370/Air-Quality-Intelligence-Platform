import React, { useState, useEffect } from 'react';
import { ArrowRight, Cpu, Brain, Database, BarChart2, ChevronDown, ChevronUp } from 'lucide-react';
import './ModelArchitecture.css';

const ModelArchitecture = ({ selectedState }) => {
    const [modelDetails, setModelDetails] = useState(null);
    const [expanded, setExpanded] = useState({});
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        if (selectedState) {
            setLoading(true);
            // Fetch model architecture details
            fetch(`http://localhost:8501/api/model_architecture?state=${selectedState}`)
                .then(response => response.json())
                .then(data => {
                    setModelDetails(data);
                    setLoading(false);
                })
                .catch(error => {
                    console.error('Error fetching model architecture:', error);
                    setLoading(false);
                });
        }
    }, [selectedState]);

    const toggleSection = (section) => {
        setExpanded({
            ...expanded,
            [section]: !expanded[section]
        });
    };

    if (!selectedState) {
        return (
            <div className="model-architecture">
                <h3>Model Architecture & Selection Reasoning</h3>
                <div className="select-state-prompt">
                    <p>Please select a state to view model architecture details</p>
                </div>
            </div>
        );
    }

    if (loading) {
        return (
            <div className="model-architecture">
                <h3>Model Architecture & Selection Reasoning</h3>
                <div className="loading-container">
                    <div className="loading-spinner"></div>
                    <p>Loading model architecture details...</p>
                </div>
            </div>
        );
    }

    if (!modelDetails) {
        return (
            <div className="model-architecture">
                <h3>Model Architecture & Selection Reasoning</h3>
                <div className="no-data">
                    <p>No architecture details available for {selectedState}</p>
                </div>
            </div>
        );
    }

    return (
        <div className="model-architecture">
            <h3>Model Architecture & Selection Reasoning</h3>
            
            <div className="model-overview">
                <div className="model-header">
                    <h4>
                        <Cpu size={20} />
                        Model Type: {modelDetails.model_type}
                    </h4>
                    <span className="model-version">Version: {modelDetails.version}</span>
                </div>
                <p className="model-description">{modelDetails.description}</p>
            </div>
            
            <div className="architecture-diagram">
                <h4>Architecture Diagram</h4>
                <div className="diagram">
                    {modelDetails.layers.map((layer, index) => (
                        <div key={index} className="layer">
                            <div className="layer-box" style={{ backgroundColor: layer.color }}>
                                <div className="layer-name">{layer.name}</div>
                                <div className="layer-params">{layer.params}</div>
                            </div>
                            {index < modelDetails.layers.length - 1 && (
                                <div className="layer-connector">
                                    <ArrowRight size={20} />
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            </div>
            
            <div className={`collapsible-section ${expanded.selection ? 'expanded' : ''}`}>
                <div className="section-header" onClick={() => toggleSection('selection')}>
                    <h4><Brain size={20} />Model Selection Reasoning</h4>
                    {expanded.selection ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
                </div>
                {expanded.selection && (
                    <div className="section-content">
                        {modelDetails.selection_reasoning.map((reason, index) => (
                            <div key={index} className="reasoning-point">
                                <span className="point-number">{index + 1}</span>
                                <p>{reason}</p>
                            </div>
                        ))}
                    </div>
                )}
            </div>
            
            <div className={`collapsible-section ${expanded.parameters ? 'expanded' : ''}`}>
                <div className="section-header" onClick={() => toggleSection('parameters')}>
                    <h4><Database size={20} />Model Parameters & Hyperparameters</h4>
                    {expanded.parameters ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
                </div>
                {expanded.parameters && (
                    <div className="section-content">
                        <div className="parameters-grid">
                            {Object.entries(modelDetails.hyperparameters).map(([param, value], index) => (
                                <div key={index} className="parameter-card">
                                    <div className="parameter-name">{param}</div>
                                    <div className="parameter-value">{value}</div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
            
            <div className={`collapsible-section ${expanded.training ? 'expanded' : ''}`}>
                <div className="section-header" onClick={() => toggleSection('training')}>
                    <h4><BarChart2 size={20} />Training Process</h4>
                    {expanded.training ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
                </div>
                {expanded.training && (
                    <div className="section-content">
                        <div className="training-details">
                            <div className="training-stat">
                                <div className="stat-label">Training Data Size</div>
                                <div className="stat-value">{modelDetails.training.data_size}</div>
                            </div>
                            <div className="training-stat">
                                <div className="stat-label">Training Time</div>
                                <div className="stat-value">{modelDetails.training.training_time}</div>
                            </div>
                            <div className="training-stat">
                                <div className="stat-label">Epochs</div>
                                <div className="stat-value">{modelDetails.training.epochs}</div>
                            </div>
                            <div className="training-stat">
                                <div className="stat-label">Batch Size</div>
                                <div className="stat-value">{modelDetails.training.batch_size}</div>
                            </div>
                        </div>
                        
                        <div className="convergence-info">
                            <h5>Training Convergence</h5>
                            <p>{modelDetails.training.convergence_notes}</p>
                        </div>
                        
                        {modelDetails.training.image && (
                            <div className="training-image">
                                <img 
                                    src={`data:image/png;base64,${modelDetails.training.image}`}
                                    alt="Training convergence visualization"
                                />
                            </div>
                        )}
                    </div>
                )}
            </div>
            
            <div className={`collapsible-section ${expanded.alternatives ? 'expanded' : ''}`}>
                <div className="section-header" onClick={() => toggleSection('alternatives')}>
                    <h4>Alternative Models Considered</h4>
                    {expanded.alternatives ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
                </div>
                {expanded.alternatives && (
                    <div className="section-content">
                        <div className="alternatives-table-container">
                            <table className="alternatives-table">
                                <thead>
                                    <tr>
                                        <th>Model</th>
                                        <th>Performance</th>
                                        <th>Pros</th>
                                        <th>Cons</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {modelDetails.alternatives.map((alt, index) => (
                                        <tr key={index} className={alt.isCurrent ? 'current-model' : ''}>
                                            <td>{alt.name} {alt.isCurrent && <span className="current-tag">Selected</span>}</td>
                                            <td>{alt.performance}</td>
                                            <td>
                                                <ul>
                                                    {alt.pros.map((pro, i) => (
                                                        <li key={i}>{pro}</li>
                                                    ))}
                                                </ul>
                                            </td>
                                            <td>
                                                <ul>
                                                    {alt.cons.map((con, i) => (
                                                        <li key={i}>{con}</li>
                                                    ))}
                                                </ul>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default ModelArchitecture;