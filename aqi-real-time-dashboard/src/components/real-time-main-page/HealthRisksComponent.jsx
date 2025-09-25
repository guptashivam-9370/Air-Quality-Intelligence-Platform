import React, { useState, useEffect } from 'react';
import './HealthRisksComponent.css';
import { Heart, Flower, Thermometer, Bug, Activity, Wind } from 'lucide-react';


const HealthRisksComponent = ({ currentAQI }) => {
    // Use a default AQI if none is provided
    const aqiValue = currentAQI || 0;

    const [selectedCondition, setSelectedCondition] = useState('asthma');

    // Determine AQI category based on value
    const getAQICategory = (aqi) => {
        if (aqi <= 50) return { level: 'Good', severity: 'Low', color: '#00e400', range: '0-50' };
        if (aqi <= 100) return { level: 'Moderate', severity: 'Moderate', color: '#ffff00', range: '51-100' };
        if (aqi <= 150) return { level: 'Unhealthy for Sensitive Groups', severity: 'Mid', color: '#ff7e00', range: '101-150' };
        if (aqi <= 200) return { level: 'Unhealthy', severity: 'High', color: '#ff0000', range: '151-200' };
        if (aqi <= 300) return { level: 'Very Unhealthy', severity: 'Very High', color: '#8f3f97', range: '201-300' };
        return { level: 'Hazardous', severity: 'Severe', color: '#7e0023', range: '301-500' };
    };

    const aqiCategory = getAQICategory(aqiValue);

    // Health condition information based on AQI category
    const healthConditions = {
        asthma: {
            title: "Asthma",
            icon: <Wind size={24} />,
            image: "person_inhaler.svg",
            info: {
                Good: {
                    intro: "Risk of Asthma symptoms is Low when AQI is Good (0-50)",
                    description: "Minimal risk of asthma symptoms. Most people can carry out normal activities.",
                    dos: [
                        "Enjoy outdoor activities normally",
                        "Keep rescue inhaler with you as a precaution",
                        "Continue regular asthma management plan"
                    ],
                    donts: [
                        "Skip prescribed medications",
                        "Ignore any unusual symptoms"
                    ]
                },
                Moderate: {
                    intro: "Risk of Asthma symptoms is Moderate when AQI is Moderate (51-100)",
                    description: "Unusually sensitive individuals may experience respiratory symptoms.",
                    dos: [
                        "Consider limiting prolonged outdoor exertion",
                        "Keep windows closed during peak traffic hours",
                        "Have your inhaler accessible"
                    ],
                    donts: [
                        "Exercise vigorously outdoors if sensitive",
                        "Expose yourself to additional irritants (smoke, strong perfumes)"
                    ]
                },
                "Unhealthy for Sensitive Groups": {
                    intro: "Risk of Asthma symptoms is Mid when AQI is Poor (101-150)",
                    description: "Moderate symptoms including frequent wheezing, noticeable shortness of breath, chest tightness, and persistent cough.",
                    dos: [
                        "Limit outdoor activities when AQI is poor",
                        "Clean indoor air with an air purifier to reduce exposure",
                        "Soothe the respiratory tract with herbal teas or warm water to help alleviate symptoms"
                    ],
                    donts: [
                        "Exercise outdoors without a mask",
                        "Stay in smoky areas with strong fumes"
                    ]
                },
                Unhealthy: {
                    intro: "Risk of Asthma symptoms is High when AQI is Unhealthy (151-200)",
                    description: "Increased likelihood of asthma attacks and worsening symptoms in all asthmatic individuals.",
                    dos: [
                        "Stay indoors with windows closed",
                        "Use air purifiers with HEPA filters",
                        "Follow your asthma action plan and keep rescue medications handy",
                        "Consider wearing N95 masks if going outdoors is necessary"
                    ],
                    donts: [
                        "Exercise outdoors",
                        "Venture outside without necessary medication",
                        "Ignore worsening symptoms"
                    ]
                },
                "Very Unhealthy": {
                    intro: "Risk of Asthma symptoms is Very High when AQI is Very Unhealthy (201-300)",
                    description: "Serious risk of asthma attacks. Significant breathing difficulties likely for asthmatics.",
                    dos: [
                        "Stay indoors in clean air environments",
                        "Use air purifiers continuously",
                        "Contact healthcare provider to adjust medication if needed",
                        "Monitor peak flow readings regularly"
                    ],
                    donts: [
                        "Go outdoors for any non-essential activities",
                        "Open windows or doors",
                        "Delay medical attention if experiencing severe symptoms"
                    ]
                },
                Hazardous: {
                    intro: "Risk of Asthma symptoms is Severe when AQI is Hazardous (301-500)",
                    description: "Emergency conditions for asthmatics. High risk of severe attacks requiring medical intervention.",
                    dos: [
                        "Stay indoors with air purification",
                        "Keep emergency medications readily available",
                        "Seek medical advice about preventive medication adjustments",
                        "Have an emergency plan ready"
                    ],
                    donts: [
                        "Go outdoors under any circumstances if avoidable",
                        "Exercise, even indoors, if sensitive",
                        "Wait to use rescue medications if symptoms begin"
                    ]
                }
            }
        },
        heart: {
            title: "Heart Issues",
            icon: <Heart size={24} />,
            image: "heart_health.svg",
            info: {
                Good: {
                    intro: "Risk to heart health is Low when AQI is Good (0-50)",
                    description: "Minimal cardiovascular risk. Safe for normal activities.",
                    dos: [
                        "Maintain regular exercise routines",
                        "Follow heart-healthy lifestyle",
                        "Take prescribed medications as usual"
                    ],
                    donts: [
                        "Neglect regular check-ups",
                        "Ignore unusual symptoms"
                    ]
                },
                Moderate: {
                    intro: "Risk to heart health is Moderate when AQI is Moderate (51-100)",
                    description: "People with pre-existing heart conditions may notice mild effects.",
                    dos: [
                        "Monitor your heart rate during outdoor activities",
                        "Stay hydrated",
                        "Consider indoor exercise if sensitive"
                    ],
                    donts: [
                        "Overexert yourself outdoors",
                        "Ignore any unusual heart symptoms"
                    ]
                },
                "Unhealthy for Sensitive Groups": {
                    intro: "Risk to heart health is Medium when AQI is Poor (101-150)",
                    description: "Increased strain on the cardiovascular system, especially for those with existing heart conditions.",
                    dos: [
                        "Reduce outdoor physical activity",
                        "Monitor blood pressure regularly",
                        "Stay well-hydrated",
                        "Take prescribed heart medications as directed"
                    ],
                    donts: [
                        "Engage in strenuous outdoor activities",
                        "Ignore chest pain or irregular heartbeats",
                        "Consume excessive caffeine or stimulants"
                    ]
                },
                Unhealthy: {
                    intro: "Risk to heart health is High when AQI is Unhealthy (151-200)",
                    description: "Significant cardiovascular stress. Heart attack risk increases.",
                    dos: [
                        "Stay indoors with filtered air",
                        "Monitor blood pressure more frequently",
                        "Limit all physical exertion",
                        "Contact healthcare provider if experiencing any cardiac symptoms"
                    ],
                    donts: [
                        "Exercise outdoors",
                        "Delay seeking medical attention for chest discomfort",
                        "Consume alcohol which can dehydrate you"
                    ]
                },
                "Very Unhealthy": {
                    intro: "Risk to heart health is Very High when AQI is Very Unhealthy (201-300)",
                    description: "High risk of cardiovascular events. Significant heart strain likely.",
                    dos: [
                        "Remain indoors with air purification",
                        "Have emergency contact information readily available",
                        "Monitor for symptoms like chest pain or shortness of breath",
                        "Consider contacting doctor about medication adjustments"
                    ],
                    donts: [
                        "Go outdoors for any non-essential reasons",
                        "Engage in physical activity, even indoors",
                        "Ignore even mild heart-related symptoms"
                    ]
                },
                Hazardous: {
                    intro: "Risk to heart health is Severe when AQI is Hazardous (301-500)",
                    description: "Emergency level risk for cardiovascular events, especially for elderly and those with heart disease.",
                    dos: [
                        "Stay in clean air environment",
                        "Keep emergency medications accessible",
                        "Have emergency numbers ready",
                        "Consider evacuation to cleaner air location if possible"
                    ],
                    donts: [
                        "Expose yourself to outdoor air",
                        "Delay seeking medical help for any cardiac symptoms",
                        "Engage in any unnecessary physical activity"
                    ]
                }
            }
        },
        allergies: {
            title: "Allergies",
            icon: <Flower size={24} />,
            image: "allergies.svg",
            info: {
                Good: {
                    intro: "Risk of allergic reactions is Low when AQI is Good (0-50)",
                    description: "Minimal risk of allergy symptoms from air pollution.",
                    dos: [
                        "Enjoy outdoor activities normally",
                        "Monitor pollen counts separately from AQI",
                        "Continue regular allergy medications as prescribed"
                    ],
                    donts: [
                        "Skip allergy medications during high pollen seasons",
                        "Confuse pollen allergies with air pollution effects"
                    ]
                },
                Moderate: {
                    intro: "Risk of allergic reactions is Moderate when AQI is Moderate (51-100)",
                    description: "Sensitive individuals may experience mild allergy symptoms.",
                    dos: [
                        "Consider taking allergy medication before outdoor activities",
                        "Rinse sinuses with saline after extended outdoor exposure",
                        "Keep windows closed during high pollution periods"
                    ],
                    donts: [
                        "Dry laundry outdoors where it can collect allergens",
                        "Spend extended time outdoors if experiencing symptoms"
                    ]
                },
                "Unhealthy for Sensitive Groups": {
                    intro: "Risk of allergic reactions is Medium when AQI is Poor (101-150)",
                    description: "Pollutants can worsen allergic responses and trigger symptoms like runny nose, itchy eyes, and sneezing.",
                    dos: [
                        "Take allergy medications proactively",
                        "Use air purifiers with HEPA filters indoors",
                        "Shower after coming indoors to remove allergens",
                        "Consider wearing wrap-around sunglasses outdoors"
                    ],
                    donts: [
                        "Leave windows open",
                        "Spend unnecessary time outdoors",
                        "Touch face or eyes while outdoors"
                    ]
                },
                Unhealthy: {
                    intro: "Risk of allergic reactions is High when AQI is Unhealthy (151-200)",
                    description: "High likelihood of intensified allergic responses, even in those with mild allergies.",
                    dos: [
                        "Stay indoors with air purification",
                        "Use nasal irrigation to clear allergens",
                        "Consider wearing masks (N95) if outdoors is necessary",
                        "Keep allergy medications readily available"
                    ],
                    donts: [
                        "Exercise outdoors",
                        "Open windows for ventilation",
                        "Mix cleaning chemicals which can create additional irritants"
                    ]
                },
                "Very Unhealthy": {
                    intro: "Risk of allergic reactions is Very High when AQI is Very Unhealthy (201-300)",
                    description: "Severe allergy symptoms likely. Existing allergies significantly worsened.",
                    dos: [
                        "Stay indoors in clean air",
                        "Consider additional allergy medications (consult doctor)",
                        "Use air purifiers in sleeping areas especially",
                        "Monitor for symptoms of respiratory distress"
                    ],
                    donts: [
                        "Venture outdoors unnecessarily",
                        "Use scented products that may add to irritation",
                        "Ignore progressive worsening of symptoms"
                    ]
                },
                Hazardous: {
                    intro: "Risk of allergic reactions is Severe when AQI is Hazardous (301-500)",
                    description: "Emergency level risk for allergy sufferers. Severe reactions requiring medical attention possible.",
                    dos: [
                        "Remain in clean air environments only",
                        "Have emergency allergy medications ready",
                        "Contact healthcare provider about preventive measures",
                        "Consider temporary relocation if possible"
                    ],
                    donts: [
                        "Go outdoors under any circumstances",
                        "Delay seeking medical help for severe reactions",
                        "Open windows or doors"
                    ]
                }
            }
        },
        sinus: {
            title: "Sinus",
            icon: <Thermometer size={24} />,
            image: "sinus_problems.svg",
            info: {
                Good: {
                    intro: "Risk of sinus issues is Low when AQI is Good (0-50)",
                    description: "Minimal impact on sinus conditions.",
                    dos: [
                        "Maintain normal sinus care routines",
                        "Stay hydrated to keep mucus thin",
                        "Get fresh air exposure"
                    ],
                    donts: [
                        "Ignore persistent sinus symptoms",
                        "Neglect proper hydration"
                    ]
                },
                Moderate: {
                    intro: "Risk of sinus issues is Moderate when AQI is Moderate (51-100)",
                    description: "Some individuals may notice mild sinus congestion or pressure.",
                    dos: [
                        "Use saline nasal sprays to keep passages moist",
                        "Drink plenty of fluids",
                        "Consider using a humidifier indoors"
                    ],
                    donts: [
                        "Use excessive decongestants which can dry mucous membranes",
                        "Spend extended periods in polluted areas if sensitive"
                    ]
                },
                "Unhealthy for Sensitive Groups": {
                    intro: "Risk of sinus issues is Medium when AQI is Poor (101-150)",
                    description: "Noticeable increase in sinus pressure, congestion, and possible headaches.",
                    dos: [
                        "Use nasal irrigation (neti pot) to flush irritants",
                        "Apply warm compresses to relieve sinus pressure",
                        "Run air purifiers in living spaces",
                        "Stay well-hydrated"
                    ],
                    donts: [
                        "Expose yourself to additional irritants like cigarette smoke",
                        "Skip sinus medications if you use them regularly",
                        "Sleep with windows open"
                    ]
                },
                Unhealthy: {
                    intro: "Risk of sinus issues is High when AQI is Unhealthy (151-200)",
                    description: "Significant worsening of sinus conditions including pain, pressure, and congestion.",
                    dos: [
                        "Stay indoors with filtered air",
                        "Use nasal irrigation more frequently",
                        "Apply steam therapy to help clear sinuses",
                        "Consider wearing an N95 mask if going outdoors"
                    ],
                    donts: [
                        "Spend time outdoors unnecessarily",
                        "Use wood-burning fireplaces or candles indoors",
                        "Ignore symptoms of sinus infections"
                    ]
                },
                "Very Unhealthy": {
                    intro: "Risk of sinus issues is Very High when AQI is Very Unhealthy (201-300)",
                    description: "Severe sinus inflammation likely. Risk of infections increases.",
                    dos: [
                        "Stay in clean air environments",
                        "Use air purifiers continuously",
                        "Contact healthcare provider if symptoms worsen",
                        "Consider prescribed anti-inflammatory medication"
                    ],
                    donts: [
                        "Go outdoors if avoidable",
                        "Use irritating cleaning products",
                        "Delay treatment for possible sinus infections"
                    ]
                },
                Hazardous: {
                    intro: "Risk of sinus issues is Severe when AQI is Hazardous (301-500)",
                    description: "Emergency level risk for sinus conditions. Severe pain and complications possible.",
                    dos: [
                        "Stay exclusively in clean air environments",
                        "Seek medical advice for preventive care",
                        "Use prescribed medications diligently",
                        "Monitor for fever or signs of infection"
                    ],
                    donts: [
                        "Expose yourself to outdoor air",
                        "Skip medications",
                        "Delay seeking medical attention for severe symptoms"
                    ]
                }
            }
        },
        coldFlu: {
            title: "Cold/Flu",
            icon: <Bug size={24} />,
            image: "cold_flu.svg",
            info: {
                Good: {
                    intro: "Impact on cold/flu recovery is Minimal when AQI is Good (0-50)",
                    description: "Clean air supports respiratory recovery from illnesses.",
                    dos: [
                        "Get fresh air if feeling up to it",
                        "Continue normal recovery practices",
                        "Stay hydrated and rest"
                    ],
                    donts: [
                        "Overexert yourself even in clean air",
                        "Ignore worsening symptoms"
                    ]
                },
                Moderate: {
                    intro: "Impact on cold/flu recovery is Slight when AQI is Moderate (51-100)",
                    description: "Some minor irritation may slow recovery for respiratory illnesses.",
                    dos: [
                        "Limit outdoor time to shorter periods",
                        "Use humidifiers to keep air moist",
                        "Increase fluid intake"
                    ],
                    donts: [
                        "Exercise outdoors while recovering",
                        "Expose yourself to crowds and additional germs"
                    ]
                },
                "Unhealthy for Sensitive Groups": {
                    intro: "Impact on cold/flu recovery is Moderate when AQI is Poor (101-150)",
                    description: "Recovery from respiratory illnesses may be delayed. Symptoms could be aggravated.",
                    dos: [
                        "Stay indoors with clean filtered air",
                        "Use steam therapy to soothe irritated airways",
                        "Take all prescribed medications",
                        "Increase rest time"
                    ],
                    donts: [
                        "Go outdoors unnecessarily",
                        "Return to normal activities prematurely",
                        "Skip flu/cold medications"
                    ]
                },
                Unhealthy: {
                    intro: "Impact on cold/flu recovery is Significant when AQI is Unhealthy (151-200)",
                    description: "Greatly prolonged recovery time and risk of complications for respiratory illnesses.",
                    dos: [
                        "Stay strictly indoors with air purification",
                        "Contact healthcare provider about management",
                        "Monitor fever and breathing carefully",
                        "Increase fluid intake significantly"
                    ],
                    donts: [
                        "Expose yourself to outdoor pollution",
                        "Delay seeking help if breathing becomes difficult",
                        "Use decongestants without medical advice"
                    ]
                },
                "Very Unhealthy": {
                    intro: "Impact on cold/flu recovery is Severe when AQI is Very Unhealthy (201-300)",
                    description: "High risk of serious complications from respiratory illnesses. Recovery significantly impaired.",
                    dos: [
                        "Stay in clean air environment exclusively",
                        "Seek medical advice even for common cold symptoms",
                        "Monitor oxygen levels if possible",
                        "Consider prescribed medications for symptom management"
                    ],
                    donts: [
                        "Leave clean air environments",
                        "Ignore any difficulty breathing",
                        "Delay medical consultation"
                    ]
                },
                Hazardous: {
                    intro: "Impact on cold/flu recovery is Critical when AQI is Hazardous (301-500)",
                    description: "Extremely high risk of pneumonia and other serious complications from respiratory illnesses.",
                    dos: [
                        "Seek medical consultation even for mild symptoms",
                        "Stay exclusively in clean air",
                        "Follow medical advice strictly",
                        "Consider evacuation to area with better air quality"
                    ],
                    donts: [
                        "Expose yourself to outdoor air at all",
                        "Self-medicate without medical guidance",
                        "Ignore even minor changes in breathing ability"
                    ]
                }
            }
        },
        copd: {
            title: "Chronic (COPD)",
            icon: <Activity size={24} />,
            image: "copd_health.svg",
            info: {
                Good: {
                    intro: "Risk to COPD patients is Low when AQI is Good (0-50)",
                    description: "Minimal impact on COPD symptoms. Safe for normal prescribed activities.",
                    dos: [
                        "Follow regular COPD management plan",
                        "Enjoy light outdoor activities as tolerated",
                        "Take medications as prescribed"
                    ],
                    donts: [
                        "Skip medications",
                        "Overexert beyond your normal capacity"
                    ]
                },
                Moderate: {
                    intro: "Risk to COPD patients is Moderate when AQI is Moderate (51-100)",
                    description: "Some COPD patients may notice slight increase in symptoms.",
                    dos: [
                        "Limit outdoor activities to cooler parts of day",
                        "Keep rescue inhalers readily available",
                        "Monitor breathing more carefully",
                        "Stay well-hydrated"
                    ],
                    donts: [
                        "Exercise vigorously outdoors",
                        "Spend extended periods in unfiltered air"
                    ]
                },
                "Unhealthy for Sensitive Groups": {
                    intro: "Risk to COPD patients is Substantial when AQI is Poor (101-150)",
                    description: "Likely worsening of COPD symptoms including increased breathlessness and coughing.",
                    dos: [
                        "Stay indoors with air purification",
                        "Use oxygen as prescribed",
                        "Monitor oxygen saturation if possible",
                        "Have action plan ready for symptom worsening"
                    ],
                    donts: [
                        "Go outdoors without a properly fitted N95 mask",
                        "Exert yourself physically",
                        "Expose yourself to other lung irritants"
                    ]
                },
                Unhealthy: {
                    intro: "Risk to COPD patients is High when AQI is Unhealthy (151-200)",
                    description: "Serious risk of COPD exacerbation requiring medical intervention.",
                    dos: [
                        "Stay strictly indoors with air purification",
                        "Use prescribed medications including oxygen",
                        "Contact healthcare provider about preventive adjustments",
                        "Monitor symptoms and oxygen levels carefully"
                    ],
                    donts: [
                        "Go outdoors for any non-essential reason",
                        "Delay using rescue medications if breathing worsens",
                        "Wait to call doctor if unusual symptoms develop"
                    ]
                },
                "Very Unhealthy": {
                    intro: "Risk to COPD patients is Very High when AQI is Very Unhealthy (201-300)",
                    description: "Severe risk of acute exacerbation and hospitalization. Emergency planning recommended.",
                    dos: [
                        "Stay exclusively in clean air environments",
                        "Have emergency contact information ready",
                        "Consider contacting doctor for preventive medications",
                        "Use nebulizer treatments as directed"
                    ],
                    donts: [
                        "Expose yourself to outdoor air at all",
                        "Delay seeking emergency care for significant breathing changes",
                        "Attempt to manage severe symptoms at home alone"
                    ]
                },
                Hazardous: {
                    intro: "Risk to COPD patients is Extreme when AQI is Hazardous (301-500)",
                    description: "Life-threatening conditions for COPD patients. Immediate precautions necessary.",
                    dos: [
                        "Contact healthcare provider immediately for guidance",
                        "Consider evacuation to area with better air quality",
                        "Use all prescribed emergency medications",
                        "Have emergency transport plan ready"
                    ],
                    donts: [
                        "Go outdoors under any circumstances",
                        "Wait to seek emergency care if breathing deteriorates",
                        "Rely solely on rescue inhalers for severe symptoms"
                    ]
                }
            }
        }
    };

    // Safely get selected condition information
    const selectedInfo = healthConditions[selectedCondition]?.info?.[aqiCategory.level] || {
        description: 'No data available for this condition.',
        dos: [],
        donts: []
    };

    // Handle condition button click
    const handleConditionClick = (condition) => {
        setSelectedCondition(condition);
    };
    const getFadedColor = (color) => {
        const fadedColors = {
            "#00e400": "rgba(174, 228, 174, 0.7)",   // Good → Faded Green
            "#ffff00": "rgba(238, 238, 187, 0.7)", // Moderate → Faded Yellow
            "#ff7e00": "rgba(255, 126, 0, 0.7)", // Unhealthy for Sensitive Groups → Faded Orange
            "#ff0000": "rgba(226, 141, 182, 0.6)",   // Unhealthy → Faded Red
            "#8f3f97": "rgba(221, 152, 227, 0.7)",// Very Unhealthy → Faded Purple
            "#7e0023": "rgba(223, 142, 165, 0.7)"   // Hazardous → Faded Dark Red
        };
        return fadedColors[color] || color; // Default to original if not found
    };

    return (
        <div className="health-risks-container">
            <h2 className="health-risks-title">Prevent Health Problems: Understand Your Risks</h2>
            <h3 className="health-risks-location">
                {/* Optionally add location or station name here */}
            </h3>

            <div className="condition-buttons">
                <button
                    className={`condition-button ${selectedCondition === 'asthma' ? 'selected' : ''}`}
                    onClick={() => handleConditionClick('asthma')}
                >
                    <Wind size={20} /> Asthma
                </button>
                <button
                    className={`condition-button ${selectedCondition === 'heart' ? 'selected' : ''}`}
                    onClick={() => handleConditionClick('heart')}
                >
                    <Heart size={20} /> Heart Issues
                </button>
                <button
                    className={`condition-button ${selectedCondition === 'allergies' ? 'selected' : ''}`}
                    onClick={() => handleConditionClick('allergies')}
                >
                    <Flower size={20} /> Allergies
                </button>
                <button
                    className={`condition-button ${selectedCondition === 'sinus' ? 'selected' : ''}`}
                    onClick={() => handleConditionClick('sinus')}
                >
                    <Thermometer size={20} /> Sinus
                </button>
                <button
                    className={`condition-button ${selectedCondition === 'coldFlu' ? 'selected' : ''}`}
                    onClick={() => handleConditionClick('coldFlu')}
                >
                    <Bug size={20} /> Cold/Flu
                </button>
                <button
                    className={`condition-button ${selectedCondition === 'copd' ? 'selected' : ''}`}
                    onClick={() => handleConditionClick('copd')}
                >
                    <Activity size={20} /> Chronic (COPD)
                </button>
            </div>

            <div className="health-content-grid">
                <div className="placeholder-image" style={{ backgroundColor: getFadedColor(aqiCategory.color), padding: '20px', borderRadius: '10px' }}>
                    {healthConditions[selectedCondition]?.image ? (
                        <img
                            src={`/${healthConditions[selectedCondition].image}`}
                            alt={healthConditions[selectedCondition]?.title}
                            style={{ width: '60%', height: 'auto', borderRadius: '10px' }}
                        />
                    ) : healthConditions[selectedCondition]?.icon && (
                        <div style={{ fontSize: '100px', display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                            {healthConditions[selectedCondition].icon}
                        </div>
                    )}

                    <div
                        className="risk-indicator"
                        style={{
                            backgroundColor: aqiCategory.color,
                            color: '#000', // Force black text
                        }}
                    >
                        {aqiCategory.severity} Chances of {healthConditions[selectedCondition]?.title}
                    </div>

                </div>

                <div className="health-info-container">
                    <h3 className="condition-title">{healthConditions[selectedCondition]?.title}</h3>

                    <p className="condition-risk">
                        <span className="risk-label">Risk of {healthConditions[selectedCondition]?.title} symptoms is </span>
                        <span className="risk-level" style={{ color: aqiCategory.color }}>{aqiCategory.severity}</span>
                        <span> when AQI is {aqiCategory.level} ({aqiCategory.range})</span>
                    </p>

                    <p className="condition-description">{selectedInfo.description}</p>

                    <div className="recommendations">
                        <div className="dos">
                            <h4>Do's :</h4>
                            <ul>
                                {selectedInfo.dos.map((item, index) => (
                                    <li key={`do-${index}`} className="do-item">
                                        <span className="check-icon">✓</span> {item}
                                    </li>
                                ))}
                            </ul>
                        </div>

                        <div className="donts">
                            <h4>Don'ts :</h4>
                            <ul>
                                {selectedInfo.donts.map((item, index) => (
                                    <li key={`dont-${index}`} className="dont-item">
                                        <span className="x-icon">✕</span> {item}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default HealthRisksComponent;