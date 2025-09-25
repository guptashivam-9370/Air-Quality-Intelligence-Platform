import React from 'react';
import './Component2.css'; // <-- Import your new CSS file

const Component2 = () => {
  return (
    <div className="component2">
      <h1 className="title">India 2019 PM2.5 Death Toll</h1>

      <div className="boxes">
        {/* PM2.5 Levels Box */}
        <div className="box-red">
          <p className="text-xl">PM2.5 annual mean levels</p>

          <div className="mt-4">
            <p className="text-8xl">83</p>
            <p className="text-lg">μg/m³</p>
            <p className="mt-2">each person's annual mean exposure</p>
            <p className="text-xl mt-1">
              <span className="font-bold">16.6</span> times WHO's guideline.
            </p>
          </div>
        </div>

        {/* Deaths Box */}
        <div className="box-death">
          <div className="flex mb-4">
            <img
              src="/pm2_5/component2.webp"
              alt="Skull and crossbones"
              className="mr-4"
            />
            <p className="text-8xl">70</p>
          </div>

          <p className="text-3xl">Deaths per 100,000 people</p>
          <p className="text-gray-300">
            attributable to fine particle pollution in 2019 (979,682 in total in
            the country).
          </p>
        </div>
      </div>

      <h2 className="subtitle">Death percentage as per diseases</h2>

      {/* Disease List */}
      <div className="disease-list">
        {diseaseData.map((disease) => (
          <div key={disease.name} className="disease-item">
            <div className="icon">{disease.icon}</div>
            <div className="name">
              <p>{disease.name}</p>
            </div>
            <div className="progress">
              <div
                className="progress-bar"
                style={{ width: `${disease.percentage}%` }}
              />
              <span className="percentage">{disease.percentage}%</span>
            </div>
          </div>
        ))}
      </div>

      <div className="source">*Source: UNEP</div>
    </div>
  );
};

// Same diseaseData array as before
const diseaseData = [
  {
    name: "Chronic Obstructive Pulmonary Disease",
    percentage: 30,
    icon: (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="24"
        height="24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M12 22a7 7 0 0 0 7-7" />
        <path d="M9 9a3 3 0 0 1 3-3" />
        <path d="M11 3a9 9 0 0 1 9 9" />
        <path d="M12 3h.393A7.5 7.5 0 0 1 19.5 6h0A1.5 1.5 0 0 1 18 7.5" />
        <path d="M13.5 13.5H12a1.5 1.5 0 0 0-1.5 1.5v0a1.5 1.5 0 0 0 3 0M12 22v-8.5"></path>
      </svg>
    ),
  },
  {
    name: "Lower Respiratory Infections",
    percentage: 24,
    icon: (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="24"
        height="24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M8 12h8m-4-4v8M9 9h6" />
        <path d="M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z" />
      </svg>
    ),
  },
  {
    name: "Stroke",
    percentage: 23,
    icon: (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="24"
        height="24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M12 12c-2.8 0-5-2.2-5-5s2.2-5 5-5 5 2.2 5 5" />
        <path d="M7 12h10" />
        <path d="M12 7v15" />
      </svg>
    ),
  },
  {
    name: "Ischemic Heart Disease",
    percentage: 22,
    icon: (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="24"
        height="24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3.332.802-4.5 2.05A5.5 5.5 0 0 0 2 8.5c0 2.29 1.5 4.04 3 5.5l7 7Z" />
      </svg>
    ),
  },
  {
    name: "Tracheal, Bronchus, And Lung Cancer",
    percentage: 21,
    icon: (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="24"
        height="24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <circle cx="12" cy="12" r="10" />
        <path d="m8 12 2 2 4-4" />
      </svg>
    ),
  },
  {
    name: "Type 2 Diabetes",
    percentage: 16,
    icon: (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="24"
        height="24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M12 2v6.5l3 1.72" />
        <path d="M15.873 15.873a7.5 7.5 0 1 0-3.873 1.127" />
      </svg>
    ),
  },
  {
    name: "Neonatal Disorders",
    percentage: 11,
    icon: (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="24"
        height="24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M9 12h.01" />
        <path d="M15 12h.01" />
        <path d="M10 16c.5.3 1.5.5 2 .5s1.5-.2 2-.5" />
        <path d="M19 6.3a9 9 0 0 1 1.8 3.9 2 2 0 0 1 0 3.6 9 9 0 0 1-17.6 0 2 2 0 0 1 0-3.6A9 9 0 0 1 5 6.3" />
        <path d="M3 13c0-2.1 4-3.6 9-3.6s9 1.5 9 3.6" />
      </svg>
    ),
  },
];

export default Component2;
