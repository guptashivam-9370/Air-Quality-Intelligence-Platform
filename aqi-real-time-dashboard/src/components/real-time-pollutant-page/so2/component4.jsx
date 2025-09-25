import React from 'react';

const SO2SourcesComponent = () => {
  const sources = [
    {
      id: 1,
      title: 'Chemical manufacturing',
      description: 'The production processes that emit sulfur dioxide.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path d="M4 14h16M4 10h16M5 6h14M4 18h16" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    },
    {
      id: 2,
      title: 'Industries',
      description: 'Emissions from manufacturing and industrial processes.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path d="M3 21h18V9l-6 3V6l-6 3V6L3 9v12z" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    },
    {
      id: 3,
      title: 'Power plants generation',
      description: 'Released during the burning of fossil fuels for electricity.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path d="M4 21h16V9l-4 2V6l-4 2V6L4 9v12z" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    },
    {
      id: 4,
      title: 'Petroleum Refineries',
      description: 'Emits from refining crude oil into various products.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path d="M4 4h16v4H4V4zM4 8v12h16V8M8 12h8M8 16h8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    },
    {
      id: 5,
      title: 'Heavy-duty vehicles',
      description: 'Exhaust from trucks and buses using diesel fuel.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path d="M4 17h16M6 9h12l1.5 5H4.5L6 9zM7 17v2a1 1 0 001 1h2a1 1 0 001-1v-2m4 0v2a1 1 0 001 1h2a1 1 0 001-1v-2" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    }
  ];

  return (
    <div className="so2-container">
      <div className="so2-content">
        <div className="so2-header">
          <h1>Uncovering the Sources of Sulfur Dioxide (SO₂):</h1>
          <h2>Where Does It Come From?</h2>
        </div>

        <div className="so2-sources-grid">
          {sources.map(source => (
            <div key={source.id} className="so2-source-card">
              <div className="so2-source-content">
                <div className="so2-source-icon">{source.icon}</div>
                <div className="so2-source-text">
                  <h3>{source.title}</h3>
                  <p>{source.description}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <style jsx>{`
        .so2-container {
          position: relative;
          width: 100%;
          background-color: rgba(0, 0, 0, 0.5);
          background-image: url('https://www.aqi.in/media/sections/pollutant-sources/backgrounds/so2.webp');
          background-size: cover;
          background-position: center;
          background-blend-mode: overlay;
        }

        .so2-content {
          max-width: 1200px;
          margin: 0 auto;
          padding: 48px 16px;
        }

        .so2-header {
          max-width: 900px;
          margin: 0 auto 48px;
          text-align: center;
        }

        .so2-header h1 {
          font-size: 32px;
          font-weight: bold;
          color: white;
          margin-bottom: 16px;
        }

        .so2-header h2 {
          font-size: 28px;
          font-weight: bold;
          color: white;
        }

        .so2-sources-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
          gap: 24px;
        }

        .so2-source-card {
          background-color: rgba(51, 51, 51, 0.7);
          border-radius: 8px;
          padding: 16px;
          backdrop-filter: blur(4px);
          border: 1px solid rgba(119, 119, 119, 0.7);
        }

        .so2-source-content {
          display: flex;
          align-items: flex-start;
        }

        .so2-source-icon {
          color: white;
          margin-right: 12px;
          flex-shrink: 0;
        }

        .so2-source-icon svg {
          width: 24px;
          height: 24px;
        }

        .so2-source-text h3 {
          color: white;
          font-weight: bold;
          margin-bottom: 4px;
          font-size: 16px;
        }

        .so2-source-text p {
          color: rgb(209, 213, 219);
          font-size: 14px;
        }

        @media (max-width: 768px) {
          .so2-header h1 {
            font-size: 24px;
          }

          .so2-header h2 {
            font-size: 20px;
          }
        }
      `}</style>
    </div>
  );
};

export default SO2SourcesComponent;
