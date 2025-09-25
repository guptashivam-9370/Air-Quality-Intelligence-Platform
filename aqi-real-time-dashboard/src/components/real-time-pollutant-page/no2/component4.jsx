import React from 'react';

const NO2SourcesComponent = () => {
  const sources = [
    {
      id: 1,
      title: 'Explosives and welding',
      description: 'Release during high-temperature processes.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path d="M12 2v2m0 4v2m0 4v2m0 4v2m-6-8h2m4 0h2m4 0h2" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    },
    {
      id: 2,
      title: 'Lighting',
      description: 'Emissions from certain types of lighting.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path d="M13 10V3L4 14h7v7l9-11h-7z" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    },
    {
      id: 3,
      title: 'Power-generating plants',
      description: 'Emissions from fossil fuels burning for electricity.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path d="M4 21h16V9l-4 2V6l-4 2V6L4 9v12z" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    },
    {
      id: 4,
      title: 'Road traffic',
      description: 'Overall contributions from vehicles on the road.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path d="M5 17l1.5-5h11L19 17H5zM7 17v2a1 1 0 001 1h2a1 1 0 001-1v-2m4 0v2a1 1 0 001 1h2a1 1 0 001-1v-2" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    },
    {
      id: 5,
      title: 'Motor vehicles',
      description: 'Exhaust gases from cars and trucks.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path d="M4 17h16M6 9h12l1.5 5H4.5L6 9zM7 17v2a1 1 0 001 1h2a1 1 0 001-1v-2m4 0v2a1 1 0 001 1h2a1 1 0 001-1v-2" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    }
  ];

  return (
    <div className="no2-container">
      <div className="no2-content">
        <div className="no2-header">
          <h1>Uncovering the Sources of Nitrogen Dioxide (NO₂):</h1>
          <h2>Where Does It Come From?</h2>
        </div>

        <div className="no2-sources-grid">
          {sources.map(source => (
            <div key={source.id} className="no2-source-card">
              <div className="no2-source-content">
                <div className="no2-source-icon">{source.icon}</div>
                <div className="no2-source-text">
                  <h3>{source.title}</h3>
                  <p>{source.description}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <style jsx>{`
        .no2-container {
          position: relative;
          width: 100%;
          background-color: rgba(0, 0, 0, 0.5);
          background-image: url('https://www.aqi.in/media/sections/pollutant-sources/backgrounds/no2.webp');
          background-size: cover;
          background-position: center;
          background-blend-mode: overlay;
        }

        .no2-content {
          max-width: 1200px;
          margin: 0 auto;
          padding: 48px 16px;
        }

        .no2-header {
          max-width: 900px;
          margin: 0 auto 48px;
          text-align: center;
        }

        .no2-header h1 {
          font-size: 32px;
          font-weight: bold;
          color: white;
          margin-bottom: 16px;
        }

        .no2-header h2 {
          font-size: 28px;
          font-weight: bold;
          color: white;
        }

        .no2-sources-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
          gap: 24px;
        }

        .no2-source-card {
          background-color: rgba(51, 51, 51, 0.7);
          border-radius: 8px;
          padding: 16px;
          backdrop-filter: blur(4px);
          border: 1px solid rgba(119, 119, 119, 0.7);
        }

        .no2-source-content {
          display: flex;
          align-items: flex-start;
        }

        .no2-source-icon {
          color: white;
          margin-right: 12px;
          flex-shrink: 0;
        }

        .no2-source-icon svg {
          width: 24px;
          height: 24px;
        }

        .no2-source-text h3 {
          color: white;
          font-weight: bold;
          margin-bottom: 4px;
          font-size: 16px;
        }

        .no2-source-text p {
          color: rgb(209, 213, 219);
          font-size: 14px;
        }

        @media (max-width: 768px) {
          .no2-header h1 {
            font-size: 24px;
          }

          .no2-header h2 {
            font-size: 20px;
          }
        }
      `}</style>
    </div>
  );
};

export default NO2SourcesComponent;
