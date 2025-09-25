import React from 'react';

const PM10SourcesComponent = () => {
  const sources = [
    {
      id: 1,
      title: 'Wind-blown dust',
      description: 'Dust lifted and spread by the wind from bare soil.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path d="M9.59 4.59A2 2 0 1 1 11 8H2m10.59 11.41A2 2 0 1 0 14 16H2m15.73-8.27A2.5 2.5 0 1 1 19.5 12H2" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      )
    },
    {
      id: 2,
      title: 'Construction sites',
      description: 'Dust and pollutants from building activities.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path d="M3 21h18M9 8v4m6-4v4M4 17l2-6h12l2 6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    },
    {
      id: 3,
      title: 'Industries',
      description: 'Releases various pollutants from different processes.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path d="M3 21h18V9l-6 3V6l-6 3V6L3 9v12z" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    },
    {
      id: 4,
      title: 'Waste burning',
      description: 'Smoke and toxins from burning waste materials.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path d="M12 3c2 1 3 3 3 5s-1 3-3 3-3-1-3-3 1-4 3-5zm0 9v9" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    },
    {
      id: 5,
      title: 'Landfills',
      description: 'Emissions from decomposing waste in big landfills.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path d="M3 22v-7l7-5v5l4-3v3l7-5v7H3z" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    },
    {
      id: 6,
      title: 'Vehicles exhausts',
      description: 'Emissions of harmful gases and particles from cars.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path d="M5 17h14l2-5-2-5H5L3 12l2 5zM6 17v2a1 1 0 001 1h2a1 1 0 001-1v-2M14 17v2a1 1 0 001 1h2a1 1 0 001-1v-2" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    }
  ];

  return (
    <div className="pm10-container">
      <div className="pm10-content">
        <div className="pm10-header">
          <h1>Uncovering the Sources of Particulate Matter (PM10):</h1>
          <h2>Where Does It Come From?</h2>
        </div>

        <div className="pm10-sources-grid">
          {sources.map(source => (
            <div key={source.id} className="pm10-source-card">
              <div className="pm10-source-content">
                <div className="pm10-source-icon">{source.icon}</div>
                <div className="pm10-source-text">
                  <h3>{source.title}</h3>
                  <p>{source.description}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <style jsx>{`
        .pm10-container {
          position: relative;
          width: 100%;
          background-color: rgba(0, 0, 0, 0.5);
          background-image: url('https://www.aqi.in/media/sections/pollutant-sources/backgrounds/pm10.webp');
          background-size: cover;
          background-position: center;
          background-blend-mode: overlay;
        }

        .pm10-content {
          max-width: 1200px;
          margin: 0 auto;
          padding: 48px 16px;
        }

        .pm10-header {
          max-width: 800px;
          margin: 0 auto 48px;
          text-align: center;
        }

        .pm10-header h1 {
          font-size: 32px;
          font-weight: bold;
          color: white;
          margin-bottom: 16px;
        }

        .pm10-header h2 {
          font-size: 28px;
          font-weight: bold;
          color: white;
        }

        .pm10-sources-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
          gap: 24px;
        }

        .pm10-source-card {
          background-color: rgba(51, 51, 51, 0.7);
          border-radius: 8px;
          padding: 16px;
          backdrop-filter: blur(4px);
          border: 1px solid rgba(119, 119, 119, 0.7);
        }

        .pm10-source-content {
          display: flex;
          align-items: flex-start;
        }

        .pm10-source-icon {
          color: white;
          margin-right: 12px;
          flex-shrink: 0;
        }

        .pm10-source-icon svg {
          width: 24px;
          height: 24px;
        }

        .pm10-source-text h3 {
          color: white;
          font-weight: bold;
          margin-bottom: 4px;
          font-size: 16px;
        }

        .pm10-source-text p {
          color: rgb(209, 213, 219);
          font-size: 14px;
        }

        @media (max-width: 768px) {
          .pm10-header h1 {
            font-size: 24px;
          }

          .pm10-header h2 {
            font-size: 20px;
          }
        }
      `}</style>
    </div>
  );
};

export default PM10SourcesComponent;
