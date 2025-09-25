import React from 'react';

const PM25SourcesComponent = () => {
  const sources = [
    {
      id: 1,
      title: 'Windblown Dust',
      description: 'Daily activities like construction or other practices',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M9.59 4.59A2 2 0 1 1 11 8H2m10.59 11.41A2 2 0 1 0 14 16H2m15.73-8.27A2.5 2.5 0 1 1 19.5 12H2" />
          <circle cx="12" cy="12" r="3" />
        </svg>
      )
    },
    {
      id: 2,
      title: 'Home-related emission',
      description: 'Household activities, such as cooking and heating',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3" />
          <path d="M12 8c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2z" />
          <path d="M12 8v8" />
          <path d="M9 11h6" />
        </svg>
      )
    },
    {
      id: 3,
      title: "Factories and industries' emission",
      description: 'Regular operations in factories and industries',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M20 10V7.5L14 4v3.5" />
          <rect x="14" y="7.5" width="6" height="11" />
          <rect x="6" y="11.5" width="8" height="7" />
          <path d="M6 18.5v-7l-6 3v4" />
          <path d="M20 18.5v-3" />
          <path d="M16 10.5v3" />
          <path d="M10 14.5v3" />
        </svg>
      )
    },
    {
      id: 4,
      title: 'Power plants generation',
      description: 'Emission from Routine energy production in power plants',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 2a3 3 0 0 0-3 3v1h6V5a3 3 0 0 0-3-3z" />
          <rect x="6" y="6" width="12" height="3" />
          <path d="M6 9v9a3 3 0 0 0 3 3h1" />
          <path d="M18 9v9a3 3 0 0 1-3 3h-1" />
          <path d="M10 12h4" />
          <path d="M10 15h4" />
          <path d="M10 18h4" />
        </svg>
      )
    },
    {
      id: 5,
      title: 'Landfill fires',
      description: 'Fires in landfills, often caused by waste mismanagement',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M8 16s1.5-2 4-2 4 2 4 2" />
          <path d="M12 12a3 3 0 0 0 0-6 3 3 0 0 0 0 6z" />
          <path d="M12 21a9 9 0 0 0 9-9H3a9 9 0 0 0 9 9z" />
          <path d="M9 8l1.5-5" />
          <path d="M14 8l-1-5" />
        </svg>
      )
    },
    {
      id: 6,
      title: 'Transportation emission',
      description: 'Diesel operated Daily vehicles produces exhaust',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M7 17m-2 0a2 2 0 1 0 4 0a2 2 0 1 0 -4 0" />
          <path d="M17 17m-2 0a2 2 0 1 0 4 0a2 2 0 1 0 -4 0" />
          <path d="M5 17h-2v-6l2-5h9l4 5h1a2 2 0 0 1 2 2v4h-2m-4 0h-6" />
          <path d="M15 6l0-4" />
        </svg>
      )
    },
    {
      id: 7,
      title: 'Human-caused emissions',
      description: 'Common practices like open burning of waste or agricultural residues',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M9 12c0 2-2 2-2 4c0 1 1 2 2 2c2 0 3-1 3-3s-1-1-1-3c0-1-.5-2-2-2s-2 1-2 2c0 3 2 3 2 5" />
          <path d="M12 12c0-5 8-1 8-5c0-2-1-3-3-3c-1 0-2 .5-2 2c0 1 0 2 1 3c2 2 3 3 3 5s-2 3-4 3s-3-2-3-4" />
        </svg>
      )
    }
  ];

  return (
    <div className="pm25-container">
      <div className="pm25-content">
        <div className="pm25-header">
          <h1>Uncovering the Sources of Particulate Matter (PM2.5):</h1>
          <h2>Where Does It Come From?</h2>
        </div>

        <div className="pm25-sources-row">
          {sources.slice(0, 3).map(source => (
            <div key={source.id} className="pm25-source-card">
              <div className="pm25-source-content">
                <div className="pm25-source-icon">
                  {source.icon}
                </div>
                <div className="pm25-source-text">
                  <h3>{source.title}</h3>
                  <p>{source.description}</p>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="pm25-sources-row">
          {sources.slice(3, 6).map(source => (
            <div key={source.id} className="pm25-source-card">
              <div className="pm25-source-content">
                <div className="pm25-source-icon">
                  {source.icon}
                </div>
                <div className="pm25-source-text">
                  <h3>{source.title}</h3>
                  <p>{source.description}</p>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="pm25-sources-row-single">
          {sources.slice(6).map(source => (
            <div key={source.id} className="pm25-source-card">
              <div className="pm25-source-content">
                <div className="pm25-source-icon">
                  {source.icon}
                </div>
                <div className="pm25-source-text">
                  <h3>{source.title}</h3>
                  <p>{source.description}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <style jsx>{`
        .pm25-container {
          position: relative;
          width: 100%;
          
          background-color: rgba(0, 0, 0, 0.5);
          background-image: url('https://www.aqi.in/media/sections/pollutant-sources/backgrounds/pm25.webp');
          background-size: cover;
          background-position: center;
          background-blend-mode: overlay;
        }

        .pm25-content {
          max-width: 1200px;
          margin: 0 auto;
          padding: 48px 16px;
        }

        .pm25-header {
          max-width: 800px;
          margin: 0 auto 48px;
          text-align: center;
        }

        .pm25-header h1 {
          font-size: 32px;
          font-weight: bold;
          color: white;
          margin-bottom: 16px;
        }

        .pm25-header h2 {
          font-size: 28px;
          font-weight: bold;
          color: white;
        }

        .pm25-sources-row {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 24px;
          margin-bottom: 24px;
        }

        .pm25-sources-row-single {
          display: grid;
          grid-template-columns: 1fr;
          gap: 24px;
          max-width: 400px;
          margin: 0 auto;
        }

        .pm25-source-card {
          background-color: rgba(51, 51, 51, 0.7);
          border-radius: 8px;
          padding: 16px;
          backdrop-filter: blur(4px);
          border: 1px solid rgba(119, 119, 119, 0.7);
        }

        .pm25-source-content {
          display: flex;
          align-items: flex-start;
        }

        .pm25-source-icon {
          color: white;
          margin-right: 12px;
          flex-shrink: 0;
        }

        .pm25-source-text h3 {
          color: white;
          font-weight: bold;
          margin-bottom: 4px;
          font-size: 16px;
        }

        .pm25-source-text p {
          color: rgb(209, 213, 219);
          font-size: 14px;
        }

        @media (max-width: 768px) {
          .pm25-sources-row {
            grid-template-columns: 1fr;
          }

          .pm25-header h1 {
            font-size: 24px;
          }

          .pm25-header h2 {
            font-size: 20px;
          }

         
        }
      `}</style>
    </div>
  );
};

export default PM25SourcesComponent;