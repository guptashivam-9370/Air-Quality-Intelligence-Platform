import React from 'react';

const OzoneSourcesComponent = () => {
  const sources = [
    {
      id: 1,
      title: 'Refineries',
      description: 'Industrial processes emit substances.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path d="M4 21h16V9l-4 2V6l-4 2V6L4 9v12z" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    },
    {
      id: 2,
      title: 'Power plants',
      description: 'Emissions from burning fossil fuels.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path d="M4 21h16V9l-4 2V6l-4 2V6L4 9v12z" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    },
    {
      id: 3,
      title: 'Paints evaporation',
      description: 'VOCs from paints react with sunlight.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path d="M3 17v-4a4 4 0 014-4h2a4 4 0 014 4v4M7 17h6m-3 0v4m-4-4v4m8-16h4m-2-2v4m4 2h4m-2-2v4m-6 2a6 6 0 00-12 0" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    },
    {
      id: 4,
      title: 'Transportation',
      description: 'Diesel operated vehicles react and form ozone.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path d="M5 17l1.5-5h11L19 17H5zM7 17v2a1 1 0 001 1h2a1 1 0 001-1v-2m4 0v2a1 1 0 001 1h2a1 1 0 001-1v-2" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    }
  ];

  return (
    <div className="o3-container">
      <div className="o3-content">
        <div className="o3-header">
          <h1>Uncovering the Sources of Ozone (O₃):</h1>
          <h2>Where Does It Come From?</h2>
        </div>

        <div className="o3-sources-grid">
          {sources.map(source => (
            <div key={source.id} className="o3-source-card">
              <div className="o3-source-content">
                <div className="o3-source-icon">{source.icon}</div>
                <div className="o3-source-text">
                  <h3>{source.title}</h3>
                  <p>{source.description}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <style jsx>{`
        .o3-container {
          position: relative;
          width: 100%;
          background-color: rgba(0, 0, 0, 0.5);
          background-image: url('https://www.aqi.in/media/sections/pollutant-sources/backgrounds/o3.webp');
          background-size: cover;
          background-position: center;
          background-blend-mode: overlay;
        }

        .o3-content {
          max-width: 1200px;
          margin: 0 auto;
          padding: 48px 16px;
        }

        .o3-header {
          max-width: 900px;
          margin: 0 auto 48px;
          text-align: center;
        }

        .o3-header h1 {
          font-size: 32px;
          font-weight: bold;
          color: white;
          margin-bottom: 16px;
        }

        .o3-header h2 {
          font-size: 28px;
          font-weight: bold;
          color: white;
        }

        .o3-sources-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
          gap: 24px;
        }

        .o3-source-card {
          background-color: rgba(51, 51, 51, 0.7);
          border-radius: 8px;
          padding: 16px;
          backdrop-filter: blur(4px);
          border: 1px solid rgba(119, 119, 119, 0.7);
        }

        .o3-source-content {
          display: flex;
          align-items: flex-start;
        }

        .o3-source-icon {
          color: white;
          margin-right: 12px;
          flex-shrink: 0;
        }

        .o3-source-icon svg {
          width: 24px;
          height: 24px;
        }

        .o3-source-text h3 {
          color: white;
          font-weight: bold;
          margin-bottom: 4px;
          font-size: 16px;
        }

        .o3-source-text p {
          color: rgb(209, 213, 219);
          font-size: 14px;
        }

        @media (max-width: 768px) {
          .o3-header h1 {
            font-size: 24px;
          }

          .o3-header h2 {
            font-size: 20px;
          }
        }
      `}</style>
    </div>
  );
};

export default OzoneSourcesComponent;
