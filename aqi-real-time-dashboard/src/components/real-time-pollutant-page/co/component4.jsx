import React from 'react';

const COSourcesComponent = () => {
  const sources = [
    {
      id: 1,
      title: 'Biomass burning',
      description: 'Burning wood, crop residues, and organic materials.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path d="M12 3c2 1 3 3 3 5s-1 3-3 3-3-1-3-3 1-4 3-5zm0 9v9" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    },
    {
      id: 2,
      title: 'Power plants',
      description: 'Emissions from burning fossil fuels in power plants.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path d="M3 21h18V9l-6 3V6l-6 3V6L3 9v12z" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    },
    {
      id: 3,
      title: 'Fireplaces',
      description: 'Emissions from burning wood or coal in fireplaces.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path d="M4 21h16V9l-4 2V6l-4 2V6L4 9v12z" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    },
    {
      id: 4,
      title: 'Road traffic',
      description: 'Overall contributions from vehicles exhaust.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path d="M4 17h16l1-5-1-5H4L3 12l1 5zM6 17v2a1 1 0 001 1h2a1 1 0 001-1v-2M14 17v2a1 1 0 001 1h2a1 1 0 001-1v-2" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    },
    {
      id: 5,
      title: 'Vehicles engines',
      description: 'Exhaust emissions from cars and trucks.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path d="M4 17h16M6 9h12l1.5 5H4.5L6 9zM7 17v2a1 1 0 001 1h2a1 1 0 001-1v-2m4 0v2a1 1 0 001 1h2a1 1 0 001-1v-2" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    }
  ];

  return (
    <div className="co-container">
      <div className="co-content">
        <div className="co-header">
          <h1>Uncovering the Sources of Carbon Monoxide (CO):</h1>
          <h2>Where Does It Come From?</h2>
        </div>

        <div className="co-sources-grid">
          {sources.map(source => (
            <div key={source.id} className="co-source-card">
              <div className="co-source-content">
                <div className="co-source-icon">{source.icon}</div>
                <div className="co-source-text">
                  <h3>{source.title}</h3>
                  <p>{source.description}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <style jsx>{`
        .co-container {
          position: relative;
          width: 100%;
          background-color: rgba(0, 0, 0, 0.5);
          background-image: url('https://www.aqi.in/media/sections/pollutant-sources/backgrounds/co.webp');
          background-size: cover;
          background-position: center;
          background-blend-mode: overlay;
        }

        .co-content {
          max-width: 1200px;
          margin: 0 auto;
          padding: 48px 16px;
        }

        .co-header {
          max-width: 800px;
          margin: 0 auto 48px;
          text-align: center;
        }

        .co-header h1 {
          font-size: 32px;
          font-weight: bold;
          color: white;
          margin-bottom: 16px;
        }

        .co-header h2 {
          font-size: 28px;
          font-weight: bold;
          color: white;
        }

        .co-sources-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
          gap: 24px;
        }

        .co-source-card {
          background-color: rgba(51, 51, 51, 0.7);
          border-radius: 8px;
          padding: 16px;
          backdrop-filter: blur(4px);
          border: 1px solid rgba(119, 119, 119, 0.7);
        }

        .co-source-content {
          display: flex;
          align-items: flex-start;
        }

        .co-source-icon {
          color: white;
          margin-right: 12px;
          flex-shrink: 0;
        }

        .co-source-icon svg {
          width: 24px;
          height: 24px;
        }

        .co-source-text h3 {
          color: white;
          font-weight: bold;
          margin-bottom: 4px;
          font-size: 16px;
        }

        .co-source-text p {
          color: rgb(209, 213, 219);
          font-size: 14px;
        }

        @media (max-width: 768px) {
          .co-header h1 {
            font-size: 24px;
          }

          .co-header h2 {
            font-size: 20px;
          }
        }
      `}</style>
    </div>
  );
};

export default COSourcesComponent;
