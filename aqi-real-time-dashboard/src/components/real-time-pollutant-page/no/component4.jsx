import React from 'react';

const NitricOxideSourcesComponent = () => {
  const sources = [
    {
      id: 1,
      title: 'Combustion Engines',
      description: 'High-temperature combustion in engines releases NO as a byproduct.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path d="M5 17l1.5-5h11L19 17H5zM7 17v2a1 1 0 001 1h2a1 1 0 001-1v-2m4 0v2a1 1 0 001 1h2a1 1 0 001-1v-2" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    },
    {
      id: 2,
      title: 'Power Stations',
      description: 'NO is emitted during fossil fuel combustion at power plants.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path d="M4 21h16V9l-4 2V6l-4 2V6L4 9v12z" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    },
    {
      id: 3,
      title: 'Aircraft Emissions',
      description: 'Jet engines release nitric oxide at high altitudes.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path d="M2 16l18-5-18-5v4l14 1-14 1v4z" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    },
    {
      id: 4,
      title: 'Waste Incineration',
      description: 'Burning of waste materials contributes to NO emissions.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path d="M9 3v2H5v14h14V5h-4V3H9zM9 9h6m-6 4h6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    },
    {
      id: 5,
      title: 'Industrial Boilers',
      description: 'NO is produced during fuel combustion in industrial heating systems.',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path d="M3 17V7a2 2 0 012-2h14a2 2 0 012 2v10a4 4 0 01-4 4H7a4 4 0 01-4-4z" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      )
    }
  ];

  return (
    <div className="no-container">
      <div className="no-content">
        <div className="no-header">
          <h1>Uncovering the Sources of Nitric Oxide (NO):</h1>
          <h2>Where Does It Come From?</h2>
        </div>

        <div className="no-sources-grid">
          {sources.map(source => (
            <div key={source.id} className="no-source-card">
              <div className="no-source-content">
                <div className="no-source-icon">{source.icon}</div>
                <div className="no-source-text">
                  <h3>{source.title}</h3>
                  <p>{source.description}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <style jsx>{`
        .no-container {
          position: relative;
          width: 100%;
          background-color: rgba(0, 0, 0, 0.5);
          background-image: url('https://airqualitynews.com/wp-content/uploads/2017/07/Drax.jpg');
          background-size: cover;
          background-position: center;
          background-blend-mode: overlay;
        }

        .no-content {
          max-width: 1200px;
          margin: 0 auto;
          padding: 48px 16px;
        }

        .no-header {
          max-width: 900px;
          margin: 0 auto 48px;
          text-align: center;
        }

        .no-header h1 {
          font-size: 32px;
          font-weight: bold;
          color: white;
          margin-bottom: 16px;
        }

        .no-header h2 {
          font-size: 28px;
          font-weight: bold;
          color: white;
        }

        .no-sources-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
          gap: 24px;
        }

        .no-source-card {
          background-color: rgba(51, 51, 51, 0.7);
          border-radius: 8px;
          padding: 16px;
          backdrop-filter: blur(4px);
          border: 1px solid rgba(119, 119, 119, 0.7);
        }

        .no-source-content {
          display: flex;
          align-items: flex-start;
        }

        .no-source-icon {
          color: white;
          margin-right: 12px;
          flex-shrink: 0;
        }

        .no-source-icon svg {
          width: 24px;
          height: 24px;
        }

        .no-source-text h3 {
          color: white;
          font-weight: bold;
          margin-bottom: 4px;
          font-size: 16px;
        }

        .no-source-text p {
          color: rgb(209, 213, 219);
          font-size: 14px;
        }

        @media (max-width: 768px) {
          .no-header h1 {
            font-size: 24px;
          }

          .no-header h2 {
            font-size: 20px;
          }
        }
      `}</style>
    </div>
  );
};

export default NitricOxideSourcesComponent;
